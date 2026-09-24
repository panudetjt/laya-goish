// Laya forward kernels for the GPU sidecar PoC (RTX 5060 Ti, sm_120).
//
// Numerics mirror the CPU engine (src/tensor.rs + src/models/laya/model.rs):
// - F16 weights convert to F32 with the same bit-exact routine as the
//   engine's gguf::half.
// - Accumulation is F32 throughout (no tensor cores, no F16 accumulate).
// - LayerNorm keeps the CPU's sequential mean/var summation order.
// - Attention keeps the CPU's serial window order: max, then a serial sum
//   of exponentials, then per-element p*v accumulation. expf is recomputed
//   per pass (deterministic) so the summation order matches exactly.
// - MatMul reduction order differs from the AVX2 kernel (tiled K loop) and
//   gelu/softmax use CUDA math lib tanhf/expf vs libm — this is the declared
//   drift: zero choice flips, |dP| <= 0.01, |dS| <= 0.05.
typedef unsigned short u16;
typedef unsigned int u32;

// Bit-exact port of gguf::half (F16 -> F32), including subnormals/inf/nan.
__device__ static float h2f(unsigned short h) {
  u32 sign = ((u32)(h & 0x8000)) << 16;
  u32 e = (h >> 10) & 31;
  u32 m = h & 1023;
  if (e == 0) {
    if (m == 0) return __int_as_float(sign);
    float v = (float)m * (1.0f / 16777216.0f);
    return sign ? -v : v;
  }
  if (e == 31) return __int_as_float(sign | 0x7f800000u | (m << 13));
  return __int_as_float(sign | ((e + 112) << 23) | (m << 13));
}

__device__ static float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

// ---------------------------------------------------------------------------
// out[M,N] = x[M,K] . W[N,K]^T (+ bias[N]); W rows are F16.
#define TILE 16
extern "C" __global__ void matmul_f16(
    float* __restrict__ out,
    const float* __restrict__ x,
    const unsigned short* __restrict__ w,
    const float* __restrict__ bias, // NULL = none
    int M, int N, int K,
    int wf) // weight format: 1 = F16 rows, 0 = F32 rows (scorer_fc2)
{
  __shared__ float xs[TILE][TILE];
  __shared__ float ws[TILE][TILE];
  int n0 = blockIdx.x * TILE + threadIdx.x;
  int m0 = blockIdx.y * TILE + threadIdx.y;
  float acc = 0.f;
  for (int kt = 0; kt < K; kt += TILE) {
    int kx = kt + threadIdx.x;
    int ky = kt + threadIdx.y;
    xs[threadIdx.y][threadIdx.x] = (m0 < M && kx < K) ? x[(size_t)m0 * K + kx] : 0.f;
    if (n0 < N && ky < K) {
      const size_t wi = (size_t)n0 * K + ky;
      ws[threadIdx.y][threadIdx.x] =
          wf ? h2f(w[wi]) : ((const float*)w)[wi];
    } else {
      ws[threadIdx.y][threadIdx.x] = 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < TILE; q++)
      acc += xs[threadIdx.y][q] * ws[q][threadIdx.x];
    __syncthreads();
  }
  if (m0 < M && n0 < N) {
    if (bias) acc += bias[n0];
    out[(size_t)m0 * N + n0] = acc;
  }
}

// ---------------------------------------------------------------------------
// LayerNorm over `rows` rows of width d, one thread per row, sequential
// reductions in the CPU engine's summation order.
extern "C" __global__ void layernorm(
    float* __restrict__ o,
    const float* __restrict__ a,
    const float* __restrict__ w, // NULL = 1
    const float* __restrict__ b, // NULL = 0
    int rows, int d, float eps)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* x = a + (size_t)r * d;
  float* y = o + (size_t)r * d;
  float mean = 0.f;
  for (int j = 0; j < d; j++) mean += x[j];
  mean /= (float)d;
  float var = 0.f;
  for (int j = 0; j < d; j++) { float z = x[j] - mean; var += z * z; }
  var /= (float)d;
  float scale = 1.0f / sqrtf(var + eps);
  for (int j = 0; j < d; j++) {
    float t = (x[j] - mean) * scale;
    y[j] = t * (w ? w[j] : 1.0f) + (b ? b[j] : 0.0f);
  }
}

// ---------------------------------------------------------------------------
// Attention. One warp per (head, query position). qkv rows are
// [q | k | v] each of width d. RoPE is applied on load exactly as the CPU
// engine's attention(): for jj < hd/2,
//   a[jj]    = u*cos[p*hd+jj]        - v*sin[p*hd+jj]
//   a[jj+h2] = v*cos[p*hd+jj+h2]     + u*sin[p*hd+jj+h2]
// The softmax keeps the CPU's serial order: lane 0 walks j ascending for
// max, serial sum of expf, per-element divide; the p*v accumulation walks
// j ascending per output element (same order as the CPU loops). Only the
// q.k dot reduction order differs (declared drift).
__device__ static float rope_load(
    const float* __restrict__ qkv, int p, int part, int h, int jj,
    int d, int hd, const float* __restrict__ cos, const float* __restrict__ sin)
{
  const float* a = qkv + (size_t)p * 3 * d + part * d + h * hd;
  // RoPE applies to q (part 0) and k (part 1) only; v (part 2) is raw,
  // matching the CPU engine's attention().
  if (!cos || part == 2) return a[jj];
  int half = hd / 2;
  if (jj < half) {
    float u = a[jj], v = a[jj + half];
    return u * cos[p * hd + jj] - v * sin[p * hd + jj];
  }
  int j = jj - half;
  float u = a[j], v = a[j + half];
  return v * cos[p * hd + j + half] + u * sin[p * hd + j + half];
}

extern "C" __global__ void attention(
    float* __restrict__ out,        // [s, d]
    const float* __restrict__ qkv,  // [s, 3d]
    const float* __restrict__ cos,  // NULL = no rope
    const float* __restrict__ sin,
    int s, int d, int hd, float scale,
    int window,                     // 0 = full attention
    int heads)
{
  extern __shared__ float sh[];     // [warps per block][s]
  int lane = threadIdx.x & 31;
  int wid = (int)threadIdx.x >> 5;
  int warps_per_block = (int)blockDim.x >> 5;
  int warp = blockIdx.x * warps_per_block + wid;
  int total = heads * s;
  if (warp >= total) return;
  int h = warp / s;
  int i = warp - h * s;
  float* score = sh + wid * s;
  int lo = 0, hi = s;
  if (window > 0) {
    lo = i - window; if (lo < 0) lo = 0;
    hi = i + window + 1; if (hi > s) hi = s;
  }
  // Scores: lane-strided over j, serial dot over hd (roped q,k on the fly).
  for (int j = lo + lane; j < hi; j += 32) {
    float dot = 0.f;
    for (int z = 0; z < hd; z++)
      dot += rope_load(qkv, i, 0, h, z, d, hd, cos, sin) *
             rope_load(qkv, j, 1, h, z, d, hd, cos, sin);
    score[j] = dot * scale;
  }
  __syncwarp();
  // Softmax in the CPU's serial order (lane 0).
  if (lane == 0) {
    float m = -INFINITY;
    for (int j = lo; j < hi; j++) if (score[j] > m) m = score[j];
    float sum = 0.f;
    for (int j = lo; j < hi; j++) score[j] = expf(score[j] - m);
    for (int j = lo; j < hi; j++) sum += score[j];
    for (int j = lo; j < hi; j++) score[j] /= sum;
  }
  __syncwarp();
  // p * v accumulation: each lane owns strided z; j walks ascending, which
  // preserves the CPU's per-element accumulation order exactly.
  for (int z = lane; z < hd; z += 32) {
    float acc = 0.f;
    for (int j = lo; j < hi; j++)
      acc += score[j] * rope_load(qkv, j, 2, h, z, d, hd, cos, sin);
    out[(size_t)i * d + h * hd + z] = acc;
  }
}

// ---------------------------------------------------------------------------
// h[s,d] = emb[ids[i]] (F16 -> F32)
extern "C" __global__ void embed_gather(
    float* __restrict__ out,
    const unsigned short* __restrict__ emb,
    const int* __restrict__ ids,
    int s, int d)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= s * d) return;
  int i = k / d, j = k - i * d;
  out[k] = h2f(emb[(size_t)ids[i] * d + j]);
}

// m[s,f] = gelu(up[:, :f]) * up[:, f:]  (GeGLU, trunk MLP)
extern "C" __global__ void geglu(
    float* __restrict__ m,
    const float* __restrict__ up,
    int s, int f)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= s * f) return;
  int i = k / f, j = k - i * f;
  m[k] = gelu(up[(size_t)i * 2 * f + j]) * up[(size_t)i * 2 * f + f + j];
}

extern "C" __global__ void relu_inplace(float* __restrict__ x, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) x[k] = fmaxf(x[k], 0.f);
}

extern "C" __global__ void gelu_inplace(float* __restrict__ x, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) x[k] = gelu(x[k]);
}

extern "C" __global__ void add_inplace(float* __restrict__ x, const float* __restrict__ y, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) x[k] += y[k];
}

// h[i,:] += type_emb[qtype,:] (F16 -> F32)
extern "C" __global__ void add_typeemb(
    float* __restrict__ h,
    const unsigned short* __restrict__ te,
    int qtype, int s, int d)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= s * d) return;
  int j = k % d;
  h[k] += h2f(te[(size_t)qtype * d + j]);
}

// out[r,:] = h[markers[r],:] for the decision heads.
extern "C" __global__ void gather_rows(
    float* __restrict__ out,
    const float* __restrict__ h,
    const int* __restrict__ markers,
    int k, int d)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= k * d) return;
  int r = t / d, j = t - r * d;
  out[t] = h[(size_t)markers[r] * d + j];
}
