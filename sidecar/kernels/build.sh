#!/bin/sh
# Build the sidecar CUDA kernels for the local GPU (sm_120, RTX 5060 Ti).
# Uses the pip-wheel nvcc from the project venv (nvidia-cuda-nvcc, CUDA 13.4)
# so no system CUDA toolkit is required. Produces laya.cubin (embedded cubin
# loaded through the driver API).
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
NVCC="${NVCC:-$(dirname "$DIR")/../.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc}"
ARCH="${ARCH:-sm_120}"
"$NVCC" -cubin -arch="$ARCH" -O3 "$DIR/laya.cu" -o "$DIR/laya.cubin"
echo "built $DIR/laya.cubin ($ARCH)"
