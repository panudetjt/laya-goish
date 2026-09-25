//! Go-compatible float formatting for `J::state()`.
//!
//! The goish engine serialized non-integer state numbers with Go's
//! `strconv.FormatFloat(n, 'g', 6, 64)` (goish/src/strconv, a port of Go
//! 1.25 strconv/ftoa.go + decimal.go). Those exact strings feed the
//! tokenizer, so the std port must reproduce them byte-for-byte. This file
//! is that goish strconv module adapted to std (Goish MIT, see
//! LICENSES/Goish-MIT.txt); the algorithm and constants are unchanged.
#![allow(non_snake_case, dead_code)]

const FLOAT32INFO: FloatInfo = FloatInfo {
    mantbits: 32 - 9,
    expbits: 8,
    bias: -127,
};
const FLOAT64INFO: FloatInfo = FloatInfo {
    mantbits: 52,
    expbits: 11,
    bias: -1023,
};
struct FloatInfo {
    mantbits: u32,
    expbits: u32,
    bias: i32,
}
/// `strconv.FormatFloat` (Go semantics; slow path only).
pub fn format_float(f: f64, fmt_b: u8, prec: i32, bit_size: i32) -> String {
    let bytes = generic_ftoa(f, fmt_b, prec, bit_size);
    String::from_utf8(bytes).unwrap()
}
fn generic_ftoa(val: f64, fmt_b: u8, prec: i32, bit_size: i32) -> Vec<u8> {
    let mut dst = Vec::with_capacity(prec as usize + 4);
    let (bits, flt) = match bit_size {
        32 => ((val as f32).to_bits() as u64, &FLOAT32INFO),
        64 => (val.to_bits(), &FLOAT64INFO),
        _ => panic!("strconv: illegal AppendFloat/FormatFloat bitSize"),
    };
    let neg = (bits >> (flt.expbits + flt.mantbits)) != 0;
    let mut exp = ((bits >> flt.mantbits) & ((1u64 << flt.expbits) - 1)) as i32;
    let mut mant = bits & ((1u64 << flt.mantbits) - 1);
    if exp == (1i32 << flt.expbits) - 1 {
        let s: &[u8] = if mant != 0 {
            b"NaN"
        } else if neg {
            b"-Inf"
        } else {
            b"+Inf"
        };
        dst.extend_from_slice(s);
        return dst;
    } else if exp == 0 {
        exp += 1;
    } else {
        mant |= 1u64 << flt.mantbits;
    }
    exp += flt.bias;
    big_ftoa(&mut dst, prec, fmt_b, neg, mant, exp, flt)
}
fn big_ftoa(
    dst: &mut Vec<u8>,
    mut prec: i32,
    fmt_b: u8,
    neg: bool,
    mant: u64,
    exp: i32,
    flt: &FloatInfo,
) -> Vec<u8> {
    let mut d = Decimal::new();
    d.Assign(mant);
    d.Shift(exp - flt.mantbits as i32);
    let shortest = prec < 0;
    if shortest {
        round_shortest(&mut d, mant, exp, flt);
        match fmt_b {
            b'e' | b'E' => prec = (d.nd - 1).max(0),
            b'f' => prec = (d.nd - d.dp).max(0),
            b'g' | b'G' => prec = d.nd,
            _ => {}
        }
    } else {
        match fmt_b {
            b'e' | b'E' => d.Round(prec + 1),
            b'f' => d.Round(d.dp + prec),
            b'g' | b'G' => {
                if prec == 0 {
                    prec = 1;
                }
                d.Round(prec);
            }
            _ => {}
        }
    }
    let digs = DecimalSlice {
        nd: d.nd,
        dp: d.dp,
        d: &d.d,
    };
    format_digits(dst, shortest, neg, digs, prec, fmt_b)
}
fn format_digits(
    dst: &mut Vec<u8>,
    shortest: bool,
    neg: bool,
    digs: DecimalSlice<'_>,
    mut prec: i32,
    fmt_b: u8,
) -> Vec<u8> {
    let (nd, dp) = (digs.nd, digs.dp);
    match fmt_b {
        b'e' | b'E' => fmt_e(dst, neg, digs, prec, fmt_b),
        b'f' => fmt_f(dst, neg, digs, prec),
        b'g' | b'G' => {
            // trailing fractional zeros in 'e' form will be trimmed.
            let mut eprec = prec;
            if eprec > nd && nd >= dp {
                eprec = nd;
            }
            // %e is used if the exponent from the conversion is less than -4
            // or greater than or equal to the precision.
            if shortest {
                eprec = 6;
            }
            let exp = dp - 1;
            if exp < -4 || exp >= eprec {
                let mut p = prec;
                if p > nd {
                    p = nd;
                }
                let next_fmt = (fmt_b as i32 + (b'e' as i32 - b'g' as i32)) as u8;
                return fmt_e(dst, neg, digs, p - 1, next_fmt);
            }
            if prec > dp {
                prec = nd;
            }
            fmt_f(dst, neg, digs, (prec - dp).max(0))
        }
        _ => {
            dst.push(b'%');
            dst.push(fmt_b);
            dst.clone()
        }
    }
}
fn round_shortest(d: &mut Decimal, mant: u64, exp: i32, flt: &FloatInfo) {
    if mant == 0 {
        d.nd = 0;
        return;
    }
    let minexp = flt.bias + 1;
    if exp > minexp && 332 * (d.dp - d.nd) >= 100 * (exp - flt.mantbits as i32) {
        return;
    }
    let mut upper = Decimal::new();
    upper.Assign(mant * 2 + 1);
    upper.Shift(exp - flt.mantbits as i32 - 1);
    let (mantlo, explo) = if mant > (1u64 << flt.mantbits) || exp == minexp {
        (mant - 1, exp)
    } else {
        (mant * 2 - 1, exp - 1)
    };
    let mut lower = Decimal::new();
    lower.Assign(mantlo * 2 + 1);
    lower.Shift(explo - flt.mantbits as i32 - 1);
    let inclusive = mant % 2 == 0;
    let mut upperdelta: u8 = 0;
    let mut ui: i32 = 0;
    loop {
        let mi = ui - upper.dp + d.dp;
        if mi >= d.nd {
            break;
        }
        let li = ui - upper.dp + lower.dp;
        let l: u8 = if li >= 0 && li < lower.nd {
            lower.d[li as usize]
        } else {
            b'0'
        };
        let m: u8 = if mi >= 0 { d.d[mi as usize] } else { b'0' };
        let u: u8 = if ui < upper.nd {
            upper.d[ui as usize]
        } else {
            b'0'
        };
        let okdown = l != m || (inclusive && li + 1 == lower.nd);
        match upperdelta {
            0 if m + 1 < u => {
                upperdelta = 2;
            }
            0 if m != u => {
                upperdelta = 1;
            }
            1 if m != b'9' || u != b'0' => {
                upperdelta = 2;
            }
            _ => {}
        }
        let okup = upperdelta > 0 && (inclusive || upperdelta > 1 || ui + 1 < upper.nd);
        if okdown && okup {
            d.Round(mi + 1);
            return;
        } else if okdown {
            d.RoundDown(mi + 1);
            return;
        } else if okup {
            d.RoundUp(mi + 1);
            return;
        }
        ui += 1;
    }
}
struct DecimalSlice<'a> {
    d: &'a [u8],
    nd: i32,
    dp: i32,
}
/// %e: -d.ddddde±dd
fn fmt_e(dst: &mut Vec<u8>, neg: bool, ds: DecimalSlice<'_>, prec: i32, fmt_b: u8) -> Vec<u8> {
    let (d, nd, dp) = (ds.d, ds.nd, ds.dp);
    if neg {
        dst.push(b'-');
    }
    let ch = if nd != 0 { d[0] } else { b'0' };
    dst.push(ch);
    if prec > 0 {
        dst.push(b'.');
        let m = nd.min(prec + 1);
        let mut i: i32 = 1;
        if i < m {
            dst.extend_from_slice(&d[i as usize..m as usize]);
            i = m;
        }
        while i <= prec {
            dst.push(b'0');
            i += 1;
        }
    }
    dst.push(fmt_b);
    let mut exp = dp - 1;
    if nd == 0 {
        exp = 0;
    }
    if exp < 0 {
        dst.push(b'-');
        exp = -exp;
    } else {
        dst.push(b'+');
    }
    if exp < 10 {
        dst.push(b'0');
        dst.push(b'0' + exp as u8);
    } else if exp < 100 {
        dst.push(b'0' + (exp / 10) as u8);
        dst.push(b'0' + (exp % 10) as u8);
    } else {
        dst.push(b'0' + (exp / 100) as u8);
        dst.push(b'0' + ((exp / 10) % 10) as u8);
        dst.push(b'0' + (exp % 10) as u8);
    }
    dst.clone()
}
/// %f: -ddddddd.ddddd
fn fmt_f(dst: &mut Vec<u8>, neg: bool, ds: DecimalSlice<'_>, prec: i32) -> Vec<u8> {
    let (d, nd, dp) = (ds.d, ds.nd, ds.dp);
    if neg {
        dst.push(b'-');
    }
    if dp > 0 {
        let m = nd.min(dp);
        dst.extend_from_slice(&d[..m as usize]);
        let mut k = m;
        while k < dp {
            dst.push(b'0');
            k += 1;
        }
    } else {
        dst.push(b'0');
    }
    if prec > 0 {
        dst.push(b'.');
        for i in 0..prec {
            let j = dp + i;
            let ch = if j >= 0 && j < nd {
                d[j as usize]
            } else {
                b'0'
            };
            dst.push(ch);
        }
    }
    dst.clone()
}
// ---------------------------------------------------------------------------
// Multiprecision decimal — port of Go 1.25 src/strconv/decimal.go (via
// goish/src/strconv/decimal.rs). Only assign and binary shifts; used by the
// float formatters above.
struct Decimal {
    d: [u8; 800], // digits, big-endian representation
    nd: i32,      // number of digits used
    dp: i32,      // decimal point
    neg: bool,    // negative flag
    trunc: bool,  // discarded nonzero digits beyond d[:nd]
}
impl Decimal {
    fn new() -> Self {
        Self {
            d: [0u8; 800],
            nd: 0,
            dp: 0,
            neg: false,
            trunc: false,
        }
    }
    fn Assign(&mut self, mut v: u64) {
        let mut buf = [0u8; 24];
        let mut n: usize = 0;
        while v > 0 {
            let v1 = v / 10;
            v -= 10 * v1;
            buf[n] = (v + b'0' as u64) as u8;
            n += 1;
            v = v1;
        }
        self.nd = 0;
        let mut k = n;
        while k > 0 {
            k -= 1;
            self.d[self.nd as usize] = buf[k];
            self.nd += 1;
        }
        self.dp = self.nd;
        trim(self);
    }
    fn Shift(&mut self, mut k: i32) {
        if self.nd == 0 {
            return;
        }
        if k > 0 {
            while k > MAX_SHIFT as i32 {
                left_shift(self, MAX_SHIFT);
                k -= MAX_SHIFT as i32;
            }
            left_shift(self, k as u32);
        } else if k < 0 {
            while k < -(MAX_SHIFT as i32) {
                right_shift(self, MAX_SHIFT);
                k += MAX_SHIFT as i32;
            }
            right_shift(self, (-k) as u32);
        }
    }
    fn Round(&mut self, nd: i32) {
        if nd < 0 || nd >= self.nd {
            return;
        }
        if should_round_up(self, nd) {
            self.RoundUp(nd);
        } else {
            self.RoundDown(nd);
        }
    }
    fn RoundDown(&mut self, nd: i32) {
        if nd < 0 || nd >= self.nd {
            return;
        }
        self.nd = nd;
        trim(self);
    }
    fn RoundUp(&mut self, nd: i32) {
        if nd < 0 || nd >= self.nd {
            return;
        }
        let mut i = nd - 1;
        while i >= 0 {
            let c = self.d[i as usize];
            if c < b'9' {
                self.d[i as usize] += 1;
                self.nd = i + 1;
                return;
            }
            i -= 1;
        }
        self.d[0] = b'1';
        self.nd = 1;
        self.dp += 1;
    }
}
fn trim(a: &mut Decimal) {
    while a.nd > 0 && a.d[(a.nd - 1) as usize] == b'0' {
        a.nd -= 1;
    }
    if a.nd == 0 {
        a.dp = 0;
    }
}
const MAX_SHIFT: u32 = 60;
fn right_shift(a: &mut Decimal, k: u32) {
    let mut r: i32 = 0;
    let mut w: i32 = 0;
    let mut n: u64 = 0;
    loop {
        if (n >> k) != 0 {
            break;
        }
        if r >= a.nd {
            if n == 0 {
                a.nd = 0;
                return;
            }
            while (n >> k) == 0 {
                n *= 10;
                r += 1;
            }
            break;
        }
        let c = a.d[r as usize] as u64;
        n = n * 10 + c - b'0' as u64;
        r += 1;
    }
    a.dp -= r - 1;
    let mask: u64 = (1u64 << k) - 1;
    while r < a.nd {
        let c = a.d[r as usize] as u64;
        let dig = n >> k;
        n &= mask;
        a.d[w as usize] = (dig + b'0' as u64) as u8;
        w += 1;
        n = n * 10 + c - b'0' as u64;
        r += 1;
    }
    while n > 0 {
        let dig = n >> k;
        n &= mask;
        if (w as usize) < a.d.len() {
            a.d[w as usize] = (dig + b'0' as u64) as u8;
            w += 1;
        } else if dig > 0 {
            a.trunc = true;
        }
        n *= 10;
    }
    a.nd = w;
    trim(a);
}
struct LeftCheat {
    delta: i32,
    cutoff: &'static [u8],
}
const LEFTCHEATS: &[LeftCheat] = &[
    LeftCheat {
        delta: 0,
        cutoff: b"",
    },
    LeftCheat {
        delta: 1,
        cutoff: b"5",
    },
    LeftCheat {
        delta: 1,
        cutoff: b"25",
    },
    LeftCheat {
        delta: 1,
        cutoff: b"125",
    },
    LeftCheat {
        delta: 2,
        cutoff: b"625",
    },
    LeftCheat {
        delta: 2,
        cutoff: b"3125",
    },
    LeftCheat {
        delta: 2,
        cutoff: b"15625",
    },
    LeftCheat {
        delta: 3,
        cutoff: b"78125",
    },
    LeftCheat {
        delta: 3,
        cutoff: b"390625",
    },
    LeftCheat {
        delta: 3,
        cutoff: b"1953125",
    },
    LeftCheat {
        delta: 4,
        cutoff: b"9765625",
    },
    LeftCheat {
        delta: 4,
        cutoff: b"48828125",
    },
    LeftCheat {
        delta: 4,
        cutoff: b"244140625",
    },
    LeftCheat {
        delta: 4,
        cutoff: b"1220703125",
    },
    LeftCheat {
        delta: 5,
        cutoff: b"6103515625",
    },
    LeftCheat {
        delta: 5,
        cutoff: b"30517578125",
    },
    LeftCheat {
        delta: 5,
        cutoff: b"152587890625",
    },
    LeftCheat {
        delta: 6,
        cutoff: b"762939453125",
    },
    LeftCheat {
        delta: 6,
        cutoff: b"3814697265625",
    },
    LeftCheat {
        delta: 6,
        cutoff: b"19073486328125",
    },
    LeftCheat {
        delta: 7,
        cutoff: b"95367431640625",
    },
    LeftCheat {
        delta: 7,
        cutoff: b"476837158203125",
    },
    LeftCheat {
        delta: 7,
        cutoff: b"2384185791015625",
    },
    LeftCheat {
        delta: 7,
        cutoff: b"11920928955078125",
    },
    LeftCheat {
        delta: 8,
        cutoff: b"59604644775390625",
    },
    LeftCheat {
        delta: 8,
        cutoff: b"298023223876953125",
    },
    LeftCheat {
        delta: 8,
        cutoff: b"1490116119384765625",
    },
    LeftCheat {
        delta: 9,
        cutoff: b"7450580596923828125",
    },
    LeftCheat {
        delta: 9,
        cutoff: b"37252902984619140625",
    },
    LeftCheat {
        delta: 9,
        cutoff: b"186264514923095703125",
    },
    LeftCheat {
        delta: 10,
        cutoff: b"931322574615478515625",
    },
    LeftCheat {
        delta: 10,
        cutoff: b"4656612873077392578125",
    },
    LeftCheat {
        delta: 10,
        cutoff: b"23283064365386962890625",
    },
    LeftCheat {
        delta: 10,
        cutoff: b"116415321826934814453125",
    },
    LeftCheat {
        delta: 11,
        cutoff: b"582076609134674072265625",
    },
    LeftCheat {
        delta: 11,
        cutoff: b"2910383045673370361328125",
    },
    LeftCheat {
        delta: 11,
        cutoff: b"14551915228366851806640625",
    },
    LeftCheat {
        delta: 12,
        cutoff: b"72759576141834259033203125",
    },
    LeftCheat {
        delta: 12,
        cutoff: b"363797880709171295166015625",
    },
    LeftCheat {
        delta: 12,
        cutoff: b"1818989403545856475830078125",
    },
    LeftCheat {
        delta: 13,
        cutoff: b"9094947017729282379150390625",
    },
    LeftCheat {
        delta: 13,
        cutoff: b"45474735088646411895751953125",
    },
    LeftCheat {
        delta: 13,
        cutoff: b"227373675443232059478759765625",
    },
    LeftCheat {
        delta: 13,
        cutoff: b"1136868377216160297393798828125",
    },
    LeftCheat {
        delta: 14,
        cutoff: b"5684341886080801486968994140625",
    },
    LeftCheat {
        delta: 14,
        cutoff: b"28421709430404007434844970703125",
    },
    LeftCheat {
        delta: 14,
        cutoff: b"142108547152020037174224853515625",
    },
    LeftCheat {
        delta: 15,
        cutoff: b"710542735760100185871124267578125",
    },
    LeftCheat {
        delta: 15,
        cutoff: b"3552713678800500929355621337890625",
    },
    LeftCheat {
        delta: 15,
        cutoff: b"17763568394002504646778106689453125",
    },
    LeftCheat {
        delta: 16,
        cutoff: b"88817841970012523233890533447265625",
    },
    LeftCheat {
        delta: 16,
        cutoff: b"444089209850062616169452667236328125",
    },
    LeftCheat {
        delta: 16,
        cutoff: b"2220446049250313080847263336181640625",
    },
    LeftCheat {
        delta: 16,
        cutoff: b"11102230246251565404236316680908203125",
    },
    LeftCheat {
        delta: 17,
        cutoff: b"55511151231257827021181583404541015625",
    },
    LeftCheat {
        delta: 17,
        cutoff: b"277555756156289135105907917022705078125",
    },
    LeftCheat {
        delta: 17,
        cutoff: b"1387778780781445675529539585113525390625",
    },
    LeftCheat {
        delta: 18,
        cutoff: b"6938893903907228377647697925567626953125",
    },
    LeftCheat {
        delta: 18,
        cutoff: b"34694469519536141888238489627838134765625",
    },
    LeftCheat {
        delta: 18,
        cutoff: b"173472347597680709441192448139190673828125",
    },
    LeftCheat {
        delta: 19,
        cutoff: b"867361737988403547205962240695953369140625",
    },
];
fn prefix_is_less_than(b: &[u8], s: &[u8]) -> bool {
    for i in 0..s.len() {
        if i >= b.len() {
            return true;
        }
        if b[i] != s[i] {
            return b[i] < s[i];
        }
    }
    false
}
fn left_shift(a: &mut Decimal, k: u32) {
    let mut delta = LEFTCHEATS[k as usize].delta;
    if prefix_is_less_than(&a.d[0..a.nd as usize], LEFTCHEATS[k as usize].cutoff) {
        delta -= 1;
    }
    let mut r: i32 = a.nd;
    let mut w: i32 = a.nd + delta;
    let mut n: u64 = 0;
    r -= 1;
    while r >= 0 {
        n += ((a.d[r as usize] - b'0') as u64) << k;
        let quo = n / 10;
        let rem = n - 10 * quo;
        w -= 1;
        if (w as usize) < a.d.len() {
            a.d[w as usize] = (rem + b'0' as u64) as u8;
        } else if rem != 0 {
            a.trunc = true;
        }
        n = quo;
        r -= 1;
    }
    while n > 0 {
        let quo = n / 10;
        let rem = n - 10 * quo;
        w -= 1;
        if (w as usize) < a.d.len() {
            a.d[w as usize] = (rem + b'0' as u64) as u8;
        } else if rem != 0 {
            a.trunc = true;
        }
        n = quo;
    }
    a.nd += delta;
    if a.nd as usize >= a.d.len() {
        a.nd = a.d.len() as i32;
    }
    a.dp += delta;
    trim(a);
}
fn should_round_up(a: &Decimal, nd: i32) -> bool {
    if nd < 0 || nd >= a.nd {
        return false;
    }
    let nd_u = nd as usize;
    if a.d[nd_u] == b'5' && nd + 1 == a.nd {
        // exactly halfway - round to even
        if a.trunc {
            return true;
        }
        return nd > 0 && (a.d[nd_u - 1] - b'0') % 2 != 0;
    }
    a.d[nd_u] >= b'5'
}
