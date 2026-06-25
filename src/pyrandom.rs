//! CPython-compatible `random.Random` (Mersenne Twister MT19937).
//!
//! Bit-exact reimplementation of CPython's `_randommodule.c` + the
//! `random.Random.seed` Python wrapper, so map/comet generation from a seed
//! matches the Kaggle env exactly. Covers both seed kinds the env uses:
//! - int seed: `random.Random(seed)` (e.g. the episode seed).
//! - str seed: `random.Random("orbit_wars-comet-{seed}-{step+1}")`, which goes
//!   through `int.from_bytes(a + sha512(a).digest(), 'big')`.

use sha2::{Digest, Sha512};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct PyRandom {
    mt: [u32; N],
    mti: usize,
}

impl PyRandom {
    /// `random.Random(n)` for a non-negative integer seed.
    pub fn from_int(n: u64) -> Self {
        let key = int_to_mt_key(&n.to_be_bytes());
        Self::from_key(&key)
    }

    /// `random.Random(s)` for a string seed (version-2 hashing path):
    /// `a = int.from_bytes(s_bytes + sha512(s_bytes).digest(), 'big')`.
    pub fn from_str(s: &str) -> Self {
        let mut bytes = s.as_bytes().to_vec();
        let digest = Sha512::digest(s.as_bytes());
        bytes.extend_from_slice(&digest);
        let key = int_to_mt_key(&bytes);
        Self::from_key(&key)
    }

    fn from_key(key: &[u32]) -> Self {
        let mut r = PyRandom { mt: [0; N], mti: N + 1 };
        r.init_by_array(key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = (1812433253u32
                .wrapping_mul(prev ^ (prev >> 30)))
                .wrapping_add(i as u32);
        }
        self.mti = N;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {
        self.init_genrand(19650218);
        let key_length = init_key.len();
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = if N > key_length { N } else { key_length };
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1664525)))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
            k -= 1;
        }
        for _ in 0..(N - 1) {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    pub fn genrand_uint32(&mut self) -> u32 {
        if self.mti >= N {
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] =
                    self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// genrand_res53: a 53-bit float in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_uint32() >> 5) as f64; // 27 bits
        let b = (self.genrand_uint32() >> 6) as f64; // 26 bits
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// CPython `getrandbits(k)` for 1 <= k <= 64.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        debug_assert!(k >= 1 && k <= 64);
        let words = (k - 1) / 32 + 1;
        let mut result: u64 = 0;
        let mut kk = k as i64;
        for i in 0..words {
            let mut r = self.genrand_uint32();
            if kk < 32 {
                r >>= (32 - kk) as u32;
            }
            result |= (r as u64) << (32 * i);
            kk -= 32;
        }
        result
    }

    /// `_randbelow_with_getrandbits(n)`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = bit_length(n);
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `randint(a, b)` inclusive.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        let width = (b - a + 1) as u64;
        a + self.randbelow(width) as i64
    }

    /// `uniform(a, b)` = a + (b - a) * random().
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }
}

#[inline]
fn bit_length(n: u64) -> u32 {
    if n == 0 {
        0
    } else {
        u64::BITS - n.leading_zeros()
    }
}

/// Convert a big-endian byte representation of a non-negative integer into the
/// little-endian u32 key array CPython's `random_seed` feeds to init_by_array
/// (the words of abs(n), least-significant first; [0] when n == 0).
fn int_to_mt_key(be: &[u8]) -> Vec<u32> {
    // Strip leading zero bytes (they don't affect the integer value).
    let start = be.iter().position(|&b| b != 0).unwrap_or(be.len());
    let stripped = &be[start..];
    if stripped.is_empty() {
        return vec![0];
    }
    // Little-endian bytes, then pack into u32 words.
    let mut le: Vec<u8> = stripped.iter().rev().cloned().collect();
    while le.len() % 4 != 0 {
        le.push(0);
    }
    le.chunks(4)
        .map(|c| {
            (c[0] as u32)
                | ((c[1] as u32) << 8)
                | ((c[2] as u32) << 16)
                | ((c[3] as u32) << 24)
        })
        .collect()
}
