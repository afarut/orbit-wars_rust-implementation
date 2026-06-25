//! Bit-exact math wrappers matching CPython on the same platform.
//!
//! Verified over 200k random inputs vs CPython 3.12 (macOS/arm64 + Linux/glibc):
//! Rust's native `cos`, `atan2`, `ln`, `sqrt` are **bit-identical** to the
//! system libm (so we use them directly — no FFI). Only two need care:
//!   - `sin`: `f64::sin` (LLVM) differs from libm `sin` on ~0.4% of inputs →
//!     call libm `sin` via FFI.
//!   - `x ** 2`: CPython computes it as libm `pow(x, 2.0)`, which differs from
//!     `x * x` in the last ULP ~0.13% of the time; LLVM const-folds
//!     `pow(x, 2.0)` to `x * x`, so we call libm `pow` via FFI with a
//!     `black_box` exponent to block the fold.

// cos / atan2 / ln / sqrt: native == libm bit-for-bit on both platforms.
#[inline]
pub fn c_cos(x: f64) -> f64 {
    x.cos()
}

#[inline]
pub fn c_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

#[inline]
pub fn c_log(x: f64) -> f64 {
    x.ln()
}

#[inline]
pub fn c_sqrt(x: f64) -> f64 {
    x.sqrt()
}

// --- Bit-exact path (default): FFI libm for sin + pow(x,2.0) + log1p ------
#[cfg(not(feature = "fast_math"))]
mod exact {
    use std::hint::black_box;
    extern "C" {
        fn sin(x: f64) -> f64;
        fn pow(x: f64, y: f64) -> f64;
        fn log1p(x: f64) -> f64;
    }
    #[inline]
    pub fn c_sin(x: f64) -> f64 {
        unsafe { sin(x) }
    }
    /// libm `pow`; `black_box` blocks LLVM folding `pow(x, 2.0)` -> `x * x`.
    #[inline]
    pub fn c_pow(x: f64, y: f64) -> f64 {
        unsafe { pow(x, black_box(y)) }
    }
    /// Python `x ** 2` == libm `pow(x, 2.0)` (NOT `x * x`).
    #[inline]
    pub fn sq2(x: f64) -> f64 {
        c_pow(x, 2.0)
    }
    /// CPython `math.log1p` == libm `log1p`.
    #[inline]
    pub fn c_log1p(x: f64) -> f64 {
        unsafe { log1p(x) }
    }
}

// --- Fast path (fast_math): native math, ~3x faster, not ULP-exact ------
#[cfg(feature = "fast_math")]
mod fast {
    #[inline]
    pub fn c_sin(x: f64) -> f64 {
        x.sin()
    }
    #[inline]
    pub fn c_pow(x: f64, y: f64) -> f64 {
        x.powf(y)
    }
    #[inline]
    pub fn sq2(x: f64) -> f64 {
        x * x
    }
    #[inline]
    pub fn c_log1p(x: f64) -> f64 {
        x.ln_1p()
    }
}

#[cfg(not(feature = "fast_math"))]
pub use exact::{c_log1p, c_pow, c_sin, sq2};
#[cfg(feature = "fast_math")]
pub use fast::{c_log1p, c_pow, c_sin, sq2};
