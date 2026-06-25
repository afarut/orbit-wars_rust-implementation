//! Pure geometry / continuous-collision primitives.
//!
//! Ported verbatim from `ow_sim/geometry.py`. To stay bit-identical with
//! CPython we route `x ** 2` and `sqrt` through `pymath` (system libm),
//! and keep `a * b` as plain multiplication exactly as the Python does.

use crate::pymath::{c_sqrt, sq2};

/// distance(p1, p2) = sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
#[inline]
pub fn distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    c_sqrt(sq2(ax - bx) + sq2(ay - by))
}

/// Minimum distance from point p to line segment v-w.
pub fn point_to_segment_distance(
    px: f64,
    py: f64,
    vx: f64,
    vy: f64,
    wx: f64,
    wy: f64,
) -> f64 {
    let l2 = sq2(vx - wx) + sq2(vy - wy);
    if l2 == 0.0 {
        return distance(px, py, vx, vy);
    }
    let mut t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2;
    // Python: max(0, min(1, t))
    t = t.min(1.0).max(0.0);
    let proj_x = vx + t * (wx - vx);
    let proj_y = vy + t * (wy - vy);
    distance(px, py, proj_x, proj_y)
}

/// True iff a fleet moving A->B and a planet moving P0->P1 come within `r`
/// of each other for some t in [0, 1]. Both segments are linear over the tick.
pub fn swept_pair_hit(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    p0x: f64,
    p0y: f64,
    p1x: f64,
    p1y: f64,
    r: f64,
) -> bool {
    let d0x = ax - p0x;
    let d0y = ay - p0y;
    let dvx = (bx - ax) - (p1x - p0x);
    let dvy = (by - ay) - (p1y - p0y);
    let a = dvx * dvx + dvy * dvy;
    let b = 2.0 * (d0x * dvx + d0y * dvy);
    let c = d0x * d0x + d0y * d0y - r * r;
    if a < 1e-12 {
        return c <= 0.0;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return false;
    }
    let sq = c_sqrt(disc);
    let t1 = (-b - sq) / (2.0 * a);
    let t2 = (-b + sq) / (2.0 * a);
    t2 >= 0.0 && t1 <= 1.0
}
