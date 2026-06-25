//! Symmetric planet map generation — faithful port of `generate_planets` in
//! `ow_sim/mapgen.py`. The order of RNG calls (and the branches that skip
//! them) must match CPython exactly for bit-identical maps from a seed.
//!
//! Note the coordinate swap in the Python source: each group's first copy is
//! stored as `[id, -1, y, x, r, ships, prod]` (x and y swapped). We replicate
//! the literal row construction so positions match field-for-field.

use crate::geometry::distance;
use crate::pymath::{c_cos, c_log, c_sin};
use crate::pyrandom::PyRandom;
use crate::state::*;

use std::f64::consts::PI;

/// Build the 4 symmetric planet rows for a group, mirroring the Python literal
/// `[[id, -1, y, x, ...], [id+1, -1, B-x, y, ...], [id+2, -1, x, B-y, ...],
///   [id+3, -1, B-y, B-x, ...]]`.
fn group_rows(id0: i64, x: f64, y: f64, r: f64, ships: i64, prod: i64) -> [Planet; 4] {
    let b = BOARD_SIZE;
    let mk = |id: i64, px: f64, py: f64| Planet {
        id,
        owner: -1,
        x: px,
        y: py,
        radius: r,
        ships,
        production: prod,
    };
    [
        mk(id0, y, x),
        mk(id0 + 1, b - x, y),
        mk(id0 + 2, x, b - y),
        mk(id0 + 3, b - y, b - x),
    ]
}

pub fn generate_planets(rng: &mut PyRandom) -> Vec<Planet> {
    let mut planets: Vec<Planet> = Vec::new();
    let num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS);
    let mut id_counter: i64 = 0;

    // Phase 1: guaranteed static groups.
    let mut static_groups = 0i64;
    for _ in 0..5000 {
        if static_groups >= MIN_STATIC_GROUPS {
            break;
        }
        let prod = rng.randint(1, 5);
        let r = 1.0 + c_log(prod as f64);
        let angle = rng.uniform(0.0, PI / 2.0);
        let min_orbital = ROTATION_RADIUS_LIMIT - r;
        let max_orbital = (BOARD_SIZE - CENTER - r) / c_cos(angle).max(c_sin(angle));
        if min_orbital > max_orbital {
            continue;
        }
        let orbital_r = rng.uniform(min_orbital, max_orbital);
        let x = CENTER + orbital_r * c_cos(angle);
        let y = CENTER + orbital_r * c_sin(angle);

        if x + r > BOARD_SIZE || x - r < 0.0 || y + r > BOARD_SIZE || y - r < 0.0 {
            continue;
        }
        if (BOARD_SIZE - x) - r < 0.0 || (BOARD_SIZE - y) - r < 0.0 {
            continue;
        }
        if (x - CENTER) < r + 5.0 || (y - CENTER) < r + 5.0 {
            continue;
        }

        let ships = rng.randint(5, 99).min(rng.randint(5, 99));
        let temp = group_rows(id_counter, x, y, r, ships, prod);

        let mut valid = true;
        'outer: for tp in &temp {
            for p in &planets {
                if distance(p.x, p.y, tp.x, tp.y) < p.radius + tp.radius + PLANET_CLEARANCE {
                    valid = false;
                    break 'outer;
                }
            }
        }

        if valid {
            planets.extend(temp);
            id_counter += 4;
            static_groups += 1;
        }
    }

    // Phase 2: fill remaining groups with the normal random loop.
    let mut attempts = 0i64;
    let max_attempts = 5000i64;
    let mut has_orbiting = false;

    while (planets.len() as i64) < num_q1 * 4 || (!has_orbiting && attempts < max_attempts) {
        attempts += 1;
        if attempts >= max_attempts {
            break;
        }
        let prod = rng.randint(1, 5);
        let r = 1.0 + c_log(prod as f64);
        let x = rng.uniform(CENTER + 15.0, BOARD_SIZE - r - 5.0);
        let y = rng.uniform(CENTER + 15.0, BOARD_SIZE - r - 5.0);

        let orbital_radius = distance(x, y, CENTER, CENTER);
        if orbital_radius < SUN_RADIUS + r + 10.0 {
            continue;
        }
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT
            && (x + r > BOARD_SIZE || x - r < 0.0 || y + r > BOARD_SIZE || y - r < 0.0)
        {
            continue;
        }

        let ships = rng.randint(5, 30);
        let temp = group_rows(id_counter, x, y, r, ships, prod);

        let mut valid = true;
        'outer2: for tp in &temp {
            let tp_orbital = distance(tp.x, tp.y, CENTER, CENTER);
            let tp_is_rotating = tp_orbital + tp.radius < ROTATION_RADIUS_LIMIT;
            for p in &planets {
                let p_orbital = distance(p.x, p.y, CENTER, CENTER);
                let p_is_rotating = p_orbital + p.radius < ROTATION_RADIUS_LIMIT;
                if distance(p.x, p.y, tp.x, tp.y) < p.radius + tp.radius + PLANET_CLEARANCE {
                    valid = false;
                    break 'outer2;
                }
                if tp_is_rotating != p_is_rotating
                    && (tp_orbital - p_orbital).abs() < tp.radius + p.radius + PLANET_CLEARANCE
                {
                    valid = false;
                    break 'outer2;
                }
            }
        }

        if valid {
            if orbital_radius + r < ROTATION_RADIUS_LIMIT {
                has_orbiting = true;
            }
            planets.extend(temp);
            id_counter += 4;
        }
    }

    planets
}
