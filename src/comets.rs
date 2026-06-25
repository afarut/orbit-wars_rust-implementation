//! Elliptical comet path generation — faithful port of `generate_comet_paths`
//! in `ow_sim/comets.py`. RNG call order per attempt: uniform(e), uniform(a),
//! perihelion check (may `continue` BEFORE phi), then uniform(phi).

use std::collections::HashSet;
use std::f64::consts::PI;

use crate::geometry::distance;
use crate::pymath::{c_atan2, c_cos, c_sin, c_sqrt, sq2};
use crate::pyrandom::PyRandom;
use crate::state::*;

/// Returns 4 rotationally-symmetric comet paths, or None on failure (300 tries).
pub fn generate_comet_paths(
    initial_planets: &[Planet],
    angular_velocity: f64,
    spawn_step: i64,
    comet_planet_ids: &HashSet<i64>,
    comet_speed: f64,
    rng: &mut PyRandom,
) -> Option<Vec<Vec<[f64; 2]>>> {
    for _ in 0..300 {
        let e = rng.uniform(0.75, 0.93);
        let a = rng.uniform(60.0, 150.0);
        let perihelion = a * (1.0 - e);
        if perihelion < SUN_RADIUS + COMET_RADIUS {
            continue;
        }
        let b = a * c_sqrt(1.0 - sq2(e));
        let c_val = a * e;
        let phi = rng.uniform(PI / 6.0, PI / 3.0);

        // Dense sample around perihelion half of orbit.
        let num = 5000usize;
        let mut dense: Vec<(f64, f64)> = Vec::with_capacity(num);
        for i in 0..num {
            let t = 0.3 * PI + 1.4 * PI * (i as f64) / ((num - 1) as f64);
            let ex = c_val + a * c_cos(t);
            let ey = b * c_sin(t);
            let x = CENTER + ex * c_cos(phi) - ey * c_sin(phi);
            let y = CENTER + ex * c_sin(phi) + ey * c_cos(phi);
            dense.push((x, y));
        }

        // Re-sample at constant comet_speed arc-length intervals.
        let mut path: Vec<(f64, f64)> = vec![dense[0]];
        let mut cum = 0.0_f64;
        let mut target = comet_speed;
        for i in 1..dense.len() {
            cum += distance(dense[i].0, dense[i].1, dense[i - 1].0, dense[i - 1].1);
            if cum >= target {
                path.push(dense[i]);
                target += comet_speed;
            }
        }

        // Contiguous on-board segment.
        let mut board_start: Option<usize> = None;
        let mut board_end: usize = 0;
        for (i, &(x, y)) in path.iter().enumerate() {
            if 0.0 <= x && x <= BOARD_SIZE && 0.0 <= y && y <= BOARD_SIZE {
                if board_start.is_none() {
                    board_start = Some(i);
                }
                board_end = i;
            }
        }
        let board_start = match board_start {
            Some(s) => s,
            None => continue,
        };
        let visible: Vec<(f64, f64)> = path[board_start..=board_end].to_vec();
        if !(5..=40).contains(&visible.len()) {
            continue;
        }

        // 4 rotationally symmetric paths (note the [y, x] swap on the first).
        let b_ = BOARD_SIZE;
        let paths: Vec<Vec<[f64; 2]>> = vec![
            visible.iter().map(|&(x, y)| [y, x]).collect(),
            visible.iter().map(|&(x, y)| [b_ - x, y]).collect(),
            visible.iter().map(|&(x, y)| [x, b_ - y]).collect(),
            visible.iter().map(|&(x, y)| [b_ - y, b_ - x]).collect(),
        ];

        // Separate planets into static / orbiting (exclude comets).
        let mut static_planets: Vec<&Planet> = Vec::new();
        let mut orbiting_planets: Vec<&Planet> = Vec::new();
        for planet in initial_planets {
            if comet_planet_ids.contains(&planet.id) {
                continue;
            }
            let pr = distance(planet.x, planet.y, CENTER, CENTER);
            if pr + planet.radius < ROTATION_RADIUS_LIMIT {
                orbiting_planets.push(planet);
            } else {
                static_planets.push(planet);
            }
        }

        let mut valid = true;
        let buf = COMET_RADIUS + 0.5;
        'visible: for (k, &(cx, cy)) in visible.iter().enumerate() {
            if distance(cx, cy, CENTER, CENTER) < SUN_RADIUS + COMET_RADIUS {
                valid = false;
                break;
            }
            let sym_pts = [
                (cy, cx),
                (b_ - cx, cy),
                (cx, b_ - cy),
                (b_ - cy, b_ - cx),
            ];
            for planet in &static_planets {
                for sp in &sym_pts {
                    if distance(sp.0, sp.1, planet.x, planet.y) < planet.radius + buf {
                        valid = false;
                        break 'visible;
                    }
                }
            }
            let game_step = (spawn_step - 1 + k as i64) as f64;
            for planet in &orbiting_planets {
                let dx = planet.x - CENTER;
                let dy = planet.y - CENTER;
                let orb_r = c_sqrt(sq2(dx) + sq2(dy));
                let init_angle = c_atan2(dy, dx);
                let cur_angle = init_angle + angular_velocity * game_step;
                let px = CENTER + orb_r * c_cos(cur_angle);
                let py = CENTER + orb_r * c_sin(cur_angle);
                for sp in &sym_pts {
                    if distance(sp.0, sp.1, px, py) < planet.radius + COMET_RADIUS {
                        valid = false;
                        break 'visible;
                    }
                }
            }
        }

        if valid {
            return Some(paths);
        }
    }
    None
}
