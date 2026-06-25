//! The interpreter step — one full game tick.
//!
//! Faithful port of the non-init body of `interpreter` in `ow_sim/engine.py`.
//! Operates on a cloned `GameState`. RNG-dependent map/comet *generation* is
//! out of scope here: at a comet-spawn tick the caller injects the comet
//! group via `SpawnInjection` (the RNG port comes later).
//!
//! Step indexing (validated against recorded replays): the transition that
//! produces recorded index `k` is driven with `cur_step = k - 1`. Comets
//! spawn when `cur_step + 1` is in COMET_SPAWN_STEPS (i.e. k in {50,150,...}).

use std::collections::HashSet;

use crate::comets::generate_comet_paths;
use crate::geometry::{point_to_segment_distance, swept_pair_hit};
use crate::mapgen::generate_planets;
use crate::pymath::{c_atan2, c_cos, c_log, c_pow, c_sin, c_sqrt, sq2};
use crate::pyrandom::PyRandom;
use crate::state::*;

/// Build the initial state from an episode seed — mirrors the `interpreter`
/// init block: angular_velocity = uniform(0.025,0.05), planets = generate,
/// initial_planets = pristine copy, then home-planet assignment.
pub fn init_from_seed(seed: u64, num_agents: usize) -> GameState {
    let mut rng = PyRandom::from_int(seed);
    let angular_velocity = rng.uniform(0.025, 0.05);
    let mut planets = generate_planets(&mut rng);
    let initial_planets = planets.clone();

    let num_groups = planets.len() / 4;
    if num_groups > 0 {
        let home_group = rng.randint(0, (num_groups - 1) as i64);
        let base = (home_group * 4) as usize;
        if num_agents == 2 {
            planets[base].owner = 0;
            planets[base].ships = 10;
            planets[base + 3].owner = 1;
            planets[base + 3].ships = 10;
        } else if num_agents == 4 {
            for j in 0..4 {
                planets[base + j].owner = j as i64;
                planets[base + j].ships = 10;
            }
        }
    }

    GameState {
        planets,
        fleets: Vec::new(),
        next_fleet_id: 0,
        comets: Vec::new(),
        comet_planet_ids: Vec::new(),
        initial_planets,
        angular_velocity,
    }
}

pub struct StepResult {
    pub state: GameState,
    pub terminated: bool,
    pub rewards: Option<Vec<i64>>,
}

pub struct StepOutcome {
    pub terminated: bool,
    pub rewards: Option<Vec<i64>>,
}

/// Convenience wrapper that clones the input state (used by the single-step
/// parity test, which compares against a freshly-built expected state). The
/// hot path (Env / full-game) should use `step_in_place`.
pub fn step(
    src: &GameState,
    actions: &[Vec<Move>],
    cur_step: i64,
    cfg: &Config,
    spawn: Option<&SpawnInjection>,
    episode_seed: u64,
) -> StepResult {
    let mut g = src.clone();
    let out = step_in_place(&mut g, actions, cur_step, cfg, spawn, episode_seed);
    StepResult { state: g, terminated: out.terminated, rewards: out.rewards }
}

/// Run one tick, mutating `g` in place. `actions[player]` is that player's
/// launch orders.
pub fn step_in_place(
    g: &mut GameState,
    actions: &[Vec<Move>],
    cur_step: i64,
    cfg: &Config,
    spawn: Option<&SpawnInjection>,
    episode_seed: u64,
) -> StepOutcome {
    let num_agents = actions.len();

    // --- 1. Comet expiration (before launch) ---------------------------
    let mut expired: Vec<i64> = Vec::new();
    for group in &g.comets {
        let idx = group.path_index;
        for (i, &pid) in group.planet_ids.iter().enumerate() {
            if idx >= group.paths[i].len() as i64 {
                expired.push(pid);
            }
        }
    }
    if !expired.is_empty() {
        remove_comets(g, &expired.iter().cloned().collect());
    }

    // --- 2. Comet spawning --------------------------------------------
    // Replay-validation mode injects the recorded comet group; full-from-seed
    // mode generates it via the CPython-compatible comet RNG.
    if COMET_SPAWN_STEPS.contains(&(cur_step + 1)) {
        let spawned: Option<(Vec<Vec<[f64; 2]>>, i64)> = if let Some(inj) = spawn {
            Some((inj.paths.clone(), inj.ships))
        } else {
            let s = cur_step + 1;
            let mut crng = PyRandom::from_str(&format!("orbit_wars-comet-{episode_seed}-{s}"));
            let comet_ids: HashSet<i64> = g.comet_planet_ids.iter().cloned().collect();
            let paths = generate_comet_paths(
                &g.initial_planets,
                g.angular_velocity,
                s,
                &comet_ids,
                cfg.comet_speed,
                &mut crng,
            );
            paths.map(|p| {
                // comet_ships = min of 4 randint(1,99), in order.
                let ships = crng
                    .randint(1, 99)
                    .min(crng.randint(1, 99))
                    .min(crng.randint(1, 99))
                    .min(crng.randint(1, 99));
                (p, ships)
            })
        };

        if let Some((paths, comet_ships)) = spawned {
            let next_id = g.planets.iter().map(|p| p.id).max().unwrap() + 1;
            let mut group = CometGroup {
                planet_ids: Vec::new(),
                paths: paths.clone(),
                path_index: -1,
            };
            for i in 0..paths.len() {
                let pid = next_id + i as i64;
                group.planet_ids.push(pid);
                g.comet_planet_ids.push(pid);
                let planet = Planet {
                    id: pid,
                    owner: -1,
                    x: -99.0,
                    y: -99.0,
                    radius: COMET_RADIUS,
                    ships: comet_ships,
                    production: COMET_PRODUCTION,
                };
                g.planets.push(planet.clone());
                g.initial_planets.push(planet);
            }
            g.comets.push(group);
        }
    }

    // --- 3. Fleet launch ----------------------------------------------
    for (player, acts) in actions.iter().enumerate() {
        let pl = player as i64;
        for mv in acts {
            let ships = mv.ships;
            if let Some(idx) = g.planets.iter().position(|p| p.id == mv.from_id) {
                let fp = &g.planets[idx];
                if fp.owner == pl && fp.ships >= ships && ships > 0 {
                    let sx = fp.x + c_cos(mv.angle) * (fp.radius + 0.1);
                    let sy = fp.y + c_sin(mv.angle) * (fp.radius + 0.1);
                    g.planets[idx].ships -= ships;
                    let fid = g.next_fleet_id;
                    g.fleets.push(Fleet {
                        id: fid,
                        owner: pl,
                        x: sx,
                        y: sy,
                        angle: mv.angle,
                        from_planet_id: mv.from_id,
                        ships,
                    });
                    g.next_fleet_id += 1;
                }
            }
        }
    }

    // --- 4. Production -------------------------------------------------
    for p in g.planets.iter_mut() {
        if p.owner != -1 {
            p.ships += p.production;
        }
    }

    // --- 5. Planet end-of-tick positions ------------------------------
    // paths[i] = (old_x, old_y, new_x, new_y, check_collision), aligned with
    // g.planets by index. Invariant: g.planets[i].id == g.initial_planets[i].id
    // for all i (both grow/shrink together on spawn/expiry), so the initial
    // position is read positionally — no id lookup needed.
    let n = g.planets.len();
    let is_comet: Vec<bool> = g
        .planets
        .iter()
        .map(|p| g.comet_planet_ids.contains(&p.id))
        .collect();

    let mut paths: Vec<(f64, f64, f64, f64, bool)> = Vec::with_capacity(n);
    for (i, p) in g.planets.iter().enumerate() {
        if is_comet[i] {
            // Filled in the comet-advance loop below; placeholder for now.
            paths.push((p.x, p.y, p.x, p.y, true));
            continue;
        }
        let ip = &g.initial_planets[i];
        let (ox, oy) = (p.x, p.y);
        let (mut nx, mut ny) = (ox, oy);
        let dx = ip.x - CENTER;
        let dy = ip.y - CENTER;
        let r = c_sqrt(sq2(dx) + sq2(dy));
        if r + p.radius < ROTATION_RADIUS_LIMIT {
            let initial_angle = c_atan2(dy, dx);
            let current_angle = initial_angle + g.angular_velocity * (cur_step as f64);
            nx = CENTER + r * c_cos(current_angle);
            ny = CENTER + r * c_sin(current_angle);
        }
        paths.push((ox, oy, nx, ny, true));
    }

    let ids: Vec<i64> = g.planets.iter().map(|p| p.id).collect();
    let mut expired_move: Vec<i64> = Vec::new();
    for group in g.comets.iter_mut() {
        group.path_index += 1;
        let idx = group.path_index;
        for (i, &pid) in group.planet_ids.iter().enumerate() {
            let pj = match ids.iter().position(|&x| x == pid) {
                Some(j) => j,
                None => continue,
            };
            let (ox, oy) = (paths[pj].0, paths[pj].1); // current pos (placeholder)
            let p_path = &group.paths[i];
            if idx >= p_path.len() as i64 {
                expired_move.push(pid);
                paths[pj] = (ox, oy, ox, oy, true);
            } else {
                let np = p_path[idx as usize];
                let check = ox >= 0.0;
                paths[pj] = (ox, oy, np[0], np[1], check);
            }
        }
    }

    // --- 6. Fleet movement + continuous collision ---------------------
    let max_speed = cfg.ship_speed;
    let radii: Vec<f64> = g.planets.iter().map(|p| p.radius).collect();
    let mut combat_lists: Vec<Vec<(i64, i64)>> = vec![Vec::new(); n];
    let mut remove_flags: Vec<bool> = vec![false; g.fleets.len()];

    let ln1000 = c_log(1000.0);
    for (fi, f) in g.fleets.iter_mut().enumerate() {
        let angle = f.angle;
        let ships = f.ships;
        let mut speed =
            1.0 + (max_speed - 1.0) * c_pow(c_log(ships as f64) / ln1000, 1.5);
        if speed > max_speed {
            speed = max_speed;
        }
        let (ox, oy) = (f.x, f.y);
        f.x += c_cos(angle) * speed;
        f.y += c_sin(angle) * speed;
        let (nx, ny) = (f.x, f.y);

        let mut hit = false;
        for j in 0..n {
            let path = paths[j];
            if !path.4 {
                continue;
            }
            if swept_pair_hit(ox, oy, nx, ny, path.0, path.1, path.2, path.3, radii[j]) {
                combat_lists[j].push((f.owner, f.ships));
                remove_flags[fi] = true;
                hit = true;
                break;
            }
        }
        if hit {
            continue;
        }
        if !(0.0 <= f.x && f.x <= BOARD_SIZE && 0.0 <= f.y && f.y <= BOARD_SIZE) {
            remove_flags[fi] = true;
            continue;
        }
        if point_to_segment_distance(CENTER, CENTER, ox, oy, nx, ny) < SUN_RADIUS {
            remove_flags[fi] = true;
            continue;
        }
    }

    // --- 7. Apply planet movement -------------------------------------
    for (i, p) in g.planets.iter_mut().enumerate() {
        p.x = paths[i].2;
        p.y = paths[i].3;
    }

    // Remove comets that left the board this tick.
    if !expired_move.is_empty() {
        remove_comets(g, &expired_move.iter().cloned().collect());
    }

    // Remove fleets that were consumed / lost this tick.
    let mut fi = 0usize;
    g.fleets.retain(|_| {
        let keep = !remove_flags[fi];
        fi += 1;
        keep
    });

    // --- 8. Combat resolution -----------------------------------------
    // Iterate planets in their pre-removal order (combat_lists is index-aligned
    // with that order via `ids`); resolve on the post-removal planet by id.
    for j in 0..n {
        if combat_lists[j].is_empty() {
            continue;
        }
        let pid = ids[j];
        let pidx = match g.planets.iter().position(|p| p.id == pid) {
            Some(i) => i,
            None => continue,
        };

        // Sum ships per owner, preserving first-appearance order (few owners).
        let mut sums: Vec<(i64, i64)> = Vec::new();
        for &(owner, ships) in &combat_lists[j] {
            if let Some(e) = sums.iter_mut().find(|e| e.0 == owner) {
                e.1 += ships;
            } else {
                sums.push((owner, ships));
            }
        }
        let mut sorted = sums;
        sorted.sort_by(|a, b| b.1.cmp(&a.1)); // stable; ties keep insertion order

        let (top_player, top_ships) = sorted[0];
        let (survivor_owner, survivor_ships) = if sorted.len() > 1 {
            let second = sorted[1].1;
            let mut ss = top_ships - second;
            if sorted[0].1 == sorted[1].1 {
                ss = 0;
            }
            (if ss > 0 { top_player } else { -1 }, ss)
        } else {
            (top_player, top_ships)
        };

        if survivor_ships > 0 {
            let pl = &mut g.planets[pidx];
            if pl.owner == survivor_owner {
                pl.ships += survivor_ships;
            } else {
                pl.ships -= survivor_ships;
                if pl.ships < 0 {
                    pl.owner = survivor_owner;
                    pl.ships = pl.ships.abs();
                }
            }
        }
    }

    // --- 9. Termination + scoring -------------------------------------
    let mut terminated = cur_step >= cfg.episode_steps - 2;
    let mut seen = [false; 4];
    for p in &g.planets {
        if p.owner >= 0 && (p.owner as usize) < 4 {
            seen[p.owner as usize] = true;
        }
    }
    for f in &g.fleets {
        if f.owner >= 0 && (f.owner as usize) < 4 {
            seen[f.owner as usize] = true;
        }
    }
    if seen.iter().filter(|&&b| b).count() <= 1 {
        terminated = true;
    }

    let rewards = if terminated {
        let mut scores = vec![0i64; num_agents];
        for p in &g.planets {
            if p.owner != -1 && (p.owner as usize) < num_agents {
                scores[p.owner as usize] += p.ships;
            }
        }
        for f in &g.fleets {
            if (f.owner as usize) < num_agents {
                scores[f.owner as usize] += f.ships;
            }
        }
        let maxs = *scores.iter().max().unwrap();
        Some(
            (0..num_agents)
                .map(|i| if scores[i] == maxs && maxs > 0 { 1 } else { -1 })
                .collect(),
        )
    } else {
        None
    };

    StepOutcome { terminated, rewards }
}

/// Remove the given comet planet ids from planets, initial_planets,
/// comet_planet_ids, every group's planet_ids, and drop empty groups.
fn remove_comets(g: &mut GameState, ids: &HashSet<i64>) {
    g.planets.retain(|p| !ids.contains(&p.id));
    g.initial_planets.retain(|p| !ids.contains(&p.id));
    g.comet_planet_ids.retain(|pid| !ids.contains(pid));
    for group in g.comets.iter_mut() {
        group.planet_ids.retain(|pid| !ids.contains(pid));
    }
    g.comets.retain(|grp| !grp.planet_ids.is_empty());
}
