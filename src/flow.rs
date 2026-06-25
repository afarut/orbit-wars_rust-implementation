//! Garrison projection and competitive flow scoring — mirrors orbit_lite's
//! `garrison_launch.py` + `intercept_aim.py` + `planner_core.py` in Rust.
//!
//! The goal is bit-level parity with the Python pipeline so our Rust self-play
//! agents can serve as drop-in deterministic opponents.

use crate::geometry::{distance, swept_pair_hit};
use crate::pymath::{c_atan2, c_cos, c_log, c_pow, c_sin, c_sqrt, sq2};
use crate::state::*;

// ── constants ────────────────────────────────────────────────────────────────

pub const HORIZON: usize = 18;

/// Source surface offset at launch (matches engine: start = src + unit*( r+0.1 ))
const LAUNCH_SURFACE_OFFSET: f64 = 0.1;
/// Extra target margin (0 → hit when fleet center enters planet radius)
const TARGET_HIT_SURFACE_OFFSET: f64 = 0.0;
/// Fixed-point iterations for continuous intercept solver
const FP_ITERS: usize = 6;

// ── geometry helpers (duplicated from agents to keep this module standalone) ──

fn fleet_speed_f(ships: i64, max_speed: f64) -> f64 {
    if ships <= 0 {
        return max_speed;
    }
    let s = 1.0 + (max_speed - 1.0) * c_pow(c_log(ships as f64) / c_log(1000.0), 1.5);
    s.min(max_speed)
}

/// Planet (or comet) centre at game step `step` starting from initial position.
fn planet_pos_at(
    initial_x: f64,
    initial_y: f64,
    angular_velocity: f64,
    step: i64,
) -> (f64, f64) {
    let dx = initial_x - CENTER;
    let dy = initial_y - CENTER;
    let r = c_sqrt(sq2(dx) + sq2(dy));
    // Matches engine: orbit when initial_radius < ROTATION_RADIUS_LIMIT.
    // We can't check planet radius here, but the caller passes initial coords
    // of the *planet* at step-0 so the threshold check is (r < limit) not
    // (r + planet_radius < limit). This is the formula used by intercept_aim.py.
    // Outer ring planets: r is large, just return fixed position.
    let theta0 = c_atan2(dy, dx);
    let theta = theta0 + angular_velocity * step as f64;
    (CENTER + r * c_cos(theta), CENTER + r * c_sin(theta))
}

/// Whether (initial_x, initial_y) is an orbiting planet (inner ring).
fn is_orbiting(initial_x: f64, initial_y: f64, planet_radius: f64) -> bool {
    let dx = initial_x - CENTER;
    let dy = initial_y - CENTER;
    let r = c_sqrt(sq2(dx) + sq2(dy));
    r + planet_radius < ROTATION_RADIUS_LIMIT
}

// ── intercept angle (continuous fixed-point, matches intercept_aim.py) ───────

/// Continuous fixed-point intercept aim.
///
/// Returns `(angle, eta_steps, viable)`. `viable` is false when the
/// straight-line shot would be blocked by the sun (simplified body screen —
/// we skip the full swept-pair planet screen for speed, matching agents.rs
/// behaviour for self-play use).
pub fn intercept_angle_fp(
    sx: f64,
    sy: f64,
    src_radius: f64,
    tgt_init_x: f64,
    tgt_init_y: f64,
    tgt_radius: f64,
    tgt_orbiting: bool,
    angular_velocity: f64,
    cur_step: i64,
    ships: i64,
    max_speed: f64,
    horizon: usize,
) -> (f64, f64, bool) {
    let speed = fleet_speed_f(ships, max_speed).max(1e-6);
    let gap = src_radius + LAUNCH_SURFACE_OFFSET + tgt_radius + TARGET_HIT_SURFACE_OFFSET;

    // Target orbit: centre-relative R and angular velocity omega.
    // Mirror orbit_lite's `position_at_slots(tgt, 0)` and `position_at_slots(tgt, 1)`:
    // these are movement.x[0] and movement.x[1], which use orbital phases
    // max(0, cur_step - 1) and max(0, cur_step) respectively.
    // At step 0: both phases are 0 → t0 ≈ t1 → omega ≈ 0 (planet appears static).
    let phase_t0 = (cur_step - 1).max(0); // max(0, cur_step - 1)
    let phase_t1 = cur_step;              // max(0, cur_step) = cur_step
    let (t0x, t0y) = if tgt_orbiting {
        planet_pos_at(tgt_init_x, tgt_init_y, angular_velocity, phase_t0)
    } else {
        (tgt_init_x, tgt_init_y)
    };
    let (t1x, t1y) = if tgt_orbiting {
        planet_pos_at(tgt_init_x, tgt_init_y, angular_velocity, phase_t1)
    } else {
        (tgt_init_x, tgt_init_y)
    };

    let r_orb = c_sqrt(sq2(t0x - CENTER) + sq2(t0y - CENTER));
    let a0 = c_atan2(t0y - CENTER, t0x - CENTER);
    let a1 = c_atan2(t1y - CENTER, t1x - CENTER);
    // Wrapped angular step per tick.
    let da = a1 - a0;
    let omega = f64::atan2(f64::sin(da), f64::cos(da));

    let h = horizon as f64;
    let d0 = c_sqrt(sq2(t0x - sx) + sq2(t0y - sy));
    let mut t_star = ((d0 - gap) / speed).max(0.0).min(h);

    for _ in 0..FP_ITERS {
        let ang = a0 + omega * t_star;
        let tx = CENTER + r_orb * c_cos(ang);
        let ty = CENTER + r_orb * c_sin(ang);
        let d = c_sqrt(sq2(tx - sx) + sq2(ty - sy));
        t_star = ((d - gap) / speed).max(0.0).min(h);
    }

    let ang_final = a0 + omega * t_star;
    let tx = CENTER + r_orb * c_cos(ang_final);
    let ty = CENTER + r_orb * c_sin(ang_final);
    let angle = c_atan2(ty - sy, tx - sx);

    // Sun/body collision is screened by the FINITE `fleet_eta` trajectory walk in
    // act() (first contact must equal the target). The old `ray_hits_sun` check used
    // an INFINITE ray and false-positived shots whose closest sun approach lies beyond
    // the target — wrongly killing viable launches (e.g. reinforcing a planet across
    // the board). Mirror orbit_lite: viability = first-contact == target only.
    let _ = ray_hits_sun(sx, sy, angle);
    let viable = true;

    (angle, t_star, viable)
}

fn ray_hits_sun(sx: f64, sy: f64, angle: f64) -> bool {
    let dx = c_cos(angle);
    let dy = c_sin(angle);
    let fx = sx - CENTER;
    let fy = sy - CENTER;
    let t = -(fx * dx + fy * dy);
    let (cx, cy) = if t > 0.0 {
        (sx + t * dx, sy + t * dy)
    } else {
        (sx, sy)
    };
    distance(cx, cy, CENTER, CENTER) < SUN_RADIUS + 1.0
}

// ── fleet ETA tracking ────────────────────────────────────────────────────────

/// Planet positions at step k (index into the pre-computed table).
/// We pre-compute positions[k][p] for k = 0..=horizon.
///
/// Mirrors orbit_lite's phase convention: `movement.x[k]` at observation step S uses
/// orbital phase `max(0, S + k - 1)` (i.e. `orbit_phase_index_from_obs_step(S + k)`).
/// At step 0 this means k=0 and k=1 both map to phase 0 (same position — the planet
/// appears static on the first observation step, omega=0 in the FP solver).
/// Public wrapper for debug bindings.
pub fn build_planet_positions_pub(g: &GameState, horizon: usize, cur_step: i64) -> Vec<Vec<(f64, f64)>> {
    build_planet_positions(g, horizon, cur_step)
}

fn build_planet_positions(
    g: &GameState,
    horizon: usize,
    cur_step: i64,
) -> Vec<Vec<(f64, f64)>> {
    let h = horizon + 1;
    let n = g.planets.len();
    let mut pos = vec![vec![(0.0f64, 0.0f64); n]; h];
    for k in 0..h {
        // orbit_phase_index_from_obs_step(cur_step + k): phase = max(0, cur_step + k - 1).
        let abs_step = cur_step + k as i64;
        let phase = if abs_step > 0 { abs_step - 1 } else { 0 };
        for (i, p) in g.planets.iter().enumerate() {
            let ip = g.initial_planets.iter().find(|ip| ip.id == p.id);
            if let Some(ip) = ip {
                if is_orbiting(ip.x, ip.y, p.radius) {
                    pos[k][i] = planet_pos_at(ip.x, ip.y, g.angular_velocity, phase);
                } else {
                    pos[k][i] = (p.x, p.y);
                }
            } else {
                pos[k][i] = (p.x, p.y);
            }
        }
    }
    pos
}

/// Simulate a fleet's trajectory for up to `horizon` steps.
/// Returns `Some((planet_idx, eta))` for the first planet it hits, or `None`.
///
/// Mirrors the engine's fleet-movement loop, using swept-pair collision.
pub fn fleet_eta(
    fleet: &Fleet,
    planet_positions: &[Vec<(f64, f64)>], // [H+1][P]
    planet_radii: &[f64],                  // [P]
    max_speed: f64,
    horizon: usize,
) -> Option<(usize, usize)> {
    let p = planet_radii.len();
    let speed = fleet_speed_f(fleet.ships, max_speed);
    let mut fx = fleet.x;
    let mut fy = fleet.y;

    for step in 0..horizon {
        let nfx = fx + c_cos(fleet.angle) * speed;
        let nfy = fy + c_sin(fleet.angle) * speed;

        // Swept-pair collision with each planet.
        let pp0 = &planet_positions[step];
        let pp1 = &planet_positions[step + 1];
        for j in 0..p {
            if swept_pair_hit(
                fx, fy, nfx, nfy,
                pp0[j].0, pp0[j].1, pp1[j].0, pp1[j].1,
                planet_radii[j],
            ) {
                return Some((j, step + 1));
            }
        }

        // OOB or sun kill.
        if !(0.0 <= nfx && nfx <= BOARD_SIZE && 0.0 <= nfy && nfy <= BOARD_SIZE) {
            return None;
        }
        if crate::geometry::point_to_segment_distance(CENTER, CENTER, fx, fy, nfx, nfy)
            < SUN_RADIUS
        {
            return None;
        }

        fx = nfx;
        fy = nfy;
    }
    None
}

// ── garrison status ───────────────────────────────────────────────────────────

/// Per-planet, per-step garrison projection.
///
/// Indices: `owner[p][k]`, `ships[p][k]`, `pre_owner[p][k]`, `pre_ships[p][k]`
/// for k = 0..=H.
/// `arrivals[p][k][a]` for k = 1..=H (index 0 is always 0; stored as [H+1][A]
/// with index 0 zeroed).
pub struct GarrisonStatus {
    pub p: usize,
    pub h: usize,
    pub a: usize,
    pub owner:     Vec<Vec<i64>>,   // [P][H+1]
    pub ships:     Vec<Vec<f64>>,   // [P][H+1]
    pub pre_owner: Vec<Vec<i64>>,   // [P][H+1]
    pub pre_ships: Vec<Vec<f64>>,   // [P][H+1]
    /// arrivals[p][k][a]: ships from player a arriving at planet p at step k.
    /// k=0 always zero (no arrivals on the current turn).
    pub arrivals:  Vec<Vec<Vec<f64>>>, // [P][H+1][A]
    pub prod:      Vec<f64>,         // [P]
    pub alive:     Vec<Vec<bool>>,   // [P][H+1]
}

/// Engine survivor rule: given per-player ships arriving, return (owner, ships).
/// owner = -1 and ships = 0 when no arrivals or complete annihilation.
fn survivor(arrivals_per_player: &[f64]) -> (i64, f64) {
    let mut top1_ships = 0.0f64;
    let mut top1_owner = -1i64;
    let mut top2_ships = 0.0f64;

    for (a, &s) in arrivals_per_player.iter().enumerate() {
        if s > top1_ships {
            top2_ships = top1_ships;
            top1_ships = s;
            top1_owner = a as i64;
        } else if s > top2_ships {
            top2_ships = s;
        }
    }

    if top1_ships == top2_ships {
        // Tie → annihilation.
        (-1, 0.0)
    } else {
        (top1_owner, top1_ships - top2_ships)
    }
}

/// Run the production→combat recurrence for a single planet over [1..=H].
///
/// Mirrors `_run_exact_recurrence` in garrison_launch.py but for one planet.
fn planet_trajectory(
    init_owner: i64,
    init_ships: f64,
    prod: f64,
    alive: &[bool],             // [H+1]
    arrivals: &[Vec<f64>],      // [H+1][A] (index 0 unused)
) -> (Vec<i64>, Vec<f64>, Vec<i64>, Vec<f64>) {
    let h = alive.len() - 1;
    let mut owner_t = vec![-1i64; h + 1];
    let mut ships_t = vec![0.0f64; h + 1];
    let mut pre_owner_t = vec![-1i64; h + 1];
    let mut pre_ships_t = vec![0.0f64; h + 1];

    owner_t[0] = init_owner;
    ships_t[0] = init_ships;
    pre_owner_t[0] = init_owner;
    pre_ships_t[0] = init_ships;

    let mut cur_owner = init_owner;
    let mut cur_ships = init_ships;

    for k in 1..=h {
        // Production: credited if owned and alive at start of step.
        if alive[k - 1] && cur_owner >= 0 {
            cur_ships += prod;
        }

        // Pre-combat snapshot.
        pre_owner_t[k] = if alive[k] { cur_owner } else { -1 };
        pre_ships_t[k] = if alive[k] { cur_ships } else { 0.0 };

        // Survivor vs garrison.
        let (surv_owner, surv_ships) = survivor(&arrivals[k]);
        if surv_ships > 0.0 && alive[k] {
            if surv_owner == cur_owner {
                // Friendly reinforcement.
                cur_ships += surv_ships;
            } else {
                // Combat.
                let diff = (cur_ships - surv_ships).abs();
                if cur_ships < surv_ships {
                    cur_owner = surv_owner;
                }
                cur_ships = diff;
            }
        }

        // End-of-step death reset.
        if !alive[k] {
            cur_owner = -1;
            cur_ships = 0.0;
        }

        owner_t[k] = cur_owner;
        ships_t[k] = cur_ships;
    }

    (owner_t, ships_t, pre_owner_t, pre_ships_t)
}

/// Compute (produced[A], combat_lost[A]) for a single-planet trajectory.
///
/// Mirrors `_flow_terms_per_planet` in garrison_launch.py for a single planet.
fn flow_terms_single(
    owner_t: &[i64],
    pre_owner_t: &[i64],
    pre_ships_t: &[f64],
    arrivals: &[Vec<f64>],  // [H+1][A]
    prod: f64,
    alive: &[bool],          // [H+1]
    a: usize,
) -> (Vec<f64>, Vec<f64>) {
    let h = owner_t.len() - 1;
    let mut produced = vec![0.0f64; a];
    let mut combat_lost = vec![0.0f64; a];

    for k in 1..=h {
        // Production credited to owner at step k-1.
        let producing_owner = owner_t[k - 1];
        if producing_owner >= 0 && alive[k - 1] {
            produced[producing_owner as usize] += prod;
        }

        // Combat at step k.
        let (surv_owner, surv_ships) = survivor(&arrivals[k]);
        if surv_ships <= 0.0 || surv_owner < 0 || !alive[k] {
            continue;
        }
        let prior_owner = pre_owner_t[k];
        let prior_ships = pre_ships_t[k];

        // Attacker losses: each player loses (their arrivals - their survival).
        for aa in 0..a {
            let arrived = arrivals[k][aa];
            let survived_for_aa = if (aa as i64) == surv_owner { surv_ships } else { 0.0 };
            let att_lost = (arrived - survived_for_aa).max(0.0);
            combat_lost[aa] += att_lost;
        }

        // Garrison losses: when survivor fights against garrison (including neutral).
        // Mirror Python: fights_garrison requires only surv_owner != prior_owner
        // (no prior_owner >= 0 requirement — neutral garrison also costs the attacker).
        let fights_garrison = surv_owner != prior_owner;
        if fights_garrison {
            let garrison_loss = prior_ships.min(surv_ships);
            // Prior garrison owner loses (only if a real player).
            if prior_owner >= 0 {
                combat_lost[prior_owner as usize] += garrison_loss;
            }
            // Survivor pays garrison_loss regardless of whether prior was neutral.
            if surv_owner >= 0 {
                combat_lost[surv_owner as usize] += garrison_loss;
            }
        }
    }

    (produced, combat_lost)
}

/// Build the full garrison status for all planets over `horizon` steps.
pub fn build_garrison_status(
    g: &GameState,
    cfg: &Config,
    player_count: usize,
    horizon: usize,
    cur_step: i64,
) -> GarrisonStatus {
    let p = g.planets.len();
    let h = horizon;
    let a = player_count;
    let _comet_ids: std::collections::HashSet<i64> =
        g.comet_planet_ids.iter().copied().collect();

    // Build planet positions [H+1][P].
    let planet_pos = build_planet_positions(g, horizon, cur_step);

    // alive[p][k]: planet p exists at step k.
    // Comets: they expire at fixed steps; for simplicity we mark them alive
    // for all steps (conservative — they disappear but won't cause crashes).
    let alive: Vec<Vec<bool>> = (0..p).map(|_| vec![true; h + 1]).collect();

    // --- Build arrivals from in-flight fleets ---
    let mut arrivals: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0f64; a]; h + 1]; p];
    let planet_radii: Vec<f64> = g.planets.iter().map(|p| p.radius).collect();

    for fleet in &g.fleets {
        if fleet.owner < 0 || fleet.owner as usize >= a {
            continue;
        }
        if let Some((pidx, eta)) = fleet_eta(fleet, &planet_pos, &planet_radii, cfg.ship_speed, h) {
            if eta <= h {
                arrivals[pidx][eta][fleet.owner as usize] += fleet.ships as f64;
            }
        }
    }

    // --- Run recurrence for each planet ---
    let prod: Vec<f64> = g.planets.iter().map(|p| p.production as f64).collect();
    let init_owners: Vec<i64> = g.planets.iter().map(|p| p.owner).collect();
    let init_ships: Vec<f64> = g.planets.iter().map(|p| p.ships as f64).collect();

    let mut owner_all: Vec<Vec<i64>> = Vec::with_capacity(p);
    let mut ships_all: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut pre_owner_all: Vec<Vec<i64>> = Vec::with_capacity(p);
    let mut pre_ships_all: Vec<Vec<f64>> = Vec::with_capacity(p);

    for i in 0..p {
        let (ot, st, pot, pst) = planet_trajectory(
            init_owners[i],
            init_ships[i],
            prod[i],
            &alive[i],
            &arrivals[i],
        );
        owner_all.push(ot);
        ships_all.push(st);
        pre_owner_all.push(pot);
        pre_ships_all.push(pst);
    }

    GarrisonStatus {
        p,
        h,
        a,
        owner: owner_all,
        ships: ships_all,
        pre_owner: pre_owner_all,
        pre_ships: pre_ships_all,
        arrivals,
        prod,
        alive,
    }
}

/// DEBUG: the enemy-pressure vector used for regroup ranking (cheap_enemy_pressure).
pub fn enemy_pressure_debug(g: &GameState, game_cfg: &Config, player: i64, step: i64, horizon: usize) -> Vec<f64> {
    let n = g.planets.len();
    let planet_pos = build_planet_positions(g, horizon, step);
    let k0 = &planet_pos[0];
    let mut pres = vec![0.0f64; n];
    for e in 0..n {
        if g.planets[e].owner == player || g.planets[e].owner < 0 { continue; }
        let e_ships = g.planets[e].ships;
        if e_ships <= 0 { continue; }
        let e_speed = fleet_speed_f(e_ships, game_cfg.ship_speed).max(1e-6);
        let reach = (e_speed * horizon as f64).max(1e-6);
        let (ex, ey) = k0[e];
        for p in 0..n {
            if p == e { continue; }   // mirror orbit_lite ~eye: no self-pressure
            let (px, py) = k0[p];
            let d = distance(ex, ey, px, py);
            pres[p] += e_ships as f64 * (1.0 - d / reach).max(0.0);
        }
    }
    pres
}

/// DEBUG: per-(owned src, enemy tgt) offensive candidate evaluation, mirroring
/// `act()`'s offensive loop but computing every gate without skipping. One tuple per
/// pair: (src_id, tgt_id, avail, eta, angle, viable, floor_ships, bodyscreen_ok,
/// score, roi_pass, in_off_shortlist, in_def_shortlist). For parity debugging vs
/// the canonical orbit_lite planner.
pub fn offensive_debug(
    g: &GameState,
    game_cfg: &Config,
    player: i64,
    step: i64,
    _horizon: usize,
) -> Vec<(i64, i64, i64, i64, f64, bool, i64, bool, f64, bool, bool, bool)> {
    let player_count = {
        let max_owner = g.planets.iter().map(|p| p.owner).max().unwrap_or(0);
        let max_fleet = g.fleets.iter().map(|f| f.owner).max().unwrap_or(0);
        (max_owner.max(max_fleet) + 1).max(2) as usize
    };
    let cfg4p: ProducerLiteConfig;
    let base = ProducerLiteConfig::default();
    let c: &ProducerLiteConfig = if player_count >= 4 {
        cfg4p = ProducerLiteConfig {
            horizon: 13, max_sources_per_lane: 6, max_defensive_targets: 2,
            max_regroup_time: 6.0, max_regroup_targets_per_source: 8, ..base.clone()
        };
        &cfg4p
    } else { &base };
    let h = c.horizon;
    let status = build_garrison_status(g, game_cfg, player_count, h, step);
    let n = g.planets.len();
    let planet_radii: Vec<f64> = g.planets.iter().map(|p| p.radius).collect();
    let planet_pos = build_planet_positions(g, h, step);
    let owned_idxs: Vec<usize> = (0..n).filter(|&i| g.planets[i].owner == player).collect();
    let comet_ids: std::collections::HashSet<i64> = g.comet_planet_ids.iter().copied().collect();

    let offensive: Vec<usize> = {
        let mut ts: Vec<(usize, f64)> = (0..n)
            .filter(|&i| g.planets[i].owner != player && !comet_ids.contains(&g.planets[i].id))
            .map(|i| {
                let md = owned_idxs.iter()
                    .filter(|&&s| g.planets[s].ships >= base.min_ships_to_launch as i64)
                    .map(|&s| {
                    let (sx, sy) = planet_pos[0][s];
                    (1..=h).map(|k| { let (tx, ty) = planet_pos[k][i]; distance(sx, sy, tx, ty) })
                        .fold(f64::INFINITY, f64::min)
                }).fold(f64::INFINITY, f64::min);
                (i, md)
            }).collect();
        ts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ts.into_iter().take(c.max_offensive_targets).map(|(i, _)| i).collect()
    };
    let defensive: Vec<usize> = {
        let mut cand: Vec<(f64, usize)> = Vec::new();
        for &i in &owned_idxs {
            let flip = (1..=h).find(|&k| status.owner[i][k] != player && status.owner[i][k] >= 0);
            if let Some(ft) = flip {
                cand.push((status.prod[i] * (h - ft) as f64 + status.ships[i][0], i));
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        cand.into_iter().take(c.max_defensive_targets).map(|(_, i)| i).collect()
    };

    let mut out = Vec::new();
    for &src_i in owned_idxs.iter() {
        let src = &g.planets[src_i];
        let drain = safe_drain_planet(&status, src_i, player as usize, h);
        let avail = (drain as i64).min(src.ships).max(0);
        for tgt_i in 0..n {
            if tgt_i == src_i { continue; }   // include owned (defensive/regroup) targets too
            let tgt = &g.planets[tgt_i];
            let tgt_ip = g.initial_planets.iter().find(|ip| ip.id == tgt.id);
            let (tix, tiy) = tgt_ip.map(|ip| (ip.x, ip.y)).unwrap_or((tgt.x, tgt.y));
            let tgt_orb = is_orbiting(tix, tiy, tgt.radius);
            let ships = avail;
            let (angle, _t_star, viable) = intercept_angle_fp(
                src.x, src.y, src.radius, tix, tiy, tgt.radius, tgt_orb,
                g.angular_velocity, step, ships, game_cfg.ship_speed, h);
            let lx = src.x + c_cos(angle) * (src.radius + LAUNCH_SURFACE_OFFSET);
            let ly = src.y + c_sin(angle) * (src.radius + LAUNCH_SURFACE_OFFSET);
            let synth = Fleet { id: -1, owner: player, x: lx, y: ly, angle, from_planet_id: src.id, ships };
            let (bodyscreen, eta) = match fleet_eta(&synth, &planet_pos, &planet_radii, game_cfg.ship_speed, h) {
                Some((hi, st)) if hi == tgt_i => (true, st.max(1).min(h)),
                _ => (false, 1usize),
            };
            let floor_ships = capture_floor(&status, tgt_i, eta, player as usize, 1.0) as i64;
            let score = if ships > 0 { score_launch(&status, src_i, tgt_i, ships, eta, player as usize) } else { f64::NEG_INFINITY };
            let roi = score > c.roi_threshold;
            out.push((src.id, tgt.id, avail, eta as i64, angle, viable, floor_ships, bodyscreen,
                      score, roi, offensive.contains(&tgt_i), defensive.contains(&tgt_i)));
        }
    }
    out
}

/// DEBUG: per-player net-ship delta (produced − combat) for a launch, plus the
/// 4 base/hyp produced+combat sums for src & tgt. Returns
/// [delta_net(A), base_prod_src(A), base_combat_src(A), hyp_prod_src(A), hyp_combat_src(A),
///  base_prod_tgt(A), base_combat_tgt(A), hyp_prod_tgt(A), hyp_combat_tgt(A)] flattened.
pub fn score_launch_delta(
    status: &GarrisonStatus, src_idx: usize, tgt_idx: usize, ships: i64, eta: usize, player_id: usize,
) -> Vec<f64> {
    let a = status.a;
    let (bps, bcs) = flow_terms_single(&status.owner[src_idx], &status.pre_owner[src_idx],
        &status.pre_ships[src_idx], &status.arrivals[src_idx], status.prod[src_idx], &status.alive[src_idx], a);
    let (bpt, bct) = flow_terms_single(&status.owner[tgt_idx], &status.pre_owner[tgt_idx],
        &status.pre_ships[tgt_idx], &status.arrivals[tgt_idx], status.prod[tgt_idx], &status.alive[tgt_idx], a);
    let src_init = (status.ships[src_idx][0] - ships as f64).max(0.0);
    let (hos, _hss, hpos, hpss) = planet_trajectory(status.owner[src_idx][0], src_init,
        status.prod[src_idx], &status.alive[src_idx], &status.arrivals[src_idx]);
    let (hps, hcs) = flow_terms_single(&hos, &hpos, &hpss, &status.arrivals[src_idx], status.prod[src_idx], &status.alive[src_idx], a);
    let mut ta = status.arrivals[tgt_idx].clone();
    ta[eta][player_id] += ships as f64;
    let (hot, _hst, hpot, hpst) = planet_trajectory(status.owner[tgt_idx][0], status.ships[tgt_idx][0],
        status.prod[tgt_idx], &status.alive[tgt_idx], &ta);
    let (hpt, hct) = flow_terms_single(&hot, &hpot, &hpst, &ta, status.prod[tgt_idx], &status.alive[tgt_idx], a);
    let mut dn = vec![0.0f64; a];
    for aa in 0..a {
        dn[aa] = (hps[aa]-bps[aa]) + (hpt[aa]-bpt[aa]) - ((hcs[aa]-bcs[aa]) + (hct[aa]-bct[aa]));
    }
    let mut out = dn;
    for v in [bps,bcs,hps,hcs,bpt,bct,hpt,hct] { out.extend(v); }
    out
}

// ── flow scoring ──────────────────────────────────────────────────────────────

/// Competitive net-ship delta for a candidate launch.
///
/// Mirrors `sparse_launch_flow_delta` + `competitive_score` from orbit_lite:
///   score = Δnet_me − Σ_opp Δnet_opp
///   where Δnet = Δproduced − Δcombat_lost
///
/// Only recomputes trajectories for the affected planets (source + target).
pub fn score_launch(
    status: &GarrisonStatus,
    src_idx: usize,
    tgt_idx: usize,
    ships: i64,
    eta: usize,  // steps until arrival (>= 1)
    player_id: usize,
) -> f64 {
    let a = status.a;
    let h = status.h;
    if ships <= 0 || eta == 0 || eta > h {
        return f64::NEG_INFINITY;
    }

    // --- Baseline flow terms for affected planets ---
    let (base_prod_src, base_combat_src) = flow_terms_single(
        &status.owner[src_idx],
        &status.pre_owner[src_idx],
        &status.pre_ships[src_idx],
        &status.arrivals[src_idx],
        status.prod[src_idx],
        &status.alive[src_idx],
        a,
    );
    let (base_prod_tgt, base_combat_tgt) = flow_terms_single(
        &status.owner[tgt_idx],
        &status.pre_owner[tgt_idx],
        &status.pre_ships[tgt_idx],
        &status.arrivals[tgt_idx],
        status.prod[tgt_idx],
        &status.alive[tgt_idx],
        a,
    );

    // --- Hypothetical: debit source, credit target ---

    // Source: ships leave now (debit at k=0).
    let src_init_ships = (status.ships[src_idx][0] - ships as f64).max(0.0);
    let (hyp_owner_src, _hyp_ships_src, hyp_pre_owner_src, hyp_pre_ships_src) =
        planet_trajectory(
            status.owner[src_idx][0],
            src_init_ships,
            status.prod[src_idx],
            &status.alive[src_idx],
            &status.arrivals[src_idx], // arrivals unchanged for source
        );
    let (hyp_prod_src, hyp_combat_src) = flow_terms_single(
        &hyp_owner_src,
        &hyp_pre_owner_src,
        &hyp_pre_ships_src,
        &status.arrivals[src_idx],
        status.prod[src_idx],
        &status.alive[src_idx],
        a,
    );

    // Target: add arrival at step `eta`.
    let mut tgt_arrivals = status.arrivals[tgt_idx].clone();
    tgt_arrivals[eta][player_id] += ships as f64;
    let (hyp_owner_tgt, _hyp_ships_tgt, hyp_pre_owner_tgt, hyp_pre_ships_tgt) =
        planet_trajectory(
            status.owner[tgt_idx][0],
            status.ships[tgt_idx][0],
            status.prod[tgt_idx],
            &status.alive[tgt_idx],
            &tgt_arrivals,
        );
    let (hyp_prod_tgt, hyp_combat_tgt) = flow_terms_single(
        &hyp_owner_tgt,
        &hyp_pre_owner_tgt,
        &hyp_pre_ships_tgt,
        &tgt_arrivals,
        status.prod[tgt_idx],
        &status.alive[tgt_idx],
        a,
    );

    // --- Compute per-player net delta ---
    let mut delta_net = vec![0.0f64; a];
    for aa in 0..a {
        let d_prod = (hyp_prod_src[aa] - base_prod_src[aa])
            + (hyp_prod_tgt[aa] - base_prod_tgt[aa]);
        let d_combat = (hyp_combat_src[aa] - base_combat_src[aa])
            + (hyp_combat_tgt[aa] - base_combat_tgt[aa]);
        delta_net[aa] = d_prod - d_combat;
    }

    // Competitive score: my delta minus sum of opponents' deltas.
    let me = delta_net[player_id];
    let opp: f64 = delta_net.iter().sum::<f64>() - me;
    me - opp
}

/// Owner-aware capture floor: minimum ships to send to guarantee capture
/// at arrival step `eta`, accounting for projected defender garrison.
///
/// Mirrors `capture_floor` in planner_core.py.
pub fn capture_floor(
    status: &GarrisonStatus,
    tgt_idx: usize,
    eta: usize,
    player_id: usize,
    overhead: f64,
) -> f64 {
    if eta == 0 || eta > status.h {
        return 0.0;
    }
    let owner_at_eta = status.owner[tgt_idx][eta];
    if owner_at_eta == player_id as i64 {
        // We'll own it at arrival — just send 1 as reinforcement.
        return 1.0;
    }
    let defenders = status.ships[tgt_idx][eta];
    (defenders + overhead).ceil()
}

// ── ETA-aware reinforcement risk ──────────────────────────────────────────────

/// Reaction-likelihood ramp ρ(eta) ∈ [0, 1].
/// `ρ = clamp((eta − eta_free) / eta_scale, 0, 1)`
pub fn reinforcement_timing_factor(eta: f64, eta_free: f64, eta_scale: f64) -> f64 {
    ((eta - eta_free) / eta_scale.max(1e-6)).clamp(0.0, 1.0)
}

// ── ProducerLite config ───────────────────────────────────────────────────────

/// Max ships that can be drained from `src_i` while its garrison stays >= 0 on every
/// turn in [1..=horizon] where we still hold it. Mirrors orbit_lite's `safe_drain`.
fn safe_drain_planet(status: &GarrisonStatus, src_i: usize, player_id: usize, horizon: usize) -> f64 {
    let h = status.h.min(horizon);
    let mut min_slack = f64::INFINITY;
    for k in 1..=h {
        if status.owner[src_i][k] == player_id as i64 && status.ships[src_i][k] > 0.0 {
            if status.ships[src_i][k] < min_slack {
                min_slack = status.ships[src_i][k];
            }
        }
    }
    if min_slack.is_infinite() {
        status.ships[src_i][0] // doomed → send everything
    } else {
        min_slack.min(status.ships[src_i][0]).max(0.0)
    }
}

/// Configuration matching `ProducerLiteConfig` defaults in orbit_lite.
#[derive(Clone)]
pub struct ProducerLiteConfig {
    pub horizon: usize,
    pub max_sources_per_lane: usize,
    pub max_offensive_targets: usize,
    pub max_defensive_targets: usize,
    pub max_waves_per_turn: usize,
    pub roi_threshold: f64,
    pub min_ships_to_launch: f64,
    pub reinforce_size_beta: f64,
    pub reinforce_eta_free: f64,
    pub reinforce_eta_scale: f64,
    pub enable_regroup: bool,
    pub max_regroup_time: f64,
    pub regroup_pressure_delta_min: f64,
    pub max_regroup_sources_per_lane: usize,
    pub max_regroup_targets_per_source: usize,
    pub regroup_time_penalty_weight: f64,
}

impl Default for ProducerLiteConfig {
    fn default() -> Self {
        ProducerLiteConfig {
            horizon: HORIZON,
            max_sources_per_lane: 12,
            max_offensive_targets: 12,
            max_defensive_targets: 4,
            max_waves_per_turn: 6,
            roi_threshold: 1.5,
            min_ships_to_launch: 4.0,
            reinforce_size_beta: 2.2,
            reinforce_eta_free: 3.0,
            reinforce_eta_scale: 12.0,
            enable_regroup: true,
            max_regroup_time: 7.0,
            regroup_pressure_delta_min: 0.25,
            max_regroup_sources_per_lane: 6,
            max_regroup_targets_per_source: 7,
            regroup_time_penalty_weight: 1e-3,
        }
    }
}

// ── ProducerLiteAgent ─────────────────────────────────────────────────────────

use crate::agents::Agent;

pub struct ProducerLiteAgent {
    pub cfg: ProducerLiteConfig,
}

impl Default for ProducerLiteAgent {
    fn default() -> Self {
        ProducerLiteAgent { cfg: ProducerLiteConfig::default() }
    }
}

impl Agent for ProducerLiteAgent {
    fn name(&self) -> &'static str { "producer_lite" }

    fn act(&self, g: &GameState, player: i64, game_cfg: &Config, step: i64) -> Vec<Move> {
        let player_count = {
            let max_owner = g.planets.iter().map(|p| p.owner).max().unwrap_or(0);
            let max_fleet = g.fleets.iter().map(|f| f.owner).max().unwrap_or(0);
            (max_owner.max(max_fleet) + 1).max(2) as usize
        };

        // Mirror Python's _config_for(player_count): use reduced config for 4P.
        // Mirror Python's _config_for(player_count): reduced params for 4P.
        let cfg4p: ProducerLiteConfig;
        let c: &ProducerLiteConfig = if player_count >= 4 {
            cfg4p = ProducerLiteConfig {
                horizon: 13,
                max_sources_per_lane: 6,
                max_defensive_targets: 2,
                max_regroup_time: 6.0,
                max_regroup_targets_per_source: 8,
                ..self.cfg.clone()
            };
            &cfg4p
        } else {
            &self.cfg
        };
        let h = c.horizon;

        // Build garrison status.
        let status = build_garrison_status(g, game_cfg, player_count, h, step);

        let n = g.planets.len();
        // Pre-compute planet positions and radii — used for shortlist and body screen.
        let planet_radii: Vec<f64> = g.planets.iter().map(|p| p.radius).collect();
        let planet_pos = build_planet_positions(g, h, step);

        // Track committed ships per source planet (index → committed).
        let mut committed = vec![0i64; n];
        let mut moves: Vec<Move> = Vec::new();

        // -----------------------------------------------------------------
        // 1. Score all (src, tgt) candidate launches and collect best.
        // -----------------------------------------------------------------
        #[allow(dead_code)]
        struct Candidate {
            score: f64,
            src_idx: usize,
            tgt_idx: usize,
            ships: i64,
            eta: usize,
            angle: f64,
            rank: usize,   // src_rank * n_tgts + tgt_rank — mirrors orbit_lite flat C-index
        }

        let mut candidates: Vec<Candidate> = Vec::new();

        let owned_idxs: Vec<usize> = (0..n)
            .filter(|&i| g.planets[i].owner == player)
            .collect();

        // Comet planets are excluded from attack targets (mirrors orbit_lite's
        // attack_target_mask = (enemy|neutral) & alive & ~comet). The earlier port
        // ranked them in the offensive shortlist, letting nearby comets crowd out
        // real targets from the top-`max_offensive_targets` slots.
        let comet_ids: std::collections::HashSet<i64> =
            g.comet_planet_ids.iter().copied().collect();

        // cheap_enemy_pressure[p]: reachability-weighted enemy ship count.
        // pressure[p] = Σ_{enemy e} ships[e] * clamp(1 - dist0(e,p)/(speed(e)*H), 0, 1)
        // Used in reinforcement floor term and regroup destination ranking.
        let k0_pos_all = &planet_pos[0];
        let enemy_pressure: Vec<f64> = {
            let mut pres = vec![0.0f64; n];
            for e in 0..n {
                if g.planets[e].owner == player || g.planets[e].owner < 0 { continue; }
                let e_ships = g.planets[e].ships;
                if e_ships <= 0 { continue; }
                let e_speed = fleet_speed_f(e_ships, game_cfg.ship_speed).max(1e-6);
                let reach_dist = (e_speed * h as f64).max(1e-6);
                let (ex, ey) = k0_pos_all[e];
                for p in 0..n {
                    if p == e { continue; }   // mirror orbit_lite ~eye: no self-pressure
                    let (px, py) = k0_pos_all[p];
                    let d = distance(ex, ey, px, py);
                    let decay = (1.0 - d / reach_dist).max(0.0);
                    pres[p] += e_ships as f64 * decay;
                }
            }
            pres
        };

        let offensive_tgt_idxs: Vec<usize> = {
            // Orbital cross-distance: min over k=1..=H of dist(src@step0, tgt@step_k).
            // Mirrors orbit_lite's min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
            // which uses cross_dist[k, s, t] = dist(src@0, tgt@k), k in [1..K].
            let mut ts: Vec<(usize, f64)> = (0..n)
                .filter(|&i| g.planets[i].owner != player && !comet_ids.contains(&g.planets[i].id))
                .map(|i| {
                    // Proximity is measured only from sources that can actually launch
                    // (ships >= min_ships_to_launch) — mirrors orbit_lite's source_mask.
                    let min_cross_d = owned_idxs.iter()
                        .filter(|&&s| g.planets[s].ships >= c.min_ships_to_launch as i64)
                        .map(|&s| {
                        let (sx, sy) = planet_pos[0][s];
                        (1..=h).map(|k| {
                            let (tx, ty) = planet_pos[k][i];
                            distance(sx, sy, tx, ty)
                        }).fold(f64::INFINITY, f64::min)
                    }).fold(f64::INFINITY, f64::min);
                    (i, min_cross_d)
                })
                .collect();
            // proximity asc; ties broken by ascending planet index (orbit_lite's stable topk).
            ts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
            ts.into_iter().take(c.max_offensive_targets).map(|(i, _)| i).collect()
        };

        let defensive_tgt_idxs: Vec<usize> = {
            // Mirrors Python's friendly_flip_targets: owned planets whose garrison
            // projection shows them flipping to an enemy within H steps.
            // urgency = prod * (H - flip_turn) + ships_now (higher = more urgent to defend).
            let mut candidates: Vec<(f64, usize)> = Vec::new();
            for &i in &owned_idxs {
                let flip_turn = (1..=h).find(|&k| {
                    status.owner[i][k] != player && status.owner[i][k] >= 0
                });
                if let Some(ft) = flip_turn {
                    let urgency = status.prod[i] * (h - ft) as f64 + status.ships[i][0];
                    candidates.push((urgency, i));
                }
            }
            // urgency desc; ties broken by ascending planet index (orbit_lite's stable topk).
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            candidates.into_iter().take(c.max_defensive_targets).map(|(_, i)| i).collect()
        };

        let all_tgts: Vec<usize> = {
            let mut v = offensive_tgt_idxs.clone();
            for &t in &defensive_tgt_idxs {
                if !v.contains(&t) {
                    v.push(t);
                }
            }
            v
        };

        // Sources ranked by garrison ships (desc), ties by planet index — mirrors
        // orbit_lite source_idx ordering; the rank drives greedy tie-breaking.
        let mut ranked_srcs: Vec<usize> = owned_idxs.clone();
        ranked_srcs.sort_by(|&a, &b| g.planets[b].ships.cmp(&g.planets[a].ships).then(a.cmp(&b)));
        ranked_srcs.truncate(c.max_sources_per_lane);
        let n_tgts = all_tgts.len();
        for (src_rank, &src_i) in ranked_srcs.iter().enumerate() {
            let src = &g.planets[src_i];

            // Fix 3: safe_drain = max ships we can shed while source garrison stays >= 0
            // over every turn k in [1..H] where we still own the planet. Mirrors
            // orbit_lite's safe_drain(). Ships already committed this turn are subtracted.
            let drain = safe_drain_planet(&status, src_i, player as usize, h);
            let avail = (drain as i64).min(src.ships - committed[src_i]).max(0);
            if avail < c.min_ships_to_launch as i64 {
                continue;
            }

            for (tgt_rank, &tgt_i) in all_tgts.iter().enumerate() {
                if tgt_i == src_i {
                    continue;
                }
                let tgt = &g.planets[tgt_i];

                // Intercept angle.
                let tgt_ip = g.initial_planets.iter().find(|ip| ip.id == tgt.id);
                let (tgt_init_x, tgt_init_y) = tgt_ip
                    .map(|ip| (ip.x, ip.y))
                    .unwrap_or((tgt.x, tgt.y));
                let tgt_orbiting = is_orbiting(tgt_init_x, tgt_init_y, tgt.radius);

                let ships = avail;

                // Compute intercept first so we use the actual t_star for the floor check.
                // Using straight-line eta_est is wrong for orbiting planets (can differ by 2×).
                let (angle, _t_star, viable) = intercept_angle_fp(
                    src.x, src.y, src.radius,
                    tgt_init_x, tgt_init_y, tgt.radius,
                    tgt_orbiting,
                    g.angular_velocity,
                    step,
                    ships,
                    game_cfg.ship_speed,
                    h,
                );
                if !viable {
                    continue;
                }

                // Body screen FIRST — its swept first-contact step IS the canonical eta.
                // orbit_lite's eta = the engine-faithful first-contact step (_analytic_first_contact),
                // NOT ceil() of the continuous FP time (which diverges from the f32 canon at integer
                // boundaries → eta off-by-one → wrong floor/score → spurious launch flips).
                let launch_x = src.x + c_cos(angle) * (src.radius + LAUNCH_SURFACE_OFFSET);
                let launch_y = src.y + c_sin(angle) * (src.radius + LAUNCH_SURFACE_OFFSET);
                let synth = Fleet {
                    id: -1, owner: player, x: launch_x, y: launch_y,
                    angle, from_planet_id: src.id, ships,
                };
                let eta = match fleet_eta(&synth, &planet_pos, &planet_radii, game_cfg.ship_speed, h) {
                    Some((hit_idx, contact_step)) if hit_idx == tgt_i => contact_step.max(1).min(h),
                    _ => continue,
                };

                // Capture floor = ceil(defenders + overhead), owner-aware (capture_floor with
                // reinforcement=None, mirroring the live canonical agent).
                let floor_ships = capture_floor(&status, tgt_i, eta, player as usize, 1.0) as i64;
                if ships < floor_ships || ships < c.min_ships_to_launch as i64 {
                    continue;
                }
                let score = score_launch(&status, src_i, tgt_i, ships, eta, player as usize);
                if score > c.roi_threshold {
                    candidates.push(Candidate { score, src_idx: src_i, tgt_idx: tgt_i, ships, eta, angle, rank: src_rank * n_tgts + tgt_rank });
                }
            }
        }

        // Sort by descending score; ties broken by ascending (src, tgt) planet index
        // to match orbit_lite's device-stable `_stable_argmax` (lowest-index argmax).
        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.rank.cmp(&b.rank))
        });

        // Greedy wave selection — mirrors orbit_lite `_greedy_select`:
        //   * one wave per target (`target_taken`),
        //   * role mutex: a target can't be a planet that already launched as a source
        //     (`used_src`), and a source can't be a planet being reinforced this turn
        //     (`defended`, set for owned/defensive targets).
        let mut target_taken = vec![false; n];
        let mut used_src = vec![false; n];
        let mut defended = vec![false; n];
        for cand in &candidates {
            if moves.len() >= c.max_waves_per_turn {
                break;
            }
            if target_taken[cand.tgt_idx] || used_src[cand.tgt_idx] || defended[cand.src_idx] {
                continue;
            }
            let avail = g.planets[cand.src_idx].ships - committed[cand.src_idx];
            if avail < cand.ships {
                continue;
            }
            committed[cand.src_idx] += cand.ships;
            target_taken[cand.tgt_idx] = true;
            used_src[cand.src_idx] = true;
            if g.planets[cand.tgt_idx].owner == player {
                defended[cand.tgt_idx] = true; // reinforcement of an owned planet
            }
            moves.push(Move {
                from_id: g.planets[cand.src_idx].id,
                angle: cand.angle,
                ships: cand.ships,
            });
        }

        // -----------------------------------------------------------------
        // 2. Regroup: move ships from low-pressure to high-pressure planets.
        //    Mirrors `_plan_regroup` in planner_core.py.
        //    Always runs (even when attacks were found) using leftover ships.
        // -----------------------------------------------------------------
        if c.enable_regroup {
            // Sources: owned planets ranked by leftover ships (descending).
            let mut src_list: Vec<(i64, usize)> = owned_idxs.iter()
                .map(|&i| (g.planets[i].ships - committed[i], i))
                .filter(|&(avail, _)| avail >= c.min_ships_to_launch as i64)
                .collect();
            src_list.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            src_list.truncate(c.max_regroup_sources_per_lane);

            // Destinations: owned, non-comet planets ranked by enemy_pressure (desc).
            // Mirrors orbit_lite dst_mask = owned & alive & ~comet.
            let mut dst_list: Vec<(f64, usize)> = owned_idxs.iter()
                .filter(|&&i| !comet_ids.contains(&g.planets[i].id))
                .map(|&i| (enemy_pressure[i], i))
                .collect();
            dst_list.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            dst_list.truncate(c.max_regroup_targets_per_source);

            // For each source, find best destination. Regroup is a SEPARATE phase from
            // the attack waves in orbit_lite (not capped by max_waves_per_turn) — bounded
            // only by max_regroup_sources_per_lane (the src_list truncation above).
            for (_, src_i) in &src_list {
                let src_i = *src_i;
                // regroup_cap = min(leftover, safe_drain − committed): never shed more than
                // keeps the source safe (orbit_lite caps the regroup send by safe_drain, not
                // the whole leftover — the old port sent everything and over-drained sources).
                let leftover = g.planets[src_i].ships - committed[src_i];
                let dr = safe_drain_planet(&status, src_i, player as usize, h) as i64;
                let ships = leftover.min((dr - committed[src_i]).max(0));
                if ships < c.min_ships_to_launch as i64 { continue; }
                let src = &g.planets[src_i];
                let src_pressure = enemy_pressure[src_i];

                let mut best: Option<(f64, f64)> = None; // (score, angle)
                for (_, dst_i) in &dst_list {
                    let dst_i = *dst_i;
                    if dst_i == src_i { continue; }
                    let gap = enemy_pressure[dst_i] - src_pressure;
                    if gap <= c.regroup_pressure_delta_min { continue; }

                    let dst = &g.planets[dst_i];
                    let dst_ip = g.initial_planets.iter().find(|ip| ip.id == dst.id);
                    let (dst_ix, dst_iy) = dst_ip.map(|ip| (ip.x, ip.y)).unwrap_or((dst.x, dst.y));
                    let orb = is_orbiting(dst_ix, dst_iy, dst.radius);
                    let (angle, _t_star, viable) = intercept_angle_fp(
                        src.x, src.y, src.radius,
                        dst_ix, dst_iy, dst.radius, orb,
                        g.angular_velocity, step, ships, game_cfg.ship_speed, h,
                    );
                    if !viable { continue; }

                    // Body screen → swept first-contact step = canonical eta (orbit_lite uses
                    // the engine first-contact step, not ceil of the continuous FP time).
                    let lx = src.x + c_cos(angle) * (src.radius + LAUNCH_SURFACE_OFFSET);
                    let ly = src.y + c_sin(angle) * (src.radius + LAUNCH_SURFACE_OFFSET);
                    let synth = Fleet { id: -1, owner: player, x: lx, y: ly, angle, from_planet_id: src.id, ships };
                    let eta = match fleet_eta(&synth, &planet_pos, &planet_radii, game_cfg.ship_speed, h) {
                        Some((hi, st)) if hi == dst_i => st.max(1).min(h),
                        _ => continue,
                    };
                    if (eta as f64) > c.max_regroup_time { continue; }
                    if status.owner[dst_i][eta] != player { continue; }   // dst still mine at arrival

                    let score = gap - c.regroup_time_penalty_weight * (eta as f64);
                    if best.as_ref().map_or(true, |(bs, _)| score > *bs) {
                        best = Some((score, angle));
                    }
                }
                if let Some((_, angle)) = best {
                    committed[src_i] += ships;
                    moves.push(Move { from_id: g.planets[src_i].id, angle, ships });
                }
            }
        }

        moves
    }
}
