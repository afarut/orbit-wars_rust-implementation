//! Single-step parity test: the Rust `step` must reproduce every recorded
//! transition of the 8 golden traces (5 local + 3 real Kaggle replays).
//!
//! For each recorded index k (1..N) we feed the state at index k-1 plus the
//! actions recorded at index k, drive `step` with cur_step = k-1 (injecting
//! the recorded comet group on spawn ticks), and compare the produced
//! planets+fleets to the recorded state at index k — exact f64 equality.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use ow_rs::engine::step;
use ow_rs::state::*;

use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn obs_of<'a>(steps: &'a Value, k: usize) -> &'a Value {
    &steps[k][0]["observation"]
}

fn planet_from(v: &Value) -> Planet {
    Planet {
        id: v[0].as_i64().unwrap(),
        owner: v[1].as_i64().unwrap(),
        x: v[2].as_f64().unwrap(),
        y: v[3].as_f64().unwrap(),
        radius: v[4].as_f64().unwrap(),
        ships: v[5].as_i64().unwrap(),
        production: v[6].as_i64().unwrap(),
    }
}

fn fleet_from(v: &Value) -> Fleet {
    Fleet {
        id: v[0].as_i64().unwrap(),
        owner: v[1].as_i64().unwrap(),
        x: v[2].as_f64().unwrap(),
        y: v[3].as_f64().unwrap(),
        angle: v[4].as_f64().unwrap(),
        from_planet_id: v[5].as_i64().unwrap(),
        ships: v[6].as_i64().unwrap(),
    }
}

fn comet_group_from(v: &Value) -> CometGroup {
    let planet_ids = v["planet_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    let paths = v["paths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|path| {
            path.as_array()
                .unwrap()
                .iter()
                .map(|pt| [pt[0].as_f64().unwrap(), pt[1].as_f64().unwrap()])
                .collect::<Vec<[f64; 2]>>()
        })
        .collect();
    CometGroup {
        planet_ids,
        paths,
        path_index: v["path_index"].as_i64().unwrap(),
    }
}

fn state_from_obs(obs: &Value) -> GameState {
    let planets = obs["planets"].as_array().unwrap().iter().map(planet_from).collect();
    let fleets = obs["fleets"].as_array().unwrap().iter().map(fleet_from).collect();
    let initial_planets =
        obs["initial_planets"].as_array().unwrap().iter().map(planet_from).collect();
    let comets = obs["comets"].as_array().unwrap().iter().map(comet_group_from).collect();
    let comet_planet_ids = obs["comet_planet_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    GameState {
        planets,
        fleets,
        next_fleet_id: obs["next_fleet_id"].as_i64().unwrap(),
        comets,
        comet_planet_ids,
        initial_planets,
        angular_velocity: obs["angular_velocity"].as_f64().unwrap(),
    }
}

fn actions_from(steps: &Value, k: usize, n_agents: usize) -> Vec<Vec<Move>> {
    let mut out = Vec::with_capacity(n_agents);
    for p in 0..n_agents {
        let mut moves = Vec::new();
        if let Some(arr) = steps[k][p]["action"].as_array() {
            for mv in arr {
                if let Some(m) = mv.as_array() {
                    if m.len() == 3 {
                        moves.push(Move {
                            from_id: m[0].as_i64().unwrap(),
                            angle: m[1].as_f64().unwrap(),
                            // ships may be float in JSON; engine ints it.
                            ships: m[2].as_f64().unwrap() as i64,
                        });
                    }
                }
            }
        }
        out.push(moves);
    }
    out
}

/// Find the comet group that newly appears at index k (spawned this tick).
fn spawn_injection(steps: &Value, k: usize) -> Option<SpawnInjection> {
    let prev = obs_of(steps, k - 1);
    let cur = obs_of(steps, k);
    let prev_ids: HashSet<i64> = prev["comet_planet_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    for grp in cur["comets"].as_array().unwrap() {
        let ids: Vec<i64> =
            grp["planet_ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        if ids.iter().any(|id| !prev_ids.contains(id)) {
            // New group. Ships = current ship count of its first comet planet.
            let first_id = ids[0];
            let ships = cur["planets"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p[0].as_i64().unwrap() == first_id)
                .map(|p| p[5].as_i64().unwrap())
                .unwrap();
            let paths = comet_group_from(grp).paths;
            return Some(SpawnInjection { ships, paths });
        }
    }
    None
}

/// Compare expected vs produced state for one step.
///
/// Discrete fields (ids, owners, ships, production, counts) must ALWAYS match
/// exactly. Continuous coordinates must match bit-for-bit when `bit_exact`
/// (same-platform traces); otherwise we only require they stay within a tiny
/// epsilon — cross-platform libm (Linux glibc vs macOS) rounds the last bit of
/// cos/sin/atan2 differently, so real Kaggle (Linux) replays diverge by ~1 ULP
/// in positions while every discrete outcome is identical. Returns the max
/// absolute coordinate delta seen.
fn compare(
    expected: &GameState,
    got: &GameState,
    k: usize,
    bit_exact: bool,
) -> Result<f64, String> {
    let mut max_delta = 0.0_f64;
    let mut chk = |a: f64, b: f64, what: &str| -> Result<(), String> {
        if bit_exact {
            if a.to_bits() != b.to_bits() {
                return Err(format!("step {k}: {what} exp={a:?} got={b:?} (bit-exact)"));
            }
        } else {
            let d = (a - b).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        Ok(())
    };

    if expected.planets.len() != got.planets.len() {
        return Err(format!(
            "step {k}: planet count exp={} got={}",
            expected.planets.len(),
            got.planets.len()
        ));
    }
    for (i, (e, g)) in expected.planets.iter().zip(got.planets.iter()).enumerate() {
        if (e.id, e.owner, e.ships, e.production) != (g.id, g.owner, g.ships, g.production)
            || e.radius.to_bits() != g.radius.to_bits()
        {
            return Err(format!("step {k}: planet[{i}] discrete\n   exp={e:?}\n   got={g:?}"));
        }
        chk(e.x, g.x, &format!("planet[{i}].x"))?;
        chk(e.y, g.y, &format!("planet[{i}].y"))?;
    }

    if expected.fleets.len() != got.fleets.len() {
        return Err(format!(
            "step {k}: fleet count exp={} got={}",
            expected.fleets.len(),
            got.fleets.len()
        ));
    }
    for (i, (e, g)) in expected.fleets.iter().zip(got.fleets.iter()).enumerate() {
        if (e.id, e.owner, e.from_planet_id, e.ships)
            != (g.id, g.owner, g.from_planet_id, g.ships)
            || e.angle.to_bits() != g.angle.to_bits()
        {
            return Err(format!("step {k}: fleet[{i}] discrete\n   exp={e:?}\n   got={g:?}"));
        }
        chk(e.x, g.x, &format!("fleet[{i}].x"))?;
        chk(e.y, g.y, &format!("fleet[{i}].y"))?;
    }
    Ok(max_delta)
}

fn run_trace(path: &PathBuf, bit_exact: bool) -> Result<(usize, f64), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let j: Value = serde_json::from_str(&text).map_err(|e| format!("parse {path:?}: {e}"))?;
    let steps = &j["steps"];
    let n_steps = steps.as_array().unwrap().len();
    let n_agents = steps[0].as_array().unwrap().len();
    let cfg = Config {
        ship_speed: j["configuration"]["shipSpeed"].as_f64().unwrap_or(6.0),
        episode_steps: j["configuration"]["episodeSteps"].as_i64().unwrap_or(500),
        comet_speed: j["configuration"]["cometSpeed"].as_f64().unwrap_or(4.0),
    };

    let mut max_delta = 0.0_f64;
    for k in 1..n_steps {
        let prev = state_from_obs(obs_of(steps, k - 1));
        let actions = actions_from(steps, k, n_agents);
        let cur_step = (k - 1) as i64;
        let spawn = if COMET_SPAWN_STEPS.contains(&(cur_step + 1)) {
            spawn_injection(steps, k)
        } else {
            None
        };
        // Replay-validation mode: comets are injected, so episode_seed unused.
        let res = step(&prev, &actions, cur_step, &cfg, spawn.as_ref(), 0);
        let expected = state_from_obs(obs_of(steps, k));
        let d = compare(&expected, &res.state, k, bit_exact)?;
        if d > max_delta {
            max_delta = d;
        }
    }
    Ok((n_steps, max_delta))
}

/// Same-platform traces (generated locally on this machine): require the Rust
/// step to reproduce every transition BIT-FOR-BIT.
#[test]
fn local_traces_bit_exact() {
    // These traces were generated on this dev machine; bit-exactness only holds
    // on the same platform. Skip when verifying on a forced (Linux) host.
    if std::env::var("OW_FORCE_BIT_EXACT").is_ok() {
        eprintln!("skipping local bit-exact test on non-generating platform");
        return;
    }
    // The fast_math build trades ULP-exactness for speed; require discrete+eps.
    let bit = !cfg!(feature = "fast_math");
    const EPS: f64 = 1e-6;
    let traces = [
        "traces/demo_2p_random_random.json",
        "traces/demo_2p_sniper_random.json",
        "traces/demo_2p_starter_sniper.json",
        "traces/demo_4p_mixed.json",
        "traces/demo_4p_starters.json",
    ];
    let mut failures = Vec::new();
    for t in traces {
        match run_trace(&root().join(t), bit) {
            Ok((n, d)) => {
                println!("PASS {t}: {n} steps (bit={bit}) max_delta={d:.2e}");
                if !bit && d > EPS {
                    failures.push(t);
                }
            }
            Err(e) => {
                println!("FAIL {t}:\n  {e}");
                failures.push(t);
            }
        }
    }
    assert!(failures.is_empty(), "local traces failed parity: {failures:?}");
}

/// Real Kaggle replays were generated on Linux (glibc libm). cos/sin/atan2
/// round the last bit differently there, so positions differ by ~1 ULP — but
/// every DISCRETE outcome (ownership, ships, captures, fleet survival) must be
/// identical. We require discrete-exact and positions within a tiny epsilon.
#[test]
fn kaggle_traces_discrete_exact() {
    const EPS: f64 = 1e-6;
    // On a Linux/glibc host matching Kaggle's runtime these replays should
    // reproduce bit-for-bit. Set OW_FORCE_BIT_EXACT=1 to require that.
    // The fast_math build is never bit-exact → discrete+eps only.
    let bit_exact =
        std::env::var("OW_FORCE_BIT_EXACT").is_ok() && !cfg!(feature = "fast_math");
    let traces = ["79616274.json", "79623944.json", "79629315.json"];
    let mut failures = Vec::new();
    for t in traces {
        match run_trace(&root().join(t), bit_exact) {
            Ok((n, max_delta)) => {
                println!("PASS {t}: {n} steps discrete-identical, max pos delta={max_delta:.2e}");
                if max_delta > EPS {
                    failures.push(format!("{t}: max delta {max_delta:.2e} > {EPS:.0e}"));
                }
            }
            Err(e) => {
                println!("FAIL {t}:\n  {e}");
                failures.push(t.to_string());
            }
        }
    }
    assert!(failures.is_empty(), "kaggle traces failed parity: {failures:?}");
}
