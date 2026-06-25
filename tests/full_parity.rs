//! End-to-end parity from a SEED ONLY: regenerate the map + comets via the
//! CPython-compatible RNG, then chain `step` with the recorded actions, and
//! compare the whole trajectory to the recorded replay. This exercises
//! init_from_seed + RNG comet spawning + the step pipeline together.
//!
//! Same-platform traces reproduce bit-for-bit; on a Linux/glibc host (set
//! OW_FORCE_BIT_EXACT=1) the real Kaggle replays reproduce bit-for-bit too.

use std::fs;
use std::path::PathBuf;

use ow_rs::engine::{init_from_seed, step_in_place};
use ow_rs::replay::{actions_at, obs_of, state_from_obs};
use ow_rs::state::*;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn compare(exp: &GameState, got: &GameState, k: usize, bit_exact: bool) -> Result<f64, String> {
    let mut max_delta = 0.0_f64;
    let mut chk = |a: f64, b: f64, what: &str| -> Result<(), String> {
        if bit_exact {
            if a.to_bits() != b.to_bits() {
                return Err(format!("step {k}: {what} exp={a:?} got={b:?}"));
            }
        } else {
            let d = (a - b).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        Ok(())
    };
    if exp.planets.len() != got.planets.len() {
        return Err(format!("step {k}: planet count {} vs {}", exp.planets.len(), got.planets.len()));
    }
    for (i, (e, g)) in exp.planets.iter().zip(got.planets.iter()).enumerate() {
        if (e.id, e.owner, e.ships, e.production) != (g.id, g.owner, g.ships, g.production)
            || e.radius.to_bits() != g.radius.to_bits()
        {
            return Err(format!("step {k}: planet[{i}] discrete exp={e:?} got={g:?}"));
        }
        chk(e.x, g.x, &format!("planet[{i}].x"))?;
        chk(e.y, g.y, &format!("planet[{i}].y"))?;
    }
    if exp.fleets.len() != got.fleets.len() {
        return Err(format!("step {k}: fleet count {} vs {}", exp.fleets.len(), got.fleets.len()));
    }
    for (i, (e, g)) in exp.fleets.iter().zip(got.fleets.iter()).enumerate() {
        if (e.id, e.owner, e.from_planet_id, e.ships) != (g.id, g.owner, g.from_planet_id, g.ships)
            || e.angle.to_bits() != g.angle.to_bits()
        {
            return Err(format!("step {k}: fleet[{i}] discrete exp={e:?} got={g:?}"));
        }
        chk(e.x, g.x, &format!("fleet[{i}].x"))?;
        chk(e.y, g.y, &format!("fleet[{i}].y"))?;
    }
    Ok(max_delta)
}

fn run_from_seed(path: &PathBuf, bit_exact: bool) -> Result<(usize, f64), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let j: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let steps = &j["steps"];
    let n_steps = steps.as_array().unwrap().len();
    let n_agents = steps[0].as_array().unwrap().len();
    let seed = j["info"]["seed"].as_u64().ok_or("missing info.seed")?;
    let cfg = Config {
        ship_speed: j["configuration"]["shipSpeed"].as_f64().unwrap_or(6.0),
        episode_steps: j["configuration"]["episodeSteps"].as_i64().unwrap_or(500),
        comet_speed: j["configuration"]["cometSpeed"].as_f64().unwrap_or(4.0),
    };

    let mut state = init_from_seed(seed, n_agents);
    let mut max_delta = compare(&state_from_obs(obs_of(steps, 0)), &state, 0, bit_exact)?;

    for k in 1..n_steps {
        let actions = actions_at(steps, k, n_agents);
        step_in_place(&mut state, &actions, (k - 1) as i64, &cfg, None, seed);
        let d = compare(&state_from_obs(obs_of(steps, k)), &state, k, bit_exact)?;
        if d > max_delta {
            max_delta = d;
        }
    }
    Ok((n_steps, max_delta))
}

const LOCAL: [&str; 5] = [
    "traces/demo_2p_random_random.json",
    "traces/demo_2p_sniper_random.json",
    "traces/demo_2p_starter_sniper.json",
    "traces/demo_4p_mixed.json",
    "traces/demo_4p_starters.json",
];
const KAGGLE: [&str; 3] = ["79616274.json", "79623944.json", "79629315.json"];

#[test]
fn full_game_from_seed() {
    const EPS: f64 = 1e-6;
    // fast_math never reproduces bit-for-bit; fall back to discrete+eps.
    let exact_ok = !cfg!(feature = "fast_math");
    let force = std::env::var("OW_FORCE_BIT_EXACT").is_ok();
    let mut failures: Vec<String> = Vec::new();

    // Local traces: bit-exact on the platform that generated them (macOS).
    // On a forced (Linux) run they were generated elsewhere — report only.
    for t in LOCAL {
        let bit = !force && exact_ok;
        match run_from_seed(&root().join(t), bit) {
            Ok((n, d)) => {
                println!("PASS local {t}: {n} steps from seed (bit={bit}) max_delta={d:.2e}");
                if !bit && d > EPS {
                    failures.push(format!("{t}: delta {d:.2e}"));
                }
            }
            Err(e) => {
                if force {
                    println!("(local {t} not bit-exact on this platform — expected)\n  {e}");
                } else {
                    failures.push(format!("{t}: {e}"));
                }
            }
        }
    }

    // Kaggle replays: bit-exact only on a matching Linux/glibc host.
    for t in KAGGLE {
        let bit = force && exact_ok;
        match run_from_seed(&root().join(t), bit) {
            Ok((n, d)) => {
                println!("PASS kaggle {t}: {n} steps from seed (bit={bit}) max_delta={d:.2e}");
                if !bit && d > EPS {
                    failures.push(format!("{t}: delta {d:.2e}"));
                }
            }
            Err(e) => failures.push(format!("{t}: {e}")),
        }
    }

    assert!(failures.is_empty(), "full-from-seed failures:\n{}", failures.join("\n"));
}
