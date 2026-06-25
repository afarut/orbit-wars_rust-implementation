//! Pure-Rust throughput: replay a real game's actions through the engine,
//! N games, no FFI / no obs construction.
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ow_rs::engine::{init_from_seed, step_in_place};
use ow_rs::replay::actions_at;
use ow_rs::state::Config;
use serde_json::Value;

fn main() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../79616274.json");
    let j: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let steps = &j["steps"];
    let n_steps = steps.as_array().unwrap().len();
    let n_agents = steps[0].as_array().unwrap().len();
    let seed = j["info"]["seed"].as_u64().unwrap();
    let cfg = Config::default();

    // Pre-extract actions once.
    let actions: Vec<_> = (0..n_steps).map(|k| actions_at(steps, k, n_agents)).collect();

    let games = 2000;
    let eff = (n_steps - 1) as u64;

    // warmup
    {
        let mut st = init_from_seed(seed, n_agents);
        for k in 1..n_steps {
            let out = step_in_place(&mut st, &actions[k], (k - 1) as i64, &cfg, None, seed);
            if out.terminated { break; }
        }
    }

    let t0 = Instant::now();
    for _ in 0..games {
        let mut st = init_from_seed(seed, n_agents);
        for k in 1..n_steps {
            let out = step_in_place(&mut st, &actions[k], (k - 1) as i64, &cfg, None, seed);
            if out.terminated { break; }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let sps = (games as u64 * eff) as f64 / dt;
    println!("pure rust: {games} games x {eff} steps in {dt:.3}s  =  {sps:.0} steps/s  ({:.1} games/s)",
        games as f64 / dt);
}
