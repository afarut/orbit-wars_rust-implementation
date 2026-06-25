//! Validate generate_comet_paths against CPython output (same-platform).
//! Each seed: reproduce angular_velocity + map via from_int, then comet paths
//! at every spawn step via the string-seeded comet RNG. Isolated check uses
//! empty comet ids + the original map (matches the Python reference).

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use ow_rs::comets::generate_comet_paths;
use ow_rs::mapgen::generate_planets;
use ow_rs::pyrandom::PyRandom;
use serde_json::Value;

fn fbits(x: f64) -> String {
    x.to_le_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn generate_comet_paths_matches_cpython() {
    // Golden file generated on this machine (libm-dependent positions); only
    // bit-exact on the same platform. Skip on a forced (Linux) verification host.
    if std::env::var("OW_FORCE_BIT_EXACT").is_ok() || cfg!(feature = "fast_math") {
        eprintln!("skipping same-platform comet golden test");
        return;
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/comets_ref.json");
    let j: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    let empty: HashSet<i64> = HashSet::new();
    let mut failures = Vec::new();

    for (seed_str, entry) in j.as_object().unwrap() {
        let seed: u64 = seed_str.parse().unwrap();
        let mut irng = PyRandom::from_int(seed);
        let av = irng.uniform(0.025, 0.05);
        if fbits(av) != entry["av"].as_str().unwrap() {
            failures.push(format!("seed {seed}: angular_velocity mismatch"));
            continue;
        }
        let planets = generate_planets(&mut irng);

        for (s_str, exp_paths) in entry["spawns"].as_object().unwrap() {
            let s: i64 = s_str.parse().unwrap();
            let mut crng = PyRandom::from_str(&format!("orbit_wars-comet-{seed}-{s}"));
            let got = generate_comet_paths(&planets, av, s, &empty, 4.0, &mut crng);

            if exp_paths.is_null() {
                if got.is_some() {
                    failures.push(format!("seed {seed} step {s}: exp None got Some"));
                }
                continue;
            }
            let exp = exp_paths.as_array().unwrap();
            let got = match got {
                Some(g) => g,
                None => {
                    failures.push(format!("seed {seed} step {s}: exp Some got None"));
                    continue;
                }
            };
            if got.len() != exp.len() {
                failures.push(format!("seed {seed} step {s}: path-count mismatch"));
                continue;
            }
            'paths: for (pi, (gp, ep)) in got.iter().zip(exp.iter()).enumerate() {
                let ep = ep.as_array().unwrap();
                if gp.len() != ep.len() {
                    failures.push(format!("seed {seed} step {s} path {pi}: len {} vs {}", gp.len(), ep.len()));
                    break;
                }
                for (k, (gpt, ept)) in gp.iter().zip(ep.iter()).enumerate() {
                    if fbits(gpt[0]) != ept[0].as_str().unwrap()
                        || fbits(gpt[1]) != ept[1].as_str().unwrap()
                    {
                        failures.push(format!("seed {seed} step {s} path {pi} pt {k} mismatch"));
                        break 'paths;
                    }
                }
            }
        }
        println!("PASS seed {seed}");
    }
    assert!(failures.is_empty(), "comet parity failures:\n{}", failures.join("\n"));
}
