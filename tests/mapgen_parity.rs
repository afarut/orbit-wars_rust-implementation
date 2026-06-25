//! Validate generate_planets against CPython output (same-platform bit-exact).

use std::fs;
use std::path::PathBuf;

use ow_rs::mapgen::generate_planets;
use ow_rs::pyrandom::PyRandom;
use serde_json::Value;

fn fbits(x: f64) -> String {
    x.to_le_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn generate_planets_matches_cpython() {
    // Golden file generated on this machine (libm-dependent positions); only
    // bit-exact on the same platform. Skip on a forced (Linux) verification host.
    if std::env::var("OW_FORCE_BIT_EXACT").is_ok() || cfg!(feature = "fast_math") {
        eprintln!("skipping same-platform mapgen golden test");
        return;
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/mapgen_ref.json");
    let j: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    let mut failures = Vec::new();
    for (seed_str, planets) in j.as_object().unwrap() {
        let seed: u64 = seed_str.parse().unwrap();
        let mut rng = PyRandom::from_int(seed);
        let got = generate_planets(&mut rng);
        let exp = planets.as_array().unwrap();
        if got.len() != exp.len() {
            failures.push(format!("seed {seed}: count exp={} got={}", exp.len(), got.len()));
            continue;
        }
        for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
            let ok = g.id == e[0].as_i64().unwrap()
                && g.owner == e[1].as_i64().unwrap()
                && fbits(g.x) == e[2].as_str().unwrap()
                && fbits(g.y) == e[3].as_str().unwrap()
                && fbits(g.radius) == e[4].as_str().unwrap()
                && g.ships == e[5].as_i64().unwrap()
                && g.production == e[6].as_i64().unwrap();
            if !ok {
                failures.push(format!(
                    "seed {seed} planet[{i}]: exp={e:?}\n   got id={} owner={} x={} y={} r={} ships={} prod={}",
                    g.id, g.owner, fbits(g.x), fbits(g.y), fbits(g.radius), g.ships, g.production
                ));
                break;
            }
        }
        if failures.is_empty() {
            println!("PASS seed {seed}: {} planets bit-identical", got.len());
        }
    }
    assert!(failures.is_empty(), "mapgen parity failures:\n{}", failures.join("\n"));
}
