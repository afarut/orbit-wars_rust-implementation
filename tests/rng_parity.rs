//! Validate the CPython-compatible MT19937 port against reference sequences
//! dumped from CPython 3.12 (`tools/gen_rng_ref.py` -> tests/rng_ref.json).

use std::fs;
use std::path::PathBuf;

use ow_rs::pyrandom::PyRandom;
use serde_json::Value;

fn fbits(x: f64) -> String {
    let b = x.to_le_bytes();
    b.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn make(seed_key: &str) -> PyRandom {
    match seed_key {
        "int_42" => PyRandom::from_int(42),
        "int_1602569207" => PyRandom::from_int(1602569207),
        "str_comet" => PyRandom::from_str("orbit_wars-comet-1602569207-50"),
        "str_comet2" => PyRandom::from_str("orbit_wars-comet-42-150"),
        other => panic!("unknown seed {other}"),
    }
}

#[test]
fn rng_matches_cpython() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/rng_ref.json");
    let j: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    let pi_half = 1.5707963267948966_f64;
    let mut failures: Vec<String> = Vec::new();

    for (seed_key, ref_data) in j.as_object().unwrap() {
        // getrandbits(32)
        let mut r = make(seed_key);
        for (i, v) in ref_data["getrandbits32"].as_array().unwrap().iter().enumerate() {
            let got = r.getrandbits(32);
            let exp = v.as_u64().unwrap();
            if got != exp {
                failures.push(format!("{seed_key} getrandbits32[{i}] exp={exp} got={got}"));
                break;
            }
        }

        // getrandbits(k) for k=1..=64
        let mut r = make(seed_key);
        for (idx, v) in ref_data["getrandbits_var"].as_array().unwrap().iter().enumerate() {
            let k = (idx + 1) as u32;
            let got = r.getrandbits(k);
            let exp = v.as_u64().unwrap();
            if got != exp {
                failures.push(format!("{seed_key} getrandbits({k}) exp={exp} got={got}"));
                break;
            }
        }

        // random()
        let mut r = make(seed_key);
        for (i, v) in ref_data["random"].as_array().unwrap().iter().enumerate() {
            let got = fbits(r.random());
            let exp = v.as_str().unwrap();
            if got != exp {
                failures.push(format!("{seed_key} random[{i}] exp={exp} got={got}"));
                break;
            }
        }

        // randint variants
        for (key, a, b) in [("randint_1_99", 1, 99), ("randint_5_30", 5, 30), ("randint_5_10", 5, 10)] {
            let mut r = make(seed_key);
            for (i, v) in ref_data[key].as_array().unwrap().iter().enumerate() {
                let got = r.randint(a, b);
                let exp = v.as_i64().unwrap();
                if got != exp {
                    failures.push(format!("{seed_key} {key}[{i}] exp={exp} got={got}"));
                    break;
                }
            }
        }

        // uniform variants
        for (key, a, b) in [("uniform_0_pihalf", 0.0, pi_half), ("uniform_60_150", 60.0, 150.0)] {
            let mut r = make(seed_key);
            for (i, v) in ref_data[key].as_array().unwrap().iter().enumerate() {
                let got = fbits(r.uniform(a, b));
                let exp = v.as_str().unwrap();
                if got != exp {
                    failures.push(format!("{seed_key} {key}[{i}] exp={exp} got={got}"));
                    break;
                }
            }
        }
    }

    if !failures.is_empty() {
        for f in &failures {
            println!("FAIL {f}");
        }
        panic!("{} rng parity failures", failures.len());
    }
    println!("PASS: all CPython RNG reference sequences match");
}
