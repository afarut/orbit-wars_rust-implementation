//! Helpers to load Kaggle/`env.toJSON()` replay JSON into engine types.
//! Shared by the parity tests and useful for replay-driven training.

use serde_json::Value;

use crate::state::*;

pub fn planet_from(v: &Value) -> Planet {
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

pub fn fleet_from(v: &Value) -> Fleet {
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

pub fn comet_group_from(v: &Value) -> CometGroup {
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

pub fn state_from_obs(obs: &Value) -> GameState {
    GameState {
        planets: obs["planets"].as_array().unwrap().iter().map(planet_from).collect(),
        fleets: obs["fleets"].as_array().unwrap().iter().map(fleet_from).collect(),
        next_fleet_id: obs["next_fleet_id"].as_i64().unwrap(),
        comets: obs["comets"].as_array().unwrap().iter().map(comet_group_from).collect(),
        comet_planet_ids: obs["comet_planet_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect(),
        initial_planets: obs["initial_planets"].as_array().unwrap().iter().map(planet_from).collect(),
        angular_velocity: obs["angular_velocity"].as_f64().unwrap(),
    }
}

pub fn obs_of<'a>(steps: &'a Value, k: usize) -> &'a Value {
    &steps[k][0]["observation"]
}

/// Actions submitted at recorded index k (the decision made from the obs at
/// step k-1; see the step-indexing note in the engine).
pub fn actions_at(steps: &Value, k: usize, n_agents: usize) -> Vec<Vec<Move>> {
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
