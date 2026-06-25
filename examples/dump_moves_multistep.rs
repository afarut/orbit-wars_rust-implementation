//! Runs a multi-step game and dumps (state, moves) at each step.
//! cargo run --example dump_moves_multistep > /tmp/rust_multistep.json
use ow_rs::agents::AgentKind;
use ow_rs::engine::{init_from_seed, step};
use ow_rs::state::Config;
use serde_json::{json, Value};

fn planet_row(p: &ow_rs::state::Planet) -> Value {
    json!([p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production])
}

fn fleet_row(f: &ow_rs::state::Fleet) -> Value {
    json!([f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships])
}

fn main() {
    let cfg = Config::default();
    let agent = AgentKind::ProducerLite.make();
    let n_steps = 20;
    let mut results: Vec<Value> = Vec::new();

    for seed in 0u64..5 {
        for num_agents in [2usize, 4] {
            let mut g = init_from_seed(seed, num_agents);
            let initial: Vec<Value> = g.initial_planets.iter().map(planet_row).collect();

            for cur_step in 0i64..n_steps {
                let planets: Vec<Value> = g.planets.iter().map(planet_row).collect();
                let fleets: Vec<Value> = g.fleets.iter().map(fleet_row).collect();

                // Collect moves for each player.
                let mut all_moves: Vec<Value> = Vec::new();
                let mut action_vecs: Vec<Vec<ow_rs::state::Move>> = Vec::new();
                for player in 0..num_agents as i64 {
                    if !g.planets.iter().any(|p| p.owner == player) {
                        all_moves.push(json!([]));
                        action_vecs.push(vec![]);
                        continue;
                    }
                    let moves = agent.act(&g, player, &cfg, cur_step);
                    let moves_json: Vec<Value> = moves.iter().map(|m| {
                        json!({ "from_id": m.from_id, "angle": m.angle, "ships": m.ships })
                    }).collect();
                    all_moves.push(json!(moves_json));
                    action_vecs.push(moves);
                }

                results.push(json!({
                    "seed": seed,
                    "num_agents": num_agents,
                    "step": cur_step,
                    "angular_velocity": g.angular_velocity,
                    "planets": planets,
                    "initial_planets": initial,
                    "fleets": fleets,
                    "rust_moves": all_moves,
                }));

                let result = step(&g, &action_vecs, cur_step, &cfg, None, seed);
                if result.terminated {
                    break;
                }
                g = result.state;
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&json!(results)).unwrap());
}
