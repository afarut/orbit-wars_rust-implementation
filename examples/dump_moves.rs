//! Dumps initial game states + ProducerLite moves as JSON for parity testing.
//! cargo run --example dump_moves > /tmp/rust_moves.json
use ow_rs::agents::AgentKind;
use ow_rs::engine::init_from_seed;
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
    let mut results: Vec<Value> = Vec::new();

    for seed in 0u64..10 {
        for num_agents in [2usize, 4] {
            let g = init_from_seed(seed, num_agents);

            // Dump planets + initial_planets + fleets
            let planets: Vec<Value> = g.planets.iter().map(planet_row).collect();
            let initial: Vec<Value> = g.initial_planets.iter().map(planet_row).collect();
            let fleets: Vec<Value> = g.fleets.iter().map(fleet_row).collect();

            // Run agent for each player
            let mut all_moves: Vec<Value> = Vec::new();
            for player in 0..num_agents as i64 {
                // Only run for players that own at least one planet
                if !g.planets.iter().any(|p| p.owner == player) {
                    all_moves.push(json!([]));
                    continue;
                }
                let moves = agent.act(&g, player, &cfg, 0);
                let moves_json: Vec<Value> = moves.iter().map(|m| {
                    json!({
                        "from_id": m.from_id,
                        "angle": m.angle,
                        "ships": m.ships
                    })
                }).collect();
                all_moves.push(json!(moves_json));
            }

            results.push(json!({
                "seed": seed,
                "num_agents": num_agents,
                "angular_velocity": g.angular_velocity,
                "planets": planets,
                "initial_planets": initial,
                "fleets": fleets,
                "rust_moves": all_moves
            }));
        }
    }

    println!("{}", serde_json::to_string_pretty(&json!(results)).unwrap());
}
