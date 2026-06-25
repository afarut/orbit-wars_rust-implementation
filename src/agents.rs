//! Built-in heuristic agents for RL training.
//!
//! Each agent implements the `Agent` trait: given a `GameState`, a player id,
//! config and step index, it returns a list of `Move` orders.
//!
//! Agents:
//!   `HoldAgent`        – never moves (lower bound / placeholder)
//!   `ProducerLiteAgent` – orbit_lite planner, the primary training opponent

use crate::state::*;

// ── trait ────────────────────────────────────────────────────────────────────

pub trait Agent: Send + Sync {
    fn act(&self, g: &GameState, player: i64, cfg: &Config, step: i64) -> Vec<Move>;
    fn name(&self) -> &'static str;
}

// ── integer enum for Python ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentKind {
    Hold = 0,
    ProducerLite = 4,
}

impl AgentKind {
    pub fn from_i32(v: i32) -> Self {
        match v {
            4 => AgentKind::ProducerLite,
            _ => AgentKind::Hold,
        }
    }

    pub fn make(&self) -> Box<dyn Agent> {
        match self {
            AgentKind::Hold => Box::new(HoldAgent),
            AgentKind::ProducerLite => Box::new(
                crate::flow::ProducerLiteAgent::default()
            ),
        }
    }
}

// ── HoldAgent ────────────────────────────────────────────────────────────────

pub struct HoldAgent;

impl Agent for HoldAgent {
    fn name(&self) -> &'static str { "hold" }
    fn act(&self, _g: &GameState, _player: i64, _cfg: &Config, _step: i64) -> Vec<Move> {
        vec![]
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::init_from_seed;

    #[test]
    fn producer_lite_smoke() {
        let g = init_from_seed(42, 2);
        let cfg = Config::default();
        let agent = crate::flow::ProducerLiteAgent::default();
        let moves = agent.act(&g, 0, &cfg, 0);
        for m in &moves {
            let p = g.planets.iter().find(|p| p.id == m.from_id).unwrap();
            assert_eq!(p.owner, 0);
            assert!(m.ships > 0);
        }
    }

    #[test]
    fn producer_lite_4p_smoke() {
        let g = init_from_seed(123, 4);
        let cfg = Config::default();
        let agent = crate::flow::ProducerLiteAgent::default();
        for player in 0..4i64 {
            let moves = agent.act(&g, player, &cfg, 0);
            for m in &moves {
                let p = g.planets.iter().find(|p| p.id == m.from_id).unwrap();
                assert_eq!(p.owner, player);
            }
        }
    }

    #[test]
    fn agent_kind_producer_lite_roundtrip() {
        assert_eq!(AgentKind::from_i32(4), AgentKind::ProducerLite);
        let agent = AgentKind::ProducerLite.make();
        assert_eq!(agent.name(), "producer_lite");
    }
}
