//! PyO3 bindings: a stateful `Env` for RL training loops, with an observation
//! that matches the original Kaggle `orbit_wars` obs field-for-field so existing
//! `agent(obs)` functions run unchanged.
//!
//! Build with maturin (`maturin develop --release --features python`). Exposes
//! `ow_rs.Env` with `reset(seed, num_agents)`, `step(actions) -> (rewards, done)`,
//! and `observation(player) -> dict`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::engine::{init_from_seed, step_in_place};
use crate::state::*;

type PlanetTuple = (i64, i64, f64, f64, f64, i64, i64);
type FleetTuple = (i64, i64, f64, f64, f64, i64, i64);

const DEFAULT_OVERAGE: f64 = 60.0;

fn planet_tuples(planets: &[Planet]) -> Vec<PlanetTuple> {
    planets
        .iter()
        .map(|p| (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production))
        .collect()
}

fn fleet_tuples(fleets: &[Fleet]) -> Vec<FleetTuple> {
    fleets
        .iter()
        .map(|f| (f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships))
        .collect()
}

/// Build a Kaggle-compatible observation dict for `player` from `state` at `cur_step`.
/// Shared by the single-game `Env` and the vectorized `VecEnv` so both expose an
/// identical obs schema (planets, fleets, angular_velocity, initial_planets,
/// next_fleet_id, comets, comet_planet_ids, remainingOverageTime, step).
pub fn build_obs_dict<'py>(
    py: Python<'py>,
    state: &GameState,
    player: i64,
    cur_step: i64,
) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("planets", planet_tuples(&state.planets))?;
    d.set_item("fleets", fleet_tuples(&state.fleets))?;
    d.set_item("player", player)?;
    d.set_item("angular_velocity", state.angular_velocity)?;
    d.set_item("initial_planets", planet_tuples(&state.initial_planets))?;
    d.set_item("next_fleet_id", state.next_fleet_id)?;
    d.set_item("comet_planet_ids", state.comet_planet_ids.clone())?;
    d.set_item("remainingOverageTime", DEFAULT_OVERAGE)?;
    d.set_item("step", cur_step)?;

    let comets = PyList::empty_bound(py);
    for g in &state.comets {
        let gd = PyDict::new_bound(py);
        gd.set_item("planet_ids", g.planet_ids.clone())?;
        let paths: Vec<Vec<(f64, f64)>> = g
            .paths
            .iter()
            .map(|path| path.iter().map(|pt| (pt[0], pt[1])).collect())
            .collect();
        gd.set_item("paths", paths)?;
        gd.set_item("path_index", g.path_index)?;
        comets.append(gd)?;
    }
    d.set_item("comets", comets)?;
    Ok(d.into_any().unbind())
}

#[pyclass]
pub struct Env {
    state: GameState,
    cfg: Config,
    seed: u64,
    num_agents: usize,
    cur_step: i64,
    done: bool,
}

impl Env {
    /// Build the per-player observation matching the Kaggle obs schema.
    fn build_obs<'py>(&self, py: Python<'py>, player: i64) -> PyResult<PyObject> {
        build_obs_dict(py, &self.state, player, self.cur_step)
    }
}

#[pymethods]
impl Env {
    #[new]
    #[pyo3(signature = (ship_speed = 6.0, episode_steps = 500, comet_speed = 4.0))]
    fn new(ship_speed: f64, episode_steps: i64, comet_speed: f64) -> Self {
        Env {
            state: GameState {
                planets: Vec::new(),
                fleets: Vec::new(),
                next_fleet_id: 0,
                comets: Vec::new(),
                comet_planet_ids: Vec::new(),
                initial_planets: Vec::new(),
                angular_velocity: 0.0,
            },
            cfg: Config { ship_speed, episode_steps, comet_speed },
            seed: 0,
            num_agents: 2,
            cur_step: 0,
            done: false,
        }
    }

    /// Start a new episode from `seed`.
    #[pyo3(signature = (seed, num_agents = 2))]
    fn reset(&mut self, seed: u64, num_agents: usize) {
        self.seed = seed;
        self.num_agents = num_agents;
        self.cur_step = 0;
        self.done = false;
        self.state = init_from_seed(seed, num_agents);
    }

    /// The per-player observation (Kaggle-compatible). Call after reset/step.
    #[pyo3(signature = (player = 0))]
    fn observation(&self, py: Python<'_>, player: i64) -> PyResult<PyObject> {
        self.build_obs(py, player)
    }

    /// Advance one tick. `actions[player]` is a list of `(from_id, angle, ships)`.
    /// Returns `(rewards_or_none, done)`.
    fn step(
        &mut self,
        actions: Vec<Vec<(i64, f64, i64)>>,
    ) -> (Option<Vec<i64>>, bool) {
        let acts: Vec<Vec<Move>> = actions
            .iter()
            .map(|a| {
                a.iter()
                    .map(|&(from_id, angle, ships)| Move { from_id, angle, ships })
                    .collect()
            })
            .collect();
        let out = step_in_place(&mut self.state, &acts, self.cur_step, &self.cfg, None, self.seed);
        self.cur_step += 1;
        self.done = out.terminated;
        (out.rewards, out.terminated)
    }

    #[getter]
    fn step_index(&self) -> i64 {
        self.cur_step
    }

    #[getter]
    fn done(&self) -> bool {
        self.done
    }

    #[getter]
    fn num_agents(&self) -> usize {
        self.num_agents
    }
}

#[pymodule]
fn ow_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<crate::vecenv::VecEnv>()?;
    // AgentKind constants: HOLD=0, PRODUCER_LITE=4
    m.add("AGENT_HOLD", 0i32)?;
    m.add("AGENT_PRODUCER_LITE", 4i32)?;
    Ok(())
}
