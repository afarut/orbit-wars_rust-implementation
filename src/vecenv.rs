//! Vectorized, multi-threaded environment for high-throughput RL from PyTorch.
//!
//! `VecEnv` holds N independent games, steps them all in parallel (rayon, GIL
//! released) per call, auto-resets finished games, and returns observations as
//! a single float32 numpy tensor — so the Python↔Rust boundary is crossed once
//! per batch instead of once per env-step.
//!
//! Observation encoding (per env, per player), a flat float32 vector:
//!   planets block: MAX_PLANETS x P_FEAT
//!     [exists, x/100, y/100, radius, ships/200, prod/5, is_mine, is_enemy,
//!      is_neutral, is_comet]
//!   fleets block:  MAX_FLEETS x F_FEAT
//!     [exists, x/100, y/100, sin(angle), cos(angle), ships/200, is_mine]
//! Owner flags are relative to the player whose view this is.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::agents::{Agent, AgentKind};
use crate::engine::{init_from_seed, step_in_place};
use crate::features as feat;
use crate::py::build_obs_dict;
use crate::state::*;

pub const MAX_PLANETS: usize = 48;
pub const MAX_FLEETS: usize = 128;
pub const P_FEAT: usize = 10;
pub const F_FEAT: usize = 7;
/// Two global scalars appended after the fleets block: [angular_velocity, cur_step/500].
pub const GLOBAL_EXTRA: usize = 2;
pub const OBS_DIM: usize = MAX_PLANETS * P_FEAT + MAX_FLEETS * F_FEAT + GLOBAL_EXTRA;

fn encode_env(g: &GameState, player: i64, cur_step: i64, out: &mut [f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let np = g.planets.len().min(MAX_PLANETS);
    for i in 0..np {
        let p = &g.planets[i];
        let b = i * P_FEAT;
        out[b] = 1.0;
        out[b + 1] = (p.x as f32) / 100.0;
        out[b + 2] = (p.y as f32) / 100.0;
        out[b + 3] = p.radius as f32;
        out[b + 4] = (p.ships as f32) / 200.0;
        out[b + 5] = (p.production as f32) / 5.0;
        out[b + 6] = if p.owner == player { 1.0 } else { 0.0 };
        out[b + 7] = if p.owner != -1 && p.owner != player { 1.0 } else { 0.0 };
        out[b + 8] = if p.owner == -1 { 1.0 } else { 0.0 };
        out[b + 9] = if g.comet_planet_ids.contains(&p.id) { 1.0 } else { 0.0 };
    }
    let foff = MAX_PLANETS * P_FEAT;
    let nf = g.fleets.len().min(MAX_FLEETS);
    for j in 0..nf {
        let f = &g.fleets[j];
        let b = foff + j * F_FEAT;
        out[b] = 1.0;
        out[b + 1] = (f.x as f32) / 100.0;
        out[b + 2] = (f.y as f32) / 100.0;
        out[b + 3] = f.angle.sin() as f32;
        out[b + 4] = f.angle.cos() as f32;
        out[b + 5] = (f.ships as f32) / 200.0;
        out[b + 6] = if f.owner == player { 1.0 } else { 0.0 };
    }
    // Global scalars at end: [angular_velocity, cur_step/500]
    let goff = foff + MAX_FLEETS * F_FEAT;
    out[goff]     = g.angular_velocity as f32;
    out[goff + 1] = cur_step as f32 / 500.0;
}

#[pyclass]
pub struct VecEnv {
    envs: Vec<GameState>,
    cfg: Config,
    seeds: Vec<u64>,
    cur_step: Vec<i64>,
    num_agents: usize,
    n: usize,
    next_seed: u64,
    /// Per-env opponent agents: opponents[env_i][slot] where slot 0 = player 1, etc.
    /// None means that slot's actions come from Python.
    opponents: Vec<Vec<Option<Box<dyn Agent>>>>,
}

impl VecEnv {
    fn encode_all(&self) -> Vec<f32> {
        let a = self.num_agents;
        let mut obs = vec![0f32; self.n * a * OBS_DIM];
        obs.par_chunks_mut(a * OBS_DIM)
            .zip(self.envs.par_iter())
            .zip(self.cur_step.par_iter())
            .for_each(|((chunk, env), &cs)| {
                for p in 0..a {
                    encode_env(env, p as i64, cs, &mut chunk[p * OBS_DIM..(p + 1) * OBS_DIM]);
                }
            });
        obs
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (num_envs, num_agents = 2, ship_speed = 6.0, episode_steps = 500, comet_speed = 4.0))]
    fn new(
        num_envs: usize,
        num_agents: usize,
        ship_speed: f64,
        episode_steps: i64,
        comet_speed: f64,
    ) -> Self {
        let opponents = (0..num_envs)
            .map(|_| (0..num_agents - 1).map(|_| None).collect())
            .collect();
        VecEnv {
            envs: Vec::new(),
            cfg: Config { ship_speed, episode_steps, comet_speed },
            seeds: Vec::new(),
            cur_step: vec![0; num_envs],
            num_agents,
            n: num_envs,
            next_seed: 0,
            opponents,
        }
    }

    /// Set opponent agents for all envs at once.
    ///
    /// `agent_kinds` is a flat list of length `num_envs * (num_agents - 1)`.
    /// Row `e` = kinds for env `e`'s opponent slots (players 1..num_agents).
    /// AgentKind: 0=Hold, 1=Greedy, 2=Producer, 3=Defender.
    fn set_opponents(&mut self, agent_kinds: Vec<i32>) {
        let slots = self.num_agents - 1;
        assert_eq!(agent_kinds.len(), self.n * slots,
            "agent_kinds must have length num_envs * (num_agents - 1)");
        for e in 0..self.n {
            for s in 0..slots {
                let kind = AgentKind::from_i32(agent_kinds[e * slots + s]);
                self.opponents[e][s] = Some(kind.make());
            }
        }
    }

    /// Step all envs where player 0 actions come from Python and opponents
    /// are computed internally by built-in agents (set via `set_opponents`).
    ///
    /// `p0_actions` is float32 [N, K, 3] — source_planet_index / angle / ships
    /// for player 0 only.  Returns (obs [N,A,OBS_DIM], rewards [N,A], dones [N]).
    fn step_p0<'py>(
        &mut self,
        py: Python<'py>,
        p0_actions: PyReadonlyArray3<f32>,
    ) -> (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<i32>>,
        Bound<'py, PyArray1<u8>>,
    ) {
        let n = self.n;
        let a = self.num_agents;
        let av = p0_actions.as_array();
        let k = av.shape()[1];

        // Parse player-0 actions (GIL held).
        let p0_acts: Vec<Vec<Move>> = (0..n)
            .map(|e| {
                let env = &self.envs[e];
                let mut mv = Vec::new();
                for kk in 0..k {
                    let ships = av[[e, kk, 2]] as i64;
                    let idx = av[[e, kk, 0]] as usize;
                    if ships > 0 && idx < env.planets.len() {
                        mv.push(Move {
                            from_id: env.planets[idx].id,
                            angle: av[[e, kk, 1]] as f64,
                            ships,
                        });
                    }
                }
                mv
            })
            .collect();

        let mut term = vec![false; n];
        let mut rew = vec![0i32; n * a];
        let mut obs_buf = vec![0f32; n * a * OBS_DIM];

        let envs = &mut self.envs;
        let cur_step = &mut self.cur_step;
        let seeds = &mut self.seeds;
        let cfg = &self.cfg;
        let next_seed = &mut self.next_seed;
        let opponents = &self.opponents;

        py.allow_threads(|| {
            // Parallel step.
            envs.par_iter_mut()
                .zip(cur_step.par_iter_mut())
                .zip(term.par_iter_mut())
                .zip(rew.par_chunks_mut(a))
                .enumerate()
                .for_each(|(e, (((env, cs), t), rslot))| {
                    // Build actions for all players.
                    let mut acts: Vec<Vec<Move>> = Vec::with_capacity(a);
                    acts.push(p0_acts[e].clone()); // player 0
                    for slot in 0..a - 1 {
                        let opp_moves = if let Some(agent) = &opponents[e][slot] {
                            agent.act(env, (slot + 1) as i64, cfg, *cs)
                        } else {
                            vec![]
                        };
                        acts.push(opp_moves);
                    }

                    let out = step_in_place(env, &acts, *cs, cfg, None, seeds[e]);
                    *cs += 1;
                    *t = out.terminated;
                    if let Some(r) = out.rewards {
                        for (i, v) in r.iter().enumerate() {
                            rslot[i] = *v as i32;
                        }
                    }
                });

            // Serial auto-reset.
            for e in 0..n {
                if term[e] {
                    let ns = *next_seed;
                    *next_seed += 1;
                    envs[e] = init_from_seed(ns, a);
                    cur_step[e] = 0;
                    seeds[e] = ns;
                }
            }

            // Parallel observation encoding.
            obs_buf.par_chunks_mut(a * OBS_DIM)
                .zip(envs.par_iter())
                .zip(cur_step.par_iter())
                .for_each(|((chunk, env), &cs)| {
                    for p in 0..a {
                        encode_env(env, p as i64, cs, &mut chunk[p * OBS_DIM..(p + 1) * OBS_DIM]);
                    }
                });
        });

        let obs_arr = PyArray1::from_vec_bound(py, obs_buf)
            .reshape([n, a, OBS_DIM])
            .unwrap();
        let rew_arr = PyArray1::from_vec_bound(py, rew).reshape([n, a]).unwrap();
        let dones = PyArray1::from_vec_bound(py, term.iter().map(|&b| b as u8).collect::<Vec<_>>());
        (obs_arr, rew_arr, dones)
    }

    /// Batched feature encoding (bit-for-bit port of core.features.encode), computed
    /// in parallel across envs (rayon, GIL released). Returns a dict of numpy arrays:
    ///   planet_feats [N,40,20], planet_mask [N,40], comet_feats [N,16,25],
    ///   comet_mask [N,16], fleet_feats [N,256,10], fleet_mask [N,256],
    ///   global_feats [N,11], place_meta [N,56,8], comet_paths [N,16,max_path,2],
    ///   comet_plen [N,16], comet_pidx [N,16], max_path (int).
    /// place_meta cols: [valid, id, owner, x, y, ships, kind(0=static/1=orbit/2=comet), ang_vel].
    fn encode_features<'py>(&self, py: Python<'py>, player: i64) -> PyResult<Bound<'py, PyDict>> {
        let n = self.n;
        let states: Vec<&GameState> = self.envs.iter().collect();
        let max_path = feat::max_comet_path_len(&states).max(1);

        struct Ef {
            p: Vec<f32>, pm: Vec<u8>, c: Vec<f32>, cm: Vec<u8>,
            f: Vec<f32>, fm: Vec<u8>, gl: Vec<f32>, meta: Vec<f64>,
            cpath: Vec<f64>, cplen: Vec<i32>, cpidx: Vec<i32>, phi: Vec<f64>,
        }

        let envs = &self.envs;
        let steps = &self.cur_step;
        let results: Vec<Ef> = py.allow_threads(|| {
            (0..n).into_par_iter().map(|e| {
                let mut ef = Ef {
                    p: vec![0.0; feat::F_MAX_P * feat::PFD], pm: vec![0; feat::F_MAX_P],
                    c: vec![0.0; feat::F_MAX_C * feat::CFD], cm: vec![0; feat::F_MAX_C],
                    f: vec![0.0; feat::F_MAX_F * feat::FFD], fm: vec![0; feat::F_MAX_F],
                    gl: vec![0.0; feat::GFD], meta: vec![0.0; feat::M_PLACES * feat::META_W],
                    cpath: vec![0.0; feat::F_MAX_C * max_path * 2],
                    cplen: vec![0; feat::F_MAX_C], cpidx: vec![0; feat::F_MAX_C],
                    phi: vec![0.0; 1],
                };
                feat::encode_env_features(
                    &envs[e], player, steps[e],
                    &mut ef.p, &mut ef.pm, &mut ef.c, &mut ef.cm, &mut ef.f, &mut ef.fm,
                    &mut ef.gl, &mut ef.meta, &mut ef.cpath, &mut ef.cplen, &mut ef.cpidx,
                    &mut ef.phi, max_path,
                );
                ef
            }).collect()
        });

        // склейка per-env буферов в плоские
        let mut p = Vec::with_capacity(n * feat::F_MAX_P * feat::PFD);
        let mut pm = Vec::with_capacity(n * feat::F_MAX_P);
        let mut c = Vec::with_capacity(n * feat::F_MAX_C * feat::CFD);
        let mut cm = Vec::with_capacity(n * feat::F_MAX_C);
        let mut f = Vec::with_capacity(n * feat::F_MAX_F * feat::FFD);
        let mut fm = Vec::with_capacity(n * feat::F_MAX_F);
        let mut gl = Vec::with_capacity(n * feat::GFD);
        let mut meta: Vec<f64> = Vec::with_capacity(n * feat::M_PLACES * feat::META_W);
        let mut cpath: Vec<f64> = Vec::with_capacity(n * feat::F_MAX_C * max_path * 2);
        let mut cplen = Vec::with_capacity(n * feat::F_MAX_C);
        let mut cpidx = Vec::with_capacity(n * feat::F_MAX_C);
        let mut phi = Vec::with_capacity(n);
        for ef in &results {
            p.extend_from_slice(&ef.p); pm.extend_from_slice(&ef.pm);
            c.extend_from_slice(&ef.c); cm.extend_from_slice(&ef.cm);
            f.extend_from_slice(&ef.f); fm.extend_from_slice(&ef.fm);
            gl.extend_from_slice(&ef.gl); meta.extend_from_slice(&ef.meta);
            cpath.extend_from_slice(&ef.cpath);
            cplen.extend_from_slice(&ef.cplen); cpidx.extend_from_slice(&ef.cpidx);
            phi.push(ef.phi[0]);
        }

        let d = PyDict::new_bound(py);
        d.set_item("planet_feats", PyArray1::from_vec_bound(py, p).reshape([n, feat::F_MAX_P, feat::PFD]).unwrap())?;
        d.set_item("planet_mask", PyArray1::from_vec_bound(py, pm).reshape([n, feat::F_MAX_P]).unwrap())?;
        d.set_item("comet_feats", PyArray1::from_vec_bound(py, c).reshape([n, feat::F_MAX_C, feat::CFD]).unwrap())?;
        d.set_item("comet_mask", PyArray1::from_vec_bound(py, cm).reshape([n, feat::F_MAX_C]).unwrap())?;
        d.set_item("fleet_feats", PyArray1::from_vec_bound(py, f).reshape([n, feat::F_MAX_F, feat::FFD]).unwrap())?;
        d.set_item("fleet_mask", PyArray1::from_vec_bound(py, fm).reshape([n, feat::F_MAX_F]).unwrap())?;
        d.set_item("global_feats", PyArray1::from_vec_bound(py, gl).reshape([n, feat::GFD]).unwrap())?;
        d.set_item("place_meta", PyArray1::from_vec_bound(py, meta).reshape([n, feat::M_PLACES, feat::META_W]).unwrap())?;
        d.set_item("comet_paths", PyArray1::from_vec_bound(py, cpath).reshape([n, feat::F_MAX_C, max_path, 2]).unwrap())?;
        d.set_item("comet_plen", PyArray1::from_vec_bound(py, cplen).reshape([n, feat::F_MAX_C]).unwrap())?;
        d.set_item("comet_pidx", PyArray1::from_vec_bound(py, cpidx).reshape([n, feat::F_MAX_C]).unwrap())?;
        d.set_item("phi", PyArray1::from_vec_bound(py, phi))?;
        d.set_item("max_path", max_path)?;
        Ok(d)
    }

    /// Как `encode_features`, но КАНОНИЗирует перспективу `player` -> p0 (поворот -φ +
    /// релейбл, бит-точно к model._canonicalize) и кодирует канон-состояние как player 0.
    /// Доп. ключ `phi_canon` [N] — φ на env (прибавить к выходному углу decode).
    /// Снимает per-obs python-encode у self-play оппонентов (главный тормоз sps).
    fn encode_features_canon<'py>(&self, py: Python<'py>, player: i64) -> PyResult<Bound<'py, PyDict>> {
        let n = self.n;
        let states: Vec<&GameState> = self.envs.iter().collect();
        let max_path = feat::max_comet_path_len(&states).max(1); // повороты не меняют длины путей

        struct Ef {
            p: Vec<f32>, pm: Vec<u8>, c: Vec<f32>, cm: Vec<u8>,
            f: Vec<f32>, fm: Vec<u8>, gl: Vec<f32>, meta: Vec<f64>,
            cpath: Vec<f64>, cplen: Vec<i32>, cpidx: Vec<i32>, phi: Vec<f64>, phic: f64,
        }
        let envs = &self.envs;
        let steps = &self.cur_step;
        let results: Vec<Ef> = py.allow_threads(|| {
            (0..n).into_par_iter().map(|e| {
                let (cstate, phic) = feat::canonicalize_state(&envs[e], player);
                let mut ef = Ef {
                    p: vec![0.0; feat::F_MAX_P * feat::PFD], pm: vec![0; feat::F_MAX_P],
                    c: vec![0.0; feat::F_MAX_C * feat::CFD], cm: vec![0; feat::F_MAX_C],
                    f: vec![0.0; feat::F_MAX_F * feat::FFD], fm: vec![0; feat::F_MAX_F],
                    gl: vec![0.0; feat::GFD], meta: vec![0.0; feat::M_PLACES * feat::META_W],
                    cpath: vec![0.0; feat::F_MAX_C * max_path * 2],
                    cplen: vec![0; feat::F_MAX_C], cpidx: vec![0; feat::F_MAX_C],
                    phi: vec![0.0; 1], phic,
                };
                feat::encode_env_features(
                    &cstate, 0, steps[e],
                    &mut ef.p, &mut ef.pm, &mut ef.c, &mut ef.cm, &mut ef.f, &mut ef.fm,
                    &mut ef.gl, &mut ef.meta, &mut ef.cpath, &mut ef.cplen, &mut ef.cpidx,
                    &mut ef.phi, max_path,
                );
                ef
            }).collect()
        });

        let mut p = Vec::new(); let mut pm = Vec::new();
        let mut c = Vec::new(); let mut cm = Vec::new();
        let mut f = Vec::new(); let mut fm = Vec::new();
        let mut gl = Vec::new(); let mut meta: Vec<f64> = Vec::new();
        let mut cpath: Vec<f64> = Vec::new(); let mut cplen = Vec::new(); let mut cpidx = Vec::new();
        let mut phi = Vec::new(); let mut phic = Vec::with_capacity(n);
        for ef in &results {
            p.extend_from_slice(&ef.p); pm.extend_from_slice(&ef.pm);
            c.extend_from_slice(&ef.c); cm.extend_from_slice(&ef.cm);
            f.extend_from_slice(&ef.f); fm.extend_from_slice(&ef.fm);
            gl.extend_from_slice(&ef.gl); meta.extend_from_slice(&ef.meta);
            cpath.extend_from_slice(&ef.cpath);
            cplen.extend_from_slice(&ef.cplen); cpidx.extend_from_slice(&ef.cpidx);
            phi.push(ef.phi[0]); phic.push(ef.phic);
        }
        let d = PyDict::new_bound(py);
        d.set_item("planet_feats", PyArray1::from_vec_bound(py, p).reshape([n, feat::F_MAX_P, feat::PFD]).unwrap())?;
        d.set_item("planet_mask", PyArray1::from_vec_bound(py, pm).reshape([n, feat::F_MAX_P]).unwrap())?;
        d.set_item("comet_feats", PyArray1::from_vec_bound(py, c).reshape([n, feat::F_MAX_C, feat::CFD]).unwrap())?;
        d.set_item("comet_mask", PyArray1::from_vec_bound(py, cm).reshape([n, feat::F_MAX_C]).unwrap())?;
        d.set_item("fleet_feats", PyArray1::from_vec_bound(py, f).reshape([n, feat::F_MAX_F, feat::FFD]).unwrap())?;
        d.set_item("fleet_mask", PyArray1::from_vec_bound(py, fm).reshape([n, feat::F_MAX_F]).unwrap())?;
        d.set_item("global_feats", PyArray1::from_vec_bound(py, gl).reshape([n, feat::GFD]).unwrap())?;
        d.set_item("place_meta", PyArray1::from_vec_bound(py, meta).reshape([n, feat::M_PLACES, feat::META_W]).unwrap())?;
        d.set_item("comet_paths", PyArray1::from_vec_bound(py, cpath).reshape([n, feat::F_MAX_C, max_path, 2]).unwrap())?;
        d.set_item("comet_plen", PyArray1::from_vec_bound(py, cplen).reshape([n, feat::F_MAX_C]).unwrap())?;
        d.set_item("comet_pidx", PyArray1::from_vec_bound(py, cpidx).reshape([n, feat::F_MAX_C]).unwrap())?;
        d.set_item("phi", PyArray1::from_vec_bound(py, phi))?;
        d.set_item("phi_canon", PyArray1::from_vec_bound(py, phic))?;
        d.set_item("max_path", max_path)?;
        Ok(d)
    }

    /// Per-env Kaggle-compatible observation dicts for `player`.
    /// Returns a list of N dicts (same schema as the single-game `Env.observation`),
    /// so the RL side can feed them straight into the proven `core.features.encode`.
    fn observation_dicts(&self, py: Python<'_>, player: i64) -> PyResult<Vec<PyObject>> {
        self.envs
            .iter()
            .zip(self.cur_step.iter())
            .map(|(env, &cs)| build_obs_dict(py, env, player, cs))
            .collect()
    }

    /// DEBUG: build the ProducerLite garrison projection for one env/player and
    /// return it flat for numerical comparison against the canonical orbit_lite
    /// `PlanetMovement.garrison_status()`. Returns
    /// `(owner_flat[P*(H+1)] i64, ships_flat[P*(H+1)] f64, P, H+1, planet_ids[P])`.
    fn garrison_debug(
        &self,
        env_idx: usize,
        player_count: usize,
        horizon: usize,
    ) -> (Vec<i64>, Vec<f64>, usize, usize, Vec<i64>) {
        let g = &self.envs[env_idx];
        let cs = self.cur_step[env_idx];
        let status = crate::flow::build_garrison_status(g, &self.cfg, player_count, horizon, cs);
        let p = status.p;
        let h1 = status.h + 1;
        let mut owner_flat = Vec::with_capacity(p * h1);
        let mut ships_flat = Vec::with_capacity(p * h1);
        for i in 0..p {
            for k in 0..h1 {
                owner_flat.push(status.owner[i][k]);
                ships_flat.push(status.ships[i][k]);
            }
        }
        let planet_ids: Vec<i64> = g.planets.iter().map(|pl| pl.id).collect();
        (owner_flat, ships_flat, p, h1, planet_ids)
    }

    /// DEBUG: per-player net-ship delta breakdown for a specific launch.
    fn score_launch_delta(&self, env_idx: usize, player: i64, src_id: i64, tgt_id: i64, ships: i64, eta: usize) -> Vec<f64> {
        let g = &self.envs[env_idx];
        let cs = self.cur_step[env_idx];
        let pc = { let mo = g.planets.iter().map(|p| p.owner).max().unwrap_or(0);
                   let mf = g.fleets.iter().map(|f| f.owner).max().unwrap_or(0); (mo.max(mf)+1).max(2) as usize };
        let status = crate::flow::build_garrison_status(g, &self.cfg, pc, 18, cs);
        let si = g.planets.iter().position(|p| p.id==src_id).unwrap();
        let ti = g.planets.iter().position(|p| p.id==tgt_id).unwrap();
        crate::flow::score_launch_delta(&status, si, ti, ships, eta, player as usize)
    }

    /// DEBUG: Rust ProducerLite moves for an arbitrary obs (JSON) — stateless,
    /// for fast cached-mismatch iteration without replaying games.
    fn pl_moves_json(&self, obs_json: String, player: i64) -> Vec<(i64, f64, i64)> {
        use crate::agents::Agent;
        let v: serde_json::Value = serde_json::from_str(&obs_json).unwrap();
        let g = crate::replay::state_from_obs(&v);
        let step = v.get("step").and_then(|x| x.as_i64()).unwrap_or(0);
        let agent = crate::flow::ProducerLiteAgent::default();
        agent.act(&g, player, &self.cfg, step)
            .into_iter().map(|m| (m.from_id, m.angle, m.ships)).collect()
    }

    /// DEBUG: garrison projection (owner+ships) on an arbitrary obs (JSON).
    /// Returns (owner_flat[P*(H+1)], ships_flat, P, H+1, planet_ids).
    fn garrison_debug_json(&self, obs_json: String, player_count: usize, horizon: usize)
        -> (Vec<i64>, Vec<f64>, usize, usize, Vec<i64>) {
        let v: serde_json::Value = serde_json::from_str(&obs_json).unwrap();
        let g = crate::replay::state_from_obs(&v);
        let cs = v.get("step").and_then(|x| x.as_i64()).unwrap_or(0);
        let status = crate::flow::build_garrison_status(&g, &self.cfg, player_count, horizon, cs);
        let p = status.p; let h1 = status.h + 1;
        let mut owf = Vec::with_capacity(p*h1); let mut shf = Vec::with_capacity(p*h1);
        for i in 0..p { for k in 0..h1 { owf.push(status.owner[i][k]); shf.push(status.ships[i][k]); } }
        let ids: Vec<i64> = g.planets.iter().map(|pl| pl.id).collect();
        (owf, shf, p, h1, ids)
    }

    /// DEBUG: per-fleet Rust arrival (fleet_eta) on an arbitrary obs (JSON).
    /// Returns list of (from_planet_id, owner, ships, hit_planet_id, arrival_step) — hit -1/step -1 if none.
    fn fleet_arrivals_json(&self, obs_json: String, horizon: usize) -> Vec<(i64, i64, i64, i64, i64)> {
        let v: serde_json::Value = serde_json::from_str(&obs_json).unwrap();
        let g = crate::replay::state_from_obs(&v);
        let cs = v.get("step").and_then(|x| x.as_i64()).unwrap_or(0);
        let pos = crate::flow::build_planet_positions_pub(&g, horizon, cs);
        let radii: Vec<f64> = g.planets.iter().map(|p| p.radius).collect();
        let mut out = Vec::new();
        for f in &g.fleets {
            let (hit, st) = match crate::flow::fleet_eta(f, &pos, &radii, self.cfg.ship_speed, horizon) {
                Some((idx, s)) => (g.planets[idx].id, s as i64),
                None => (-1, -1),
            };
            out.push((f.from_planet_id, f.owner, f.ships, hit, st));
        }
        out
    }

    /// DEBUG: offensive candidate eval on an arbitrary obs (JSON).
    fn offensive_debug_json(&self, obs_json: String, player: i64) -> Vec<(i64, i64, i64, i64, f64, bool, i64, bool, f64, bool, bool, bool)> {
        let v: serde_json::Value = serde_json::from_str(&obs_json).unwrap();
        let g = crate::replay::state_from_obs(&v);
        let step = v.get("step").and_then(|x| x.as_i64()).unwrap_or(0);
        crate::flow::offensive_debug(&g, &self.cfg, player, step, 18)
    }

    /// DEBUG: enemy-pressure vector (regroup gradient) for one env/player.
    fn enemy_pressure_debug(&self, env_idx: usize, player: i64) -> Vec<f64> {
        crate::flow::enemy_pressure_debug(&self.envs[env_idx], &self.cfg, player, self.cur_step[env_idx], 18)
    }

    /// DEBUG: the Rust ProducerLite agent's actual moves for one env/player.
    /// Returns list of (from_planet_id, angle, ships).
    fn producer_lite_moves(&self, env_idx: usize, player: i64) -> Vec<(i64, f64, i64)> {
        use crate::agents::Agent;
        let g = &self.envs[env_idx];
        let cs = self.cur_step[env_idx];
        let agent = crate::flow::ProducerLiteAgent::default();
        agent.act(g, player, &self.cfg, cs)
            .into_iter()
            .map(|m| (m.from_id, m.angle, m.ships))
            .collect()
    }

    /// DEBUG: per-(owned src, enemy tgt) offensive candidate eval for one env/player.
    /// See `flow::offensive_debug` for the tuple layout.
    fn offensive_debug(
        &self,
        env_idx: usize,
        player: i64,
    ) -> Vec<(i64, i64, i64, i64, f64, bool, i64, bool, f64, bool, bool, bool)> {
        let g = &self.envs[env_idx];
        let cs = self.cur_step[env_idx];
        crate::flow::offensive_debug(g, &self.cfg, player, cs, 18)
    }

    /// Step with player-0 actions given as planet IDs; opponents computed
    /// internally (set via `set_opponents`). Auto-resets finished envs.
    /// `actions[e]` is a list of `(from_planet_id, angle, ships)` for player 0.
    /// Returns `(rewards [N,A] int32, dones [N] u8)` — call `observation_dicts`
    /// afterwards for the post-step obs.
    fn step_p0_ids<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<Vec<(i64, f64, i64)>>,
    ) -> (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray1<u8>>) {
        let n = self.n;
        let a = self.num_agents;
        assert_eq!(actions.len(), n, "actions must have length num_envs");

        let p0_acts: Vec<Vec<Move>> = actions
            .iter()
            .map(|env_acts| {
                env_acts
                    .iter()
                    .filter(|&&(_, _, ships)| ships > 0)
                    .map(|&(from_id, angle, ships)| Move { from_id, angle, ships })
                    .collect()
            })
            .collect();

        let mut term = vec![false; n];
        let mut rew = vec![0i32; n * a];

        let envs = &mut self.envs;
        let cur_step = &mut self.cur_step;
        let seeds = &mut self.seeds;
        let cfg = &self.cfg;
        let next_seed = &mut self.next_seed;
        let opponents = &self.opponents;

        py.allow_threads(|| {
            envs.par_iter_mut()
                .zip(cur_step.par_iter_mut())
                .zip(term.par_iter_mut())
                .zip(rew.par_chunks_mut(a))
                .enumerate()
                .for_each(|(e, (((env, cs), t), rslot))| {
                    let mut acts: Vec<Vec<Move>> = Vec::with_capacity(a);
                    acts.push(p0_acts[e].clone());
                    for slot in 0..a - 1 {
                        let opp_moves = if let Some(agent) = &opponents[e][slot] {
                            agent.act(env, (slot + 1) as i64, cfg, *cs)
                        } else {
                            vec![]
                        };
                        acts.push(opp_moves);
                    }
                    let out = step_in_place(env, &acts, *cs, cfg, None, seeds[e]);
                    *cs += 1;
                    *t = out.terminated;
                    if let Some(r) = out.rewards {
                        for (i, v) in r.iter().enumerate() {
                            rslot[i] = *v as i32;
                        }
                    }
                });

            for e in 0..n {
                if term[e] {
                    let ns = *next_seed;
                    *next_seed += 1;
                    envs[e] = init_from_seed(ns, a);
                    cur_step[e] = 0;
                    seeds[e] = ns;
                }
            }
        });

        let rew_arr = PyArray1::from_vec_bound(py, rew).reshape([n, a]).unwrap();
        let dones = PyArray1::from_vec_bound(py, term.iter().map(|&b| b as u8).collect::<Vec<_>>());
        (rew_arr, dones)
    }

    /// Step with ALL players' actions given as planet IDs (self-play; no internal
    /// opponents). `actions[e][p]` is a list of `(from_planet_id, angle, ships)`
    /// for player `p` in env `e`. Auto-resets finished envs. Returns
    /// `(rewards [N,A] int32, dones [N] u8)`; call `observation_dicts` for obs.
    fn step_ids<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<Vec<Vec<(i64, f64, i64)>>>,
    ) -> (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray1<u8>>) {
        let n = self.n;
        let a = self.num_agents;
        assert_eq!(actions.len(), n, "actions must have length num_envs");

        let acts_all: Vec<Vec<Vec<Move>>> = actions
            .iter()
            .map(|env_acts| {
                (0..a)
                    .map(|p| {
                        env_acts
                            .get(p)
                            .map(|moves| {
                                moves
                                    .iter()
                                    .filter(|&&(_, _, ships)| ships > 0)
                                    .map(|&(from_id, angle, ships)| Move { from_id, angle, ships })
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect();

        let mut term = vec![false; n];
        let mut rew = vec![0i32; n * a];

        let envs = &mut self.envs;
        let cur_step = &mut self.cur_step;
        let seeds = &mut self.seeds;
        let cfg = &self.cfg;
        let next_seed = &mut self.next_seed;

        py.allow_threads(|| {
            envs.par_iter_mut()
                .zip(cur_step.par_iter_mut())
                .zip(term.par_iter_mut())
                .zip(rew.par_chunks_mut(a))
                .enumerate()
                .for_each(|(e, (((env, cs), t), rslot))| {
                    let out = step_in_place(env, &acts_all[e], *cs, cfg, None, seeds[e]);
                    *cs += 1;
                    *t = out.terminated;
                    if let Some(r) = out.rewards {
                        for (i, v) in r.iter().enumerate() {
                            rslot[i] = *v as i32;
                        }
                    }
                });

            for e in 0..n {
                if term[e] {
                    let ns = *next_seed;
                    *next_seed += 1;
                    envs[e] = init_from_seed(ns, a);
                    cur_step[e] = 0;
                    seeds[e] = ns;
                }
            }
        });

        let rew_arr = PyArray1::from_vec_bound(py, rew).reshape([n, a]).unwrap();
        let dones = PyArray1::from_vec_bound(py, term.iter().map(|&b| b as u8).collect::<Vec<_>>());
        (rew_arr, dones)
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.n
    }
    #[getter]
    fn num_agents(&self) -> usize {
        self.num_agents
    }
    #[getter]
    fn obs_dim(&self) -> usize {
        OBS_DIM
    }

    /// Seed every env. `seeds` has length num_envs. Auto-resets draw seeds
    /// continuing past max(seeds). Returns the initial obs tensor.
    fn reset<'py>(&mut self, py: Python<'py>, seeds: Vec<u64>) -> Bound<'py, PyArray3<f32>> {
        assert_eq!(seeds.len(), self.n, "seeds length must equal num_envs");
        let a = self.num_agents;
        self.next_seed = seeds.iter().copied().max().unwrap_or(0) + 1;
        let cfg_agents = a;
        self.envs = seeds.par_iter().map(|&s| init_from_seed(s, cfg_agents)).collect();
        self.seeds = seeds;
        self.cur_step = vec![0; self.n];
        let obs = self.encode_all();
        PyArray1::from_vec_bound(py, obs)
            .reshape([self.n, a, OBS_DIM])
            .unwrap()
    }

    /// Step all envs. `actions` is float32 [N, num_agents, K, 3] where each row
    /// is (source_planet_index, angle, num_ships); ships>0 launches from the
    /// planet at that index in the observation's planet ordering (the env maps
    /// index->id and the engine rejects it if the player doesn't own it).
    /// Finished envs auto-reset. Returns (obs [N,A,OBS_DIM], rewards [N,A]
    /// int32, dones [N] bool-as-uint8).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray4<f32>,
    ) -> (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<i32>>,
        Bound<'py, PyArray1<u8>>,
    ) {
        let n = self.n;
        let a = self.num_agents;
        let av = actions.as_array();
        let k = av.shape()[2];

        // Parse actions (GIL held). Action[..,0] is a planet INDEX into the
        // current (pre-step) planet ordering; map it to the planet id here.
        let acts: Vec<Vec<Vec<Move>>> = (0..n)
            .map(|e| {
                let env = &self.envs[e];
                (0..a)
                    .map(|p| {
                        let mut mv = Vec::new();
                        for kk in 0..k {
                            let ships = av[[e, p, kk, 2]] as i64;
                            let idx = av[[e, p, kk, 0]] as usize;
                            if ships > 0 && idx < env.planets.len() {
                                mv.push(Move {
                                    from_id: env.planets[idx].id,
                                    angle: av[[e, p, kk, 1]] as f64,
                                    ships,
                                });
                            }
                        }
                        mv
                    })
                    .collect()
            })
            .collect();

        let mut term = vec![false; n];
        let mut rew = vec![0i32; n * a];
        let mut obs = vec![0f32; n * a * OBS_DIM];

        let envs = &mut self.envs;
        let cur_step = &mut self.cur_step;
        let seeds = &mut self.seeds;
        let cfg = &self.cfg;
        let next_seed = &mut self.next_seed;

        py.allow_threads(|| {
            // Parallel step (no reset inside).
            envs.par_iter_mut()
                .zip(cur_step.par_iter_mut())
                .zip(term.par_iter_mut())
                .zip(rew.par_chunks_mut(a))
                .enumerate()
                .for_each(|(e, (((env, cs), t), rslot))| {
                    let out = step_in_place(env, &acts[e], *cs, cfg, None, seeds[e]);
                    *cs += 1;
                    *t = out.terminated;
                    if let Some(r) = out.rewards {
                        for (i, v) in r.iter().enumerate() {
                            rslot[i] = *v as i32;
                        }
                    }
                });

            // Serial auto-reset of finished envs (few per step).
            for e in 0..n {
                if term[e] {
                    let ns = *next_seed;
                    *next_seed += 1;
                    envs[e] = init_from_seed(ns, a);
                    cur_step[e] = 0;
                    seeds[e] = ns;
                }
            }

            // Encode observations in parallel.
            obs.par_chunks_mut(a * OBS_DIM)
                .zip(envs.par_iter())
                .zip(cur_step.par_iter())
                .for_each(|((chunk, env), &cs)| {
                    for p in 0..a {
                        encode_env(env, p as i64, cs, &mut chunk[p * OBS_DIM..(p + 1) * OBS_DIM]);
                    }
                });
        });

        let obs_arr = PyArray1::from_vec_bound(py, obs)
            .reshape([n, a, OBS_DIM])
            .unwrap();
        let rew_arr = PyArray1::from_vec_bound(py, rew).reshape([n, a]).unwrap();
        let dones_arr =
            PyArray1::from_vec_bound(py, term.iter().map(|&b| b as u8).collect::<Vec<u8>>());
        (obs_arr, rew_arr, dones_arr)
    }
}
