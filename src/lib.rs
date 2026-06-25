//! Orbit Wars simulator — Rust port of the Kaggle env (`ow_sim`).
//!
//! Phase 1 (core): the `step` function and geometry, validated bit-for-bit
//! against recorded replays by feeding state + actions (RNG-free). Map/comet
//! generation and the PyO3 bindings come in later phases.

pub mod agents;
pub mod comets;
pub mod features;
pub mod flow;
pub mod engine;
pub mod geometry;
pub mod mapgen;
pub mod pymath;
pub mod pyrandom;
pub mod replay;
pub mod state;

#[cfg(feature = "python")]
mod py;
#[cfg(feature = "python")]
mod vecenv;
