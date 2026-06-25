//! Game state model + constants. Mirrors the obs0 structure the Python
//! interpreter mutates (`ow_sim/engine.py`).

pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER: f64 = 50.0;
pub const SUN_RADIUS: f64 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f64 = 50.0;
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i64 = 1;
pub const COMET_SPAWN_STEPS: [i64; 5] = [50, 150, 250, 350, 450];
pub const PLANET_CLEARANCE: f64 = 7.0;
pub const MIN_PLANET_GROUPS: i64 = 5;
pub const MAX_PLANET_GROUPS: i64 = 10;
pub const MIN_STATIC_GROUPS: i64 = 3;

/// [id, owner, x, y, radius, ships, production]
#[derive(Clone, Debug, PartialEq)]
pub struct Planet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
}

/// [id, owner, x, y, angle, from_planet_id, ships]
#[derive(Clone, Debug, PartialEq)]
pub struct Fleet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub path_index: i64,
}

/// One launch order: [from_planet_id, angle, num_ships].
#[derive(Clone, Debug)]
pub struct Move {
    pub from_id: i64,
    pub angle: f64,
    pub ships: i64,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub ship_speed: f64,
    pub episode_steps: i64,
    pub comet_speed: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config { ship_speed: 6.0, episode_steps: 500, comet_speed: 4.0 }
    }
}

/// Comet data injected at a spawn step during core-parity validation
/// (substitutes for the RNG path generation, ported later).
#[derive(Clone, Debug)]
pub struct SpawnInjection {
    pub ships: i64,
    pub paths: Vec<Vec<[f64; 2]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameState {
    pub planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub next_fleet_id: i64,
    pub comets: Vec<CometGroup>,
    pub comet_planet_ids: Vec<i64>,
    pub initial_planets: Vec<Planet>,
    pub angular_velocity: f64,
}
