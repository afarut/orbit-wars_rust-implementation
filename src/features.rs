//! Feature encoder — bit-for-bit port of `core/features.py::encode` so RL training
//! consumes the SAME features as Kaggle inference, but computed in parallel Rust
//! (no Python GIL). All transcendentals go through `pymath::c_*` (CPython-libm
//! matched); math is done in f64 and cast to f32 only at store time, exactly like
//! Python (math → list[float] → np.float32).
//!
//! Output layout matches `EncodedObs`:
//!   planet_feats [40,20], comet_feats [16,25], fleet_feats [256,10], global [11]
//!   + bool masks, + place_meta [56,8] for decode (build intercept Target in Python).

use crate::pymath::{c_cos, c_log, c_log1p, c_pow, c_sin, c_sqrt, sq2};
use crate::state::*;

// фикс. паддинг как в FeatureConfig (max_planets/comets/fleets)
pub const F_MAX_P: usize = 40;
pub const F_MAX_C: usize = 16;
pub const F_MAX_F: usize = 256;
pub const PFD: usize = 20; // PLANET_FEAT_DIM
pub const CFD: usize = 25; // COMET_FEAT_DIM
pub const FFD: usize = 10; // FLEET_FEAT_DIM
pub const GFD: usize = 11; // GLOBAL_FEAT_DIM
pub const M_PLACES: usize = F_MAX_P + F_MAX_C; // 56
pub const META_W: usize = 8; // valid, id, owner, x, y, ships, kind, ang_vel
pub const EPISODE_STEPS: f64 = 500.0;
const COMET_SPAWNS: [i64; 5] = [50, 150, 250, 350, 450];
const HORIZONS: [f64; 3] = [4.0, 8.0, 16.0];

// kind_code в place_meta
const KIND_STATIC: f32 = 0.0;
const KIND_ORBIT: f32 = 1.0;
const KIND_COMET: f32 = 2.0;

#[inline]
fn dist_sun(x: f64, y: f64) -> f64 {
    c_sqrt(sq2(x - CENTER) + sq2(y - CENTER))
}

/// predict_position для орбиты: текущую (x,y) повернуть на omega*t вокруг центра.
/// Точная копия Python intercept.predict_position (kind="orbit").
#[inline]
fn predict_orbit(x: f64, y: f64, omega: f64, t: f64) -> (f64, f64) {
    let theta = omega * t;
    let ct = c_cos(theta);
    let st = c_sin(theta);
    let rx = x - CENTER;
    let ry = y - CENTER;
    (CENTER + ct * rx - st * ry, CENTER + st * rx + ct * ry)
}

/// Канонизация перспективы игрока — бит-точное зеркало `model._canonicalize`:
/// поворот всех позиций/углов на -φ вокруг CENTER + релейбл владельцев (owner-player)%n,
/// player->0. n выводится из ТЕКУЩИХ владельцев (4 если max owner>=2 иначе 2), как в Python.
/// φ: n==4 -> (π/2)·[0,1,3,2][player]; n==2 -> 2π·player/n. Возвращает (canon_state, φ).
pub fn canonicalize_state(game: &GameState, player: i64) -> (GameState, f64) {
    if player == 0 {
        return (game.clone(), 0.0);
    }
    let mut maxo: i64 = -1;
    for pl in &game.planets {
        if pl.owner >= 0 && pl.owner > maxo { maxo = pl.owner; }
    }
    for fl in &game.fleets {
        if fl.owner >= 0 && fl.owner > maxo { maxo = fl.owner; }
    }
    let n: i64 = if maxo >= 2 { 4 } else { 2 };
    let pi = std::f64::consts::PI;
    let phi = if n == 4 {
        (pi / 2.0) * [0.0, 1.0, 3.0, 2.0][player as usize]
    } else {
        2.0 * pi * (player as f64) / (n as f64)
    };
    let ang = -phi;
    let cs = c_cos(ang);
    let sn = c_sin(ang);
    let rot = |x: f64, y: f64| -> (f64, f64) {
        let dx = x - CENTER;
        let dy = y - CENTER;
        (CENTER + dx * cs - dy * sn, CENTER + dx * sn + dy * cs)
    };
    let rl = |o: i64| -> i64 { if o < 0 { o } else { (o - player).rem_euclid(n) } };

    let mut g2 = game.clone();
    for pl in &mut g2.planets {
        pl.owner = rl(pl.owner);
        let (x, y) = rot(pl.x, pl.y);
        pl.x = x; pl.y = y;
    }
    for fl in &mut g2.fleets {
        fl.owner = rl(fl.owner);
        let (x, y) = rot(fl.x, fl.y);
        fl.x = x; fl.y = y;
        fl.angle += ang;
    }
    for grp in &mut g2.comets {
        for path in &mut grp.paths {
            for pt in path.iter_mut() {
                let (x, y) = rot(pt[0], pt[1]);
                pt[0] = x; pt[1] = y;
            }
        }
    }
    (g2, phi)
}

/// Python intercept.fleet_speed (s=max(1,ships); frac=clamp(log s/log1000,0,1)^1.5).
#[inline]
fn fleet_speed(ships: i64) -> f64 {
    let s = (ships as f64).max(1.0);
    let mut frac = c_log(s) / c_log(1000.0);
    if frac < 0.0 {
        frac = 0.0;
    } else if frac > 1.0 {
        frac = 1.0;
    }
    frac = c_pow(frac, 1.5);
    1.0 + 5.0 * frac
}

/// Найти активный путь кометы для планеты pid: (path, group.path_index).
fn find_comet<'a>(g: &'a GameState, pid: i64) -> Option<(&'a Vec<[f64; 2]>, i64)> {
    for group in &g.comets {
        for (k, &gpid) in group.planet_ids.iter().enumerate() {
            if gpid == pid && k < group.paths.len() && !group.paths[k].is_empty() {
                return Some((&group.paths[k], group.path_index));
            }
        }
    }
    None
}

/// Кометная позиция через h шагов (дискретный lookup, как Python predict_position
/// при целом горизонте: f=path_index+h, frac=0 → path[clamp(f,0,n-1)]).
#[inline]
fn comet_pos_at(path: &[[f64; 2]], path_index: i64, h: f64) -> (f64, f64) {
    let n = path.len() as i64;
    let lo = path_index + h as i64;
    let idx = if lo < 0 {
        0
    } else if lo >= n - 1 {
        n - 1
    } else {
        lo
    };
    let p = path[idx as usize];
    (p[0], p[1])
}

/// Максимальная длина пути кометы среди всех env (для размера comet_paths).
pub fn max_comet_path_len(states: &[&GameState]) -> usize {
    let mut m = 0usize;
    for g in states {
        for group in &g.comets {
            for p in &group.paths {
                if p.len() > m {
                    m = p.len();
                }
            }
        }
    }
    m
}

/// Закодировать фичи одной игры для игрока `player` на шаге `cur_step`.
/// Заполняет переданные срезы (нулёванные заранее вызывающим кодом):
///   p[F_MAX_P*PFD], pm[F_MAX_P], c[F_MAX_C*CFD], cm[F_MAX_C], f[F_MAX_F*FFD],
///   fm[F_MAX_F], gl[GFD], meta[M_PLACES*META_W],
///   cpath[F_MAX_C*max_path*2], cplen[F_MAX_C], cpidx[F_MAX_C]
#[allow(clippy::too_many_arguments)]
pub fn encode_env_features(
    g: &GameState,
    player: i64,
    cur_step: i64,
    p: &mut [f32],
    pm: &mut [u8],
    c: &mut [f32],
    cm: &mut [u8],
    f: &mut [f32],
    fm: &mut [u8],
    gl: &mut [f32],
    meta: &mut [f64], // decode-метаданные в f64: угол intercept_angle == model.act (inference)
    cpath: &mut [f64],
    cplen: &mut [i32],
    cpidx: &mut [i32],
    phi: &mut [f64], // [1] шейпинг-потенциал Φ=prod_adv для player (НЕ вход модели)
    max_path: usize,
) {
    let av = g.angular_velocity;

    // мои планеты (для fleet incoming): (x, y, radius)
    let my_planets: Vec<(f64, f64, f64)> = g
        .planets
        .iter()
        .filter(|pl| pl.owner == player)
        .map(|pl| (pl.x, pl.y, pl.radius))
        .collect();

    let mut p_slot = 0usize; // индекс планеты в planet_rows
    let mut c_slot = 0usize; // индекс кометы в comet_rows

    for pl in &g.planets {
        let x = pl.x;
        let y = pl.y;
        let radius = pl.radius;
        let ships = pl.ships as f64;
        let prod = pl.production as f64;
        let owner = pl.owner;

        let (mine, enemy, neutral) = if owner == player {
            (1.0f64, 0.0, 0.0)
        } else if owner == -1 {
            (0.0, 0.0, 1.0)
        } else {
            (0.0, 1.0, 0.0)
        };

        // комета? (id в comet_planet_ids ИЛИ активный путь) — как Python is_comet
        let comet = find_comet(g, pl.id);
        let in_comet_ids = g.comet_planet_ids.contains(&pl.id);
        let is_comet = in_comet_ids || comet.is_some();

        // target kind (для horizon offsets, is_orbiting, place_meta)
        let ds = dist_sun(x, y);
        let (kind_code, is_orb): (f32, f64) = if comet.is_some() {
            (KIND_COMET, 0.0)
        } else if ds + radius < ROTATION_RADIUS_LIMIT {
            (KIND_ORBIT, 1.0)
        } else {
            (KIND_STATIC, 0.0)
        };

        // база 20 фич (точный порядок _place_base)
        let mut base = [0f64; PFD];
        base[0] = x / 100.0;
        base[1] = y / 100.0;
        base[2] = (x - CENTER) / 50.0;
        base[3] = (y - CENTER) / 50.0;
        base[4] = ds / 50.0;
        base[5] = is_orb;
        base[6] = av * 20.0;
        base[7] = mine;
        base[8] = enemy;
        base[9] = neutral;
        base[10] = ships / 100.0;
        base[11] = c_log1p(ships.max(0.0)) / 5.0;
        base[12] = prod / 5.0;
        base[13] = radius / 5.0;
        // horizon offsets 14..19
        for (hi, &h) in HORIZONS.iter().enumerate() {
            let (fx, fy) = if let Some((path, pidx)) = comet {
                comet_pos_at(path, pidx, h)
            } else if kind_code == KIND_ORBIT {
                predict_orbit(x, y, av, h)
            } else {
                (x, y) // static: predict == pos
            };
            base[14 + hi * 2] = (fx - x) / 50.0;
            base[14 + hi * 2 + 1] = (fy - y) / 50.0;
        }

        // place index: планета → p_slot, комета → F_MAX_P + c_slot
        let place_idx = if is_comet { F_MAX_P + c_slot } else { p_slot };
        if place_idx < M_PLACES {
            let mo = place_idx * META_W;
            meta[mo] = 1.0; // valid
            meta[mo + 1] = pl.id as f64;
            meta[mo + 2] = owner as f64;
            meta[mo + 3] = x; // f64 — координаты для intercept_angle в decode
            meta[mo + 4] = y;
            meta[mo + 5] = ships;
            meta[mo + 6] = kind_code as f64;
            meta[mo + 7] = av;
        }

        if is_comet {
            if c_slot < F_MAX_C {
                // база 0..19
                let off = c_slot * CFD;
                for d in 0..PFD {
                    c[off + d] = base[d] as f32;
                }
                // comet extras 20..24 (_comet_extras)
                let (plen, remaining, pidx_f, vx, vy) = if let Some((path, pidx)) = comet {
                    let n = path.len() as i64;
                    let i = pidx.clamp(0, (n - 2).max(0)) as usize;
                    let (vx, vy) = if n >= 2 {
                        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                    } else {
                        (0.0, 0.0)
                    };
                    (path.len() as f64, path.len() as f64 - pidx as f64, pidx as f64, vx, vy)
                } else {
                    // is_comet но без активного пути: target orbit/static
                    let (vx, vy) = if kind_code == KIND_ORBIT {
                        (av * -(y - CENTER), av * (x - CENTER))
                    } else {
                        (0.0, 0.0)
                    };
                    (0.0, 0.0, 0.0, vx, vy)
                };
                let _ = plen;
                c[off + 20] = 1.0;
                c[off + 21] = (pidx_f / 200.0) as f32;
                c[off + 22] = (remaining / 200.0) as f32;
                c[off + 23] = (vx / 6.0) as f32;
                c[off + 24] = (vy / 6.0) as f32;
                cm[c_slot] = 1;
                // comet path для decode
                if let Some((path, pidx)) = comet {
                    cplen[c_slot] = path.len() as i32;
                    cpidx[c_slot] = pidx as i32;
                    let pbase = c_slot * max_path * 2;
                    for (k, pt) in path.iter().enumerate().take(max_path) {
                        cpath[pbase + k * 2] = pt[0]; // f64
                        cpath[pbase + k * 2 + 1] = pt[1];
                    }
                }
                c_slot += 1;
            }
        } else if p_slot < F_MAX_P {
            let off = p_slot * PFD;
            for d in 0..PFD {
                p[off + d] = base[d] as f32;
            }
            pm[p_slot] = 1;
            p_slot += 1;
        }
    }

    // ── fleets ────────────────────────────────────────────────────────────────
    let mut f_slot = 0usize;
    for fl in &g.fleets {
        if f_slot >= F_MAX_F {
            break;
        }
        let owner = fl.owner;
        let mine = if owner == player { 1.0f64 } else { 0.0 };
        let enemy = if owner != player && owner != -1 { 1.0 } else { 0.0 };
        let hx = c_cos(fl.angle);
        let hy = c_sin(fl.angle);
        let ships = fl.ships as f64;

        // incoming: вражеский флот, летящий на одну из моих планет
        let mut incoming = 0.0f64;
        if enemy != 0.0 {
            for &(px, py, pr) in &my_planets {
                let rx = px - fl.x;
                let ry = py - fl.y;
                let proj = rx * hx + ry * hy;
                if proj <= 0.0 {
                    continue;
                }
                let perp = (rx * hy - ry * hx).abs();
                if perp < pr + 3.0 {
                    incoming = 1.0;
                    break;
                }
            }
        }

        let off = f_slot * FFD;
        f[off] = (fl.x / 100.0) as f32;
        f[off + 1] = (fl.y / 100.0) as f32;
        f[off + 2] = mine as f32;
        f[off + 3] = enemy as f32;
        f[off + 4] = (ships / 100.0) as f32;
        f[off + 5] = (c_log1p(ships.max(0.0)) / 5.0) as f32;
        f[off + 6] = hx as f32;
        f[off + 7] = hy as f32;
        f[off + 8] = (fleet_speed(fl.ships) / 6.0) as f32;
        f[off + 9] = incoming as f32;
        fm[f_slot] = 1;
        f_slot += 1;
    }

    // ── global 11 фич (_global_features) ─────────────────────────────────────
    let step = cur_step as f64;
    let remaining = EPISODE_STEPS - step;
    // totals по владельцам (planets ships + fleets ships), owner>=0
    let mut totals = [0.0f64; 4];
    let mut max_owner = 0i64;
    for pl in &g.planets {
        if pl.owner >= 0 {
            totals[pl.owner as usize] += pl.ships as f64;
            if pl.owner > max_owner {
                max_owner = pl.owner;
            }
        }
    }
    for fl in &g.fleets {
        if fl.owner >= 0 {
            totals[fl.owner as usize] += fl.ships as f64;
            if fl.owner > max_owner {
                max_owner = fl.owner;
            }
        }
    }
    let n_players = if max_owner >= 2 { 4.0 } else { 2.0 };
    let mine = totals[player as usize];
    let mut max_opp = 0.0f64;
    for (o, &t) in totals.iter().enumerate() {
        if o as i64 != player && t > max_opp {
            max_opp = t;
        }
    }
    let adv = (mine - max_opp) / (mine + max_opp + 1.0);
    let mut countdown = 50.0f64;
    for &s in &COMET_SPAWNS {
        if s >= cur_step {
            countdown = (s - cur_step) as f64;
            break;
        }
    }
    gl[0] = (step / EPISODE_STEPS) as f32;
    gl[1] = (remaining / EPISODE_STEPS) as f32;
    gl[2] = (n_players / 4.0) as f32;
    for i in 0..4 {
        gl[3 + i] = if player == i as i64 { 1.0 } else { 0.0 };
    }
    gl[7] = (mine / 500.0) as f32;
    gl[8] = (max_opp / 500.0) as f32;
    gl[9] = adv as f32;
    gl[10] = (countdown / 50.0) as f32;

    // Φ = prod_adv = (my_prod - enemy_prod)/(my_prod+enemy_prod+1) — шейпинг-потенциал.
    let mut mine_prod = 0.0f64;
    let mut enemy_prod = 0.0f64;
    for pl in &g.planets {
        if pl.owner == player {
            mine_prod += pl.production as f64;
        } else if pl.owner >= 0 {
            enemy_prod += pl.production as f64;
        }
    }
    phi[0] = (mine_prod - enemy_prod) / (mine_prod + enemy_prod + 1.0);
}
