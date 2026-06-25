# ow_rs

Rust crate for the Orbit Wars environment. See the [top-level README](../README.md)
for the Python API, build/wheel instructions, performance, and correctness.

## Build

```bash
maturin develop --release          # install `ow_rs` into the active venv
maturin build   --release          # -> target/wheels/ow_rs-*.whl
cargo test      --release          # bit-exact parity tests (needs ../traces + ../*.json)
```

Cargo features: `python` (PyO3 bindings + numpy, on for wheels), `fast_math`
(native math, ~3.3x faster, faithful but not ULP-identical to Kaggle — used by
the shipped wheel via `pyproject.toml`).

## Module map (`src/`)

| file | role |
|---|---|
| `state.rs` | `Planet`/`Fleet`/`CometGroup`/`GameState`/`Move`/`Config` + constants |
| `pymath.rs` | math wrappers; libm-exact by default, native under `fast_math` |
| `geometry.rs` | `distance`, `point_to_segment_distance`, `swept_pair_hit` |
| `pyrandom.rs` | CPython-compatible MT19937 (`random.Random`) + int/str seeding |
| `mapgen.rs` | `generate_planets` (symmetric map) |
| `comets.rs` | `generate_comet_paths` (elliptical orbits) |
| `engine.rs` | `init_from_seed` + `step_in_place` (one game tick) |
| `replay.rs` | load `env.toJSON()` / Kaggle replay JSON into engine types |
| `py.rs` | PyO3 `Env` (single, Kaggle-schema obs) |
| `vecenv.rs` | PyO3 `VecEnv` (vectorized, multi-threaded, tensor obs) |

`tests/`: `rng_parity`, `mapgen_parity`, `comets_parity` (vs CPython),
`parity` (single-step), `full_parity` (whole game from seed).
Set `OW_FORCE_BIT_EXACT=1` on a Linux/glibc host to require bit-for-bit Kaggle parity.
