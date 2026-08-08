# cube444 TPU beam — runbook

The JAX SPMD V-only beam ported to the 4x4x4 color cube. Everything here is
CPU-validated (see `EXPERIMENTS.md` 2026-07-23 "TPU beam port"); this doc is the
recipe to run it at width on a v6e.

## Files

| file | role |
|---|---|
| `jax_model.py` | pure-JAX ResMLPDistance forward, one-hot + embedding |
| `jax_beam_spmd_v_only.py` | SPMD shared-beam engine (6-slice port of megaminx) |
| `gcp_beam_cube444.py` | driver: one-hot V + color-cube sym-ensemble, no NISS |
| `json_to_csv.py` | merge result JSONs -> verified submission CSV |
| `test_parity.py` | JAX vs PyTorch parity (run once after any model change) |

## Artifacts the driver needs (stage into `$CUBE_DATA`, default `/mnt/data/cube444`)

```
puzzle_info.json          test.csv
rotations_24.npy          color_maps_24.npy          move_relabel_inv_24.npy
c_bells2_epoch_0399.pt    # the 3.3M V; rename from models/c_bells2/epoch_0399.pt
```

Build the sym tables with `python -c "from cube444.symmetry import save_tables; ..."`
(already in `cube444/data/`). Deploy the 3.3M, NOT the 15M: identical solve rate,
3x faster, and width is the whole lever (see EXPERIMENTS capacity section).

## Provision the v6e

Use the `megaminx-tpu-provision` skill — it handles flex-start status/stockout
and the Windows plink-noise filter. Project `gen-lang-client-0977634337`,
`ct6e-standard-4t` FLEX_START in `us-east5-a/b` or `us-central1-a` (per
[[tpu_builders_program_gcp_setup]]). One v6e-4 ~= one Kaggle v5e-8 in throughput.

Runtime setup mirrors megaminx's `setup_v6e_runtime.sh`: `jax[tpu]` + `torch`
(CPU, only for `load_params_from_pt`), then verify `jax.device_count()==4`.

## Run

```bash
# smoke: 1 pid, moderate width, prove it solves + verifies on TPU
CUBE_DATA=/mnt/data/cube444 python3 gcp_beam_cube444.py \
  --b-global 1048576 --alpha 2 --internal-bs 16384 --num-steps 120 \
  --start-pid 299 --end-pid 300 --sym-ensemble 1 \
  --out /tmp/cube_out/smoke.json

# production shard: a pid range at width, sym-4, prefix-nested frames
CUBE_DATA=/mnt/data/cube444 python3 gcp_beam_cube444.py \
  --b-global 4194304 --alpha 2 --internal-bs 32768 --num-steps 140 \
  --start-pid 0 --end-pid 200 --sym-ensemble 4 --sym-seed 0 \
  --parent-chunk 262144 \
  --out /tmp/cube_out/shard_0_200.json
```

- `--parent-chunk` streams the neighbor materialization (needed past ~B=2M/chip
  to avoid HBM OOM). Must divide `b_local = b_global / n_dev`.
- `--sym-positions lo:hi` slices the frame list for splitting sym frames across
  two boxes (both must pass the same `--sym-ensemble`/`--sym-seed`).
- `--stop-at-frame-solve` stops a pid after the first frame solves it (faster,
  slightly worse — skips the min-over-frames). Leave off for best length.

Merge + verify + min against the community floor (Rule 26):

```bash
python3 cube444/tpu/json_to_csv.py --results /tmp/cube_out/*.json --out tpu_raw.csv
python3 cube444/scripts/06_merge.py --out best.csv \
    cube444/community/submission_54754_merge33.csv tpu_raw.csv
python3 cube444/scripts/00_verify.py --submission best.csv
```

## Persistent compilation cache (MANDATORY for fast-solving puzzles)

`gcp_beam_cube444.py` enables JAX's on-disk compilation cache
(`jax_compilation_cache_dir=/tmp/jax_cube_cache`, min-size/min-time = 0). Without
it the beam recompiles the step_fn (~90s at 2^20 on v6e-8) on EVERY `run_beam`
call -- the kernel rebuilds the closure per call, so the in-memory jit cache
misses. cube444 pids solve in 1-60 steps, so ~90s compile would dominate: a full
sym-4 run would be ~5 days of pure recompile. The disk cache is keyed by HLO hash
(not callable identity), so the first call compiles (~105s) and every later call
loads in ~2s (~50x). Keep `/tmp/jax_cube_cache` across pids/frames (and reuse it
across runs at the same width -- it warm-starts the first compile too). Set
`JAX_CACHE_DIR` to relocate. megaminx didn't need this (hour-long pids amortize
compile); any puzzle with sub-minute solves does.

## Gotchas carried from megaminx

- Rule 26: always min-merge against the community 54,754 before claiming a win.
- Every path is host-verified in the driver AND again in json_to_csv, so a sym
  translation bug can only LOSE a solve, never ship a bad one.
- CPU emulation validates plumbing, NOT TPU-specific numerics/perf
  ([[jax_tpu_gotchas]]). The smoke run on real silicon is the first trustworthy
  throughput number.
- bf16 V is the compute dtype; expect 0-2 moves/pid drift vs the fp32 GPU beam.
- One batch-TPU session gotchas do not apply (this is GCP, not Kaggle), but the
  flex-start VM can be preempted — checkpoint result JSON per pid (the driver
  already does: it dumps `--out` after every pid).

## Width economics (projection to calibrate the run)

Measured on GPU: 2^16 sym-4 = 100% solve, mean 62.67 on 12 hard pids. The
organiser width curve (their model) is 2^16->71.88, 2^24->54.5 = -17.4 moves.
Extrapolated: 2^24 sym-4 ~= 45.3 mean (community floor 53.88; Rokicki 46,298).
This is optimistic (two-point extrapolation, assumes width+sym additive) but even
half the gain clears the floor. Start at 2^22, measure the real slope, scale.
