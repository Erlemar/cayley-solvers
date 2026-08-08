# TPU v4-8 session findings — 2026-06-12

Work on a shared **TPU v4-8** (4 chips, 128 GiB HBM, 400 GiB RAM; project `tpu-research-468103`).
Headline: **pid 991 went 90 → 73 verified** (3 moves under the prior community/bridge floor of 76),
and we **found + fixed a silent backpointer-overflow bug** that was capping every large V-only beam.

---

## TL;DR

- **qshort hurts path quality** on hard pids (V-only beats V+qshort by ~15 moves), but it's a *speed*
  mechanism — dropping it makes big beams compute-bound.
- **Bigger beam = monotonically shorter path** (48M→75, 60M→74, 64M→73), confirming the heuristic.
- **The 23-bit `parent_local` packing bug**: the v_only kernel silently corrupted the walkback for
  `b_local > 2^23 = 8.39M` → "found but verify=False". Fixed to 24-bit (`24/3/5`). 64M now verifies.
- **NISS implemented but neutral** here (inverse pass ties the forward at 73).
- **Corrected an earlier wrong conclusion**: the hard tail is *not* bridge-floored — that was a qshort
  artifact. A big enough V-only beam beats the bridge.

---

## 1. Machine + environment (Ubuntu 20.04 / glibc 2.31)

- Pre-installed **`torch_xla 2.9` is DOA** — it needs glibc ≥ 2.34 (Ubuntu 22.04+); this image is 2.31.
- Fix = two separate `uv` envs (zakhar's `.venv` untouched):
  - **JAX**: `~/envs/jax-tpu/bin/python` — jax/jaxlib **0.10.1**, libtpu 0.0.41. Zero-config, 4× "TPU v4".
    This is what our `tpu_beam_spmd_jax` pipeline uses.
  - **torch_xla**: `~/envs/xla-tpu/bin/python-tpu` — torch 2.4.1 + torch_xla **2.4.0** (newest pair
    predating the glibc-2.34 wall). The `python-tpu` launcher sets `PJRT_DEVICE=TPU` + `LD_LIBRARY_PATH`.
- The kernel (written for Kaggle-TPU jax, only ever patched to 0.6.2) **runs as-is on jax 0.10.1** with
  just the `pcast`→`pvary` shim — no other API breakage.
- **General rule for any new TPU/GPU VM**: `ldd --version` first. torch_xla ≥2.5 needs glibc 2.34;
  jaxlib only ~2.27. On an old base, JAX is the safe path.

## 2. Beam-size ceiling on 4 chips

- **meta-materialize is impractically slow to compile at 48M+** (>9 min, never finished). The production
  recipe uses `--no-meta` anyway (meta only ever existed for a higher OOM ceiling we can't use here).
- **V-only streaming-body tuning matters enormously**: default `parent_chunk=131072` + `internal_bs=16384`
  gives **360s/step at 48M**. Bumping to `parent_chunk=1048576` (96→12 chunks) + `internal_bs=131072`
  (8× bigger V batches) → **132s/step (~2.7×)**. The remaining ~100s floor is the V-score + selection
  over ~300M candidates — the cost of no qshort pruning.
- Per-step wall (tuned, no-meta): 48M ≈ 132s, 60M ≈ 170s, 64M ≈ 192s. Compile ≈ 55–70s (V-only graph
  is small).
- **64M (b_local = 2²⁴ = 16.78M) is the practical HBM ceiling** on 4 chips. 96M would need an 8-chip slice.

## 3. qshort hurts quality, but provides speed

- **V-only @ 48M = 75** vs **V+qshort @ 32M = 90** → a **15-move regression from qshort** (it misaligns
  with the AZ v4 V — Rule 15). Even the pid-0 smoke showed it (V-only@2M=56 < qshort@4M=64).
- BUT qshort's real job is **speed**: it uses the Q model to prune each parent's 24 moves to a shortlist
  *before* the expensive V-scoring. Without it, V-only must V-score + select over all `24 × b_local ≈ 300M`
  candidates per step. So "drop qshort and go bigger" trades a path-quality win for a ~2× per-step tax.

## 4. Bigger beam = better (search level), monotonic

| Beam | b_local | found len |
|---|---|---|
| 48M | 12.58M | 75 |
| 60M | 15.73M | 74 |
| 64M | 16.78M | 73 |

## 5. THE BUG — 23-bit `parent_local` overflow (found + fixed)

**Symptom:** 60M and 64M returned `found=True` but `verify=False` (a *shorter* path that doesn't solve),
while 48M verified fine.

**Root cause** (`jax_beam_spmd_v_only.py`): the per-step backpointer was packed into a uint32 as
`parent_local (23 bits) | rank<<23 | move<<26`. For `b_local > 2²³ = 8,388,608`, any parent at index
≥ 2²³ had its high bits **spill into the rank/move fields** → corrupt walkback. 48M's winning chain
happened to stay in low indices (so it verified); 60M/64M didn't. This is the same "packed-backptr
23-bit overflow" git fixed in the **qshort** kernel — the **v-only kernel never got the fix**.

**Why it wasn't a false "found":** found-detection uses the **full int64 hash**
(`sum(state·hash_vec) == V0_hash`), so collisions are ~3e-12 — the beam genuinely reaches V0. Only the
host-side **walkback reconstruction** corrupts. The driver's `verify()` caught every bad path (reported
0/1, nothing wrong banked).

**Fix:** the old layout wasted 1 bit (23+3+5 = 31 of 32). Reallocated it → **24/3/5**
(`parent_local` 24 bits @0-23, rank @24-26, move @27-31), covering `b_local` up to 2²⁴ = the HBM ceiling.
Applied to all 6 sites (both write bodies, the seed pack, the walkback read); kept 3 rank bits so it's
still 8-rank-safe. Round-trip unit-tested for every bug-zone value, then confirmed end-to-end: the
**identical 64M config now gives `verify=True len=73`** (independently re-verified).

→ This **unlocks the full beam range (up to 64M) for every future V-only run.**

## 6. NISS — implemented, validated, neutral

- Added `--niss` (forward + inverse passes, keep shorter) and `--invert` (inverse-only) to
  `gcp_beam_v_only.py`. Inverse scramble = **`argsort(s0)`** (clean because `central_state` is the
  identity perm); solve it, invert the word (`[inv_move_idx[m] for m in reversed(Q)]`), verify against s0.
- Round-trip unit-tested + plumbing-tested (pid 0 @ 2M: `dir=inv verify=True`).
- **pid 991: inverse@64M = 73, ties forward 73 → 0 improvement.** Both directions reach near-optimal
  independently, so there's no fwd/inv asymmetry to exploit. Confirms the 2026-05-27 ablation that NISS
  is marginal/redundant. **Skip NISS for the sweep** (not worth 2× wall).

## 7. pid 991 result + corrected lesson

- **90 (qshort@32M) → 75 (48M) → 73 (64M, fixed kernel), verified.** 3 moves under the prior floor of 76
  (`bridge_tpu_b1m.csv`). Saved to `submissions/rescue_pid991_vonly64m_73.csv`.
- **Corrected earlier conclusion**: "the hard tail is bridge-floored, don't fresh-beam it" was wrong — it
  was a *qshort* artifact. A big enough **V-only** beam beats the bridge/community floor on a hard pid.

## 8. Artifacts

- **Driver**: `kaggle_notebooks/tpu_beam_spmd_jax/gcp_beam_v_only.py` — V-only, single-view,
  `--niss`/`--invert`. (The qshort driver `gcp_beam_v6e.py` is unchanged.)
- **Fixed kernel**: `kaggle_notebooks/tpu_beam_spmd_jax/jax_beam_spmd_v_only.py` (24/3/5 packing).
- **Box staging**: `~/mm/v6e/` (V model, Q model, data, both kernels, driver, diagnostics).
  Local GCS-download cache: `C:\Users\and-l\_mm_stage\`.
- **Connect**: `ssh tpu-v4` (key at `C:\Users\and-l\.ssh\tpu-dev-key`).

## 9. Open / next

- **Hard-tail sweep at 64M V-only** (no qshort, no NISS) across pids ~985–1000 — the productive path now
  that big beams verify. ~4h/pid.
- **Re-stage the fixed `jax_beam_spmd_v_only.py` to GCS** (`gs://mm-tpu-staging-0977634337/v6e/`) so a
  fresh VM picks it up.
- **Merge** the pid-991 73 into the full-1001 submission (per-pid min) and submit.
