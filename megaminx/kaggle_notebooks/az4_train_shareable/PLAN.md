# PLAN: `cayleypy-az4-trainer-megaminx` — shareable modular AZ-v4 training kernel

Status: **SHIPPED 2026-07-04.** Built and pushed same day; this doc is the spec it was
built from (kept for reference). Deviations from the plan below:

- Decisions taken (user, 2026-07-04): (1) policy data = best PUBLIC CSV =
  `submission_73731.csv` (the newer 73,614/71,362 merges contain our unpublished paths —
  NOT shipped); (2) kernel + dataset public immediately; (3) v1 without PDB stage-4
  retrain; (4) solve block defaults to a short pid list, `[]` switches to full-1001.
- Dataset layout is FLAT (no subfolders — `kaggle datasets create` skips folders without
  `--dir-mode`); README table updated accordingly.
- Kernel: https://www.kaggle.com/code/artgor/cayleypy-az4-trainer-megaminx
  (sources: `cells/*.py|md` + `build_ipynb.py` + `smoke_run.py` in this folder).
- Dataset: https://www.kaggle.com/datasets/artgor/megaminx-az4-training-assets (~2.4GB).
- Local smoke (all 5 stages + export + canary + cayleypy bench + solve, quick_test):
  PASSED 2026-07-04 in 2.5 min on the 4090. Canary on the shipped reference
  `m_az_v4_v_only`: V(solved)=0.020, V(d1)=1.008, BFS-d6 MAE <= 0.15, sat gap +3.6,
  V@80=29.5, std@20=2.76 — all PASS (validates the canary thresholds).
- `az.epochs` defaults to 30 (not 200) with `lr_t_max=200` preserving the original cosine
  schedule at ep24; `checkpoint_every=5` so ep 19/24/29 can be compared.

## v5 (2026-07-05): MERGED MODEL POOL (user request)

`CFG['model']` is now the pool: `model_type` selector + one parameter block per
registered model. Registered: `ResMLPDistance` (ours, AZ v4 default) and
`PilgrimAttnRes` (ogurtsov's one-hot+BatchNorm family, `residual`/`attn_res` blocks,
ported into the model cell with `features()`/`feature_dim`/chunked eval). **Any
registered model trains through all five stages**: stages 1-4 use its native head,
stage 5 wraps the backbone in a generic `DualHeadModel` (policy+value heads over
`backbone.features`). Warm-start/export are registry-generic (each entry declares its
native head attr; export writes a NATIVE checkpoint of that model type - ResMLP exports
keep the exact legacy format for our local tooling). Also adopted their name-based
factories: per-stage `optimizer: {name, params}` (any torch.optim; lr stays the stage
key) and `loss: {name, params}` for pretrain/curriculum (incl. their PinballLoss /
LogCoshLoss). Guards: shipped warm-starts only for the default 6M ResMLP - other
models/shapes must run from 'pretrain' (clear AssertionError otherwise); the az quick
canary switches to eval mode (BatchNorm batch-of-1). Smokes: default chain (trajectory
matches pre-refactor), Pilgrim full chain, Pilgrim az-only guard - all pass.

## v6 (2026-07-05): full-chain default + wall-guard + auto-resume (user request)

Default `stages_to_run` = all five stages. Because a Kaggle run killed by the session
limit saves NO output, training stops gracefully at `max_wall_hours` (default 8.5h,
minus `eval_reserve_minutes`=20): every trainer checks the deadline per epoch, saves a
resume checkpoint (with optimizer/scheduler state), and returns (path, completed).
Dispatch writes `models/chain_state.json` (fingerprinted by model_type+params+seed;
posix rel paths) with completed stages + the partial stage's resume point. The kernel
mounts ITS OWN previous output (`kernel_sources` self-reference - accepted by the API;
mounts under /kaggle/input/notebooks/<owner>/<slug> presumably, probed like the other
mounts) and auto-continues: completed stages are skipped (warm-starts read from the
mount), the partial stage resumes at its saved epoch. "Save & Run All" repeatedly until
CHAIN COMPLETE (~25-30h T4 = 3-4 sessions). Smokes: regression, pre-stage guard trip,
mid-stage guard trip (az paused at ep0), continuation (4 stages skipped + az resumed at
ep1 + complete) - all pass. Chain progress (measured on T4, 8.23h/session):
- Session 1 (v6, 2026-07-05): pretrain 0 -> 2507/4000 (11.7 s/ep), paused, output saved.
- Session 2 (v7): AUTO-RESUME VERIFIED on Kaggle - found prev output at
  `/kaggle/input/cayleypy-az4-trainer-megaminx`, resumed pretrain @2508, FINISHED
  pretrain (ep3999), chained into curriculum to ep3454 (4.1 s/ep), paused.
- Session 3 (v8, 2026-07-06): curriculum FINISHED (early stop ep7799, best val 68.66 @
  ep6799 - slightly better than the original 69.29) + bellman to 183/500 (76 s/ep).
- Session 4 (v9, 3.4h quota-safe budget): bellman 184 -> 317/500 (loss 0.069, on the
  original trajectory).
- Session 5 (v10): FAILED in 38s - **chain bug**: Kaggle mounts only the LATEST
  successful output, which contains only files THAT run wrote; the resuming bellman
  stage redundantly loaded its warm-start (curriculum best_ema) which session 4 never
  wrote -> FileNotFoundError. Sessions 2-4 worked by luck (needed file always one
  session back).
- FIX (v11, 2026-07-06): (a) resuming stages skip the warm-start load entirely (resume
  restores all weights anyway); (b) completed-stage + partial artifacts are COPIED
  FORWARD into every run's output (self-contained outputs; stale entries tolerated with
  a warning).
- Session 6 (v11, 0.75h probe): resume fix WORKED (bellman 318->335) but the run then
  DIED in the eval cell loading a STALE completed-stage pointer -> errored notebook ->
  output discarded (epochs lost). Second fix: eval only touches checkpoints that exist,
  walking back the chain and falling back to the shipped reference; solve cell asserts
  its checkpoint exists. Both failure modes now covered by local smokes (stale-eval
  repro, broken-prev heal, fresh regression - all pass).
- Session 7 (v12, 0.6h probe, 2026-07-06): FULL MACHINERY VALIDATED on Kaggle - resumed
  bellman from the copy-forwarded local file, warned on stale pointers, banked 318->329,
  paused, eval fell back to reference, clean COMPLETE (0.34h). Chain resume point:
  bellman@329 in a self-contained output.
- 2026-07-06 (user idea): added `CFG['experiment']` - an explicit chain-identity label
  in the resume fingerprint (experiment, model, seed). New label = fresh from-scratch
  chain with no input/seed surgery; enables fixed-seed recipe A/Bs. Old chain-states
  without the field are treated as 'az4-baseline' (backward-compat verified by smoke:
  old-schema state resumes; new label starts fresh). Ships with the finale push.
- REMAINING: ~171 bellman ep (~3.7h) + bellman_dd (~1.1h) + az (~0.5h) + eval = ~5.5h ->
  ONE 8.5h session after the weekly GPU quota resets (likely Sat 2026-07-11). Source
  already restored to max_wall_hours=8.5; to finish:
  `kaggle kernels push -p megaminx/kaggle_notebooks/az4_train_shareable --accelerator NvidiaTeslaT4`
  (PowerShell + token env vars per CLAUDE.md 7d), or click "Save & Run All" AFTER pushing
  the 8.5h code (the saved v12 still has the 0.6h probe budget).

## First public run: VERIFIED (kernel v4, 2026-07-04, Tesla T4, 33 min total)

- Default config (az stage, 30 ep at ~59 s/ep fp32) reproduced the original trajectory:
  ep24 p_loss 2.299 / v_loss 0.110 / top-1 30.1% (original 76,304-data run: 2.357 / 0.110 /
  28.5% — slightly faster policy convergence, consistent with the tighter 73,731 data).
- Canary on the fresh ep24 export: ALL PASS and near-identical to the shipped reference
  (V(solved)+0.077 vs +0.020, V(d1) 1.009 vs 1.008, sat gap +3.6 = +3.6, V@80 29.5 = 29.5,
  std@20 2.74 vs 2.76). Bench (cayleypy iterated, beam 4096): 3/3 solved, lens 100/116/108.
- Kaggle environment gotchas fixed during rollout (kernel v1-v3 failures):
  1. **Mount paths changed**: datasets now mount at
     `/kaggle/input/datasets/<owner>/<slug>` and competitions at
     `/kaggle/input/competitions/<slug>` (not the classic `/kaggle/input/<slug>`).
     The mounts cell probes both conventions.
  2. **The current Kaggle torch (2.10.0+cu128) has NO sm_60 kernels — P100 is dead**
     (`CUDA error: no kernel image is available`). Must request T4:
     `kaggle kernels push --accelerator NvidiaTeslaT4` and/or
     `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json. Valid enum (from
     kagglesdk): `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`; invalid values are
     SILENTLY ignored (defaults to P100). The env cell also probes a CUDA op at startup
     and fails with instructions if the GPU is unsupported.

Goal: a public Kaggle notebook in the style of
[ogurtsov/cayleypy-rw-modelbaselines-megaminx-adamw](https://www.kaggle.com/code/ogurtsov/cayleypy-rw-modelbaselines-megaminx-adamw)
where teammates can (a) reproduce AZ v4 (our best megaminx model: strat-5 51/51, mean 87.5
standalone V) end-to-end, and (b) swap any stage's hyperparameters / data / trunk shape from
a single config cell.

---

## 1. Reference notebook analysis (what to mirror)

Structure of ogurtsov's notebook (65 cells, GPU, internet on, competition mount
`cayley-py-megaminx`):

1. Big markdown intro (what/why, CayleyPy links, paper links, credits).
2. Params markdown documenting the config groups.
3. `!pip install git+https://github.com/cayleypy/cayleypy.git@<pinned-commit> --no-deps`.
4. **ONE flat `cfg` dict** — the whole notebook is driven by it. Groups: manage flags
   (`mode_train`, `only_train`, `load_ckpt`, `list_states_to_solve` for quick tests),
   model arch (`model_type`, `hd1/hd2/nrd`, ...), training (`rw_width/rw_length/rw_mode`,
   `num_epochs`, `batch_size`), and **name-based factories**: `optimizer: {name, params}`,
   `scheduler: {name, params}`, `loss: {name, params}` resolved via `getattr(torch.optim, name)`
   etc. Plus optional EMA/SWA weight averaging block. cfg is dumped to `cfg.json`.
5. Load competition data from the mount (`puzzle_info.json`, `test.csv`, `sample_submission.csv`)
   + path-verify utilities.
6. Build `CayleyGraph` (cayleypy).
7. **Model-zoo cell**: architectures defined inline (`PilgrimAttnRes` with `residual` /
   `attn_res` block types), selected by `cfg['model_type']`.
8. Train-utilities cell: factory builders + `train_model` loop (fresh random walks each epoch
   via `graph.random_walks`, val on next epoch's walks, MA50 val loss printed).
9. Checkpoint save/load cells (model + optimizer + scheduler + averaged weights) → resume
   across sessions.
10. Analysis cells: loss curves (+loglog), V-predictions along RW trajectories, V on test states.
11. `sys.exit(0)` if `only_train`.
12. Beam-search block: cayleypy `graph.beam_search` + `Predictor(graph, model)` per test state,
    per-state fallback to sample submission, correctness check via `graph.apply_path`.
13. Submission save + stats + `paper_dict` → they print a **LaTeX table row** for the CayleyPy
    community results table.

What it trains: plain RW distance regression (CayleyPy-1 baseline) — i.e., the equivalent of
our Stage 1 only. Our kernel adds the 4 stages above that, which is the whole point.

Patterns to keep verbatim: single cfg dict + cfg.json dump; name-based factories; quick-test
flags; resume pattern; sanity/analysis plots; cayleypy beam + submission block at the end;
paper_dict row. Pattern to add: **STAGE selector** (their notebook has one stage; ours has 5).

---

## 2. AZ v4 recipe — full lineage (verified from configs + training-log headers)

All models are ResMLP `hidden_dims=(2048,512)`, `num_res_blocks=2`, `encoding=embedding`,
`embed_dim=16`, `state_size=num_classes=120` → 6.05M params (AZ dual-head: 6.06M).
Wall times are local 4090 Laptop (bf16 + compile).

| # | Stage | Script | Warm-start | Key hyperparams | Wall | Output |
|---|-------|--------|-----------|-----------------|------|--------|
| 1 | RW MSE pretrain (`m07`) | `02_train.py --config m07_big_k80_local.yaml` | scratch | 4000 ep x 1M samples, batch 16384, k_max 80, n_back 1, lr 2e-3 cosine, MSE | ~80 min (1.2 s/ep) | `m07_big_k80/epoch_3999.pt` |
| 2 | Curriculum + EMA (`m_curr_v0`) | `48_train_curriculum.py` | m07 ep3999 | k=35 warmup 5000 ep, then k mix [50,70,80,100]; EMA tau 1e-3; val every 100, patience 1000; budget 50000 ep, stopped at 6199 (best 5199, val_mse 69.29) | ~2h (1.2 s/ep) | `m_curr_v0/best_ema.pt` |
| 3 | Anchorless Bellman (`m_curr_v3`) | `60_train_admissible.py --config m_curr_v3.yaml` | m_curr_v0 best_ema | 500 ep x 500K, batch 8192, k80, lr 5e-4; Bellman target `clip(1+min_a V_tgt(child), 0, walk_depth)`, target-net refresh /10 ep; frontier mixin 25% (300K states, bootstrap targets, cap 200); BFS-d6 mixin 10% (exact d<=6) | ~2.6h (19 s/ep) | `m_curr_v3/epoch_0499.pt` |
| 4 | Anchored Bellman + PDB (`m_dd_v0`) | `60_train_admissible.py --config m_dd_v0.yaml` (run with n_epochs=50) | m_curr_v3 ep499 | stage-3 recipe + anchor mixin (32x V0 target 0; 4x each of 24 d=1 children target 1) + PDB lower-bound penalty lambda=5 (4 disjoint corner PDBs) | ~20 min (22 s/ep) | `m_dd_v0/epoch_0049.pt` = canonical V |
| 5 | AZ dual-head fine-tune (`m_az_v4`) | `71_train_az_v3.py` | m_dd_v0 ep49 (trunk + value head) | `ResMLPGFlowNet` (trunk + policy head 24 + value head 1); value = stage-4 Bellman recipe minus frontier/PDB (RW 500K/ep k80 n_back1 + anchors 32/4 + BFS-d6 10%, tgt refresh /10); policy = CE on `az_dataset_76304.pt`; loss `1.0*CE + 1.0*MSE`; rw_batch 8192 / policy_batch 1024 (Rule 13: NOT the script defaults 4096/512); AdamW lr 5e-4 cosine T=200, wd 0; bf16 autocast; seed 71 | 200 ep budget, **stop at ep24** (~6 min to ep24, 14 s/ep) | `m_az_v4/epoch_0024.pt` |
| 6 | Export | pattern of `103_export_az_pi_only.py` | — | copy `embedding./input_stack./res_blocks.` + `head <- value_head` into a `ResMLPDistance(output_dim=1)` ckpt (pi head likewise for output_dim=24) | seconds | `m_az_v4_v_only.pt` (+ `_pi_only.pt`) |

**Why ep24** (from `m_az_v4_training.log` + 10-pid bench): the trunk specializes for policy
memorization past ~ep25 at the expense of V calibration. ep24 = p_loss 2.36 / v_loss 0.110 /
top-1 28.5% -> 10-pid bench 874 (best); ep49 = 902; ep74 = 958; ep99 = 898 with 9/10.
Training loss is NOT the selection signal — the bench is (Rule 12).

**Policy dataset**: `67_build_az_dataset.py --submission <csv>` -> for each pid, replay the
path from `test.csv`'s scramble; emit (state int8[120], action id, remaining-distance) per
step. v4 used `merge_v9_with_77152.csv` (min-merge of our 78,029 + community 79,911 + 77,152
= 76,304 total moves -> 76,304 tuples). Merged best-of-N matters: single-source (v3, 78,029)
converges slower and was over-trained; the merged set's consistency is why early-stop at 24
works. The June az15 pipeline used the same builder on the 73,614 floor -> parameterize.

**Data artifacts** (with builders):
- `bfs_d6_train.pt` — 19,352,405 states (int8 (N,120)) + exact distances (int8), ~2.3 GB.
  Built by `14_build_bfs_d6_dataset.py` from `bfs_bytes_d6.pkl`. Used by stages 3/4/5 + canary.
- `frontier_states.pt` — 300K deduped beam-frontier states (states only; targets computed by
  Bellman on the fly). Built by `42_log_frontier_states.py` from m05-era beam runs. Stages 3/4
  only. Not regenerable in-kernel without a solver stack -> ship as artifact.
- `az_dataset_76304.pt` — ~9 MB. Rebuildable in-kernel from any submission CSV + test.csv.
- Corner PDBs (stage 4 lambda_pdb=5 only) — 4 x 452 MB.
- `puzzle_info.json`, `test.csv`, `sample_submission.csv` — from the competition mount.

**Acceptance gates** (project rules): V-canary — V(V0)~0, V(d=1)~1, BFS-d6 calibration;
saturation — V@d80-V@d40 <= 10 AND V@d80 near diameter ~29, not collapsed (Rule 23 +
dodeca-CNN false-positive lesson); mid-depth variance at d~20 (repr-bundle lesson); 10-pid
bench for checkpoint selection; strat-51 with the production recipe as the binding gate
(Rule 21) — the last one stays OUT of the kernel (too heavy), documented as "what we do".

---

## 3. Kernel design

Name: `artgor/cayleypy-az4-trainer-megaminx` (title "CayleyPy AZ4 Trainer Megaminx" — title
must slugify to the id, Rule 7d). GPU on, internet on (pip for cayleypy only),
competition source `cayley-py-megaminx`, dataset source = new assets dataset (sec. 4).

### Stage selector (the core addition over ogurtsov's)

```python
cfg = {
  # ---- manage ----
  'stages_to_run': ['az'],      # any contiguous subset of
                                # ['pretrain','curriculum','bellman','bellman_dd','az']
  'quick_test': False,          # tiny epochs/samples smoke mode
  'resume_ckpt': None,          # resume a long stage across sessions
  'warmstart': 'auto',          # 'auto' = previous stage's output if it ran this session,
                                # else the shipped artifact for that stage from the dataset
  ...
}
```

Default `['az']` reproduces AZ v4 from the shipped `m_dd_v0/epoch_0049.pt` in well under
an hour, then exports v_only/pi_only, runs the canary, and (optional flag) a small cayleypy
beam bench. Users who want to go deeper flip the list; every stage's warm-start input is
either the freshly trained output or the shipped checkpoint, so any suffix of the chain runs.

### Config groups (one dict, JSON-dumpable, mirroring their conventions)

- `model`: hidden_dims, num_res_blocks, embed_dim (default (2048,512)x2 = the 6M cluster;
  doc note: bigger trunks regress under this recipe — Rule 14 — and 15M needs the full
  pretrain chain at matched shape, Rule 18: warm-start silently skips mismatched shapes).
- `pretrain`: epochs 4000, samples 1M, batch 16384, k_max 80, n_back 1, lr 2e-3,
  optimizer/scheduler/loss as name+params factories (keep their factory helpers).
- `curriculum`: warmup_k 35, warmup_epochs, k_mix [50,70,80,100], ema_tau 1e-3,
  val_every 100, patience 1000, budget 50000.
- `bellman` / `bellman_dd`: epochs (500 / 50), batch 8192, lr 5e-4, target_update 10,
  clip_upper/lower, frontier_fraction 0.25, bfs_d6_fraction 0.10, n_anchor_v0 (0 / 32),
  n_anchor_d1 (0 / 4), lambda_pdb (v1: fixed 0 — see "Scope cuts").
- `az`: epochs 200, stop_hint 24, rw_batch 8192, policy_batch 1024, alpha 1.0, beta 1.0,
  samples_per_epoch 500K, k_max 80, n_back 1, lr 5e-4, target_update 10, checkpoint_every 25
  (add every-checkpoint canary print), policy_csv (path into mounted dataset) OR prebuilt
  `az_dataset` path, seed 71.
- `precision`: 'auto' | 'fp32' | 'fp16' | 'bf16' — **critical port**: our trainers hardcode
  bf16 autocast; P100 (sm60) has no bf16 and T4 emulates it catastrophically (confirmed ~40x
  on beam). auto = bf16 only on Ampere+, else fp32. 6M model at fp32 on P100 is fine.
- `compile`: auto-gate exactly like their cell 30 (off on P100; ResMLP compile is safe, Rule 2/22).
- `eval`: canary on/off + n_samples, bench_pids (default 2-3 easy pids), beam_width,
  full-solve block on/off.
- Echo the resolved config at start of every stage (Rule 13) and dump `cfg.json`.

### Cell map

1. Intro markdown: what AZ v4 is, the 5-stage lineage diagram, results table
   (m05 89.4 -> m_dd_v0 89.4-ish/871-bench -> AZ v4 87.5 / 51/51 strat-5 standalone),
   link to competition + CayleyPy, credit ogurtsov's notebook as the structural template.
2. Params markdown: config groups, stage semantics, session-budget table, "which stage
   should I run?" guidance.
3. cfg cell (above) + cfg.json dump.
4. Environment cell: device + precision autodetect, prints; pip install cayleypy@pinned
   (only needed by the bench/solve cells — guard the import).
5. Data cell: competition mount + assets-dataset mount; friendly asserts listing exact
   required dataset slug when missing.
6. Vendored library cells (bannered "library code — no knobs here"):
   a. `megaminx/puzzle.py` (90 ln) — Megaminx loader from puzzle_info.json.
   b. `cayley/data.py` subset (~230 ln) — GeneratorTable + generate_walks_torch (n_back).
   c. `cayley/model.py` ResMLPDistance + `cayley/gflow_model.py` ResMLPGFlowNet (~510 ln).
7. Dataset-builder cell: inline port of `67_build_az_dataset.py` — build az_dataset from ANY
   submission CSV in the mounts (or load the prebuilt .pt). Path-verifies each row.
8. Trainer cells, one per stage (pretrain / curriculum / bellman(+dd) / az), each a pure
   function of (cfg, warm_ckpt) with our exact log format, checkpointing, resume, and the
   AZ stage's per-checkpoint canary hook. Bellman cell is a trimmed port of the
   `cayley/bellman.py` loop (no PDB, no rotation-aug, no solver-trace — just the paths the
   recipe uses: RW + frontier + BFS-d6 + anchors + target net).
9. Stage-dispatch cell: run `stages_to_run` sequentially, wiring warm-starts.
10. Export cell: v_only + pi_only extraction (mirror `103_export_az_pi_only.py`).
11. Canary cell: V(V0), V(d=1 children), BFS-d6 MAE by depth (reads bfs_d6_train.pt — NOT
    the pkl), RW saturation probe (V@d40 vs V@d80, absolute level vs diameter ~29), V-std
    at d~20. Prints PASS/WARN lines with the thresholds from the project rules.
12. Optional mini-bench cell: cayleypy `graph.beam_search` + `Predictor` on `bench_pids`
    with the exported v_only model (their cell-51 pattern, `beam_mode='iterated'`; NEVER
    'advanced' — Rule 3). Honest caveat in markdown: our production numbers come from
    KhoruzhiiSolver + sym-ensemble multi-pass, so absolute lengths here differ; this bench
    is for relative comparison between checkpoints and with the community baselines in the
    SAME harness.
13. Optional solve+submit block (their cells 51-59 shape, off by default) + paper_dict row.
14. Caveats markdown: qshort incompatibility (Rule 15 — don't pair AZ v4 V with m23_v2),
    over-training cliff (don't train past bench peak), bigger-trunk trap (Rules 14/18),
    bf16-on-T4 trap.

### Scope cuts for v1 (explicit)

- No PDB penalty in the vendored Bellman (lambda_pdb pinned 0 + markdown note). Exact
  stage-4 used lambda=5; default users are unaffected because the SHIPPED m_dd_v0 ep49 IS
  the lambda=5 artifact. Full PDB port (+1.8 GB assets + pdb_heuristic code) is v1.1 if
  anyone actually wants to retrain stage 4 exactly.
- No KhoruzhiiSolver port; bench via cayleypy.
- No strat-51 gate in-kernel (documented instead).
- Stages 1-3 support resume but are documented as multi-session on P100/T4.

---

## 4. Assets to publish — ONE new Kaggle dataset `artgor/megaminx-az4-training-assets`

| File | Size | Role |
|------|------|------|
| `checkpoints/m07_big_k80_epoch3999.pt` | ~24 MB | stage-2 input |
| `checkpoints/m_curr_v0_best_ema.pt` | ~24 MB | stage-3 input |
| `checkpoints/m_curr_v3_epoch0499.pt` | ~24 MB | stage-4 input |
| `checkpoints/m_dd_v0_epoch0049.pt` | ~24 MB | stage-5 input (DEFAULT path) |
| `checkpoints/m_az_v4_epoch0024.pt` + `m_az_v4_v_only.pt` | ~48 MB | reference outputs |
| `data/bfs_d6_train.pt` | 2.3 GB | stages 3/4/5 mixin + canary |
| `data/frontier_states.pt` | ~36 MB | stages 3/4 mixin |
| `data/az_dataset_76304.pt` | ~10 MB | stage-5 policy data (exact v4 repro) |
| `data/solutions_merge_v9_with_77152.csv` | ~1 MB | builder demo input (DECIDE: see 7.1) |

Total ~2.5 GB. Checkpoints go in the dataset (not Kaggle Models) — one mount, no
Models-path quirks, versioned together with the data. Publish via PowerShell with
`$env:KAGGLE_API_TOKEN`/`PYTHONUTF8=1` per-call (Rule 7d).

## 5. Session budget (P100/T4 fp32 no-compile, est. 3-4x local — verify on first run)

| Stage | Local | Kaggle est. | Fits one 12h session? |
|-------|-------|-------------|----------------------|
| az (to ep24 + export + canary + smoke bench) | ~10 min | 30-60 min | YES (default) |
| az full 200-ep sweep w/ benches | ~50 min | 3-4h | yes |
| bellman_dd (50 ep) | 20 min | 1-1.5h | yes |
| bellman (500 ep) | 2.6h | 8-12h | borderline -> resume |
| curriculum (~6.2K ep realized) | 2h | 5-8h | yes w/ resume |
| pretrain (4000 ep) | 80 min | 5-7h | yes w/ resume |
| FULL chain | ~7h | 20-30h | 2-3 sessions (30h/wk quota) |

## 6. Build + validate workflow

1. Assemble assets locally (export checkpoint copies, verify each loads; build
   `az_dataset_76304.pt` fresh from the CSV to confirm builder parity) -> push dataset.
2. Write the .ipynb locally (source-of-truth .py cells in this folder; ASCII-only prints,
   `encoding='utf-8'` everywhere — Rule 24).
3. Local smoke: run the notebook code paths with `quick_test=True` (2 epochs/stage,
   10K samples) on the 4090 against local files standing in for mounts.
4. Push kernel (private first) via `/kaggle-push` flow; run `stages_to_run=['az']`,
   `quick_test=True` on Kaggle -> then full az run.
5. Acceptance: Kaggle-trained ep24 canary within tolerance of local reference
   (V(V0)<0.1, V(d1) in [0.9,1.1], saturation gap <=10, BFS-d6 MAE comparable), p_loss/v_loss
   trajectory ~matches the log table above (fp32 vs bf16 will wiggle numbers slightly),
   smoke-bench solves its pids. Then flip public + share with team.

## 7. Decisions needed from Andrey

1. **Which solutions CSV to publish** for the policy-dataset builder: exact-repro
   `merge_v9_with_77152.csv` (76,304; contains community-merged paths) vs current best floor
   (73,614; also community-derived) vs our-solutions-only. Publishing community-derived paths
   in a public dataset should be a deliberate call.
2. Kernel + dataset public immediately, or private with team-share first?
3. v1 without PDB stage-4 retrain (recommended) — OK?
4. Include the optional full solve+submission block (slow: cayleypy beam over 1001 pids at
   useful widths is hours) or cap it at `list_states_to_solve` short lists like ogurtsov does
   (recommended)?
