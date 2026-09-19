# IHES PieceTransformer: training plan (2026-09-16)

> **USER DECISIONS (2026-09-16, override the sections below where they differ)**
> 1. **Goal.** Reaching the same lengths as the 21,870 file is fine. The deliverable is the
>    *model*: other people will run it with 256M-or-wider beams. Our own beam campaign
>    (section 5) is not the target; wins are a bonus.
> 2. **Training length.** After each stage, measure the path lengths the beam finds at
>    checkpoints every 200 epochs. Keep training while those lengths keep shortening.
>    `n_epochs` in the configs is only an upper bound.
> 3. **No arm C.**
> 4. **One arm on a GCP GPU** (arm B, on-demand A100 40 GB `ihes-tf-b`, us-central1-f).
>
> **Execution log**
> - 2026-09-16, ~20:34 local: arm A launched on the 4090 (`models/run_ihes_tf_a.bat`).
>   Config `configs/ihes_tf_a.yaml`; 3,508,755 params; ~15 s/epoch.
> - ~20:43: arm B launched on the A100 (`~/run_ihes_tf_b.sh`, resumes from the newest
>   checkpoint). Config `configs/ihes_tf_b.yaml`, run at batch 2048 x 256 steps; ~29 s/epoch.
> - Checkpoint-trend watchers: `scripts/85_watch_eval.py` runs `scripts/84_solve_tf.py` on
>   `data/ihes_gate54.json` (54 pids, floor 1215) at B=65,536, 1 forward frame, bf16,
>   endgame d6, no merge. Results go to `submissions/eval/<arm>/trend.csv`. Score = total
>   length with an unsolved pid charged floor + 10.
> - Stage-3 inputs are built:
>   - `data/ihes_q_anchors_d6.pt`: 10.93M states x 18 exact columns. Exclusion rate
>     93.2479% vs 93.2478% predicted; 0 violations.
>   - `data/ihes_path_bank.pt`: 20,655 states, gate pids held out.
>   - Trainer: `scripts/86_train_q_bellman.py`.
> - New helpers:
>   - `scripts/80_build_ihes_move_relabel.py`: (48,18) relabel table, brute-force verified.
>   - `scripts/81_build_ihes_gate.py`: the fixed gate set.
>   - `data/ihes_piece_layout.json`: 26 pieces.
>   - `data/bfs_d5_anchor_states.pt`: the d<=5 slice for arm B.
>   - `tetraminx/scripts/51_train_sparse_q.py` gained `puzzle: ihes`, `sym_prefix`, and a
>     `depths` key fallback. All are backward compatible.
> - Matched control, E6 V at B=65,536, 1 forward frame, endgame d6, through
>   `84_solve_tf.py`: **54/54 solved, total 1355 vs floor 1215 (+140), 9 ties, 0 wins**
>   (`submissions/eval/e6_b16/`).
> - Evals moved to the A100 (~22:30). All trend numbers from here on are **A100, bf16,
>   B=65,536, 1 forward frame, endgame d6**, so they compare under rule 28.
>   - `models/sync_ihes_tf_a.sh` copies each 200th arm-A checkpoint to the VM, where a
>     second watcher evaluates it.
>   - The local watcher was stopped. On the 4090 it took 23 minutes per eval and tripled
>     arm A's epoch time while it ran.
> - **Epoch 200, matched (A100):**
>
>   | model | total | vs floor | ties |
>   |---|---|---|---|
>   | E6 | 1355 | +140 | 9 |
>   | arm A | 1353 | +138 | 5 |
>   | **arm B** | **1333** | **+118** | 9 |
>
>   E6 reproduces 1355 exactly on both machines. Arm A read 1355 on the 4090 and 1353 on
>   the A100, a hardware bf16 effect.
> - The same 4090 arm-A epoch-200 run vs E6 decorrelated by pid: shorter on 11, longer on 12;
>   the per-pid min of the two is 1331.
> - Bellman probe (~22:40): `86_train_q_bellman.py` from **arm B epoch 200** on the A100,
>   6000 steps, a checkpoint every 500. A watcher gates every 1000th step into
>   `submissions/eval/ihes_tf_b_e200_s3/`.
> - Trend (A100 gate score; lower is better; floor 1215; E6 1355):
>
>   | model | e200 | e400 | notes |
>   |---|---|---|---|
>   | arm A | 1353 | **1339** (8 ties) | improving, so training continues |
>   | arm B | 1333 | -- | |
>   | arm B e200 + Bellman | step 1000: **1321** (+106, 13 ties) | -- | Bellman pays on IHES too |
>
>   In the Bellman probe, `E[target]` held at ~6.9 (walk states) and ~12.3 (path states)
>   through step 3500. No collapse; it ran at ~0.28 s/step on a shared A100.
> - **Bellman probe curve** (arm B e200, A100 gate):
>
>   | Bellman steps | 0 | 1000 | **2000** | 3000 | 4000 | 5000 |
>   |---|---|---|---|---|---|---|
>   | gate score | 1333 | 1321 | **1313** (+98, 14 ties) | 1317 | 1315 | 1315 |
>
>   Flat after 2k, the same shape cube555 measured. `E[target]` drifted down slightly by
>   step 5000 (walk states 6.91 -> 6.69, path states 12.29 -> 12.05). The probe's 26 minutes
>   plus its six evals slowed arm B from 29 to 93 s/epoch on the shared A100. **Schedule
>   future Bellman runs and their evals so they don't overlap stage-1 training on the same
>   GPU.**
> - Stage-1 trend (A100 gate-54), 2026-09-17 ~00:50:
>
>   | model | e200 | e400 | e600 | e800 |
>   |---|---|---|---|---|
>   | arm A | 1353 | 1339 | **1309** (+94, 14 ties) | 1327 |
>   | arm B | 1333 | **1315** (+100, 13 ties) | | |
>
>   Arm A is non-monotone, like 444's s1L sweep; its 18-move swing is 1.4%, near what 54
>   pids can resolve.
> - **Arm A stage 1 stopped at epoch 1043** (~01:25). Scores 1353 / 1339 / **1309** / 1327 /
>   1319, i.e. two evals without beating e600 (the user's rule).
>   - Stage 3 on the 4090 (`models/run_ihes_tf_a_s3.bat`): Bellman 3000 steps from e600, then
>     from e1000, both on bank-h108. Then a sequential gate-54 pass over every 500th step.
>   - These are 4090 numbers, a trend within those runs only. The finalists will be
>     re-scored on the A100 with gate-108.
> - **Arm A stage 3, matched 4090 gate-54:**
>
>   | init | step 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
>   |---|---|---|---|---|---|---|---|
>   | e600 | **1309** | 1315 | 1311 | 1315 | 1313 | 1315 | 1307 |
>   | e1000 | 1317 | 1307 | 1307 | 1307 | **1305** (18 ties) | 1311 | 1307 (19 ties) |
>
>   Bellman pulls arm A's checkpoints to a common ~1305-1310 level: +12 for e1000, nothing
>   for the already-good e600. `E[target]` was stable (~7.0 walk, ~12.0 path).
>   e600/e1000 on the 4090 (1309/1317) vs the A100 (1309/1319): cross-hardware drift <= 2.
> - Arm A LR step-down (queued behind the baselines): resume e1000 at lr 3e-5 for 200 epochs
>   (`--resume-lr`, added to 51), then gate the result at e1100 and e1200 on the 4090.
> - Arm B: e600 = **1299** (+84, 18 ties) on the A100, the best so far; still improving.
> - **Arm A LR step-down (4090 gate-54):** e1000 at 1317 -> anneal **e1100 = 1299** (+84,
>   19 ties) -> e1200 = 1309. The step-down pays: both annealed checkpoints beat every
>   constant-LR arm-A checkpoint on the same machine. Bellman from anneal e1100 is next
>   (`models/run_ihes_tf_a_anneal_s3.bat`).
> - Arm B e800 = **1297** (+82, 17 ties), A100. Gains are shrinking: 1333 / 1315 / 1299 / 1297.
> - Leaders on gate-54 (1297-1305) sit within one gate's noise of each other. The final
>   choice will use gate-108 on one machine.
> - **Bellman from anneal e1100 (4090 gate-54):** 1301 / 1301 / 1299 / 1299 / 1299 / 1301 at
>   steps 500-3000, flat versus its start of 1299. Bellman lifts weak checkpoints (e1000:
>   1317 -> 1305) but not good ones.
> - **Finalists, matched on the 4090** (`models/run_ihes_finalists_4090.bat`, from ~05:39):
>   - A1 = anneal e1100; A2 = A1 + Bellman 2000; B8 = arm B e800.
>   - Scored on gate-108 at B=65,536, then on gate-54 at B=262,144 (closer to the deployment
>     width).
>   - Results go to `models/ihes_finalists_4090.log`.
>   - **Gate-108 @ B=65,536 (4090):**
>
>     | model | gate-54 half | extra-54 half | total (floor 2423) | ties |
>     |---|---|---|---|---|
>     | A1 | 1299 | 1314 | 2613 (+190) | 33 |
>     | A2 | 1299 | **1298** | **2597** (+174) | 36 |
>     | B8 | 1297 | **1298** | **2595** (+172) | 34 |
>
>     - **Bellman does generalise.** A2 ties A1 on the gate-54 half and is 16 better on the
>       extra half, which was held out of its path bank. Gate-54 alone was too small to show
>       it.
>     - B8 reads 1297 on both machines.
>     - **The models are complementary:** the per-pid min over A1/A2/B8 is 2563 (+140),
>       32 better than the best single model. Shipping one arm-A and one arm-B model lets
>       the people running 256M beams take the per-pid min.
>   - **Gate-54 @ B=262,144 (4090):** A1 = 1285 (+70, 23 ties); **A2 = 1267 (+52, 29 ties)**.
>     Bellman's edge grows with width: tied at 2^16, 18 moves at 2^18, as on 444. Since the
>     model will run at 256M+, **ship the Bellman-refined version**.
>     B8 (arm B stage 1 only) = 1279 (+64, 25 ties).
> - **v1a packaged** (`exports/ihes_tf_v1a/`, ~07:50) = A2
>   (`models/ihes_tf_a_e1100a_s3/step_02000.pt`).
>   - Gate-108 @2^16 = 2597 (+174, 36 ties); gate-54 @2^18 = 1267 (+52, 29 ties).
>   - npz parity PASS. It is the best model so far.
>   - Recipe: stage 1 for 1000 epochs at lr 3e-4, 100 epochs at lr 3e-5, then 2000
>     Bellman steps.
> - **Arm B refinements (gate-54 @2^16):**
>   - Bellman from e800 (4090): 1289 / 1295 / 1295 / 1291 / 1289 / 1291 at steps 500-3000.
>     Every step beats B8's 1297.
>   - LR step-down (A100): **e1300 = 1291** (+76, 20 ties), versus 1303 at e1200 just
>     before it.
>   - The anneal pays for arm B too.
> - **Arm B final (4090, ~08:50, `models/run_ihes_b_final_4090.bat`):**
>   - Bellman 3000 steps from anneal e1300.
>   - Matched finalists: B8S = e800 + Bellman 2000 and BAS = anneal e1300 + Bellman 2000,
>     on gate-54 @2^18 and gate-108 @2^16, against v1a (A2).
> - Arm B anneal e1400 = 1299 (A100), worse than e1300 (1291), so the anneal ended at 1400
>   and **e1300 is arm B's annealed pick**.
> - **Gate-54 @ B=262,144 (4090), all candidates:**
>
>   | model | total | vs floor | ties |
>   |---|---|---|---|
>   | A1 | 1285 | +70 | 23 |
>   | B8 | 1279 | +64 | 25 |
>   | A2 = v1a | 1267 | +52 | 29 |
>   | B8S (B e800 + Bellman 2000) | 1263 | +48 | 32 |
>   | **BAS (B anneal e1300 + Bellman 2000)** | **1255** | **+40** | **35** |
>
>   **The full recipe (stage 1 -> LR step-down -> Bellman) on arm B is the best model.**
> - **v1b packaged** (`exports/ihes_tf_v1b/`, ~10:00) = BAS
>   (`models/ihes_tf_b_e1300a_s3/step_02000.pt`); npz parity PASS (max|dQ| 1.7e-5).
> - Deployment variant queued on the 4090 (`models/run_ihes_b_full_4090.bat`): the same as
>   BAS but with the full path bank (`data/ihes_path_bank_full.pt`, all 1003 floor paths).
>   Wide-beam users solve all 1003 pids, so their states are fair game. Its gate numbers
>   are **in-sample** and are logged as such.
> - **Decisive matched comparison on the A100** (idle after the anneal, ~09:40):
>   - A2, BAS and B8S on **gate-108 @ B=262,144**, run by `~/run_g108_b18.sh` on the VM.
>   - Results go to `models/ihes_finalists_a100.log`.
>
>   | model | gate-54 half | extra half | total (floor 2423) | ties |
>   |---|---|---|---|---|
>   | A2 = v1a | 1267 | 1270 | 2537 (+114) | 54 |
>   | B8S | 1263 | 1270 | 2533 (+110) | 57 |
>   | **BAS = v1b** | **1255** | **1256** | **2511 (+88)** | **65** |
>
>   - At 2^18 the gate-54 halves reproduce the 4090 numbers exactly (1255 / 1267 / 1263).
>   - v1b ties the floor on 60% of held-out pids at only 2^18.
>   - Portfolio at 2^18: min(v1a, v1b) = 2505 and min of all three = 2503, so **v1b alone
>     captures nearly everything**; v1a is an optional second model.
> - Gate-108 @2^16 (4090): A1 2613, A2 2597, B8 2595, B8S 2585, **BAS 2573**. The ranking is
>   the same at both widths.
> - **v1b = the deliverable.**
> - Follow-up on the idle A100 (`~/run_bas_curve.sh`), applying the "continue while
>   shortening" rule to the Bellman stage at 2^18:
>   - v1b's step curve on gate-54 @2^18 (steps 500-3000);
>   - a second Bellman round from v1b (2000 steps), gated at 1000 and 2000.
> - **Bellman stage at 2^18 (A100, gate-54):**
>
>   | run | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
>   |---|---|---|---|---|---|---|
>   | round 1 (v1b's run) | 1263 | 1263 | 1257 | **1255** | 1259 | 1257 |
>   | round 2 | | 1251 | | 1255 | | |
>   | round 3 | | 1251 | | 1257 | | |
>
>   Round 2's step 1000 scored 1262 on the extra half (v1b: 1256), i.e. **2513 vs 2511 on
>   gate-108**. That is a plateau, so the Bellman stage is closed and **v1b is final**.
> - Deployment variant BASF (full path bank): in-sample 1257 (g54 @2^18) and 2571
>   (g108 @2^16), equal to v1b. Not needed.
> - **2026-09-17 ~12:55: A100 VM `ihes-tf-b` STOPPED** (disk kept). All results were pulled
>   to `submissions/eval/` and `models/`. The log entry is in EXPERIMENTS.md
>   (2026-09-16/17).
> - **v1b width scaling** (gate-54, 1 forward frame, 4090):
>
>   | beam | total | vs floor | ties |
>   |---|---|---|---|
>   | 2^16 | 1283 | +68 | 22 |
>   | 2^18 | 1255 | +40 | 35 |
>   | 2^20 | **1243** | **+28** | **41** |
>
>   At 2^20 all 7 L24 pids are at the floor. The L21 pids stay +12 with one frame, so wide
>   runs should add the inverse frame. The v1b README (`exports/ihes_tf_v1b/`) was
>   re-packaged with these numbers.
>
> **STATUS 2026-09-17 ~14:05: DONE.**
> - The deliverable is `exports/ihes_tf_v1b/`.
> - All jobs are stopped and the A100 VM is stopped.
> - **Arm B stage 1 stopped at epoch 1210** (~07:45). e1200 = 1303 was the second eval
>   without beating e800 (1297).
>   - A100: LR step-down from e1200 at 3e-5 for 200 epochs, gate-54 every 100 epochs
>     (`models/ihes_tf_b_anneal`).
>   - 4090 (queued after the finalists): Bellman 3000 steps from B e800 on bank-h108, gated
>     every 500 steps (`models/run_ihes_tf_b_e800_s3.bat`).
> - **Selection gate-108** (`data/ihes_gate108.json`, floor 2423): the 54 plus 54 more
>   disjoint pids.
>   - Future Bellman runs use `data/ihes_path_bank_h108.pt` (19,447 states, all 108 pids
>     held out), so final candidates are compared on pids none of them trained on.
>   - The trend keeps using gate-54 for continuity.
> - **v0 packaged** (`exports/ihes_tf_v0/`, weights gitignored): arm B e200 + Bellman 2000.
>   Gate 1313; npz parity PASS (max|dQ| 2.3e-5, argmin agreement 1.0000). It's a
>   preliminary artefact for testing a kernel port, not the final model.
> - **Deliverable path for 256M+ runs by others** (target kernel:
>   `artgor/cayleypy-cube444-q-beam-tpu-256m-explained`):
>   - That kernel hard-codes cube444 constants and **ReLU**, and names the Q output
>     `output_layer`. Our checkpoints are **SiLU** with a `head` output.
>   - `scripts/87_export_ihes_tf_npz.py` writes the kernel's npz layout plus numeric
>     `meta/*` constants, including `meta/silu`.
>   - `kaggle_notebooks/tpu_beam_ihes_tf/ihes_jax_q_models.py` is the generic JAX forward:
>     a full version, the kernel's CLS-only final block, and a HIGHEST-precision readout.
>   - Parity on epoch 50, 656 states: max|dQ| 2e-5, argmin agreement 1.0000, max|dV| 4e-6.
> - The IHES port of the tetraminx 256M-capable JAX driver
>   (`kaggle_notebooks/tpu_beam_ihes_tf/gcp_beam_ihes_tf.py`) passed an 8-device CPU
>   smoke on epoch 50 with forward and inverse frames, the endgame splice and replay
>   verification.

> **Beam-AVI on v1b (started 2026-09-17 ~15:30 local)**
>
> The user asked to improve v1b by training it on its own beam results, the way Q-training
> works. That is **Beam-AVI** (`BEAM_AVI_METHOD.md`). The priors are poor: on tetraminx it
> was **rejected for Q-space flattening**. There, deep gap01 fell from 0.438 to 0.270, one
> round tied, and five rounds were worse (`tetraminx/BEAM_AVI_PLAN.md`). v1b is in the same
> situation as that tetraminx incumbent, since it is already Bellman-refined.
>
> - **IHES ports:**
>   - `scripts/89_ihes_gen_harvest.py`: harvest with a beam hook. The sparse stream takes
>     8192 children per step with 1 + min Q(child) targets. The dense stream fully expands
>     256 parents per step. d<=6 table hits override the targets.
>   - `scripts/90_ihes_train_avi.py`: trains with 2.0 x exact d<=6 anchors, the value head
>     and an optional gap-margin term.
>   - `scripts/91_ihes_avi_loop.py`: harvest, train, level check and gate, then the stop
>     rule. It is resumable.
>   - `scripts/92_ihes_level_check.py`: a fixed-population flattening check (btw_sd, gap01,
>     n_red, anchor top-1).
> - **Per round:**
>   - **Harvest:** 50 pids at B=2^18. They are non-gate pids (both gate sets excluded), drawn
>     with seed 1000 + round. The target net is the student (solo).
>   - **Training:** warm from the previous round, 4000 steps, lr 2e-5.
>   - **Gate:** gate-54 at 2^18 (1 frame, bf16, no merge) against **v1b = 1255 on both
>     machines**, paired W/L/T per pid.
> - **Arms:**
>   - **A** (A100, `runs/ihes_avi/A`): faithful, no gap term.
>   - **B** (4090, `runs/ihes_avi/B`, `models/run_ihes_avi_B_4090.bat`): adds cube444's
>     gap-margin anti-flattening term (weight 0.3, batch 512, walk depth <= 12).
> - **Stop rule** (the user's rule, applied to rounds):
>   - Continue only while the gate total gets strictly shorter.
>   - Stop when the best entry is 2 rounds old. v1b counts as round -1, and ties go to the
>     earlier entry.
> - **Promotion bar:** a round must beat 1255 on gate-54 **and** 2511 on gate-108, both at
>   2^18 on the A100. Only then is it packaged as v2.
> - **Pre-registered predictions** (written before any round finished):
>   1. Arm A r000: deep gap01 and btw_sd fall below v1b, and n_red rises. The gate ties or
>      loses (the delta is within about +-4 or positive).
>   2. Arm B: gap01 holds, because the gap term pins it. The gate is closer to v1b than
>      arm A's.
>   3. If both are right, neither arm passes the promotion bar and v1b stays the deliverable.
>   A gate delta of 2 or less is within noise: hardware bf16 alone moved one model by 2.
>
> **PATH LENGTHS: the only metric that decides** (user, 2026-09-17; everything below them is
> proxy). Gate-54, B=2^18, 1 forward frame, bf16, no merge. v1b = **1255** on both machines.
>
> | arm | round | machine | total | vs v1b | W/L/T | loop |
> |---|---|---|---|---|---|---|
> | A (faithful) | r000 | A100 | 1255 | +0 | 2/2/50 | |
> | A | r001 | A100 | 1255 | +0 | 3/3/48 | **STOP**: no shortening in 2 rounds |
> | B (gap term, k<=12) | r000 | 4090 | 1259 | +4 | 2/4/48 | |
> | B | r001 | 4090 | 1257 | +2 | 3/4/47 | **STOP** |
> | R (rehearsal w=16) | r000 | A100 | 1255 | +0 | 1/1/52 | |
> | R | r001 | A100 | 1255 | +0 | 1/1/52 | **STOP** |
>
> **DONE 2026-09-17 ~17:40: negative.**
> - No AVI checkpoint shortened a path total.
> - No gate path beat a best-known pid length (0 wins vs the floor in all six gates).
> - **v1b stays the deliverable.**
> - The A100 VM is stopped. Results are in `runs/ihes_avi/`; the summary is in
>   EXPERIMENTS.md 2026-09-17.
>
> **USER CORRECTION (2026-09-17 ~17:45): the rerun (arm A20).** "The beam should have been
> 2^20. And 1 round of generation is not enough -- you need to do 5-10 rounds."
> - The runs above generated and gated at 2^18 and stopped after two rounds. They are **not**
>   a verdict on the method.
> - cube444's own result: gate at B=2^20; about 10 useful rounds (r005 -154, r010 -214,
>   nothing after); ~1 pass per round (9,000 steps).
> - **A20 on the A100** (`runs/ihes_avi/A20`, `91 --no-gate`), launched ~17:50:
>   - Faithful method, generation at **B=2^20**, **10 rounds, no early stop**.
>   - 12,288 sparse rows/step and full-expand 512, so each round's shard is ~one pass.
>   - Train 9,000 steps per round, lr 2e-5.
>   - Round 0 warm-starts from v1b.
>   - A 2^20 harvest takes ~45 min/round, so the 10 rounds take about 8.5 h.
> - **Gates on the 4090 at B=2^20** (`scripts/95_ihes_avi_gate_watch.py`,
>   `models/run_ihes_avi_A20_gate_4090.bat`):
>   - Matched against v1b's 4090 2^20 gate, **1243**.
>   - One gate takes ~100 min, so the watcher gates the newest finished round while the loop
>     runs.
>   - After the loop it gates r004 and r009 (cube444's r005/r010) and the last round.
>   - Results go to `runs/ihes_avi/A20/gate20/trend.json`.
> - **The A100 stops as soon as the loop ends** (user, ~18:00).
>   - `scripts/96_ihes_vm_reaper.py` (`models/run_ihes_avi_A20_reaper.bat`) copies every
>     finished checkpoint and the logs, writes `A20/remote_done.json`, then runs
>     `gcloud compute instances stop`.
>   - The gate watcher then works offline from the local copies.
>   - Both were tested end to end on a fake finished loop.
>
> **A20 RESULTS (2026-09-18): the method does pay at 2^20.** Gate-54 at B=2^20 on the 4090,
> v1b = 1243 (+28 over the 1215 floor, 41/54 already at the floor).
>
> | round | total | vs v1b | W/L/T | note |
> |---|---|---|---|---|
> | 0 | 1243 | +0 | 0/0/54 | 35 of 54 paths DIFFER at equal length (weights differ, max |dW| 0.10) |
> | 2 | 1239 | -4 | 3/1/50 | |
> | **4** | **1237** | **-6** | 4/1/49 | best; cube444 peaked at its r005 too |
> | 6 | 1243 | +0 | 3/3/48 | |
> | 8 | 1245 | +2 | 3/4/47 | |
> | 9 | pending | | | |
>
> - **The first pass was simply too small**: at 2^18 with 2 rounds everything tied. The gain
>   needs both the width and the rounds.
> - The dose-response peaks mid-run and gives the gain back, the same shape as cube444
>   (r010 best of 40) and tetraminx (round 1 tie, round 5 worse).
> - All 10 rounds harvested 50 fresh non-gate pids, 8.7-9.0M sparse rows each, 50/50 reaching
>   the endgame table; 9,000 training steps at lr 2e-5 per round (~47 min/round on the A100).
> - **The harvested level inflates every round**: shard target mean 14.19, 14.57, 15.00, 15.54,
>   16.03, 16.54, 17.21, 17.78, 18.25, 18.61. The d<=6 anchors hold the shallow end (anchor
>   loss ~0.01). Ranking survives to r004 and degrades after, which is consistent with the
>   inflation eventually distorting deep comparisons.
> - **Held-out check queued** (`models/run_ihes_a20_heldout_4090.bat`): v1b and r004 on the
>   54 pids of gate-108 that are NOT in gate-54 (floor 1208), also at 2^20. Those pids were
>   excluded from every harvest, so they test whether the -6 generalises.
> - The A100 stopped itself at 01:37 after copying all 10 checkpoints and the logs.
>
> **Round 0 diagnostics (2026-09-17 ~16:00; proxies only).**
> - **Arm A r000: gate 1255 vs 1255 (+0), W2/L2/T50.** A tie, the same as tetraminx's
>   first round. Prediction 1 holds.
> - **Level check (fixed population):** deep gap01 fell 0.387 -> 0.237 and deep n_red rose
>   5.40 -> 9.25, the same relative drop as tetraminx (-38%). V_hat rose 13.79 -> 14.27.
>   Anchors held (top-1 0.995).
> - **Arm B r000 flattens the same way** (deep gap01 0.240, n_red 9.16). The k<=12 gap term
>   does nothing at depth. **Prediction 2 is falsified.** v1b already sits at gap mean 2.00 on
>   those shallow pivots.
> - **Which stream flattens?** Trained on A's r000 shard with each stream alone:
>   - Sparse only: deep gap01 0.231.
>   - Dense only: 0.253 (128 parents/step) and 0.245 (512/step).
>
>   The unbiased dense stream flattens almost as much, so beam **selection bias is not the
>   cause** (`scripts/90 --sparse-weight`).
> - **Target sharpness** (`scripts/93_ihes_target_sharpness.py`, 8,192 beam parents):
>   - On deep parents the harvested target is **sharper** than v1b: gap01 0.403 vs 0.362,
>     n_red 5.65 vs 6.04.
>   - r000 fitted on that target ends **flatter than it**: gap01 0.233, n_red 9.37.
>   - So this is MSE regression-to-the-mean on confusable children, not a flat target.
>   - Frame-averaging (8 frames) makes every row flatter, so gap01 and n_red partly measure
>     noise. But r000 stays flatter than v1b after averaging too (n_red 10.75 vs 7.23).
> - **Walk-state forgetting** (`scripts/94_ihes_gap_by_depth.py`, next-minus-undo gap on random-walk
>   pivots):
>
>   | walk depth | v1b gap / n_red | A r000 gap / n_red |
>   |---|---|---|
>   | 9-12 | 1.78 / 2.03 | 1.11 / 4.66 |
>   | 13-16 | 1.18 / 3.42 | 0.78 / 6.78 |
>
>   Fitting only beam states plus d<=6 anchors erodes the mid-depth ranking v1b learned on
>   walks. v1b's own Bellman stage trained on 50% walk states.
> - **Rehearsal guard** (`--rehearse-weight`: frozen-v1b MSE on walk pivots, k 2-22):
>   - Walk depth 9-12 gap: 1.26 at w=1 and 1.45 at w=4.
>   - Beam deep gap01 barely moves: 0.243 and 0.260.
>   - Per-column MSE is a weak constraint on a two-column gap.
> - **Arm R launched ~16:20** on the A100: rehearsal w=16, round 0 on A's r000 shard (same
>   target net), and level checks on A's population. It asks whether a forgetting guard
>   changes the dose-response.

**Status: v1b delivered. Beam-AVI on v1b finished negative (see above); v1b is unchanged.**
**Deadline: 2026-09-22 22:00 UTC** (Kaggle CLI, checked 2026-09-16 18:50 UTC), about 147 h away.
Live leaderboard: Rokicki 21,840 (unchanged since 2026-03-01). CayleyPy, Absolute cinema, fle3n
and kiriill kholin are tied at 21,870. We are at 21,972.

Sources: `cube444/kaggle_inference/cube444_inference/reports/*.md`, `TRANSFORMER_AZ_RECIPE.md`,
`cube555_pull/cube555_handoff_2026_08_22/{README,RESULTS,APPROACH}.md`, `EXPERIMENTS.md`
(Exp 25-28 and 2026-08-05), `IHES_SEARCH_2026-09-06.md`, and Vlad Kuznetsov's public
`cayleypy-training-core/configs/ihes_p901_t000_piece_transformer.json`. Two code surveys of the
trainer and inference paths supplied the file:line references below.

---

## 0. TL;DR

- **What s1 / s1L / s3 were on 444.** s1 and s1L were continuations of *Vlad's already-trained*
  transformer on his own sparse-Q objective. The beam, not the loss, picked the early checkpoint
  `s1L_4k`. s3 was a 21-minute warm-started Q-Bellman refinement on exact d<=6 anchors plus
  solution-path states.
  - On the held-out 54 pids the whole chain was worth -80 moves.
  - **The blend flag was worth another -72 for free**, and the exact endgame -8.
- **IHES has no pretrained transformer.** Vlad's repo ships the IHES *config* but no weights,
  and none are on disk. Stage 1 is therefore trained from scratch, and its beam-best checkpoint
  plays the role of `s1L_4k`. s3 then runs as on 444, with cube555's correction: sweep 1k-6k
  Bellman steps, because 2k beat 20k there.
- **The model only matters if it finds a path 2 moves shorter than the 21,870 file on at least
  one pid.** Every generator is an odd permutation, so savings come in steps of 2.
  - One win = 21,868 = sole #2. Matching Rokicki needs 15 wins.
  - About 768 pids are not proven optimal.
- **Plan.**
  - Day 0: data prep, port the trainer, launch stage 1 overnight.
  - Day 1: port Bellman, run it, start gating.
  - Day 2: promote or kill against a matched control.
  - Days 3-5: campaign on the unproven pids, on the local 4090 plus Kaggle TPU (the tetraminx
    kernel already runs a transformer Q head natively).
  - Day 6: merge and verify.

---

## 1. What the model has to do

| incumbent length | pids | proven optimal | can still shorten |
|---|---|---|---|
| <= 20 | 52 | all | 0 |
| 21 | 171 | 146 | <= 25 |
| 22 | 457 | 37 (5 in Aug + 32 in Sep) | <= 420 |
| 23 | 316 | 0 (threshold 21 is unaffordable for exact search) | <= 316 |
| 24 | 7 (106 592 680 706 764 810 936) | only some openings excluded | <= 7 |

**Base rate.** If Rokicki's file is optimal, about 15 of the ~768 open pids are shortenable,
roughly 2%.

**What the earlier beam campaigns showed.** E6 and AZ-cube v1/v2 at 32-64M width, with up to
96 roots, tied the floor on all seven L24 pids and on a 36-pid L23 sample (Exp 26-28). On
pid 106 the per-step min-V trajectory at 64M was identical to the one at 32M. So the **scorer**,
not width, is binding. That is the same diagnosis that made the transformer pay on 444.

**What they did NOT show.** They did not show that shorter paths are absent. The expected number
of shortenable pids in that ~50-pid sample was about 1, so "0 wins" is weak evidence.

**The unrun sweep.** We have never swept the full unproven set with a floor-competitive scorer.
The only full-set runs were E6 at 1M and QE6 at 4M in April, both at 23,5xx quality.

---

## 2. The 444 chain, and what carries over to IHES

| step | cube444 (measured) | IHES analogue |
|---|---|---|
| orig | Vlad's transformer: sparse-Q, `k~U[2,45]`, `p~U[1,k-1]`, batch 512, lr 1e-4, no augmentation, no anchors; its 16,384-epoch schedule never finished | **does not exist**, so stage 1 runs from scratch |
| s1 | +3k steps of the same recipe, batch 4096, lr ~5e-5 with warmup | covered by the same stage-1 run |
| s1L | +20k steps. Beam curve 944/976/980/974/960 while the loss stayed flat; best at 4k | pick the stage-1 checkpoint **by beam**, never by loss |
| s3 | Q-Bellman, 6k steps from s1L_4k (details below) | same recipe at IHES size; sweep the step count |
| inference | blend 0.4 with a 47M MLP (-72), endgame d6 (-8); history_depth relaxation was harmful (+16) | blend with QE6 / E6-as-Q / arm C, endgame d6, forward + inverse frames |

s3 recipe on 444:
- batch 768 (384 random-walk + 384 path states), plus 256 d<=6 anchors per step at weight 2.0;
- lr 2e-5 flat; target network hard-copied every 500 steps;
- dense MSE on all 24 columns; 21 min on an A100.

**Corroboration and corrections from the other puzzles.**

- **cube555** (ResMLP-Q, same Bellman recipe).
  - Warm Bellman was "the single biggest lever".
  - **2k steps beat 20k** (18/24 vs 10/24 solved); 4k, 6k and 8k were indistinguishable.
  - The min over six checkpoints solved 24/24, so **keep every Bellman checkpoint for the merge**.
  - The double-Q min-bias fix is worth 0.138 moves, i.e. nothing.
- **444 and 555.**
  - Bellman from near-scratch fails.
  - Anchors in every batch are load-bearing: without them the bootstrap drifts to V(solved)~2 or
    to the Q==0 fixed point.
- **444 lessons.**
  - Adding label sources to a working sparse-Q objective lost 2 out of 2 times (`tmx`, `d6a`).
  - Path *labels* were catastrophic (the 6-change `s2` arm).
  - No offline metric tracked beam quality (four confirmations).
  - The 18-pid gate gave the wrong verdict twice, so use >= 54 pids.
  - The beam is deterministic, so add pids, not seeds.
  - Longer training was dead and EMA was neutral.
- **tetraminx** (trained from scratch).
  - Winning recipe: symmetry coverage + d<=5 anchors + pivot tilt 0.5 + AZ value head.
  - The AZ head mainly helps trainability: loss kept descending to ep900-1500.
  - At inference, qv-consistency gave -4. cube555 measured +8, so A/B it.

**Open question.** Vlad-faithful vs the tetraminx recipe is genuinely undecided for a
from-scratch IHES model, hence two arms.

---

## 3. The IHES recipe

### 3.1 Model

**Architecture.** `PieceTransformerQ` from `tetraminx/src/tetraminx/models.py`, with the IHES
layout:
- **Tokens:** 26 pieces (8 corners x3 stickers, 12 edges x2, 6 centres x4) + CLS, so T = 27.
- **Trunk:** d_model 256, 8 heads, 4 layers, ff 1024, SiLU, dropout 0.
- **Heads:** Q head 256->18 plus AZ value head 256->1. About 3.4M params.
- **Config values:** `state_size: 72` and **`num_classes: 72`**. This is a permutation puzzle; the
  444 trap `num_classes = 6` does not apply.
- **Attention:** `attn_impl: sdpa` on GPU, `einsum` on TPU.

These trunk sizes are the same as Vlad's IHES config.

**Why not bigger.** T=27 is about half of tetraminx's 51, so a 384-wide, 6-layer model would be
affordable. But cube555 measured no difference between 24.8M and 14.2M, and there is no time for
a capacity A/B.

### 3.2 Data prep (Day 0, about 3 h)

1. **Piece layout.**
   `tetraminx/scripts/50_derive_piece_layout.py --puzzle data/puzzle_info.json --out data/ihes_piece_layout.json`.
   The survey ran it without `--out`: 26 pieces, 0 block violations.
2. **Move relabel table** `data/cube_move_relabel.npy`, shape (48,18).
   - Use T51's convention: `sigma[k,m] = index(P[k][g_m[P_inv[k]]])`, with P = `cube_symmetries.npy`.
   - The IHES code conjugates the other way (P.s.P^-1, `scripts/09_symmetry_antisymmetry.py`).
   - T51 checks transport over all 48 x 18 at startup (`51_train_sparse_q.py:147-164`), so keep
     `verify_symmetry: true`.
   - Coverage: 12 outer moves + 6 slice moves give 14 rows per sample, covering 6-18 columns.
3. **Trainer port** (`tetraminx/scripts/51_train_sparse_q.py`).
   - Puzzle: swap to `cayley.puzzle.PictureCube` (:66-69, :572).
   - Symmetry file names: :125-127.
   - Anchor key: :324 reads `blob["distances"]`, but `data/bfs_d6_train.pt` stores `depths`
     (10.93M uint8 states, d<=6).
   - `data/bfs_table_d6_hash.npz` loads unchanged: Zobrist `ztab` 72x72, `max_depth` 6. The
     `max_anchor_depth < table_depth` assert therefore allows stage-1 anchors at d<=5
     (790,588 states).
   - Cosmetic: data-dir default (:524) and the log line (:588).
4. **d<=6 Q-anchors for stage 3, by exclusion labelling** (the 444 trick).
   - For each of the 10.93M states and each of the 18 moves, look the child up in the d<=6
     table. A miss is exactly depth 7.
   - Size: uint8, about 10.93M x 90 bytes = 0.98 GB, GPU-resident. Never `.long()` it.
   - Self-checks, from the level sizes 1 / 18 / 261 / 3,732 / 52,620 / 733,956 / 10,142,306:
     - the mean target per depth is d + ~0.87-0.89;
     - a d6 state has a down-degree of ~1.22, so **~93% of its children should miss**.
5. **Path-state bank.**
   - Rebuild with `scripts/10_build_az_dataset.py` from the **21,870** file, with a pid column.
     The existing `az_cube_dataset.pt` comes from the 21,872 floor and has no pid column.
   - Expand each state by 48 conjugations x inverse = 96 images, about 2.1M states.
   - Exclude the gate pids.
6. **GATE-54**, fixed now and excluded from the path bank:
   - 7 L24 + 20 L23 + 10 unproven L22 + 10 proven L22 + 7 proven L21.
   - On the 17 proven pids the floor *is* the optimum, so excess over floor is exact.
7. **Exact-depth probe** (a diagnostic, never a selector).
   - Every state on a proven-optimal path has exact depth L-i. Its toward-solved child is at
     exactly d-1 and its parent-side child at exactly d+1.
   - Scale: about 235 pids x 21 states x 96 images.
   - Use it for the 444 "method A" table (gap / pair / top-1 by depth), which on IHES has
     *exact* labels.
8. **Label-consistency pre-flight** (about 30 min, CPU).
   - Port `tetraminx/scripts/57_label_consistency.py`.
   - IHES axes commute (f0/f1/f2) and every generator has order 4. So `x y x^-1` (= y) and
     `x x x` (= x^-1) both pass the non-backtracking filter.
   - Expect worse than tetraminx's 25.9% of in-table pivots mislabelled.
   - The result decides whether arm B gets a canonical-walk filter.

### 3.3 Stage 1: sparse-Q from scratch (stands in for orig + s1 + s1L)

**Arm A: "Vlad-faithful + AZ head".** Primary arm, local 4090.

```yaml
seed: 0
model:
  arch: transformer
  az_head: true
  state_size: 72
  num_classes: 72
  d_model: 256
  nhead: 8
  num_layers: 4
  ff_dim: 1024
  dropout: 0.0
  layout_path: data/ihes_piece_layout.json
training:
  n_epochs: 1500
  steps_per_epoch: 256
  batch_size: 2048
  k_min: 2
  k_max: 23            # Vlad's IHES value; ~1.08x the counting bound (~21.3)
  pivot_tilt: 0.0      # p = floor(u*(k-1))+1 -> p ~ U[1,k-1], exactly Vlad's rw_middle_sparse
  lr: 3.0e-4           # Vlad: 1e-4 constant (fp32); T51/tetraminx: 3e-4 constant. Short budget.
  weight_decay: 3.0e-3
  grad_clip: 1.0
  sym_coverage: false  # Vlad: identity augmentation
  anchor_batch: 0      # Vlad: no anchors
  value_weight: 1.0
  top1_margin_weight: 0.0
  amp: true
  compile_model: true
  fused_optimizer: true
  verify_symmetry: true
  checkpoint_every_epochs: 50
  val_size: 8192
```

- **Cost** (estimate): tetraminx ran 340 ms per 7,680-row step at T=51 on this laptop. That
  suggests ~43k rows/s at T=27, i.e. ~12 s/epoch and **~5 h for 1500 epochs**. Measure it in
  the smoke run.
- **Data volume:** 524k fresh pivots per epoch, ~0.8B in total. Tetraminx plateaued in that range.
- **Depth coverage:** only ~8% of pivots land at depth >= 15, and ~3% at >= 18. That is
  intentional: the ranking label is sound only where walks are still locally geodesic. Stage 3
  owns the deep band.
- **Launch detached**, per memory `local_detached_training_launch`.

**Arm B: "tetraminx recipe".** Needs a second GPU, or runs after A.
- Starts from arm A and adds:
  - `sym_coverage: true, sym_rows: 4` (random frames; the full 14 rows would cost 14x);
  - `anchor_batch: 512, max_anchor_depth: 5`;
  - `pivot_tilt: 0.5`, `k_max: 26`;
  - optionally, depending on the pre-flight, a canonical-walk filter and the split
    ranking/level streams (section 7).
- Run at 512 x 1024 steps: about 2,560 rows/step, ~60 s/epoch, 8-9 h for 500 epochs.
- This deliberately bundles changes. It is a **portfolio arm** (a decorrelated model for the
  merge), not an ablation.

**Arm C: "ResMLP-Q, same objective".** Cheap; fills GPU gaps.
- `arch: resmlp` (T51's default), 1.6-10M params, with an 18-wide Q head and a value head, and
  arm A's sampler.
- About 69x cheaper per row than the transformer, so it trains in about an hour.
- Three uses:
  - the architecture-vs-objective A/B that `RESEARCH_SYNTHESIS_2026-09-04.md` asked for;
  - a Q-native blend partner, playing the role of 444's MLP;
  - a wide-beam scorer.
- Then give it its own s3. cube555's deployed stack is exactly this recipe.

**Selection.** Keep every 50-epoch checkpoint. Beam-gate (section 4) around epochs 100, 200,
400, 800 and 1500. Expect a non-monotone curve, as with 444's s1L.

### 3.4 Stage 3: Q-Bellman (the s3 step)

**Source.** Port `cube555_pull/cube555_handoff_2026_08_22/cube555/scripts/22_bellman.py` onto
T51's model and anchor classes. Its target is at :110-112, the refresh at :97-98, the anchors at
:114-121. It imports cube555's `qtrain.py`.

```
init           best stage-1 checkpoint by beam (required: from scratch fails on 444 and 555)
target         Q(s,a) = 0 if the child is solved, else 1 + min_a' Q_target(child)
value target   min_a target(s,a)   (keeps the AZ head consistent, so qv-consistency stays valid)
target net     hard copy refreshed every 500 steps (not EMA)
states/step    768 = 384 random-walk (arm-A sampler) + 384 path-bank states
anchors        +256/step, d<=6 exclusion-labelled, all 18 columns, weight 2.0
loss           dense MSE on all 18 cols + 2.0 * anchor MSE + value MSE
               fp32 loss, bf16 forward; NO Huber, NO reweighting (444 stage-2 post-mortem)
optimizer      AdamW (0.9, 0.95), wd 3e-3, lr 2e-5, 200-step warmup, then flat; clip 1.0
steps          6000, save every 500; gate 1k, 2k, 4k, 6k and keep all of them
children       768*18 = 13,824 no-grad forwards/step, chunked at <= 16384
cost (est.)    0.2-0.3 s/step on the 4090 -> 20-30 min
```

**What to watch.** Watch **`E[target]`**, not the loss.
- It should sit near the mean depth of the sampled states and drift *up* slightly
  (444: 12.8 -> 14.3).
- If it slides toward 0, the model has found the Q==0 fixed point. Stop.
- On 444 the loss rose after step 3,500 while the beam was at its best.

**Run this variant alongside, since it only costs 30 min: `s3-rw30`.**
- Change: random-walk states are walk *endpoints* with `k~U[1,30]`, instead of the shallow
  pivots.
- IHES test states behave like uniform random states: mean 21.8 vs a counting bound of ~21.3
  (lnZ 56.02 at the measured branching factor of 13.8). Walks of length >= ~25 sample exactly
  that distribution.

**Why Bellman should matter more here than on 444.**
- The sparse-Q ranking label is sound only while the walk is locally geodesic.
- IHES walks reach the typical distance around 22, which is where every solve *starts*.
- Bellman propagates from the d<=6 anchors without assuming anything about walks.

### 3.5 Inference stack (needed for the gate and the campaign)

**Base.** Fork `tetraminx/scripts/30_solve.py` (T30) into a new IHES driver. It already has:

| feature | lines |
|---|---|
| transformer loader | :99-108 |
| `BlendedQ` | :290-304 |
| qv-consistency / rerank | :307-329 |
| history | :351-354 |
| chunk 4096 | :194 |
| Zobrist endgame | :49-93 |
| shallowest-hit goal test | :374-388 |
| tail splice | :432-434 |
| inverse and symmetry frames | :134-139, :408-438 |
| replay asserts | :439, :451 |

**Edits for IHES.**
- Puzzle class -> `PictureCube` (:45, :235).
- Data paths, floor, and endgame -> `data/bfs_table_d6_hash.npz` (:152-157, :247).
- Symmetry files plus the derived relabel table (:242-244; frame convention at :412).
- Resolve the checkpoint's relative `layout_path`.
- **Add an E6-as-Q wrapper** (score the 18 children with a V model, ~20 lines). It gives a
  *matched* E6 / az_cube_v2 control through the same dedup, endgame and frames, and a
  blend partner.

**Do not use `scripts/02_solve.py` for the transformer.**
- `src/cayley/search.py:245-259` has no arch branch, so the transformer falls into the ResMLP
  builder and raises `KeyError: hidden_dims`.
- `--history-depth` never reaches the solver (`02_solve.py:85-87`).

**Smoke test first** with `models/qe6/epoch_0999.pt` (an 18-wide ResMLP Q head distilled from
E6). It proves frames, endgame splice and replay before any transformer exists.

**Defaults to carry.**
- **chunk 4096.** At 32768 the forward pages through WDDM and runs 9.2x slower, silently.
- **Endgame d6.**
- **Frames:** k0 forward + k0 inverse.
- **history:** measure it against a matched run. It was harmful on 444 (a relaxation there) and
  is set to 1 on the tetraminx TPU.
- **bf16:** only after an argmin-agreement check against fp32 on the real card. On 444,
  CPU-emulated bf16 flipped 8% of Q argmins.
- **Junction reduction:** check whether T30's splice cancels moves where the beam path meets the
  table tail. On 444 this was worth ~2 moves/pid. IHES same-axis moves commute and have order 4.
  Replay verification catches errors either way.

---

## 4. Gate

**Set.** GATE-54 (section 3.2 item 6). The beam is deterministic, so add pids, not seeds, when a
margin is under ~2%.

**Arms**, all at B=2^16 with 2 frames and endgame d6, through the same driver:
- controls: **E6-as-Q** (matched), az_cube_v2-as-Q, QE6;
- stage-1 checkpoints;
- s3 / s3-rw30 checkpoints;
- search settings: blend {0, 0.2, 0.4} x qv-consistency {0, 0.3}.

**Cost** (estimate): with the endgame the beam stops near depth 6, so ~16 steps at ~0.53 s each.
That is about 15-25 min per transformer arm and ~5 min per ResMLP arm.

**Score.**
- Excess over floor (exact on the 17 proven pids), floor ties, and wins.
- Use the per-pid min against the floor, never the standalone mean.

**Width check.** Run the best two arms at B=2^20 on a 12-pid subset.
- On 444, s3's edge grew with width.
- On 555, 2^22 solved *fewer* pids than 2^21, so width is not monotone.

**Promotion rule.** Promote the best transformer configuration if, at matched B, it has **fewer
excess moves and more floor ties than E6-as-Q**.
- Why this proxy: az_cube_v2 reached three L24 floor ties at 65K that E6 needed 32M for.
- "Reaches the floor at lower width" is the observable form of "better scorer".

**Kill rule, end of Day 2.** If no transformer configuration beats E6-as-Q at matched B on
GATE-54, stop the model line.

**Wins during the gate.** Any win goes straight to replay-verify and merge.

---

## 5. Deployment

### Route 1: beam campaign over the ~768 unproven pids

**Order.**
1. L24 (7 pids).
2. L22, ranked by `scripts/29_rank_twsearch_targets.py` (grouped-CV AUC 0.816 for "reducible").
3. L23.
4. The unproven L21 pids.

**Local 4090 at 2^20.** About 8.5 s/step x ~16 steps, i.e. ~3 min per pid-frame, or ~450
pid-frames/day (estimates).
- If only one frame fits, run the **inverse** frame: on tetraminx it carried the win.
- On IHES the winning frames scattered across the whole 96-element group.

**Kaggle TPU v5e-8.** Use `tetraminx/kaggle_notebooks/tpu_beam_tetraminx/`. It already runs the
transformer Q natively, with qv-consistency, history and the Zobrist endgame. The kernel's
defaults are already 18 moves / 72 slots (`jax_beam_spmd_v_only.py:1329-1330`).
- **Driver edits in `gcp_beam_tetraminx.py`:**
  - :127, change the assert from (24,88) to (18,72);
  - :135-137, symmetry files;
  - :155-156, layout;
  - :62-63, ResMLP dims (1024/256, 1 block, for a QE6 blend);
  - :216, endgame file.
- **Notebook builder edits:** `build_notebook.py:195,257,265-267` and
  `build_notebook_q.py:113-114,174`.
- **Validate** on 8-device CPU emulation against the T30 fork at small B.
- **Cost (not measured).** Tetraminx's blend costs ~199 s/step at 64M. IHES should be roughly
  half that: 27 vs 51 tokens and 18 vs 24 children.
  - If that holds, 4M is about 2 min per pid-frame, and the whole unproven L22 tier fits one
    weekly TPU quota.
  - At 16M it is about 7 min per pid-frame.
- **Operations:**
  - rule 7: arm the watch with `persistent: true`;
  - rule 7f: pull outputs before the next push;
  - one batch TPU session per account.

**Merge nightly** with `scripts/23_merge_all_ihes.py` (discovers sources by content and replays
them) against 21,870. See rules 26 and 26b.

### Route 2: model-ranked exact search (optional; run the pre-test first)

**Idea.** The Sep campaign's prefix-split exact search (`scripts/27_prefix_split_ladder.py`, the
native solver with the additive PDB hook, the Kaggle CPU fleet) proves optimality by exhausting
every opening. Order the openings by the model and truncate, and it becomes a discovery search
that is *exact after k moves*.
- That is robust to the beam's failure mode: losing a narrow optimal path at any one of ~16 steps.

**Pre-test** (about 1 h, no search).
- `scripts/36_ihes_opening_order_probe.py` put the incumbent's 3-move opening at median
  1,042.5 / 5,832 under E6 (~18th percentile) on 42 pids.
- Re-run it with the transformer on proven-optimal pids.
- If the median moves into the top ~2%, a top-2% sweep of 3-move openings over L22 costs about
  1/50 of a full refutation.

**Caveats.**
- A no-hit from a truncated sweep proves nothing. Log hit / none / timeout separately.
- The search campaign was stopped on 2026-09-12 and needs an explicit go-ahead.

---

## 6. Timeline (UTC)

| when | local 4090 | elsewhere |
|---|---|---|
| Sep 16 evening (T+0-6h) | data prep (section 3.2); port T51; 200-step smoke to measure rows/s; launch arm A detached, overnight | label-consistency pre-flight on CPU |
| Sep 17 | port Q-Bellman; fork T30 + E6-as-Q; QE6 smoke; gate arm-A checkpoints; s3 + s3-rw30 (~1 h); arm C | arm B if a second GPU is authorized; start the TPU patch |
| Sep 18 | GATE-54 on the s3 checkpoints, plus the blend and qv sweep; 2^20 subset; **promote or kill** | CPU-emulated TPU validation; opening-rank pre-test |
| Sep 19-21 | 2^20 campaign | Kaggle TPU campaign; nightly merge |
| Sep 22, by 18:00 | final merge + independent replay | submitting is your call; 4 h buffer before 22:00 |

---

## 7. Brainstorm, ranked

| # | idea | why it fits IHES | evidence | verdict |
|---|---|---|---|---|
| 1 | Warm Q-Bellman + d<=6 anchors (s3) | fixes the walk-label failure exactly where solves start | best scorer on 444; biggest lever on 555 | **use** |
| 2 | Sweep Bellman checkpoints and keep all of them | step count is not monotone | 555: 2k >> 20k; min over 6 checkpoints = 24/24 | **use** |
| 3 | Blend with a Q-native ResMLP | cheap second opinion | 444: -72, worth as much as the whole training chain | **use** (QE6 / E6-as-Q now, arm C later) |
| 4 | AZ value head + qv-consistency | the head helps training; qv is free at inference | tetraminx -4; 555 +8 | **train it; A/B at inference** |
| 5 | Long-walk Bellman states (`s3-rw30`) | test states behave like uniform random states | the counting bound matches the mean path | **A/B beside s3** |
| 6 | Canonical-walk sampler: forbid `x x x`, `x x^-1`, commuting `x y x^-1`; fix same-axis order | commuting axes make non-backtracking walks non-geodesic | tetraminx: 25.9% in-table mislabels, with no commuting moves | **pre-flight decides** (arm B) |
| 7 | Split ranking and level streams (the `k_max = 0.9M` rule) | rank pairs only up to p~19; deeper pivots train level only | `IDEAS_DETAILED_2026-09-04.md` section 2 | arm B option |
| 8 | Replace walk labels inside the table | pivots inside d<=6 get exact rows | `IDEAS_DETAILED_2026-09-04.md` section 3 | arm B option |
| 9 | 48-frame symmetry coverage | covers more of the 18 columns per sample | won on tetraminx; `tmx` lost on 444 | arm B, 4 random frames |
| 10 | Exact values for Bellman children found in the table | one hash lookup | small effect | optional |
| 11 | Model-ranked exact search | exact after k moves | E6 opening rank ~18th percentile | Route 2, after the pre-test |
| 12 | PDB lower bound (corners + centre-sum, H = A + B) as an input token | an exact structured signal the network cannot compute | as a beam *clamp* it cost +4 on 6 pids (untested as a feature) | after the deadline |
| 13 | Limited-horizon (2-step) Bellman | stronger propagation, at 324 forwards per state | on 555's idea list, untested | after the deadline |
| 14 | Expert iteration on beam solutions | more deep path states | close to Beam-AVI, which was rejected on tetraminx | skip |

**Skip, with measured reasons.**
- d7 anchors: negative on 444 and tetraminx, and there is no d7 table here.
- Path *labels*: 444 `s2`; tetraminx +3.07 moves/pid.
- Longer training and EMA.
- Capacity scale-up.
- Double-Q.
- Huber loss or reweighting.
- Beam-AVI.
- history relaxation without a matched run.
- Gating on loss, pair accuracy or top-1.

---

## 8. Traps

- `num_classes` is 72, not 6.
- T51 conjugates as P^-1.s.P; the IHES code uses P.s.P^-1. Keep `verify_symmetry` on.
- The anchor key in `bfs_d6_train.pt` is `depths`, not `distances`.
- Use chunk 4096. WDDM spills past VRAM silently (rule 31).
- Use `torch.compile` in the beam only with padding (rule 2). The IHES `khoruzhii_search.py`
  has no padding hook.
- `BlendedQ` silently disables `--qv-consistency` (synthesis section 5.5). Read the run header.
- Keep the anchor blob in its stored dtype.
- Rule 29: grep for `add_argument` before running any script with `--help`.
- Rule 28: run the matched control before quoting a delta. Byte-identical results mean the flag
  is not wired.
- The optimal-path probe labels are exact, but **never select a checkpoint with them**.

---

## 9. Decisions needed

1. **Compute.** Local + free Kaggle only, as in the Sep campaign? Or paid GCP (an A100 spot for
   arm B, a v6e for a wider campaign)?
2. **Route 2.** Go or no-go on model-ordered exact search, after the opening-rank pre-test?
3. **Submission.** A win would be our own model beating the community file on a pid, which is
   the condition you set for submitting the merged file.
