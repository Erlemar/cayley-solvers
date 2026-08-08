# Megaminx — handoff doc for the next Claude session

**Read this first.** Everything you need to pick up productively, in one file.
For deep history: `EXPERIMENTS.md` (ledger), `IDEAS.md` (backlog), `speed_optimizations.md`,
`multitask_training_ideas.md`, `tensorrt_gcp_plan.md`. For active work: `to_do_shortlist.md`.

---

## 1. Competition + score

- **Comp**: [CayleyPy Megaminx](https://www.kaggle.com/competitions/cayley-py-megaminx). Deadline **2026-08-31**, 16 teams, kudos prize.
- **State**: 120-element permutation, 24 generators (12 faces × {CW, CCW}), each face order-5. Optimal solver doesn't exist for full diameter — ML+search is the only viable approach.
- **Metric**: total moves across 1001 test puzzles, lower = better.
- **Test ordering**: `test.csv` is sorted by *random-walk LENGTH*, NOT by true scramble depth. pid 0 has a 72-move walk; pid 1000 has a 925-move walk. Bucket by `pid // 100` for difficulty stratification.
- **Kaggle account**: slug **`artgor`** (display name "Andrew Lukyanenko"), team displayed as "andlukyane". KAGGLE_API_TOKEN: `KGAT_630ac26efca89d28c5b2d496b238b71c`. All `kaggle kernels push` use `id: artgor/...`.

## 2. Score progression (latest first)

| date | source | score | notes |
|---|---|---|---|
| 2026-06-15 | working floor: exact relink + interior half-split repair (`half_split_rank201_400_b16k.csv`) | **73,441** | local/GCP verified 1001/1001; half-split saved -133 over `relink_v16_top200_bfsd6.csv` across 70 pids after top-200 + rank-201..400 slices. Not yet a submitted Kaggle score line. |
| 2026-05-18 | min-merge of our 75,961 base + new community v4 CSV (75,355 standalone; v3 78,196 → v4 75,355 strictly dominates v3); 290 v4 wins beat our base, 71 base wins kept (`merge_v14_plus_min_count_v4.csv`). POLICY EXCEPTION (user-authorized) | **75,200** | current submitted best |
| 2026-05-18 | min-merge of AZ v4 prod-1001 (79,606 standalone, 134h GCP L4) + our 77,877 + 76,251 community-best → 114 unique AZ v4 wins in mid buckets 4-7, -290 moves (`merge_v13_az_v4_plus_community.csv`). POLICY EXCEPTION (user-authorized) | 75,961 | superseded |
| 2026-05-12 | min-merge of 76,304 community-best + m_dd_v0_50ep_prod_1001 (84,132 standalone, 33.9h GCP L4) → 20 unique wins, 53 moves saved (`merge_v11_community_plus_m_dd_v0.csv`). POLICY EXCEPTION (user-authorized) | 76,251 | superseded |
| 2026-05-09 | min-merge of 78,029 (us) + 2 colleague CSVs (79,911 + 77,152) → 76,304, `merge_v9_with_77152.csv`. POLICY EXCEPTION (user-authorized) | 76,304 | superseded |
| 2026-05-09 | rescue: m_curr_v3 + m_pi_v2 top-200 (-18 from 78,047 → 78,029), `merge_v7_curr_v3_pi_v2_rescue.csv` | 78,029 | superseded |
| 2026-05-08 | session: -250 from 78,408 via path relink + m_pi_v0/v1 rescues + m_fr_v0/v2 (frontier replay + BFS-d6 boundary), `merge_v4_with_m_fr_v2.csv` | 78,158 | superseded |
| 2026-05-05 | community-pushed shareable-kernel runs (alexandervc + fedmug forks of `cayleypy-megaminx-beam-shareable`, B=1M with K=8 long-tail) merged with 79,522 base (`merge_v19b_plus_community.csv`); 324 community wins across all buckets | 78,408 | superseded |
| 2026-05-03 | TPU v19b partial (B=1M K=4 chunk=32768 pids 500-1000, 405 pids covered before 9h kill) merged with 79,946 base (`merge_tpu_v19b.csv`); 141 wins in buckets 5-8 | 79,522 | superseded |
| 2026-05-03 | TPU v19a-fix + GCP T1.6 v2 top-200 SA min-merge (`merge_tpu_v19a_plus_t16v2.csv`); 75/1001 pids replaced from rescue | 79,946 | superseded |
| 2026-05-03 | TPU v19a-fix partial (B=1M K=4 chunk=32768, 80% pairs done, pids 0-405 covered) merged with 80,602 base (`merge_tpu_v19a_fix.csv`) | 80,212 | superseded |
| 2026-05-02 | TPU v17 FULL (v17 partial + v17b completing pids 819-1000) merged with 80,739 base (`merge_tpu_v17_full.csv`) | 80,602 | superseded |
| 2026-05-02 | TPU v17 partial (K=8 beam=262k, 82% pairs done before 9h kill) merged with 81,357 base (`merge_tpu_v17.csv`) | 80,739 | superseded — broke 80K |
| 2026-05-02 | merge_tpuv16_t22_t16: TPU v16 ∪ T2.2 tail-resolve full-1001 ∪ T1.6 SA top-100 (`merge_tpuv16_t22_t16.csv`) | 81,357 | superseded |
| 2026-05-02 | merge_t22_t16_v1: T2.2 + T1.6 vs older 82,481 base (missed TPU v16 wins) | 81,516 | superseded by re-merge |
| 2026-05-01 | TPU v16 full-1001 (xmp.spawn + qshort + sym K=4 + beam 131k) merged with prior best (`merged_tpu_v16.csv`) | 82,225 | superseded |
| 2026-04-30 | merge_plus_sym8_top20: + K=8 sym + qshort + beam 524k on top 20 long-tail (`merge_plus_sym8_top20.csv`) | 82,481 | superseded |
| 2026-04-30 | merge_plus_sym4_top80: + K=4 sym + qshort + beam 524k rescue on top 80 long-tail of merge (`merge_plus_sym4_top80.csv`) | 82,646 | superseded |
| 2026-04-30 | merge of phase_b_plus198 + GCP m05+qshort+524k+TRT full-1001 | 83,362 | superseded |
| 2026-04-29 | phase_b_plus198: + sym-ensemble K=2 + qshort+524k rescue on top 50 (`phase_b_plus198.csv`) | 85,812 | superseded |
| 2026-04-29 | phase_b_plus148: + qshort+524k rescue on top 148 long-path pids (`phase_b_plus148.csv`) | 86,329 | superseded |
| 2026-04-27 | Phase B (m05 fresh on local 4090, beam 131k, all 1001) + beam-stack rescue of pid 490+920 | 88,195 (rank #3 at the time) | superseded |
| 2026-04-26 | Phase 1 (m07 GCP) + Phase 2 (m05 retry) + min vs pp_bfs6 | 95,682 | superseded |
| 2026-04-24 | m07+NISS strat-2 | 407,563 (#4) | first ML submission |
| 2026-04-24 | pp_bfs5_fallback (no model) | 415,521 (#8) | post-proc baseline |

**Top LB (snapshot 2026-04-27)**: #1 Kuznetsov 79,971; #2 DrozdovDan 81,946; #3 us 88,195; #4 Rokicki 93,606. **Goal: <70K. Stretch: ~60K.**

To go from 86K → 70K is **−19% moves**. Heuristic gain, not a speed gain. Speed wins only count if they unlock larger beams.

**Recipe ceiling discovered 2026-04-29**: every single-recipe Bellman variant
trained at 6M params (m17, m22 K=2, m26/m26b 12-13M, m27 family with BFS-d6
mixin, m28 Double Bellman, m29 n_back=4, m30 n_back=16, m31 rotation
augmentation, SWA m05) lands within strat-5 mean 88-97. m29 (88.98) is the
only sub-89 result; that gain is +1 solve / -0.42 mean. Recipe levers are
exhausted at this arch scale. Score-race headroom is on the beam side now
(sym-ensemble at inference, hard-tail rescue at bigger beams, multi-seed
merges).

## 3. Hardware fleet

- **Local RTX 4090 Laptop** (16 GB, Windows 11, Python 3.14, torch 2.11+cu128). Primary for training + small solves.
- **GCP `cayley-gpu`** (us-east1-b, L4 24 GB, sm_89, ~$0.70/hr running). PyTorch 2.9.1+cu129 system-wide. SSH playbook: `~/.claude/projects/.../memory/reference_gcp_cayley_vm.md`. Current IP changes on restart — fetch via `gcloud.cmd compute instances describe`.
- **Kaggle** (P100 Tesla, sm_60). 30h/week GPU quota, CPU unlimited. Paths: see `reference_kaggle_pipeline.md` memory. **GPU quota currently exhausted** (m26+m26b ate it 2026-04-27); CPU still open for cheap evals.

## 4. Layout (key paths)

```
cayley/
├── src/cayley/                      shared lib (model, search, post_process, etc.)
├── megaminx/
│   ├── src/megaminx/                Megaminx puzzle, mitm_solver, bfs_bytes, post_process
│   ├── scripts/                     03_solve.py (production), 02_train.py, 05_bellman_refine.py, ...
│   ├── beam_lab/                    benchmarking sandbox: beam_search.py, beam_search_qshort.py,
│   │                                run_benchmark.py, export_tensorrt.py
│   ├── configs/                     m0N_<slug>.yaml per training run
│   ├── models/m0N_<slug>/epoch_NNNN.pt + _training.log
│   ├── data/                        puzzle_info.json, test.csv, pp_bfs6_fallback.csv,
│   │                                bfs_bytes_d{5,6}.pkl (the .pkl files are gitignored)
│   ├── submissions/                 mNN_<beam>_<strategy>.csv + _solve.log
│   ├── EXPERIMENTS.md               canonical ledger (chronological journal)
│   ├── IDEAS.md                     prioritized backlog
│   ├── speed_optimizations.md       speed work tracker
│   ├── multitask_training_ideas.md  multi-task / training ideas
│   ├── tensorrt_gcp_plan.md         TRT execution plan
│   └── to_do_shortlist.md           active to-do (remove items as done)
└── .venv/Scripts/python.exe         ALWAYS use this (Windows venv). Never `python` / `python3`.
```

## 4a. Public AZ4 training kernel (shipped 2026-07-04)

Teammate-facing modular training notebook (ogurtsov-style config-first) that reproduces
AZ v4 end-to-end: **kernel `artgor/cayleypy-az4-trainer-megaminx`** + **dataset
`artgor/megaminx-az4-training-assets`** (~2.4GB: all 5 stage warm-start checkpoints,
bfs_d6_train.pt, frontier_states.pt, public `submission_73731.csv` + prebuilt policy
dataset). Sources: `megaminx/kaggle_notebooks/az4_train_shareable/` (`cells/*` ->
`build_ipynb.py` -> `kaggle kernels push`; local smoke via `smoke_run.py`, passed
2026-07-04). Stage names: pretrain/curriculum/bellman/bellman_dd/az; default = az-only
warm from m_dd_v0 ep49, epochs 30 with lr_t_max=200 (schedule parity at ep24). POLICY:
only PUBLIC solutions CSVs go in the dataset (user decision 2026-07-04) — our private
merges (73,614 / 71,362) stay local. **Run-verified on Kaggle T4 (33 min): ep24 trajectory
matches the original, canary identical to reference, cayleypy bench 3/3.** NEW Kaggle
gotchas encoded there: dataset mounts moved to `/kaggle/input/datasets/<owner>/<slug>`,
and the torch 2.10 image DROPPED P100 — push kernels with `--accelerator NvidiaTeslaT4`
(invalid names silently ignored). Full spec + deviations: `az4_train_shareable/PLAN.md`;
memory `az4-trainer-kernel-shipped`.

## 5. What works (currently DEPLOYED)

| component | what / where | state |
|---|---|---|
| **m05** (V model, 6.0M params) | `models/m05_bellman_warm/epoch_0499.pt`. Bellman warmstart from m07, 500 ep, lr 5e-4. Final loss 0.094. Strat-5: 50/51 solves, mean 89.4. | production teacher |
| **m23** (Q-shortlister, 12.4M params) | `models/m23_q_shortlister/epoch_0499.pt`. Q-head output_dim=24. Trained MSE+KL against m05. Recall=100% at α=2. | production student; 4.4× wall speedup at beam 131k |
| **m23_v2 sym-aware** (12.4M params) | `models/m23_v2_sym_aware/epoch_0499.pt`. Same arch as m23 but trained with rotation augmentation (R*s*R_inv before teacher Q-target, prob 0.5). Recall: α=1 98.7%, α=2 100%. Strat-5 m05+m23_v2+sym4+qshort: 51/51 / mean 88.41. | use INSTEAD of m23 when paired with --sym-ensemble |
| **--sym-ensemble** (`scripts/03_solve.py`) | K rotations from `data/rotations.npy` (360 elements, group A_5 x C_6). Per pid: K rotations including identity, transform state, full beam, translate path back via R_inv·g·R conjugation. Take min. **Strat-5 m05+sym4 (no qshort): 51/51 / mean 88.20** — best single result. | **first inference-side mechanism to break the m05 cluster ceiling.** K=4 is sweet spot (K=8 gives diminishing returns). |
| **Beam-stack rescue** | `beam_lab/beam_search_stack.py` + `scripts/12_beam_stack_rescue.py`. | rescued pid 490 (407→126) + pid 920 (758→137) for the 88,195 submission. Helps catastrophic failures (NOT already-converged-suboptimal pids). |
| **`--compile` (mode=reduce-overhead) + `pad_to_batch_size=True`** | `beam_lab/beam_search.py`. Inductor fusion + CUDAGraphs. | -27% wall, paths IDENTICAL. Always-on. |
| **TensorRT FP16 engine on GCP** | `beam_lab/export_tensorrt.py`. fp16 sm_89-specific. | DEPLOYED. 12-puzzle L4 sweep: TRT@165k 1043 paths beats compile@131k 1049 by -6 at +1% wall (noise). Used in production for full-1001 GCP solves. |
| **BFS-d6 window post-processing** | `megaminx/src/megaminx/post_process.py: full_post_process(...)`. | -0.2% on top of BFS-d5; SATURATED at d=6. d=7 needs 250M states (infeasible). |
| **Macro-augmented beam (PROTOTYPE)** | `KhoruzhiiSolver(macros=[(perm, word_list), ...])` parameter. Action_cost tensor + cost-aware V adjustment. Path reconstruction expands macro actions to gen words. | Mechanism shipped, tested with 6/30 brute-force d=4 commutators on 5 hard pids: **net +9 to +41 moves (HURT).** Brute-force commutators don't pay; needs curated speedcubing macros. |
| **`scripts/03_solve.py`** | Production solver. Flags: `--niss`, `--mitm`, `--qshort-student`, `--qshort-alpha`, `--qshort-internal-batch-size`, `--tensorrt-engine`, `--ensemble-seeds`, **`--sym-ensemble K`**, **`--sym-rotations PATH`**, **`--sym-seed`**, `--bfs-table`, `--stratified K`, `--strat-seed`, `--pids`, `--resume`. | THE entrypoint for any submittable solve. |

## 6. What didn't work (DON'T RETRY without new info)

| idea | result | why |
|---|---|---|
| More Bellman rounds (m17 r2 from m05) | REJECTED | Training loss flat from epoch 0; m05 already at fixed point. |
| Bigger arch + Bellman (m26 12M, m26b 13M) | REJECTED 2026-04-28 | Both converge to same plateau as m05's 6M (loss ~0.094-0.10). m26_bell ep499 strat-5: 51/51 / mean 91.2 vs m05's 50/51 / 89.4 (+1 solve, +1.8 path = WORSE). **Capacity scaling at the Bellman signal is fully exhausted.** |
| Walk-depth top1=0.998 → strat-5 win | REJECTED ("m12 trap") | High top1 in cheap-eval doesn't translate to beam quality. m12, m26_bell ep499 both had top1=0.998 with mediocre solves. |
| Manual CUDA Graphs alone | REJECTED 2026-04-27 | -34% slower than `--compile` on 3-puzzle. Inductor's kernel fusion is the real win, not graph capture. |
| beam_decay (geometric narrowing) | REJECTED 2026-04-27 | -33% wall on 3-puzzle but +2-3.5% paths; pid 492 +7 moves. Quality regression. |
| stochastic beam (Gumbel-top-k) | NOT A SPEED LEVER | Same compute per step, just different which-states-survive. Adds randomness. Never tested at depth. |
| Async pipeline (skip-syncs, deferred solved-check) | REJECTED 2026-04-27 | Skip-syncs: -0.7% (within noise). Deferred check: +0.7% wall AND +0.8% paths. The 96% model_s ceiling caps async wins at ~4%; in practice <1%. |
| Adaptive beam escalation (16k→65k→131k early-exit) | REJECTED 2026-04-27 | -78% wall but +13% paths. Quality loss too large. |
| `internal_batch_size > 16384` | REJECTED 2026-04-27 | Tied at 32k=65k on 4090/L4 (compute-bound). 16k is the right default. |
| Multi-puzzle batched beams (K=4 lockstep) | REJECTED 2026-04-26 | 3.7× SLOWER on L4 (compute-bound). Might help on H100/A100. |
| cayleypy `iterated` mode + `history_depth` | REJECTED 2026-04-27 | +124% wall AND +10 paths. Russian commenter explicitly warned "this slows down". m05's sharp Bellman doesn't need non-backtracking enforcement. |
| MITM via cayleypy `simple` + `hashed_neighbourhood` | REJECTED 2026-04-27 | +17% wall vs ours, paths similar. m05 already navigates d≤6 shell as a side-effect. |
| Distillation to smaller V model (m06 from m07) | REJECTED 2026-04-25 | 10× wall reduction confirmed but solve rate 20/51 (lost ordering). |
| NISS on m05 | REJECTED 2026-04-26 | Doubled wall, no clear win. m05's sharper heuristic doesn't benefit from path-diversity that NISS provides. (`--niss` flag exists; not used in production.) |
| `k_max > 80` in random walks | REJECTED 2026-04-24 | No improvement; effective Megaminx diameter < 80 for the training distribution. |
| Transformer at n=120 (m18) | REJECTED 2026-04-26 | Did converge but 225s/epoch on Kaggle P100 → 12h limit hits before reaching MLP-equivalent. Not infeasible in principle, just slower-per-wall. |
| Muon optimizer (m09-m12) | REJECTED 2026-04-25 | Plateau at MSE ~64-67, same as AdamW. Lowest training MSE (m12 ep99=59) had WORST solve rate (9/51) — `m12 trap`. |
| `m21` learn-to-hit-shell (clamp(walk_depth-6, 0)) | REJECTED 2026-04-26 | Path-sum 1197 vs m05's 1053 (-14%). Shifted target without Bellman bootstrap loses sharpness. |
| **m22 K=2 lookahead** | REJECTED 2026-04-28 | Strat-5 ~91 (cluster). K-step adds compute without breaking ceiling. |
| **m27 50% BFS-d6 mixin in pretraining** | REJECTED 2026-04-28 | Strat-5 51/51 / 91.49. Mixin in pretraining doesn't break cluster. |
| m27b 25% / m27c 10% BFS-d6 mixin | REJECTED 2026-04-28 | Lower mixin doesn't help either. |
| **m28 Double Bellman** (van Hasselt 2010) | REJECTED 2026-04-28 | Bias decorrelation doesn't break cluster — strat-5 ~91. |
| **SWA m05 ckpts 399/449/499** | REJECTED 2026-04-28 | Strat-5 mean 96.75 (REGRESSES). Late-cycle ckpts diverged enough that averaging blurs the heuristic. |
| **m30 n_back=16** | REJECTED 2026-04-28 | 26% lower training loss but worse strat-5 (89.69). m12-trap; over-narrows walks. |
| **m31 rotation augmentation in Bellman training** | REJECTED 2026-04-29 | Strat-5 50/51 / 95.76 (REGRESSES + lost 1 solve). At 6M params, augmentation across 360 orbit-equivalents dilutes signal. **Hypothesis falsified**: orbit coverage NOT the binding constraint on cluster. |
| **m32 target_update_every=5** | REJECTED 2026-04-30 | Strat-5 51/51 / 94.65 (+5.25 mean over m05). Faster target refresh → over-fits to short-term gradient noise. m05's value of 10 is calibrated. |
| **NISS+qshort (m05+m23+--niss)** | REJECTED 2026-04-30 | Strat-5 51/51 / 93.35 (REGRESSES vs sym4's 88.20). m23 was distilled on forward-state V; recall drops on inverted states. |
| **m29+qshort+524k full-1001 pairing** | ABORTED 2026-04-30 (at pid=99) | m23 distilled from m05 NOT m29 — fb-wins on 58/100 early pids. **Don't pair m23 with non-m05 teacher** without re-training m23 against THAT teacher. |
| **K=8 sym-ensemble vs K=4** (saturation test) | DIMINISHING RETURNS confirmed | K=4 saves 10/pid; K=8 saves 8.25/pid additional at 2x wall. K=4 is sweet spot. K=8 only useful for marginal final passes. |
| **Commutator window-replacement post-processing** | REJECTED 2026-04-30 | Built 37K-entry library (depths 4-8). 29K perms NEW beyond BFS-d6. Tested W=7,8 on 82,481: **0 matches**. Beam paths' structured perms don't fall on commutator atoms. Same as IHES finding. |
| **Macro-augmented beam with brute-force d=4 commutators** | REJECTED 2026-04-30 | Mechanism shipped (works), but 6-30 d=4 commutators as macros: net +9 to +41 moves on 5 hard pids. V isn't macro-aware; brute-force commutators dilute candidate pool. T1.1 needs curated speedcubing macros (multi-day scrape). |
| **m_v11 (11.8M two-stage)** | REJECTED 2026-05-19 | Trunk (3072,768)+3rb scale-up; two-stage pipeline (50ep pretrain → 200ep Bellman λ_pdb=0 lr 2e-4 → 25ep AZ fine-tune on 75,200 dataset). V-cal looked great end-to-end (V(V0)=0.012, V(d=1)=1.00 at final ckpt). 10-pid bench gave **10/10 / 991** (vs Stage 1b 9/10 / 910) — looked like a win. BUT strat-51 stuck at 16/20 model solves for 1h+ (vs AZ v4 51/51 in ~75 min same recipe). 995-998 production-recipe head-to-head: m_v11 **+43% moves/pid worse than AZ v4** (96.25 vs 67.5 avg). **Definitive**: bigger trunks regress on this puzzle regardless of recipe — extends Rule 14 (20.5M) down to 11.8M. Information-bound ceiling at 6M holds. Don't retry trunk scale-ups without a fundamentally different signal source. |

## 7. Critical gotchas (read these — see also `megaminx_gotchas.md` memory)

1. **`run_benchmark.py` does NOT save move sequences** — only stats. Cost: 14h GCP burn for an unsubmittable result on 2026-04-28. **Always use `scripts/03_solve.py` for any potentially-submittable solve.**
2. **`sed -i` on Windows MINGW silently truncates files to 0 bytes.** Use the `Edit` tool, or `sed 'pattern' f > f.new && mv f.new f`.
3. **MSE plateaus at ~64 (k_max=80) regardless of arch capacity.** Don't retry "just scale up the model". Three confirmed plateaus: m02 (2.4M), m07 (6M), m26 (12M), m26b (13M).
4. **The Bellman fixed point is data-bound, not arch-bound.** m05/m17/m26/m26b all converge to ~0.094-0.10. More params at the same target are a no-op.
5. **`cheap-eval top1=0.998` is not a reliable proxy for solve rate.** Always strat-5 before declaring a model better.
6. **`torch.compile` on beam search needs `pad_to_batch_size=True`** to prevent recompile thrash. Without padding: 5.8× slowdown (recompile loops). With padding (in `beam_lab/beam_search.py`): -27% wall.
7. **Reuse one `CayleyGraph` per session** — fresh graphs have different random hash vectors and corrupt cross-call state tracking.
8. **`--bfs-table return_all_hashes=True`** is NOT default in cayleypy 0.1; without it MITM silently degrades to plain beam.
9. **Kaggle GPU quota is 30h/week** — easy to blow with one big training run. Plan budgets, use early-stopping.
10. **Kaggle `kernels output` returns nothing while RUNNING** — only `status` is queryable. Use `_Tee` trick to write `/kaggle/working/run.log` so post-completion logs are recoverable.

## 8. Where we ended (2026-06-13 — v6e-8 256M/512M TPU beam mechanics; 2026-06-12 own-research program still queued)

### THIS SESSION (2026-06-13) — pid 991 wide-beam TPU run + 512M fit work

**256M/8-chip full run completed, identity-only, no score win.** On TPU VM
`mm-v6e8` (`gen-lang-client-0977634337`, `us-east5-a`), the launched command was
`gcp_beam_v6e.py --b-global 268435456 --start-pid 991 --end-pid 992
--k-sym 1 --num-steps 90 --student-alpha 24 --receive-alpha 1.25
--alpha-req 1.25 --parent-chunk 524288 --internal-bs 131072 --nbhd-radius 4`.
It was a real 256Mi global beam over 8 chips (`b_local=33,554,432`) and found a
verified path of **74** moves (`exact hit step 73`, `wall=53764s`, output
`/mnt/data/out/pid991_256m_full_a24_recv1p25.json`). It did **not** run
symmetries: `k_sym=1`, `rot_idxs=[0]`, `rot_idx=0`; the `_sym` in the model name
is only the loaded Q/student artifact, not a search ensemble. The best public
community CSV we used as the floor reference has pid 991 at **69**, so this is
valid engineering data, not a merge candidate.

**512M/8 now fits and runs on v6e-8, but the tested alpha4 qshort setting should
NOT be expanded to a full 90-step record run.** The qshort kernel was widened to
26/3/5 uint64 backpointers (parent/rank/move), removing the old per-chip
`b_local <= 2^24` ceiling and supporting `b_local=67,108,864 = 512M/8`.
Additional fit patches removed the `(B_local,8)` request-position matrix, tiled
response materialization, reused the donated frontier as the scatter base instead
of allocating an 8 GiB zero frontier, and built the padded seed on host NumPy
before `make_array_from_callback` so executable loading did not collide with an
extra 8 GiB device seed. Remote and local backpointer tests/py_compile passed.

512M smoke command family:

```bash
gcp_beam_v6e.py --b-global 536870912 --start-pid 991 --end-pid 992 \
    --k-sym 1 --student-alpha 4 --receive-alpha 1.03125 --alpha-req 1.03125 \
    --parent-chunk 262144 --internal-bs 131072 --nbhd-radius 4
```

Successful smoke (`--num-steps 3`): compile `143.9s`; step 1 `2515.6s` device,
step 2 `2509.3s`; `min_v=28.625 -> 27.875`; output
`/mnt/data/out/pid991_512m_smoke3_a4_recv1p03125_hostseed.json`. Tile4 response
materialization (`REQ_TILE=4*MAT_CHUNK`) was neutral (`2512.9s` for step 1), so
the remaining bottleneck is the broad selection/materialization/HBM structure,
not the number of response-all-to-alls.

**Next gate before any full 512M run:** run 512M short smokes at
`student-alpha=8` and/or `12`, plus optionally alpha24 V-all for a quality/speed
anchor. Resize scratch first for a full attempt: 512M x 90 steps x 8 ranks x
uint64 backptr is ~386.5 GB decimal before logs/slack, while the current 400GB
disk is marginal. Use >=600-800GB. Only launch a full run if early `min_v` looks
competitive with the 256M alpha24 trace (step 2 there was ~12.75; alpha4 was
27.875).

### THIS SESSION (2026-06-12) — community chat distilled + two original mechanisms built and measured

**Context**: full read of the CayleyPy chat export (3,736 msgs, May 24-Jun 11;
searchable dump `chat_export_dump.txt`). The outside world moved: Ivan Litvak's
C++/CUDA MultiGPUBeamSearch runs 86M beams on Kaggle 2xT4 / 700M on 8xA100;
Vlad Kuznetsov's 24-output Q-models feed it (his teacher beats AZ v4 86.05 vs
92.2 @ 2^16 on pids 900-1000); community merged floor ~72,161. Catch-up plan +
Rule-9 reconciliations (NISS = the chat's "dual" trick; m06-vs-modern-Q;
m_sym_v0-tie vs Vlad's recipe): `chat_brainstorm_2026-06-12.md`. Original
(non-replication) program: `own_research_directions_2026-06-12.md` (1A-1D
search-aware training, 2A-2B search-serving models, 3A-3C symmetry economics).

**3C sym-pooled beam — built, validated through 4 escalating A/Bs (EXPERIMENTS.md
2026-06-12 x2)**. One beam of width K*B seeded with all K rotated (+ inverse)
copies; per-root widths from a frame-bias-free progress softmax (per-root V-descent
since its own start, EMA-smoothed); within-root selection unchanged; K=1 ==
production solver bit-exactly. v0 (global top-k over raw V across frames) FAILED
— V is deliberately non-invariant across frames; the pool defected from the
leading root at the finish. Results at equal candidate budget (AZ v4 V fp32,
4090): K=2 smoke -18 + rescued a both-rotations-failed pid; K=4 @16k total -9
(991: 97 vs 107); pool+inverse 8 roots @16k total: 504 vs seq 525 (991: 92);
**production-width gate (24 tail pids, 16k/rot vs 8 roots @65k): pool 2201 vs
seq 2217 = -16, 8W/8T/8L, wall -7 pct, headline 991: 101 -> 82 (-19)**. All
losses one bounded mechanism (width committed before path-length info exists;
max +5). Inverse frames won 6/8 — the sym x inverse grid revives NISS at zero
extra wall (Rule 11 dropped it for 2x wall, not uselessness).

**1B certified stagnation anchors — pipeline shipped, finding logged.**
Harvester (`scripts/89_harvest_stagnation_anchors.py`) certifies beam states
claiming V < 6.5 against `bfs_bytes_d6` (exact d <= 6 / PROVEN d >= 7).
**Finding: AZ v4 V has NO Fedor-style V~2-at-d~7 stall bug (V in [0,3): 0 pct
provably wrong — the V0/d1 anchor fix likely covers it). Its real failure is an
OPTIMISM BAND: V in [4,5) -> 59 pct provably d>=7, V in [5,6) -> 99 pct (mean
certified gap >= 1.36)** — the beam's endgame ranking zone. Dataset
`data/stagnation_anchors_v0.pt` (5,177 exact + 31,150 certified-LB anchors —
a NEW label type). Trainer hook wired into `src/cayley/bellman.py` (flag-gated:
`stag_anchor_path`, `n_anchor_stag` exact-MSE rows, `n_anchor_stag_lb` +
`lambda_stag_lb` one-sided hinge), smoke-validated (`configs/stag_anchor_smoke.yaml`).

### Files created this session

- `chat_brainstorm_2026-06-12.md`, `own_research_directions_2026-06-12.md`,
  `chat_export_dump.txt` (greppable chat transcript)
- `beam_lab/beam_search_sympooled.py` (SymPooledSolver), `scripts/88_sympooled_ab.py`
  (A/B driver + K=1 self-test), `scripts/89_harvest_stagnation_anchors.py`
- `src/cayley/bellman.py` stagnation-anchor mixin (default-off flags)
- `configs/stag_anchor_smoke.yaml`, `data/stagnation_anchors_v0.pt`
- `results/sympooled_{smoke_k2,k4_hard,k4_hard_inv,prod_tail}.json` (occupancy traces inside)

### Open at end of session

1. **3C deploy queue**: (a) inverse-frame axis in TPU notebooks — zero kernel
   surgery (feed R*inv(s)*R^-1 starts, invert path notebook-side); (b)
   beam_lab pooled+inverse rescue passes on current-best's stubborn pids
   (ready now, local/GCP); (c) JAX-kernel port of the allocator for 48M scale
   — GO after a+b (rotation tag needs 3 backptr bits; 24/3/5 layout is full at
   96M — re-layout, mind the 2026-06-08 overflow).
2. **1B refine run**: Rule-13 recipe check first; re-harvest with the model
   being refined (selection is model-specific; labels are certified facts).
   Gates: calibration table improves in [4,6.5) + d~20 variance canary +
   10-pid bench + strat-51.
3. **Tier-0 score**: export CSVs min-merge = **73,984** (-1,216 vs submitted
   75,200; driver = Liuda's `d_submission_74116.csv`, in the Telegram export
   files) — awaiting user authorization per the policy-exception process.
4. Own-research update: 1A cross-width expert iteration is DONE as a diagnostic
   and REJECTED as a scalar-delta scorer (labels learn, solves regress). Still
   queued: 3A orbit-mean distillation economics A/B, 2A/3B ensemble heads,
   2B uncertainty head, 1C kill-gated REINFORCE, 1D learned beam schedule.

## 8a. Prior session (2026-05-25 — §13 batch concluded: PHS validated-but-marginal; rank/sym/frontier-regret neutral)

**Submitted score: 75,200** (unchanged). **Standalone best: 77,086** (unchanged this session).

### THIS SESSION (2026-05-24/25) — strategy-doc §13 batch (6 ideas → 4 themes)

Developed all 6 ideas from `megaminx_architecture_and_path_shortening_strategy.md` §13 into
concrete specs in the doc, then implemented the lead ones. No submission change. Full ledger
in EXPERIMENTS.md; memories [[phs-cumulative-validated-marginal]] + [[repr-upgrade-bundle-rejected]].

- **PHS cumulative path scoring (§13.1) — VALIDATED, deploy-MARGINAL** (the one §13 idea that
  works). `--phs-cumulative` in `03_solve.py` + `self._phs_cum` accumulator in
  `beam_lab/beam_search.py` + `beam_search_qshort.py` (6/6 tests, `tests/test_phs_cumulative.py`).
  score = V(child) + w_p·Σ(−log π) over the PATH (vs the regressed memoryless local penalty,
  AZ v4 V+π λ=0.05 → −156). rand50 (buckets 1-8): **−67/50 STANDALONE** over its own w=0 arm,
  difficulty-monotonic (800-899 −4.6/pid). BUT vs the merged best (merge_v12) only **−13/50
  (3 wins)** — redundant with existing diversity; ~−200 extrapolated for ~1-2 days GCP = not
  worth a deploy run. Safe w≈0.03; collapses at w≥0.2. CAVEAT: AZ v4 π is OOD on 3/4 sym
  rotations (not rot-aug).
- **Symmetry consistency loss (§13.4) — TIE, λ_sym exonerated.** `m_sym_v0` (λ_sym=0.1 on
  m_dd_v0, `configs/m_sym_v0.yaml`): variance-SAFE (d20 std flat 2.65→2.64), beam-neutral
  (model_avg ≈88.1 ≈ m_dd_v0 89.4). λ_sym is NOT the m_repr_v0 variance culprit (joins λ_rank
  ⇒ narrows to 20% solver-trace ± λ_sat).
- **Frontier-regret (§13.2) — labels benign, explains the rank/sym ties.** Harvest
  (`scripts/84_harvest_frontier_regret.py`): V "misranks" 56.9% of verified-path steps, but
  these are benign alternative-optima (beam trusts V, solves 50/51 → can't be real mistakes).
  Dataset `data/frontier_regret_triples.pt` (noisy, kept). Macro-mining (§13.3) NOT started.

**Takeaway:** V's child-ordering is already near-optimal at 6M; its disagreements with verified
paths are benign alt-optima. **The road to <70K is pure-inference (sym-ensemble scaling,
multi-seed, rescue, merges), not more V/scoring training.** GCP VM stopped.

### Bridge compression (2026-05-22/24) — shipped end-to-end, ceiling found

Full story in [[bridge-compression-findings]] memory. Quick read:
- **Mechanism shipped:** Phase 1 (V-trajectory + Hamming + mixed grain + iterate +
  Hamming-similar + macro cache + harvest) in `81_bridge_compression.py`. Phase 2
  (sym + NISS + qshort; qshort dropped for off-distribution regression) via
  `bridge_solver.py`. TPU port: `82_generate_residuals_for_tpu.py` →
  kernel `artgor/cayleypy-megaminx-bridge-residuals-b1m-shareable` (B=1M, K_SYM=4,
  NISS=on) → `83_splice_tpu_residuals.py`.
- **Works on loose paths.** Our 77,214 → 77,086 (−128 across 112 distinct top-long
  pids, 76 wins / ~2200 attempts).
- **Fails on tightest community paths.** TPU on 4-pid composite of best per-pid
  community paths (997=73, 998=71, 999=72, 1000=70 from JAX 720-sym kernel) at
  B=1M + sym4 + NISS = **0 wins / 15 attempts.** V saturation past d≈30 makes
  the predicted-save score noise on deep residuals — V is the ceiling, not beam.
- **Pid 1000 specifically:** tried beam 65k local, 256k GCP, 131k local + sym4
  + NISS, 65k from community 71, 1M TPU + sym4 + NISS from JAX 70-move — every
  single attempt: 0 wins. Pid 1000 is genuinely at the V's bridging ceiling.
- **Don't pursue further** without a fundamentally different mechanism (bridge
  model `D(s,t)` per §3.5, or qshort retrained on residual-distribution states).

### THIS SESSION (2026-05-24) — representation-upgraded ResMLP + loss bundle + child-rank isolation

### THIS SESSION (2026-05-24) — representation-upgraded ResMLP + loss bundle + child-rank isolation

User asked to pursue doc §3.1 "Representation-upgraded ResMLP" with complementary bundle
ideas, then (after the bundle failed) isolate the child-rank loss. **Both rejected; no
deployable model.** Full story in [[repr-upgrade-bundle-rejected]] memory + EXPERIMENTS.md.

**m_repr_v0 (capacity-preserving repr upgrade + aggressive bundle): REJECTED.**
- Built `encoding="features"` (`src/cayley/model.py`): per-slot SUM-pooled fusion of
  gated channels (inv_state/face/lpos/piece/ori/scalar), all α init=0 ⇒ bitwise-identical
  to `encoding="embedding"` at init; in_dim stays 1920 (NO capacity inflation — the fix
  vs the rejected state_inv concat). +`megaminx/src/megaminx/piece_features.py` (NEW),
  bundle knobs in `bellman.py` (λ_sym consistency, λ_sat saturation, λ_rank child-rank,
  per_depth_diagnostic), `megaminx/scripts/v_canary.py` (NEW).
- Warmstart from m_dd_v0 via bridge ckpt (features@α=0 ≡ embedding) — fixes the
  saturate-from-scratch problem (a fresh-RW-pretrain warmstart can't saturate in 50ep;
  m_dd_v0 only did because it warmstarted from already-saturated m_curr_v3).
- **Finding 1: repr channels INERT** — all α→0, forcing α=0 changes nothing. Working V
  already learns the bijection implicitly. doc §3.1 now **0/2** (state_inv + features).
- **Finding 2: bundle TOXIC** — tripled mid-depth V variance (d=20 std 1.7→5.3). Mean
  stayed saturated (V@d=80=30, canary GREEN) so it LOOKED fine, but per-state V scatter
  killed beam ranking → strat-51 single-pass 2/20 (Rule 21 stall). Catastrophic.

**THE LESSON: saturation-MEAN canary is necessary but NOT sufficient. V VARIANCE at
d≈20 is the binding beam-quality proxy** (m_dd_v0 ≈1.7; >~2.5 ⇒ expect beam trouble).
`v_canary.py` prints std — weight it. Generalizes Rule 21 with a cheap pre-strat-51 check.

**m_rank_v0 (child-rank CE isolated, λ_rank=0.1): NEUTRAL → effectively rejected.**
- Isolated the one bundle term with independent appeal (doc §4.2) on m_dd_v0 (embedding,
  no repr channels, no other bundle terms). Control arm (λ_rank=0) confirmed re-fine-tune
  is a no-op. Variance-SAFE (V identical to m_dd_v0). Child-choice differs ~7% of states.
- **Strat-51 sp beam-65k: 51/51 by model, mean 89.9 vs m_dd_v0 89.4 = TIE (within ±2).**
- Child-rank EXONERATED as the m_repr_v0 variance culprit (→ it's sym/sat/solver-trace,
  prime suspect solver-trace). But no beam headroom on a working V (6M ceiling). Don't pursue.

**Code state:** all new code is clean + tested + inert-by-default (bundle knobs default
off, `encoding="features"` opt-in) — existing recipes unaffected. GCP VM STOPPED.
Rejected checkpoints (~400MB: m_repr_v0, m_repr_v0_pretrain, m_rank_v0, m_rank_ctrl +
GCP copy) safe to delete. Eval CSVs in submissions/ are not submittable improvements.

**Next-session priorities (untested high-EV doc ideas, per the §9 queue + the strategy doc):**
1. Bridge model `D(s,t)` (§3.5) — lift shipped bridge-compression beyond −69; attacks why
   cross-relink failed (V saturates wrong on high-Hamming residuals).
2. Multi-teacher union qshort (§2.4) — composes with AZ v4 + sym4 production stack.
3. Suffix specialist (§2.9) — narrow dist, good labels, low risk.
(Optional forensic: isolate which of sym/sat/solver-trace caused the m_repr_v0 variance.)

---

## 8b. Prior session (2026-05-22/23 — bridge compression + state_inv rejected)

**Submitted score: 75,200** (unchanged — see prior section for community-merge details).
**Standalone best: 77,145** (was 77,214; saved -69 via bridge compression on top-50 longest pids).

### THIS SESSION (2026-05-22/23) — bridge compression infra + V-trajectory + cross-relink dead end + state_inv rejected

User asked to pursue two ideas from `megaminx_architecture_and_path_shortening_strategy.md`:
neural bridge compression (doc §2.1) and representation-upgraded ResMLP with inv_state (doc §3.1).

**state_inv encoding REJECTED** (see [[inv-state-encoding-rejected]] memory):
- Shipped `encoding="state_inv"` in `src/cayley/model.py` — concatenates embeddings of
  state[i] and inv_state[i] (slot containing sticker i, computed via scatter).
- m_inv_v0 = 9.98M params (vs 6M baseline). Two-stage train per Rule 18: 50ep RW pretrain
  → 50ep Bellman refine with the m_dd_v0 recipe.
- **Near-solved calibration excellent**: V(V0)=0.020, V@d=1=1.01, undershoot 396/8000
  (better than all baselines).
- **High-depth V doesn't saturate** — V@d=80=38 vs working baselines saturating ~29.
  Same Rule 23 failure pattern as GraphTransformer V.
- **Strat-51 single-pass beam 65k: mean 109 vs m_dd_v0 baseline 89.4 = +22% worse**.
- **Rule 23 extends to ResMLP family at 10M params, not just GT.** The extra capacity
  from the inv_state channel gets used to encode walk-depth past the true diameter
  instead of saturating. Capacity beyond the cluster ceiling actively hurts.

**Bridge compression SHIPPED + working modestly**:
- Math: residual `X = inv(S_j)[S_i]` produces a state that, when solved, gives a bridge
  B with `apply_path(S_i, B) = S_j`. Verified via `tests/test_megaminx_bridge.py` (5/5).
- Driver: `megaminx/scripts/81_bridge_compression.py`. Loads AZ v4 V eager bf16 +
  KhoruzhiiSolver in-process, iterates windows on top-N longest pids.
- **V-trajectory window selection** (the binding upgrade over uniform stride): rank
  positions by predicted-save `window_len - V(residual)`. Doubles win rate (1.3% → 2.2%
  on 75,200 base; 4.2% on 77,214 base).
- Results:
  - On 75,200 community base: 75,200 → 75,162 (−38 moves, top-50). Not submittable per
    no-community-merge policy. Artifact: `bridge_top50_vtraj.csv`.
  - On 77,214 standalone (our work): 77,214 → 77,145 (−69 moves, 41 wins / 987 attempts,
    4.2% rate). Submittable. Artifact: `bridge_our_top50.csv`.
- Win pattern: window 60 highest avg saving (1.92 moves/win), pid 104 single biggest win
  (-6 via window-40 bridge from 40 → 34). Front-of-path detours are common.
- Currently running: top 100 on 77,145 base (~5h ETA).

**Cross-solution relinking ATTEMPTED, NEGATIVE RESULT**:
- Script: `megaminx/scripts/82_cross_relinking.py`. For each pid with multiple paths
  across CSVs (916 pids have ≥3 paths), try `A[:i] + bridge + B[j:]`.
- Smoke on top 10 long pids with 5 OWN CSVs: **0 wins / 80 actual solves**.
- Root cause: V saturates at ~30 for deep states. Cross-path mid-states are typically
  Hamming 35-66 apart (deep residuals). V-score predicts ~30 for everything, producing
  false positives. Solver max-steps cap is reached without finding a bridge.
- Math is also unfavorable for short paths: with len(best)=87, i=20, Hamming=35:
  `saved = 87 - 20 - 35 - len(B[j:]) - 1`. Often negative.
- **Verdict: relinking on own-only-data isn't productive for the 80+-move long pids.**
  Could work for medium-length pids where multiple paths might converge more, but
  unverified.

### Files created this session

```
src/cayley/model.py                                    encoding="state_inv" added
megaminx/configs/m_inv_v0_pretrain.yaml, m_inv_v0.yaml two-stage state_inv configs
megaminx/models/m_inv_v0_pretrain/epoch_0049.pt        REJECTED (kept ~140 MB)
megaminx/models/m_inv_v0/epoch_0049.pt                 REJECTED (kept ~140 MB)
megaminx/src/megaminx/bridge.py                        residual math helpers
tests/test_megaminx_bridge.py                          5/5 unit tests
megaminx/scripts/81_bridge_compression.py              bridge driver (stride + v_trajectory)
megaminx/scripts/82_cross_relinking.py                 cross-relinking driver
megaminx/submissions/bridge_top50_vtraj.csv            75,162 (community-tainted)
megaminx/submissions/bridge_our_top50.csv              77,145 (new standalone best)
~/.claude/.../memory/inv_state_encoding_rejected.md    documented negative result
```

### Open at end of session

- Bridge top-100 on 77,145 base is running in background (~5h ETA). Expected ~30-50 more
  moves saved.

---

## OLD State (2026-05-18 — AZ v4 prod-1001 + 3-way merge)

**Score: 75,200** (current submitted best as of 2026-05-18; min-merge of prior 75,961 base + new community v4 CSV (75,355 standalone) → 290 v4 wins beat our base + 71 base wins kept, -761 moves vs prior submitted best, file `merge_v14_plus_min_count_v4.csv`).

**Prior: 75,961** (2026-05-18; AZ v4 prod-1001 (79,606 standalone, 134h GCP L4) min-merged with our 77,877 + 76,251 community-best yielded 114 unique AZ v4 wins concentrated in mid buckets 4-7, -290 moves vs prior).

**Prior: 76,251** (2026-05-12; m_dd_v0 50ep prod-1001 (84,132 standalone) min-merge with 76,304 community-best, 20 wins / -53 moves).

### THIS SESSION (2026-05-11 PM) — AZ v4 breakthrough

User asked to try a "larger, better, longer-trained" AZ model. Two experiments + one breakthrough.

**Experiment 1 (REJECTED): bigger V trunk (20.5M params)**
- Pretrain (`m_big_pretrain`) at hidden=(4096,1024)+4rb=20.5M then Bellman-refine (`m_dd_v_big`) 200ep mirroring m_dd_v0 recipe
- Final loss 0.0748 matched m_dd_v0's 0.0724 — but **all 4 benched checkpoints regressed**: best (ep49) 9/10 / 930 vs m_dd_v0 baseline 10/10 / 871, worst (ep99) 7/10 / 733
- Conclusion: cluster ceiling at 6M extends beyond V-only too. Don't scale trunk under this recipe. Memory: `bigger_v_trunk_regression.md`. Artifacts kept (~640 MB) at `megaminx/models/m_dd_v_big/` and `m_big_pretrain/`.

**Experiment 2 (BREAKTHROUGH): AZ v4 — 6M dual-head + augmented data + early stop**
- Same 6M ResMLPGFlowNet arch as AZ v3. Recipe identical to v3 EXCEPT:
  - Policy dataset: `az_dataset_76304.pt` (built fresh from `merge_v9_with_77152.csv` — the min-merged best-of-3 submission that produced 76,304) vs v3's single-source 78,029
  - Trained 200ep planned, **stopped at epoch 24** based on bench trajectory
- Same launch flags as v3: `--rw-batch-size 8192 --policy-batch-size 1024` (CAUTION: these are NOT the script's defaults — the script defaults to 4096/512 which collapses to memorization in ~25 ep)
- **10-pid bench**: 10/10 solved, **874 / mean 87.4** — essentially ties m_dd_v0 V-only (871) and beats AZ v3 (943) by 69 moves
- **Strat-5 bench (51 pids @ beam 65k, KhoruzhiiSolver, no qshort/sym/NISS)**: **51/51 solved, total 4,465, mean 87.5** — **FIRST 6M model to break sub-89 strat-5 STANDALONE in this project** (m05+sym4 inference trick was previously the only sub-89 result at 88.2)

**Why v4 works where v3 didn't (corrected diagnosis)**:
- Previously believed: 8% dual-head trunk-sharing tax (943 vs 871)
- Real cause: **over-training**. The 76,304 merged dataset has more optimal-path consistency than v3's single-source 78,029 → policy CE converges *much* faster → past epoch ~25 the trunk specializes for policy memorization at the V head's expense
- Trajectory shows v_loss climbs 0.096 → 0.481 between epoch 0 and 99 while top-1 acc hits 91%. Earlier stop captures policy signal (top-1 28.5%, well above 4.2% random) without sacrificing V calibration

**Experiment 3 (DIAGNOSTIC): does AZ v4 V compose with the existing inference stack?**
- Strat-5 with AZ v4 V + m23_v2 qshort + AZ v4 policy (λ=0.05) → 51/51 in CSV (47 by model), total **4,621, mean 90.6**. **WORSE than V-only by 156 moves.**
- Strat-5 with AZ v4 V + m23 qshort, NO policy → 48 by model, total **4,705, mean 92.3**. **Worse still — qshort is the problem.**
- AZ v4 V + m_pi_v2 policy + qshort: COULDN'T MEASURE locally (3 models on 17GB OOM at beam ≥32k); to be measured on GCP if (a) below underperforms
- **Diagnosis**: m23_v2's top-k action ordering reflects m05's V landscape, NOT AZ v4's. Adding the AZ v4 policy at λ=0.05 partially clawed back qshort's damage but didn't fully recover

**Inference decision for full-1001 GCP run**: **drop qshort**, keep sym-ensemble K=4 + NISS + multi-pass beam. Recipe (a):
```
03_solve.py --checkpoint megaminx/models/m_az_v4_v_only.pt \
  --sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 \
  --niss --bf16 --resume --out megaminx/submissions/m_az_v4_prod_1001.csv
```

**Files created this session**:
```
megaminx/configs/m_big_pretrain.yaml, m_dd_v_big.yaml          (bigger-trunk experiments)
megaminx/models/m_big_pretrain/, m_dd_v_big/                   (kept for diagnostic; safe to delete)
megaminx/models/m_az_v4/epoch_0024.pt                          (the breakthrough)
megaminx/models/m_az_v4_v_only.pt                              (extracted V head, ResMLPDistance output_dim=1)
megaminx/models/m_az_v4_pi_only.pt                             (extracted π head, ResMLPDistance output_dim=24)
megaminx/models/m_az_v4_smallbatch_bug/                        (training-config-bug run; useful as memorization data point)
megaminx/data/az_dataset_76304.pt                              (76,304 augmented dataset)
megaminx/submissions/m_az_v4_strat5.csv                        (the 51/51 / 4465 result; mean 87.5)
megaminx/submissions/m_az_v4_strat5_pi05.csv                   (V+π+qshort regressed result)
megaminx/submissions/m_az_v4_strat5_qshort_only.csv            (V+qshort regressed result)
```

**Memory files written**:
- `~/.claude/.../memory/bigger_v_trunk_regression.md`
- `~/.claude/.../memory/az_v4_breakthrough.md` ← READ THIS FIRST in next session

**Auto-poll loop (2026-05-11)**: DID NOT fire. `.claude/scheduled_tasks.lock` was deleted when the session ended, so the wake-up never executed. m_dd_v0 finished at 84,132 (1001/1001 valid) overnight 2026-05-12; AZ v4 had to be launched manually the next session.

**Update (2026-05-12)**: m_dd_v0 50ep prod-1001 run finished overnight (84,132 total, 1001/1001 valid). Min-merge with 76,304 community-best produced new submitted best **76,251** (-53 moves, 20 unique pid wins). AZ v4 V-only prod-1001 launched manually on cayley-gpu PID 28722 with recipe (a) below. Output: `megaminx/submissions/m_az_v4_prod_1001.csv`. Watcher b8k7jjwi0 polls every 15 min and notifies on terminal state.

**Update (2026-05-18)**: AZ v4 prod-1001 finished after ~134h wall on GCP L4 (5.6 days). Standalone 79,606 (946/1001 model-solved, 55 fallback; pass-1 16k→43 solves, pass-2 65k→958 solves). 3-way min-merge AZ v4 + our 77,877 (`merge_v10_our_plus_m_dd_v0.csv`) + community 76,251 → new submitted best **75,961** (-290 moves, file `merge_v13_az_v4_plus_community.csv`). AZ v4 unique-win attribution: 114 pids (concentrated in difficulty buckets 4-7, the regime m_dd_v0 was weak on). 6M cluster ceiling cracked — see `az_v4_breakthrough.md` memory.

**Update (2026-05-18, later)**: New community CSV arrived (`_incoming_min_count_v4.csv`, 75,355 standalone — strictly dominates the prior v3 78,196: 562/1001 pids shorter, 0 longer). Min-merge with our 75,961 → submitted best **75,200** (-761 moves, file `merge_v14_plus_min_count_v4.csv`). 290 v4 pids beat our base, 71 base pids beat v4 (so v4 alone is not the new floor — our merge_v13 still contributes). Community-merge policy exception again.

**Update (2026-05-19/20 session — m_v11 retired, qshort distilled, TPU kernel mixed)**: Three threads.

1. **m_v11 11.8M trunk (Track B from earlier plan): REJECTED.** Two-stage train (50ep random-walk pretrain + 200ep Bellman λ_pdb=0 lr 2e-4 + 25ep AZ-style fine-tune on `az_dataset_75200.pt`). Stage 1b finished clean (V(V0)=0.012, V(d=1)=1.00) and 10-pid bench showed 10/10 vs Stage 1b 9/10 — looked promising. **But**: strat-51 stuck at 16/20 model solves in 1h+ (vs AZ v4's 51/51 in ~75min). Pids 995-998 head-to-head with production recipe: m_v11 96.25 avg vs AZ v4 67.5 avg = **+43% moves/pid worse**. Bigger trunks regress on this puzzle regardless of recipe — extends Rule 14 down to 11.8M. The 10-pid bench was misleading because its pid set didn't include the hardest puzzles where the V approximation actually matters. Don't retry trunk scale-ups without a fundamentally new signal source. See EXPERIMENTS.md row.

2. **`m23_v3_az_v4_sym` Q-shortlister distilled from AZ v4 V: KEPT.** Same arch as m23_v2 (12.4M, hidden=(2048,1024)+3rb), distilled with rotation augmentation prob=0.5. Final MSE 0.355, KL 0.088 (matches m23_v2's quality on its own training distribution). Crucially, **GLOBAL recall at α=2 = 100%** vs AZ v4 V (verified via `09_eval_q_recall.py`). Unlocks `AZ v4 V + qshort + sym-ensemble` production stack (which the previous Rule 15 mismatch prevented). Strat-51 result: AZ v4 V + m23_v3 + sym4 = 4340 / mean 85.1 vs AZ v4 V-only baseline 4465 / mean 87.5 = **-125 moves (-2.8%)**. PASS the acceptance gate. Saved at `models/m23_v3_az_v4_sym/epoch_0199.pt`.

3. **TPU shareable qshort kernel built but mixed result.** New private kernel `artgor/cayleypy-megaminx-48m-720-qshort-shareable` (clone of 48M_720 V-only + qshort prefilter in JAX SPMD body). Initial v1 had a correctness bug — implemented per-parent top-α (62% recall vs V) instead of GLOBAL top-αB (99.6% recall, matches PyTorch `QShortlisterSolver` line 161). v2 fixes this. **pid 0 result on TPU at B=48M sym K=8 SYM_POSITIONS=range(0,4)**: 60 moves vs v_only's ~55 = +5 moves qshort-tax. Quality decrease deemed too high for general use on easy pids. **Not worth deploying broadly**; the TPU kernel sits ready if a future use case (long-tail-only rescue) emerges. Wall savings vs v_only also smaller than theoretical (1.5× actual vs 6-12× theoretical) — V eval isn't the bottleneck at B=48M; chunk_n expansion + all_to_all dominate.

**Net session outcome**: No submission change (still 75,200). New production-stack artifact (`m23_v3_az_v4_sym/epoch_0199.pt`) ready for a GCP-side full-1001 run with AZ v4 V + qshort + sym4 + NISS multi-pass — projected ~77K standalone, ~74.5-75K after min-merge with 75,200. **NOT launched yet** — user paused at this decision point. The TPU 48M qshort kernel is functional but not recommended for new runs given the per-pid quality tax. m_v11 artifacts (`m_v11_pretrain/`, `m_v11_bellman/`, `m_v11_az/`, `m_v11_az_v_only.pt`, `az_dataset_75200.pt`) kept on GCP for forensic reference; safe to delete to free ~250 MB.

### Previous session's work (2026-05-10 → 2026-05-11 AM)

User asked for SOTA-quality pursuit of: admissibility-aware loss, trajectory balance (arXiv:2603.01786), AlphaZero-style training, dataset distillation. Plus prior PDB+IDA*.

**Outcomes summary:**

| Model / experiment | Result | Status |
|---|---|---|
| PDB+IDA* (corner K=5 PDBs, max-of-4) | +4 moves on 20-pid bench (neutral) | INFRA built, not useful for beam |
| m_adm_v0 (admissibility loss only, 50ep) | V undershoot 77→15%; V(V0) unchanged | learned-from |
| **m_dd_v0 50ep (V0/d=1 anchor + λ_pdb=5)** | V(V0): 1.99→0.009; **20-pid: -26 vs baseline** | **ACCEPTED — canonical baseline** |
| m_dd_v0_full (184ep, killed) | smoothed loss 0.0688 → overfit; bench regressed | REJECTED (lesson: long training drifts) |
| m_tb_v0 (raw TB, length-30 walks) | 0/10 solved | training-distribution mismatch |
| **m_tb_v1 (warm-start trunk + walks 5-100)** | 10/10 solved; total 1016 (+16% vs m_dd_v0) | dual-head works with proper data |
| m_az_v0/v1/v2 (synthetic / paths / +siblings) | 0/10 solved each | training-distribution mismatch |
| **m_az_v3 (Bellman value + policy CE hybrid)** | 10/10 solved; total 943 (+8% vs m_dd_v0); **top-1 acc 50.7%** | dual-head + Bellman value works |

### Key learnings (binding insights for next session)

1. **The V(V0)≈2 bug was real and fixable.** Anchor mixin (32 V0 + 24×4 d=1 children with exact targets per batch) drops V(V0) from 1.99 to 0.009. This took 25+ recipe variants to diagnose but is now a fixed `BellmanConfig.n_anchor_v0` + `n_anchor_d1` field.
2. **Long training without per-bench validation drifts.** m_dd_v0_full trained to 184ep with `min_delta=1e-4` smoothed-loss early stopping. Final loss was lower (0.0688 vs 50ep's 0.0724) but beam quality REGRESSED severely (fallback rate jumped). Lesson: validate via small bench every ~50 epochs during long runs.
3. **The value head needs broad off-path coverage.** Training only on solver paths (even with sibling expansion from a teacher V) fails in beam — beam expands 23 off-path children per candidate. Random walks naturally span the manifold; solver paths cover a narrow tube.
4. **Dual-head architecture trades ~8-16% value quality for policy availability.** m_tb_v1: +16% moves. m_az_v3: +8% moves. But m_az_v3 has a 50.7% top-1 policy head — comparable to m_pi_v2's Q-shortlister but with shared trunk.
5. **PDB+IDA* doesn't improve beam.** V_neural already saturates above PDB's max-depth (14) for far states; near-solved is already optimal via BFS-d6. Max-combine integration is a wash. Infrastructure reusable for other uses.
6. **Single-pass beam (no escalation, no NISS) is broken for hard pids.** First GCP eval (m_dd_v0_full + `--beams 65536 --max-steps 120`) had ~0% solve rate for pids 60-99. Production recipe `--beams 16384,65536 --max-steps 60,150 --niss` recovers to ~78% solve at pid 280. THE production recipe stays the default — single-pass is for smoke only.

### GCP eval (incomplete as of session end)

**Running on `cayley-gpu`, PID 11761.** Model: `m_dd_v0 50ep`. Recipe: `--beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume`. Output: `/home/and-l/cayley/megaminx/submissions/m_dd_v0_50ep_prod_1001.csv`. Log: `/home/and-l/cayley/megaminx/models/m_dd_v0_50ep_prod_eval.log`.

**Last checkpoint: pid 299/1001, model:237 fb:63, total 21,924 moves, 32661s wall (~9h elapsed).** Rate: ~130s/pid with NISS+escalation. ETA: ~25 more hours = next session likely catches it mid-run or done.

**Action for next session**: 
1. `ssh -i /c/Users/and-l/.ssh/google_compute_engine and-l@<IP> 'tail -20 ~/cayley/megaminx/models/m_dd_v0_50ep_prod_eval.log'` to check progress. IP via `gcloud.cmd compute instances describe cayley-gpu --zone=us-east1-b --format='value(networkInterfaces[0].accessConfigs[0].natIP)'`.
2. If eval finished: `scp` the CSV back, verify, compare to 76,304, submit if better.
3. If still running: let it finish or kill at next milestone and use partial CSV (resume mode wrote per-pid as it went).

### Deliverables (new files this session)

```
src/cayley/bellman.py           — added lambda_pdb, n_anchor_v0, n_anchor_d1, early_stop_* fields
src/cayley/gflow_model.py       — NEW: ResMLPGFlowNet (dual head + log_Z) + TB loss helper
src/cayley/khoruzhii_search.py  — added pdb_combine_mode parameter ("replace_in_set"|"max")
src/megaminx/{corner_coord,edge_coord,pdb_corner,pdb_edge,pdb_heuristic}.py — PDB infra
megaminx/configs/m_adm_v0.yaml, m_dd_v0.yaml, m_dd_v0_full.yaml
megaminx/scripts/
    58_corner_pdb_beam.py       — PDB-augmented beam runner
    60_train_admissible.py      — admissibility-aware + anchor mixin trainer (m_adm, m_dd)
    62_train_tb.py              — raw TB trainer (REJECTED)
    63_train_alphazero.py       — synthetic-walk AZ trainer (REJECTED)
    65_bench_dual_head.py       — bench adapter for ResMLPGFlowNet checkpoints
    66_train_tb_proper.py       — TB v1 (warm-start + variable walks) — WORKS
    67_build_az_dataset.py      — build (state, action, remaining) tuples from CSV
    68_train_az_v1.py           — AZ on path-only data (REJECTED)
    69_build_az_dataset_v2.py   — add 23 siblings/state with V_teacher labels
    70_train_az_v2.py           — AZ on path + siblings (REJECTED)
    71_train_az_v3.py           — AZ Bellman+policy hybrid — WORKS
megaminx/data/
    corner_tables.pkl, edge_tables.pkl        — perm/ori tables for PDB
    pdb_corner_K5{,_p1,_p2,_p3}.pkl           — 4 disjoint K=5 corner PDBs (1.8 GB)
    az_dataset_78029.pt, az_dataset_v2.pt     — AZ training datasets
megaminx/models/
    m_adm_v0/, m_dd_v0/{epoch_0049.pt}, m_dd_v0_full/best.pt  — V models
    m_tb_v0/, m_tb_v1_proper/{epoch_0499.pt}                  — TB models
    m_az_v0/, m_az_v1/, m_az_v2/, m_az_v3/{epoch_0099.pt}     — AZ models
```

### Next-session priorities

**Read `~/.claude/.../memory/az_v4_breakthrough.md` first** for the recipe + inference details.

1. **MONITOR AZ V4 PROD-1001 RUN** (launched manually 2026-05-12, PID 28722 on cayley-gpu):
   - Recipe: `--sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume`
   - Output CSV: `~/cayley/megaminx/submissions/m_az_v4_prod_1001.csv`
   - Eval log: `~/cayley/megaminx/models/m_az_v4_prod_eval.log`
   - Status check: `gcloud compute ssh cayley-gpu --zone=us-east1-b 'pgrep -af 03_solve.py; wc -l ~/cayley/megaminx/submissions/m_az_v4_prod_1001.csv'`
   - Watcher b8k7jjwi0 should fire on terminal state — if not, check the output file.
   - The 2026-05-11 auto-launch via ScheduleWakeup did NOT fire (`.claude/scheduled_tasks.lock` was deleted at session end).

2. **When AZ v4 CSV is done**: scp back, verify, compare to current best (76,251 / our private 77,877), submit if win. Naive projection from strat-5 delta (-1.9 moves/pid at 51 pids → roughly -1900 moves at 1001), but tail behavior is the unknown — sym-ensemble 4 should help long-tail but also 4× the V forward cost.

3. **Open follow-ups for AZ v4** (only if (1) doesn't burn the time):
   - Distill a NEW qshort from AZ v4 forward states (m23-v3 style). m23_v2 misalignment with AZ v4 V is the binding issue preventing the production stack from composing cleanly. Probably ~30 min training.
   - Try other early-stop points (ep 20, 28, 32) — could find an even better checkpoint cheaply (just bench, training already done in `m_az_v4_smallbatch_bug/` or re-train ≤30 ep at correct batch).
   - Measure m_pi_v2 + AZ v4 V on GCP (needs > 17GB to load 3 models at beam 65k) — diagnostic skipped locally due to OOM.

4. **Don't pursue**:
   - Bigger V trunks under the m_dd_v0 recipe — confirmed regression today (see `bigger_v_trunk_regression.md`).
   - Long training without per-bench validation — m_dd_v0_full + m_dd_v_big both burn loss while regressing bench.
   - More PDB / AZ-without-Bellman / TB variants — AZ v3 superseded by AZ v4; TB v1 is worse than V-only.

---

## OLD State (2026-05-09 session — for reference)

**Score: 76,304** (current submitted best; min-merge with two colleague CSVs as user-authorized policy exception. Our own contribution stack still produces 78,029 — see superseded row). Production stack (our work):

```
m_curr_v3 V teacher (curriculum k=35 warmup → mix-K {50,70,80,100} body, then Bellman + 25% frontier + 10% BFS-d6 anchor; loss 0.0721, lowest ever)
  + m23_v2 Q-shortlister (sym-aware)
  + m_pi_v2 policy head (solved-path CE, warmstart from m_curr_v0 body), λ=0.05
  + --sym-ensemble K=4 (360 rotations, A_5 x C_6 group)
  + beam 524k for hard-tail rescue, 131k for top-200 rescue, 65k for strat-5
  + TRT FP16 engine on GCP (sm_89-specific) for full-1001
```

Daily score progression in this session:
88,195 → 86,329 → 85,812 → 83,362 → 82,646 → 82,481 → 82,225 (-5,970 net).

### What we learned (binding constraints)

1. **Cluster ceiling at 6M params: confirmed thoroughly.** Every single-recipe
   training-side variant (m05, m17, m22 K=2, m26 12M, m27 family, m28
   Double Bellman, m29 n_back=4, m30 n_back=16, m31 rot-aug, m32
   target_update=5, SWA) lands within strat-5 mean **88-97**. m29 (88.98) is
   the only sub-89 training-side. **Recipe levers exhausted at this arch.**
2. **Sym-ensemble at INFERENCE is the unlock**: m05+sym4 → 88.20 (no
   training change), m05+sym4+qshort+m23_v2 → 88.41 at 6× faster wall.
   First mechanism to break the cluster meaningfully.
3. **m23_v2 is sym-aware** (rotation-augmented during distillation). Pair it
   with --sym-ensemble; pair m23 (original) with non-rotated solves.
4. **K=4 is the sym-ensemble sweet spot.** K=8 saves ~80% as much per pid
   at 2× wall (diminishing returns).

## 9. Active queue for next session

The top of the list reflects the highest-EV moves NOT YET TRIED. Items
explicitly saved-for-last (per discipline) are flagged.

### Highest EV (multi-day mega-bet, saved for last)

- **A5 — Full-1001 sym-ensemble + qshort + 524k + TRT on GCP.** Run the
  proven m05+m23_v2+sym-ensemble K=2 or K=4 stack across all 1001 pids on
  GCP. Cost: K=2 ≈ 20-30h GCP, K=4 ≈ 40-60h. Expected: -1500 to -3500 moves
  vs current 82,481 (extrapolating from strat-5 win × 1001/51). The stack
  is fully validated; this is just compute.
- **Multi-seed beam ensemble at full-1001.** Run m05+sym4 at 3 different
  RNG seeds (different hash_vec → different beam trajectories), merge
  per-pid min. Orthogonal to sym-ensemble's diversity. Cost ~3× a single
  full-1001. Already proven on hard-tail (-1,866 in earlier work).

### Building toward T1.1 properly (multi-day)

- **T1.1 — Curated speedcubing macros.** Mechanism is shipped
  (`KhoruzhiiSolver(macros=...)` with cost-aware V adjustment). Brute-force
  d=4 commutators don't help. Need: scrape ~100 named macros from
  speedsolving.com / cubingdb.com / jPerm, map to our notation, validate.
  Effort: 3-5 days per tier doc. Expected: -10K to -20K alone (per
  tier doc: "the lever that crosses 70K alone" if it pays).

### Other "reasonable" items (low/medium ROI given cluster ceiling)

- **A4 — More aggressive beam-stack rescue.** Currently triggered only on 2
  catastrophic failures. Sweep all model failures from the merged
  full-1001, run beam-stack on each. ~2.5h. Expected -100 to -300.
- **B9 — Beam-frontier replay (DAgger-style).** Log frontier states from
  real `03_solve.py` runs, mix into Bellman epochs at ~25%. Fixes the
  *distribution* (RW vs beam) not the target. ~80 lines impl. Untried.
- **B10 — m24 V + π multi-task.** `11_train_policy_head.py` exists.
  Beam scoring `V(child) + λ·(-log π(a|parent))`. Expected +3% on hard.
- **B11 — cayleypy nbt/bfs walks A/B.** Tighter walk-distance labels than
  ours, paired with Bellman. Caveat: m21 (tightened target without
  Bellman) regressed.
- **B7 — Per-depth target shrinkage diagnostic.** ~15 lines. Tells us
  whether bias is the binding mechanism. Run before any further bias
  mitigation.
- **B8 — BFS-d6 as Dirichlet boundary IN the Bellman target.** Replace
  `1 + min_a V_target` with EXACT distance when state is in d≤6 shell.
  Different mechanism than m27's pretraining mixin. ~20 lines.

### Long-list (multi-week research)

- **T1.2 multi-agent ensemble** (CayleyPy paper recipe — diverse agents +
  selector). 1+ week.
- **Transformer at scale** — m18 converges but 9× slower per-wall.
  Local 4090 viable but slow.
- **A* / IDA*** — admissible lower bound + best-first. Big code, uncertain
  timeline win.
- **Macro-Q shortlisting** — Q-head scores 2-6 move macros mined from
  solved paths.
- **Listwise rank loss training** (T2.3) — calibrate V on relative
  ordering, the actual signal beam uses.
- **PDB (Pattern Databases)** — Korf-style on Megaminx subsets.
- **Bidirectional with learned front-to-front scoring** — research project.

## 10. Active gotchas + recipes (read these first)

See also `~/.claude/projects/.../memory/megaminx_gotchas.md` for full list.
Highest-impact ones in order of recency:

- **m23 student MUST match m05 teacher** (not m29, etc.). Pairing m23 with
  another teacher tanks recall and produces longer paths. Train a new Q-
  shortlister against the new teacher first.
- **m23_v2 (sym-aware) is the right student** when using `--sym-ensemble`.
- **`run_benchmark.py` does NOT save move sequences.** Use 03_solve.py.
- **`sed -i` on Windows MINGW silently truncates files.** Use Edit tool.
- **Windows `tasklist /FI` in Bash gets path-mangled.** Use PowerShell
  `Get-Process` instead.
- **Beam max-steps must exceed the longest expected path.** ≥150 for
  production solves; 60-80 only viable as the first pass of multi-pass
  escalation.
- **Monitor cadence**: filter to milestone epochs (every 25), not every line.
- **`torch.compile` on beam needs `pad_to_batch_size=True`** (5.8x
  slowdown otherwise).
- **Reuse one `CayleyGraph` per session** — fresh graphs have different
  hash vectors.
- **Sym-ensemble K=4 is sweet spot.** K=8 only worth it for hardest pids.
- **Macro-augmented beam REQUIRES curated macros**, not brute-force
  commutators. The mechanism is shipped but useless without good macros.

## 11. Useful project conventions

- **Submissions go to `submissions/<name>.csv`**. Always run `verify_submission` before `kaggle submit` — a typo in a generator name silently produces a no-improvement score.
- **Training configs in `configs/<name>.yaml`**. Outputs to `models/<name>/epoch_NNNN.pt` + `models/<name>_training.log`.
- **Stratified eval is `--stratified 5 --strat-seed 0`** (51 puzzles). Never `--stratified 3` (too noisy).
- **Acceptance gate for any new heuristic**: ≥+3 strat-5 solves AND mean model_avg ≤ 0.95× current best, no bucket regresses by more than 1.
- **Don't submit community-merged results** (per user policy). Continue until our own models beat community on at least some puzzles.
- **Never use `sample_fallback.csv` as fallback** — it's sample-quality (500K+ moves). Always `data/pp_bfs6_fallback.csv` (414,678 floor).
- **Discipline rule (CLAUDE.md #9)**: before adding a NEW experiment, check `to_do_shortlist.md` first. Don't propose ideas already there or already tried/rejected.

## 12. Quick-start commands

```bash
# Local 4090, single solve at beam 131k:
.venv/Scripts/python.exe megaminx/scripts/03_solve.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --out megaminx/submissions/m05_b131k.csv \
    --beams 131072 --max-steps 150 --num-attempts 1 --bf16

# Local 4090, qshort + beam 524k, FULL submission format:
.venv/Scripts/python.exe megaminx/scripts/03_solve.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --qshort-student megaminx/models/m23_q_shortlister/epoch_0499.pt \
    --qshort-alpha 2 \
    --out megaminx/submissions/qshort_524k.csv \
    --beams 524288 --max-steps 150 --bf16

# CURRENT BEST RESCUE STACK: sym-ensemble K=4 + m23_v2 + qshort + beam 524k
# (rescue top long-tail pids of current submission; ~6h on local 4090 for 80 pids)
.venv/Scripts/python.exe megaminx/scripts/03_solve.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --qshort-student megaminx/models/m23_v2_sym_aware/epoch_0499.pt \
    --qshort-alpha 2 \
    --out megaminx/submissions/sym4_rescue.csv \
    --pids <comma-separated long-pids from current submission> \
    --beams 524288 --max-steps 150 --bf16 \
    --sym-ensemble 4

# Merge a rescue CSV into a base submission (per-pid take min):
.venv/Scripts/python.exe megaminx/scripts/16_merge_rescue.py \
    --base megaminx/submissions/<base>.csv \
    --rescue megaminx/submissions/<rescue>.csv \
    --out megaminx/submissions/<merged>.csv

# Stratified-5 eval (51 puzzles, the canonical metric):
.venv/Scripts/python.exe megaminx/scripts/03_solve.py \
    --checkpoint <ckpt> --out <out.csv> --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 --bf16

# Stratified-5 eval WITH sym-ensemble (the new gold standard for V models):
.venv/Scripts/python.exe megaminx/scripts/03_solve.py \
    --checkpoint <ckpt> --out <out.csv> --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 --bf16 \
    --qshort-student megaminx/models/m23_v2_sym_aware/epoch_0499.pt \
    --qshort-alpha 2 --sym-ensemble 4

# Cheap-eval (BFS-true MSE, top-1/3, walk MSE) on multiple ckpts:
.venv/Scripts/python.exe megaminx/scripts/validate_model.py \
    --checkpoints <ckpt1> <ckpt2> ... --bf16

# GCP TRT export (on the L4 VM):
ssh and-l@<IP> 'cd ~/cayley && python3 megaminx/beam_lab/export_tensorrt.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --batch-size 16384 --precision bf16 \
    --output megaminx/models/m05_trt_bf16_b16384_sm89.ts'

# Kaggle submit:
export KAGGLE_API_TOKEN=KGAT_630ac26efca89d28c5b2d496b238b71c
.venv/Scripts/kaggle.exe competitions submit -c cayley-py-megaminx \
    -f submissions/<file> -m "<short description>"
```

## 13. Fast next-step recommendation (if you don't know where to start)

The recipe-ceiling track is exhausted. Today's session validated sym-ensemble
at inference as the cluster-breaker. The clearest highest-EV next moves:

**OPTION 1 (~30h, low risk, biggest expected score impact)** — Full-1001
sym-ensemble on GCP. Mechanism is fully validated at strat-5; this is just
scaling to all 1001. Run on GCP L4 with:

```
ssh and-l@<IP> 'cd ~/cayley && nohup python3 megaminx/scripts/03_solve.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --qshort-student megaminx/models/m23_v2_sym_aware/epoch_0499.pt \
    --qshort-alpha 2 \
    --tensorrt-engine megaminx/models/m05_trt_fp16_b16384_sm89.ts \
    --out megaminx/submissions/full_1001_sym4_v2_trt.csv \
    --beams 524288 --max-steps 150 --bf16 \
    --sym-ensemble 4 \
    > megaminx/submissions/full_1001_sym4_v2_trt.log 2>&1 &'
```

Then merge with current `submissions/merge_plus_sym8_top20.csv` (82,481).
Expected: -1500 to -3000 vs 82,481. The mega-bet that most likely cracks 80K.

**OPTION 2 (~3-5 days, biggest theoretical gain if it works)** — T1.1 curated
speedcubing macros. Mechanism shipped; fill in the macros via web scrape.
See `tier1-tier2-tier3-merged.md` T1.1. Per tier doc: "−10K to −20K" if
plays out; could cross 70K alone. Higher uncertainty though.

**OPTION 3 (continued tail-rescue, ~6-12h each)** — Iterate the
hard-tail rescue cycle: identify top long-pids of current submission, run
sym4 + qshort + 524k on them, merge. Per A2.2 today: -716 moves on 80 pids
in ~6h. Diminishing per pass but additive; 2-3 more passes plausibly worth
a few hundred moves each.

**The 70K target requires sym-ensemble at scale OR macros, not more recipe
tweaking.** TRT helps by funding bigger beams; sym-ensemble multiplies that.
