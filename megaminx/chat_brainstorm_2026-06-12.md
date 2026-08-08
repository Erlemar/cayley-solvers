# Megaminx: directions from the CayleyPy chat export (May 24 - Jun 11), 2026-06-12

Source: Telegram export `ChatExport_2026-06-12` (3,736 messages), dumped to
`megaminx/chat_export_dump.txt` for grepping. This doc distills what changed,
reconciles it against our EXPERIMENTS.md / to_do_shortlist.md verdicts (Rule 9),
and proposes a prioritized plan.

---

## 1. What changed in the outside world

1. **Ivan Litvak shipped a C++/CUDA multi-GPU beam search** that changes the
   compute regime entirely:
   - GitHub `TryDotAtwo/MultiGPUBeamSearch` (~20K lines C++), Kaggle notebook
     `trydotatwo/cayley-beam-gpu-runner` (public, accepts any model weights).
   - Beam widths: **86M on Kaggle 2xT4** (~50-64 s/step), **250-260M on a free
     molab RTX PRO 6000 96GB** (~52 s/step; molab registration now closed),
     **650-700M on 8xA100-40G** (~85 s/step; cluster access via Valeriya).
   - Tricks: fp16 inference; adaptive score-threshold pre-pruning BEFORE
     dedup/top-k (this is how selection scales); history (backptrs) spilled to
     RAM+disk arena (depth limit ~72-80); **radius-4 solved-neighborhood early
     stop** (last ~9 moves free + rescues near-misses).
   - Honest speed vs CayleyPy on same model/hardware: ~3.7x on P100, ~2x/GPU
     scaling; the qualitative jump is that *selection* no longer binds, so width
     scales to VRAM.

2. **24-output Q-models are what feed it.** Vlad Kuznetsov's 38M Q (one forward
   on the parent scores all 24 children) is the workhorse. At fixed wall the
   trade "24x cheaper per candidate -> 24x wider beam" empirically WINS:
   - Vlad Q @ 2^16x24 beats 1-output models @ 2^16 (85.5 vs 92.2 avg, hard tail).
   - Our AZ v4 (1-output) at 700M on pid 991: 70 - no record (69 stands).
     Chat consensus: 1-output at giant beam is a dead end ("nuzhna model 24
     autputa horoshaya").

3. **New per-pid records from width + the inverse trick** (700M + Vlad Q):
   pid 0 (superflip) 54 -> **53**; 1000: 70 -> **68**; 999: 72 -> **68**;
   997: 73 -> **71**; 994: 73 -> **71**; 993: 70 -> **69**. But 991, 992, 995,
   996, 998 did NOT fall - their records came from 20-28 rotation runs at 48M.
   **Width and rotation-diversity are complementary; nobody has run both.**

4. **Vlad's training recipe is the first to clearly beat our AZ v4 line** at
   matched beam on the hard tail: his 1-output teacher = **86.05 avg @ 2^16 on
   pids 900-1000** vs AZ v4's 92.2 (same range, same beam; his eval adds fp32 +
   tail-BFS-6 completion, worth maybe 1-2 of the 6). Recipe:
   - very long random-walk pretrain -> SHORT Bellman finetune (best ~ep 11,
     diverges later - matches our Rule 12 experience);
   - **symmetry-augmented batches (3-4 syms per sample) + a variance penalty
     across symmetric states** (not our isolated lambda_sym MSE);
   - distill into 24-output Q with MSE+KL;
   - NEW (Jun 11, in flight): **direct-Q training without distillation** via
     sym-completed sparse-RW targets - 91.84 @ 2^18 (300/300), approaching the
     distilled model; he calls it a small gamechanger if it lands.

5. **Community totals**: 75,163 (May 26) -> 74,517 -> Vlad-merge 72,986 (Jun 4)
   -> **72,161** (Jun 7, Vlad's 3 solutions + Liuda's 74,357). A new solo player
   (Lukin Alexey) appeared at top-2/3 without blending. The export contains
   Liuda's CSVs (74,116 best) - **n-way min with our 75,200 = 73,984 (-1,216)**,
   sitting on disk for free.

6. **Vlad's half-split re-solve**: split each path in half, re-solve each half
   as its own puzzle at 2^20 -> **-38 moves on the last 200 pids overnight**
   (~-200 projected over 1001). Works on already-merged paths.

7. **Beam stagnation quantified** (Fedor): near the end of stalled solves the
   model predicts V ~ 1.9-2.1 for states at true distance 6-8, repeatedly.
   Alexander proposes active learning on exactly these states.

8. Misc with evidence: Alexander's **color encoding** suggestion (use 12 colors,
   not 120 sticker ids: "on cubes this strongly simplified life"); LR scheduler
   on the RW baseline gave a big loss + decent length gain; beam-width jitter
   (+-1 .. +-10M) is a cheap diversity axis; Vlad's **sym-prescreen** proposal
   (solve k syms at B/k, run the winner at full B = 2x wall for k-fold
   diversity); Kirill's 2-generator cube as a stress testbed; Ivan's torchrun
   BFS (PR #188, 6.25x) and his repeated offers to run our models on the
   cluster + requests for a better 24-output model.

---

## 2. Rule-9 reconciliations (so we don't re-litigate or miss nuance)

| Chat idea | Our prior verdict | Reconciliation |
|---|---|---|
| "Dual"/inverse-state solve (Ivan) | = our **NISS**, dropped by Rule 11 (0 rescues at 16k/65k + sym4) | Rule 11 measured small beams WITH sym diversity. At 700M with NO syms, inverse took half the new records. **Re-enable the inverse arm specifically in giant-beam runs**; it is not a production-recipe change. |
| 24-output Q as scorer | m06 Q-distill REJECTED (Apr 25: 51->20 solves) | m06 distilled from weak m07 pre-Bellman, deployed at SAME beam. New regime: teacher is AZ v4 V, KL+MSE distill already proven (m23_v3: 100% recall @ alpha=2), and the 24x goes into width. Chat provides direct evidence the trade wins. Justified retry. |
| Sym-consistency loss | m_sym_v0 lambda_sym isolated = TIE (Section 13.4) | Vlad's win is the RECIPE (long RW pretrain + short Bellman + sym-aug batches + variance penalty + KL), not the isolated loss. Replicate the bundle, not the term. His 38M also contradicts our trunk-scaling regressions (Rule 14, m_v11) - those were *our recipe* at bigger trunks. |
| Half-split re-solve | Bridge compression ceiling ([[bridge-compression-findings]]); tail_resolve (T2.2) shipped | Bridge failed because **V-scored window selection** collapses past d~30. Half-split needs no window scoring - it re-solves fixed segments outright (a depth-38 state is in-distribution). Our T2.2 is suffix-only; the PREFIX arm is untried. Vlad validated on merged paths. |
| qshort on TPU at 48M | NOT RECOMMENDED (+5 moves tax) | That was qshort-as-FILTER with V still scoring. Q-as-scorer (no V at all) is a different mechanism; Vlad's Q shows no such tax at width. |
| Color encoding | no entry in EXPERIMENTS.md; Section 3 closure covers *scorer architectures* | This is an input-vocab reduction (120 -> 12) on the canonical ResMLP, lossless (every piece has a unique color combo). Not the additive-channel state_inv that failed. Untried, cheap. |
| Active learning on stagnation states | B8/B9 on shortlist (untried); frontier-regret v0 says misranks are benign | Different signal: frontier-regret = *ranking* mid-path (benign); stagnation = *absolute calibration* near solved (V=2 vs true 6-8). The anchor-mixin pattern already fixed exactly this class once (V(V0)~2 bug). |

---

## 3. Prioritized plan

### Tier 0 - free, this week

**T0.1 Merge the export CSVs.** `d_submission_74116.csv` + our merge_v14 ->
**73,984** (-1,216), 251 community wins / 750 ours kept. Verify + submit per
the standing policy-exception process (needs user OK). All later "wins" must be
measured against THIS floor (Rule 26).

**T0.2 Re-enable the inverse arm for big-beam runs.** NISS plumbing exists in
`03_solve.py` and is trivial in the JAX notebooks (invert the start state,
invert the found path). Add `orig x inverse` to every TPU sweep config.

### Tier 1 - the regime change (highest EV)

**T1.1 Q-as-scorer distillation (the enabler).** Train a 24-output ResMLP
(6M and 12.4M arms) to regress `V_AZv4(child_a(s))` for all 24 children
(absolute values, MSE + KL-on-softmin; reuse m23_v3 training infra, change the
target from shortlist-rank to absolute V). bf16/fp16-export-safe.
Gates: (a) recall vs teacher (`09_eval_q_recall.py`), (b) strat-51 with
Q-as-scorer at matched wall (so 65k x ~24 effective width vs V at 65k).
Deploy:
- (a) **Ivan's Kaggle notebook at 86M on 2xT4** (it accepts any 24-out weights
  via `STREAM1_MODEL_TESTS`); hard-tail sweep, orig + inverse.
- (b) **Convert our JAX qshort kernel to Q-only scoring** (drop the V forward;
  the 24-output plumbing exists) -> v4-8 / v6e runs at 96-268M with the full
  sym x inverse grid - the diversity Ivan's cluster runs lack.
- (c) Offer the model to Ivan for 700M cluster passes (he is asking for exactly
  this), and ask Vlad for his 38M weights/notebook (he offered to publish).

**T1.2 Replicate Vlad's training recipe.** Two arms: (a) faithful ~38M
(long RW pretrain -> short Bellman -> sym-aug + sym-variance -> Q distill),
(b) recipe deltas on our 6M m_dd_v0 line. Gate: pids 900-1000 @ 2^16 vs AZ v4's
92.2 (the cleanest apples-to-apples the chat gives us), then strat-51 binding
gate. This is the only externally validated escape from our 6M/hard-tail
ceiling - it attacks the training distribution, not the architecture (Section 3
closure untouched).

**T1.3 Giant-beam x diversity campaign on the hard tail.** Top ~100 longest
pids of the 73,984 baseline at >=96M (zakhar v4-8 validated; v6e-8 for 268M)
x {2-4 rotations} x {orig, inverse} x beam-width jitter. Records data says
width took 5/10 hardest, rotations took the other 5 - run both.

**T1.4 Tail-BFS-d6 completion + early stop in our kernels.** We already own
`bfs_bytes_d6` (19.4M states). Sorted-hash membership check of the top-k states
per step; on hit, complete with the exact optimal tail. Guaranteed >= 0 gain
per pid, saves ~6 steps of wall, rescues near-misses (Ivan radius-4, Vlad
radius-6 both run this as standard).

### Tier 2 - cheap, validated add-ons

**T2.1 Half-split re-solve pass** over the current best CSV: split at 1/2 (and
1/3, 2/3), conjugate each segment to a fresh puzzle, re-solve at ~1M beam with
sym4 + inverse on v6e, splice + verify. Vlad: -38/200 overnight. Target the
mid pids (100-900) where our bridge already showed the mechanism works.

**T2.2 Sym-prescreen A/B** (one afternoon): for 10 pids, do 16 syms @ 8k and
check whether the small-beam winner predicts the 65k/1M winner. If yes,
production becomes screen-16 -> deep-4 at the same wall.

**T2.3 LR-schedule A/B** on the Bellman refine (cosine vs constant) - Alexander
saw a large training-loss gain + decent length gain on the small baseline.

### Tier 3 - one-run experiments (while Tier-1 compute runs)

**T3.1 Color encoding** (vocab 120 -> 12, value//10) on the m_dd_v0 recipe.
Lossless; shares embedding strength across the 10 stickers of a face; cube
precedent. Local 4090 ~1 day incl. gates. Watch the d=20 V-variance canary.

**T3.2 Stagnation-anchor Bellman.** Harvest beam local-minima states from
production logs (V < 3 while unsolved for >= 10 steps), label exactly via
BFS-d6 (and small-beam re-solve for d 7-10), mix in as anchors (the proven
n_anchor pattern) +- B8 Dirichlet-boundary labels in the Bellman target.
Gate: Fedor-style near-solved calibration plot + strat-51.

### Tier 4 - research bets (pick at most one, after Tier 1)

- **Verified-regret / beam-utility training** (our `new_iteration.md` Track 2):
  now unblocked - 86M-700M runs provide the "stronger teacher search" labels
  that were the missing ingredient (wider-beam survivors as strong labels,
  verified shorter continuations as gold).
- **Large transformer-Q with Ivan's deepspeed cluster** (he has flash-attn 2 +
  deepspeed staged and asks for an architecture; Vlad suggests 300-500M first).
  Note our GT-Q distill is the one unfalsified GT variant.
- **MITM / bidirectional with Liuda's mapping trick** - watch Ivan's prototype
  before investing.

---

## 4. Operational cautions

- **fp16 export discipline**: Ivan's runner converts weights to fp16; one
  4000-epoch model silently failed to solve at 2^23 because of it. Ship
  checkpoints that survive half precision (our 50ep-style, not 184ep-style) or
  get his bf16 path.
- His kernel's **depth limit is ~72-80** (history = width x depth x 16B);
  fine for second-pass solves where we know length <= 80.
- **Rule 26**: every claimed win compares against the 73,984 n-way floor (and
  whatever newer community CSV exists that day).
- Watch **Vlad's direct-Q no-distill run** (left training Jun 11) - if it
  lands, adopt his loss instead of distilling.
- Kaggle CLI status/output is currently degraded (memory) - push kernels via UI.

## 5. Why this can reach 70K

From 73,984: Tier-1 model + width work is worth ~2-6 moves/pid on the ~300
longest pids if the chat numbers transfer (700M+dual already took 2-4 off half
the hardest; a Vlad-class model is worth ~6/pid on the tail at 2^16, less after
merging) -> plausibly -1,500 to -3,000. Tier-2 adds -200 to -500 of cheap,
verified, non-regressing edits. The two stack because they attack different
slack: model quality vs path-local detours.
