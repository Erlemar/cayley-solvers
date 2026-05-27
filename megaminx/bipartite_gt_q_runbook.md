# Bipartite Slot-Sticker Graph Transformer Q-shortlister — runbook

> **RESULT (2026-05-25): TESTED → REJECTED (no architecture win).** Trained on GCP L4 to
> epoch 30/120, then stopped. Bipartite recall tracks the flat GT-Q essentially identically
> (e9 tied), and both stay far below the production ResMLP-Q on deep buckets and improve too
> slowly to close it. Better MSE did not translate to better recall (the GT-V lesson in
> Q-space). Combined with frontier-regret (V ordering near-optimal → ~0 solve headroom) and
> the GT's ~30-50× inference cost, the bet doesn't pay. Full write-up: `EXPERIMENTS.md`
> (2026-05-25) + memory `bipartite-gt-q-shortlister`. Code is retained + validated below for
> reference / possible future revisit at a different scale or with a teacher that has real headroom.

Operational companion to `megaminx_architecture_and_path_shortening_strategy.md` §3.2
and `graph_transformer_plan.md`. Built 2026-05-25. This is the *action-centric* graph
model: Q(a) is read from a dedicated action node that attends to exactly the ~25 slots
action `a` permutes (and through them, via dynamic features, to the stickers currently
in those slots) — not from a single pooled vector.

## Why this exists / the bet

The GraphTransformer **as a V model failed** beam search in every configuration
(`gt_v_no_saturation` memory, CLAUDE rule 23): its value landscape doesn't saturate at
the puzzle diameter. The one variant never falsified, and the variant §3.2 explicitly
prescribes, is **GT as a Q-shortlister** distilled from a saturating teacher — relative
sibling ranking, not an absolute distance field. The bipartite action-node structure is
the architectural reason this Q could beat the flat GT-Q (and is the candidate mechanism
for fixing the depth-recall sag if the flat GT-Q shows one).

Two constraints govern deployment (be honest about these):
1. **GT inference is ~50x slower than ResMLP** -> it can't be a live qshort student.
   The endgame is **GT-Q as teacher -> distill into a ResMLP-Q** (fast, deployable).
2. **V child-ordering is already near-optimal at 6M** (`phs_cumulative_validated_marginal`).
   So the realistic prize is *speed* (lower alpha at fixed recall) or the *hard-pid tail*,
   not mainline path quality. Gate accordingly.

## Files

| file | role |
|---|---|
| `scripts/74_build_bipartite_features.py` | static tables -> `data/bipartite_features.pt` (264-token layout, relation/dist matrices, sparse local mask, action-affects edges) |
| `src/megaminx/graph_transformer_bipartite.py` | `BipartiteGraphTransformerQ` (GraphGPS: local masked attn + global biased attn; Q from 24 action nodes). Wired into `cayley.search.load_model_checkpoint`. |
| `scripts/76_train_gt_q_bipartite.py` | distill `Q(s,a)=V_teacher(apply(s,a))`; MSE+KL; rotation aug; **early-stop on depth-stratified recall, not loss** |
| `scripts/09b_eval_q_recall_by_depth.py` | depth-stratified recall@alpha (the saturation diagnostic) |

`data/bipartite_features.pt` is already built and validated (T=264; SELF=264, HOME=240,
AFFECTS=1200, INV=24; local graph 3288 edges incl. self).

## Stage 0 (running first) — flat GT-Q gate

Decides whether the bipartite build is warranted at all.

```
.venv/Scripts/python.exe megaminx/scripts/75_train_gt_q.py \
  --teacher megaminx/models/m_az_v4_v_only_e99.pt \
  --out-dir megaminx/models/m_gt_q_v0 --n-epochs 120 \
  --rotations-path megaminx/data/rotations.npy --rotation-aug-prob 0.25
.venv/Scripts/python.exe megaminx/scripts/09_eval_q_recall.py --teacher ... --student megaminx/models/m_gt_q_v0/epoch_0119.pt --alpha 1,1.5,2,3,4 --bf16
.venv/Scripts/python.exe megaminx/scripts/09b_eval_q_recall_by_depth.py --teacher ... --student megaminx/models/m_gt_q_v0/epoch_0119.pt --alpha 1,1.5,2,3,4 --bf16
```

Baseline bar: `m23_v3_az_v4_sym/epoch_0199.pt` (ResMLP-Q from the same AZ v4 teacher).
**Proceed to Stage 1 iff** flat GT-Q hits recall@a>=0.99 at *lower* alpha than the ResMLP-Q,
**or** 09b shows a depth-recall sag (flat-Q falls with depth) that the bipartite action
nodes could plausibly fix. If flat GT-Q can't even match the ResMLP-Q, stop.

## Stage 1 — bipartite training

```
.venv/Scripts/python.exe megaminx/scripts/76_train_gt_q_bipartite.py \
  --teacher megaminx/models/m_az_v4_v_only_e99.pt \
  --out-dir megaminx/models/m_gt_q_bip_v0 --n-epochs 150 \
  --rotations-path megaminx/data/rotations.npy --rotation-aug-prob 0.25
```

Run it ALONE on the GPU (don't overlap with another training job). It saves `best.pt`
by worst-bucket recall and early-stops if that stalls. **Gate:** `09b` recall@a=2 >= 0.99
*flat across depth buckets* AND beating flat GT-Q / ResMLP-Q at equal-or-lower alpha.

## Stage 2 — distill to ResMLP-Q + binding gate (before any deploy)

Because of the 50x slowdown, deploy a ResMLP-Q distilled from the GT-Q's child orderings
(reuse the `09_train_q_shortlister.py` machinery with the GT-Q as the scoring teacher, or
add a small distill script analogous to `78_distill_v_from_gt.py` but for the 24-vector Q).

- **Binding acceptance (CLAUDE rule 21):** strat-51 `--stratified 5 --strat-seed 0` with the
  PRODUCTION recipe `--sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --niss --bf16`,
  student = distilled ResMLP-Q, teacher = AZ v4 V. NOT single-pass beam-65k.
- **Comparison floor (CLAUDE rule 26):** compare path lengths to the n-way per-pid min over
  ALL current CSVs, not one base.

## Gotchas baked in

- **No `torch.compile`** on this model (CLAUDE rule 22: hangs on custom SDPA-with-attn_mask).
  Train eager-bf16.
- All output ASCII; `open(..., encoding="utf-8")` (rule 24).
- State enters via node features (slot knows its sticker, sticker knows its slot), NOT a
  per-state attention mask — that would be a (B,T,T) blow-up. All attention bias/masks static.
- `--teacher` must be a V-model (output_dim=1). AZ v4 V-only (saturating), not m05, unless
  the student will only ever pair with m05 (rule 15: Q distilled from teacher X misranks under Y).
```
