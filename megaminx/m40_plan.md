# m40 plan: m05 Bellman + L_upper

**Status:** queued (waiting for free GPU slot — Kaggle preferred after m37 finishes)
**Hypothesis:** asymmetric one-sided penalty `max(0, f(s) - L)^2` helps the model respect the provable upper bound `d(s) <= L` more tightly than `clip_upper` alone, breaking the cluster ceiling.

## Background

We've hit a robust cluster ceiling: every Bellman recipe variant lands at 50/51 / mean 89-100 on strat-5. The pattern across m17, m22, m26, m26b, m27, m28, m31, m34, m36, m38 strongly suggests the bottleneck is **information-bound** — random-walk depth labels are too noisy.

A peer proposed a four-term composite loss (`L_sup + λ_lip·L_lip + λ_upper·L_upper + λ_anchor·L_anchor`) targeting the noise from multiple angles. After analysis (see chat 2026-05-03), the only term that's clearly novel-to-us, free, and worth isolating is **L_upper**. The rest is either redundant (anchor, μ_d via solver-trace = m37, μ_d via BFS-d6 = m22 family) or already-tried-and-failed (L_lip ≈ m38 listwise rank loss).

m40 isolates L_upper as a single-variable test of the broader proposal.

## Recipe

```yaml
# Identical to m05_bellman_warm except for the new bellman.lambda_upper field.
bellman:
  warmstart_path: megaminx/models/m07_big_k80/epoch_3999.pt
  target_update_every_epochs: 10
  clip_upper: true   # already clips target to walk_depth
  clip_lower: true
  lambda_upper: 0.1  # NEW: penalize pred > walk_depth as well
```

Loss term added in `_train_bellman_step`:

```python
if bcfg.lambda_upper > 0:
    pred_rw = pred[: bs_rw.size(0)]                                     # RW portion only
    overshoot = (pred_rw.float().flatten() - bd_rw).clamp(min=0.0)      # 0 if pred <= L
    upper_loss = (overshoot ** 2).mean()
    loss = loss + bcfg.lambda_upper * upper_loss
```

BFS-d6 mixin portion is excluded — its targets are exact distances, so `pred ≈ true_d <= L` trivially.

## Run plan

1. **Primary** — `m40_upper_penalty.yaml` with `lambda_upper=0.1`. ~8h on Kaggle P100, ~5h on local 4090.
2. **Diagnostic during training**: log `mean(upper_loss)` per epoch. If always near zero from epoch 1, the term doesn't fire and the experiment is moot — bail at epoch 100.
3. **Eval**: standard strat-5 acceptance gate vs m05 (50/51 / mean 89.4).

## Decision matrix

| Outcome | Next |
|---|---|
| ≥51/51 AND mean ≤84.9 | Run m40b (λ=0.5), m40c (λ=0.01). Then m41 = m40 + L_anchor + L_lip combined. |
| 50-51/51 AND mean ∈ [85, 89] | Tied; re-validate seed 1 then drop. |
| <50/51 OR mean >95 | Recipe regression — dead end. The full proposal is unlikely to recover this. |
| `mean(upper_loss) ≈ 0` throughout | Term never fires — `clip_upper` alone already pushes pred below L. The L_upper idea is moot for our current setup. Drop without further ablation. |

## What this experiment can and cannot tell us

**Can tell us:** whether the prediction itself benefits from explicit upper-bound pressure (vs only the target being clipped).

**Cannot tell us:** whether L_lip, L_anchor, or `μ_d(L)` would help. Those need separate experiments — but L_anchor is mostly redundant with target-net-on-goal, L_lip is essentially what m38 did, and `μ_d(L)` is m37 if μ_d comes from solver traces.

## Implementation surface

Three small edits, all done:
1. `src/cayley/bellman.py`: added `lambda_upper: float = 0.0` to `BellmanConfig`, added the loss term inside the autocast block.
2. `megaminx/configs/m40_upper_penalty.yaml`: new config, identical to m05_bellman_warm except for `lambda_upper: 0.1`.
3. `megaminx/m40_plan.md`: this file.

No new training script needed — `05_bellman_refine.py` reads from yaml.

## When to launch

After m37 finishes on Kaggle (frees the slot), or after m39a finishes on local 4090. Whichever first.
