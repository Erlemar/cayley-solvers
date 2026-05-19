# m46 — Transformer at scale on GCP (scoping)

**Status**: scoped, not yet started. Multi-day commitment.

## Background

m18 (Kaggle, 2026-04-26) trained a 4-layer Transformer encoder, d_model=128,
824k params. Loss converged from MSE 548 → 59 over 190 epochs (~225s/epoch on
P100), but the kernel hit Kaggle's 12h kill limit before reaching MLP-equivalent
training depth (m07 used 4000 epochs at 1.2s/epoch). Skip-evaluated per user;
filed as "tractable but per-wall infeasible on Kaggle".

GCP L4 has no wall kill, so we can run the transformer to convergence.

## Proposed configuration

Aim for ~30M params for capacity comparable to m45 (49M MLP) but in transformer
inductive bias.

**Architecture**: encoder stack on the 120-element state token sequence

| Knob | Value | Rationale |
|---|---|---|
| d_model | 512 | Wide enough for non-trivial representations |
| n_layers | 8 | Match m05's effective depth (2 ResBlocks ≈ 8 transformer-block-equivalent) |
| n_heads | 8 | Standard ratio (head_dim=64) |
| ffn_dim | 2048 | Standard 4× d_model |
| dropout | 0.0 | Match MLP regime (m05 had no dropout) |
| pos encoding | learned positional embedding for 120 tokens | Absolute since tokens are sticker positions |
| input embedding | 120 classes × 32 dim | Each sticker as a class token |
| readout | mean-pool + linear → 1 (V) | Or [CLS]-style |

**Total params**: ~30M (verify on instantiation).

## Training plan

Walk-depth pretrain (matches m07 / m18 pattern):
- n_epochs: 4000 (or until convergence)
- samples_per_epoch: 1M
- batch_size: 4096 (transformer is memory-heavier than MLP)
- k_max: 80
- lr: 5e-4 with linear warmup 1000 steps + cosine decay
- AMP bf16, compile_model=True

**Wall estimate** (worst case; assume 100s/epoch on L4):
- 4000 epochs × 100s = 110h ≈ 4.5 days
- That's the WORST case — m18 was 225s/epoch on slower P100.
- Realistic: 50-80s/epoch on L4 → 55-90h ≈ 2.3-3.7 days

Bellman (after pretrain):
- 500 epochs × ~120s = 17h
- Or skip if pretrain alone hits acceptable strat-5

Strat-5 eval: ~80-100 min on L4

## Open code items

1. **TransformerForDistance model class** — need to check if `cayley/model.py` has
   one or write from scratch. Quick scan first.
2. **Training script** — `02_train.py` may handle transformer if model class
   plugs into the training loop. Verify.
3. **Bellman support** — `train_bellman` uses `_bellman_targets` which calls
   `target_model(states)`. Should be transformer-agnostic if the model has the
   same forward signature.

## Risks / gotchas

1. **Per-wall infeasibility**: at 100s/epoch, 4000 ep is 4.5 days. Monitor cost
   on GCP; consider stopping early if loss plateaus. Or train fewer epochs
   (500-1000) and accept lower convergence.
2. **CayleyPy paper claim**: "transformers fail at n>15." Megaminx is n=120.
   m18 partially refuted this (loss DID converge), but maybe it never reaches
   MLP-equivalent quality. Possible null result.
3. **Memory**: 30M transformer + activations at batch 4096 may be tight on L4
   (24 GB). Reduce batch to 2048 if OOM.

## When to launch

After m45 pretrain finishes on GCP (~12h from m45 launch). Sequencing:
1. m45 pretrain (12h GCP)
2. m45 Bellman (2h GCP)  
3. m45 strat-5 eval (~100 min GCP)
4. ↓ If m45 lands at cluster, then start m46 transformer (free GCP for ~3-4 days)
5. m46 pretrain (3-4 days GCP) → Bellman → strat-5

If m45 BREAKS the cluster, m46 is less urgent — we'd focus on scaling MLP further.

## Decision

Hold m46 launch until after m45 result lands. We'll know in ~14h whether
"capacity at 50M MLP" matters; that information shapes whether transformer
is worth a 4-day commitment.
