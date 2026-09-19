# ihes_tf_v1a: IHES picture-cube PieceTransformer Q head

Trained 2026-09-17 for the CayleyPy IHES cube (kaggle.com/competitions/cayleypy-ihes-cube).

## Contract

- **Input:** a (B, 72) integer state. Entry i is the facelet label (0..71) currently in slot i,
  in the convention of `puzzle_info.json`. Solved is the identity. A move applies as
  `new[i] = state[gen[i]]`.
- **Output:** (B, 18). `Q(s, a)` estimates the distance to solved of `apply(s, a)`. Lower is
  better. The column order is the move order of `puzzle_info.json`:
  `f0 -f0 f1 -f1 f2 -f2 r0 -r0 r1 -r1 r2 -r2 d0 -d0 d1 -d1 d2 -d2`.
- **Beam use:** score B parents once and take a global top-B over all (parent, move) pairs.
  `min_a Q(s, a) = d(s) - 1`.
- **Value head** (AZ): V(s) ~ d(s) from the same trunk. It is optional and can be used for
  qv-consistency.

## Architecture (all constants are also in `meta/*` inside the npz)

| | |
|---|---|
| tokens | 26 physical pieces + CLS = 27; pieces hold up to 4 facelets (`ihes_piece_layout.json`) |
| embedding | (max_piece_size x 72) value table, folded with piece_projection; piece position + type embeddings |
| trunk | d_model 256, 8 heads, 4 pre-norm blocks, ff 1024, LayerNorm eps 1e-5, CLS pooling |
| **activation** | **SiLU (x * sigmoid(x)), NOT ReLU** |
| params | 3,508,755 |

**The cube444 256M kernel hard-codes ReLU and cube444 constants** (6 colours, 96 slots,
24 actions, 56 pieces, max piece size 3). Loading this npz into that module unchanged runs
without error and gives wrong scores. Use `ihes_jax_q_models.py` (`kaggle_notebooks/tpu_beam_ihes_tf/`),
which reads `meta/silu` and the other constants. Its `apply_piece_transformer_mixed`
reproduces the kernel's CLS-only final block and HIGHEST-precision readout.

## Provenance

- checkpoint: `models/ihes_tf_a_e1100a_s3/step_02000.pt` (epoch 1100, bellman_step 2000)
- parity (JAX npz vs PyTorch, fp32, 656 states): see MANIFEST.json `parity`
- gate: 108-pid held-out gate, B=65536, 1 frame, endgame d6: total 2597 vs floor 2423 (+174), ties 36, solved 108/108; gate-54 at B=262,144, 1 frame, RTX 4090 bf16: total 1267 vs floor 1215 (+52), ties 29 (same setting: pre-Bellman parent 1285, arm-B stage-1 e800 1279, E6 V at B=65,536 1355); recipe: sparse-Q from scratch (Vlad's IHES sampler, k 2-23) + AZ value head, 1000 epochs lr 3e-4, 100 epochs lr 3e-5, Q-Bellman 2000 steps (d<=6 exact anchors, path states)
