# ihes_tf_v1b: IHES picture-cube PieceTransformer Q head

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

- checkpoint: `models/ihes_tf_b_e1300a_s3/step_02000.pt` (epoch 1300, bellman_step 2000)
- parity (JAX npz vs PyTorch, fp32, 656 states): see MANIFEST.json `parity`
- gate: 108-pid held-out gate, B=262144, 1 frame, endgame d6: total 2511 vs floor 2423 (+88), ties 65, solved 108/108
- Same 108-pid gate, B=262,144, A100 (the selection run): v1a 2537, arm-B e800 + Bellman 2533, Bellman rounds 2/3 of this model 2513 (plateau).
- Width scaling on gate-54 (floor 1215), 1 forward frame, RTX 4090 bf16: B=65,536 -> 1283 (+68, 22 ties); B=262,144 -> 1255 (+40, 35 ties); B=1,048,576 -> 1243 (+28, 41 ties; all 7 length-24 pids at the floor). Gains shrink per 4x of width.
- At wide beams also run the INVERSE frame: the short length-21 pids stay +12 at 2^20 with the forward frame alone.
- Recipe (arm B): sparse-Q from scratch + 48-frame symmetry coverage (4 rows) + exact d<=5 anchors + pivot tilt 0.5, k_max 26, AZ value head; 1200 epochs at lr 3e-4 (batch 2048 x 256 steps); 100 epochs at lr 3e-5; Q-Bellman 2000 steps (exact d<=6 anchors, path states, 108 gate pids held out).
