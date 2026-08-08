# Training a GFlowNet solver for the IHES Picture Cube

This notebook **trains a Generative Flow Network (GFlowNet)** to solve the
[CayleyPy SuperCube: IHES Picture Cube](https://www.kaggle.com/competitions/cayleypy-ihes-cube)
on a **TPU v3-8** with JAX + Equinox.

It implements *Learning Shortest Paths with Generative Flow Networks*
(Morozov, Maksimov, Tiapkin, Samsonov, 2026 — [arXiv:2603.01786](https://arxiv.org/abs/2603.01786)),
adapted to the IHES cube (72 facelets, 18 generators, group order ≈ 2.13·10²⁴).

## How GFN solving differs from a value/beam heuristic

The usual pipeline learns a **distance-to-solved** `V(s)` and beam-searches down it.
GFN instead flips the graph — the **solved state is the source**, every state is
terminal with uniform reward, so the partition function `Z = |group|` is known
exactly. Two policies are learned jointly:

* a **forward** policy that *scrambles* away from solved, and
* a **backward** policy `P_B` that *solves*.

A theorem (Morozov et al.) shows that **minimizing total flow forces `P_B` onto
shortest paths only** — geodesy falls out as a flow property, with *no distance
regression anywhere*. Training uses a regularized **trajectory-balance** loss.
At inference (companion beam notebook) you beam-search on cumulative `log P_B`.

## What this notebook produces

A trained `P_B` checkpoint (`train.eqx`) you can feed to the companion
**inference/beam notebook** to generate a submission. The exact hyperparameters
below were found by a sweep (`λ = 5·10⁻⁸`, `eps_explore = 0`); see the final
"Notes & honest results" cell for what to expect.
