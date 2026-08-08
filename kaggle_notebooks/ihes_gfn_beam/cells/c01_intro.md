# GFlowNet beam solver for the IHES Picture Cube (inference)

This notebook loads a **trained GFlowNet backward policy** `P_B` and **beam-searches
on cumulative `log P_B`** to solve the
[CayleyPy SuperCube: IHES Picture Cube](https://www.kaggle.com/competitions/cayleypy-ihes-cube)
test set on a **TPU v3-8**, then writes a `submission.csv`.

It is the inference companion to **`cayleypy-ihes-gfn-trainer`** and implements
*Learning Shortest Paths with Generative Flow Networks*
([arXiv:2603.01786](https://arxiv.org/abs/2603.01786)).

## Why the beam is one forward pass per state

A GFN backward policy outputs a distribution over the moves of a state in a
**single forward pass**, so scoring all 18 children of a beam node costs one NN
eval — versus a value/distance model that needs one eval *per child*. The beam
keeps the top-`W` partial solutions ranked by the sum of `log P_B` along the path
(text-decoding-style beam search), and stops when a child equals the solved state.

## What this notebook does

1. Loads `train.eqx` + `config.json` from the `ihes-gfn-checkpoint` dataset
   (swap in your own trained checkpoint by forking + attaching your dataset).
2. Reads `test.csv` from the competition mount.
3. Runs a **memory-efficient device-resident beam** (int8 states, builds only the
   `W` selected children per step) at a configurable width up to a few million.
4. Verifies every path and writes `submission.csv`.
