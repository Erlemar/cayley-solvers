# Notes

* **Output:** `/kaggle/working/submission.csv` — submit directly to the competition.
* **Width:** the IHES GFN policy is **width-saturated** — length stops improving
  well below 1M (verified: identical lengths at 16k, 64k, 1M, 3.7M even on the
  hardest puzzles). Start at `width = 65536` for a fast run; go wider only if some
  puzzles are unsolved. True multi-million widths need sharding the beam across all
  8 TPU cores (SPMD); a single core tops out near ~3.7M.
* **Expected quality:** the reference checkpoint solves the full set at ≈ 25–26
  moves/puzzle. That is a **valid, self-contained GFN solution** but a few percent
  above the best community solutions (≈ 21.8) — GFN ties, rather than beats, a
  strong value beam here. Use it as a decorrelated arm or a method reference.
* **Symmetry ensemble** (48 cube symmetries) shaves ~1–2 moves on some puzzles;
  it is omitted here for clarity — solve each of a few conjugated copies
  `P·s·P⁻¹` and keep the shortest, mapping moves back through the symmetry.

Companion trainer: **`cayleypy-ihes-gfn-trainer`**. Method:
[arXiv:2603.01786](https://arxiv.org/abs/2603.01786).
