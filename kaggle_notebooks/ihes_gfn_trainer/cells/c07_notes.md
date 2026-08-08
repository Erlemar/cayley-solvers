# Notes & honest results

**Output:** `/kaggle/working/train.eqx` (+ `config.json`) — feed these to the
companion **inference/beam notebook** (`cayleypy-ihes-gfn-beam`) to generate a
submission, or publish them as a dataset for others.

## What to expect (measured)

* The `iters = 200000` demo (~2–3 h on v3-8) gives a **working but under-trained**
  policy. The reference model (`800k` iters, v6e-8) beam-solves the whole test set
  and averages ≈ **25–26 moves/puzzle**.
* This is a **faithful GFN**, not a state-of-the-art solver. On the IHES cube a
  well-tuned GFN roughly **ties a strong value-guided beam** and lands a few
  percent above the near-optimal community solutions — consistent with the paper's
  own finding that GFN ≈ ties CayleyPy Cube. Its appeal is *method* (no distance
  regression; one forward pass scores all children) rather than raw score.

## Tuning tips (from a sweep)

* `reg_coef` (λ) is the one puzzle-specific knob and has a **single-peaked
  optimum** — here `5e-8`. Larger over-regularizes (fails to solve), smaller
  under-shortens. The paper's rule of thumb: *pick the largest λ that still solves*.
* `eps_explore` **must be 0** for cube-family puzzles. At this graph size any
  forward exploration collapses `P_B` (an `eps=0.1` control failed to solve
  anything). Exploration is only needed on much larger graphs.
* `nmax` (training trajectory length) can be well below the diameter and still
  generalize — the paper uses 24 for 3×3×3; 30 here.

## References

* Morozov, Maksimov, Tiapkin, Samsonov, *Learning Shortest Paths with Generative
  Flow Networks*, 2026 — [arXiv:2603.01786](https://arxiv.org/abs/2603.01786),
  code [github.com/GreatDrake/gfn-pathfinding](https://github.com/GreatDrake/gfn-pathfinding).
* Malkin et al., *Trajectory Balance*, NeurIPS 2022.
