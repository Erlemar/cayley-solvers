# 256M Tetraminx beam confirmed on Kaggle v5e-8

The private notebook [Tetraminx 256M Kaggle capacity](https://www.kaggle.com/code/artgor/tetraminx-256m-kaggle-capacity),
version 2, completed a full 268,435,456-slot shared-beam solve on one Kaggle v5e-8
host. PID 67 returned a 29-move path, independently replayed on the TPU host and
locally with the project puzzle implementation. The prior GCP v6e-8 256M run also
returned 29 moves; the new result demonstrates Kaggle capacity, not another
path-length improvement over that prior result.

| Measurement | Result |
|---|---:|
| Compiled program memory per chip | 12,583,625,216 bytes (11.72 GiB) |
| Full search wall | 16,766.67 s (4 h 39 m) |
| Search compilation | 465.8 s |
| Last-five median device step | 895.2 s |
| Entire validation and solve campaign | 19,984.05 s (5 h 33 m) |
| Search steps to endgame hit | 22 |
| Verified solution length | 29 |

The memory figure includes arguments, outputs minus aliases, temporaries, and
generated code reported by the compiler. It is not a sampled allocator peak.
The capacity probe and full puzzle execution both passed the 15 GB/chip guard.
The campaign total excludes allocation queue time.

The main change stores each state in 25 bytes: four bits identify each of the
50 pieces and its reachable orientation. Full 88-sticker states exist only in
bounded scoring, hash, and endgame tiles. Generator moves can be applied directly
to packed records. The exchange carries 25 state bytes, four parent/move bytes,
and a two-byte score. Ownership, scoring precision, selection, exact history,
endgame semantics, and wide ancestry are preserved.

The recipe is transformer ep1500 plus ResMLP Q at 0.8/0.2, Q/V consistency 0.3,
alpha 2, exact history 1, d6 endgame, inverse frame k0i, parent chunks 262144,
model batches 8192, 32 levels, and 64 GiB of host ancestry in /tmp. It uses
Kaggle's installed JAX/jaxlib 0.10.2 and libtpu 0.0.17. Version 1 failed before
search because its manifest incorrectly retained the GCP libtpu 0.0.42 pin;
the failure and corrected version are both preserved.

Correctness evidence includes 156 identical TPU search-output comparisons,
1,152,192 codec move checks across dataset/inverse/symmetry/padding states,
30,000 chained moves, and complete reconstruction/replay. The matched real-model
4M control and packed runs produced exactly the same 30-move path.

Compression has a measured cost: that matched 4M search took 207.45 s for the
control and 258.52 s packed, a 24.6% increase in this one-puzzle test. This is a
capacity improvement; a speed improvement has not been demonstrated. Timing
against the prior v6e run is not an isolated codec comparison because the
hardware and libtpu versions differ. The next useful work is reducing this
overhead while retaining the proven Kaggle memory budget.

Source and evidence are frozen under
`tetraminx/tpu_experiments/kaggle256_20260906_v2/`:
`packed/`, `notebook/`, `verification.json`, `completion.local.json`,
`result_source_lock.json`, and `downloaded_v2/`. The staged single-puzzle CSV is
`staged/verified_kaggle256_pid67.csv`. All 111 events bind the same immutable
manifest; completion validation has zero errors and warnings. The existing
production dataset was not replaced, and no competition submission was made.
