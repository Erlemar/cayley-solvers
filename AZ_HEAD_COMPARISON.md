# AZ dual-head: megaminx vs tetraminx

How the AlphaZero-style policy head was trained and used on each puzzle.
Written 2026-08-24 from the code and the two EXPERIMENTS/HANDOFF files.

**Verdict: the trainers are near-identical (one is an explicit port of the other);
the USAGE is opposite. Megaminx extracted the policy head and searched with it.
Tetraminx dropped it at export and never wired it into search.**

---

## 1. Shared recipe

`tetraminx/scripts/21_train_az.py` states in its docstring that it is a port of the
megaminx AZ v4 recipe (via `scripts/11_train_az_cube.py`). Both train a
`ResMLPGFlowNet` with:

- shared trunk, warm-started from a Bellman V checkpoint (`--warmstart-trunk`),
  with shape-mismatched tensors skipped LOUDLY (megaminx rule 18);
- **value head trained exactly like a standalone Bellman V**: random walks +
  BFS-d6 mixin (10%) + V0/d=1 anchors (32/4) + target net refreshed every 10 epochs;
- **policy head trained by plain cross-entropy** on the action taken at each state
  of an existing solution path;
- loss `alpha * CE + beta * MSE`, both defaulting to 1.0, plus an optional
  `--gamma-path-value` term.

The value head's beam usability is the existence proof that the recipe works; the
policy head rides along for free.

---

## 2. Training differences

| | megaminx `71_train_az_v3.py` | tetraminx `21_train_az.py` |
|---|---|---|
| policy labels from | our OWN solver's 78,029 submission (`67_build_az_dataset.py`) | the PUBLIC FLOOR csv, 29,622 moves (`20_build_az_dataset.py`) |
| policy augmentation | none; v2's 23-sibling expansion is VALUE-side only (policy target stays on-path) | **x48** -- 24 spatial symmetries + inverse antisymmetry, ~1.42M samples, every variant replay-asserted before emission |
| `--k-max` (random walk) | 80 | 32 (tetraminx diameter ~27) |
| rw / policy batch defaults | 4096 / 512 | **8192 / 1024**, hard-coded |
| trunk | 2048,512 x2 res blocks | same |
| policy top-1 | 50.7% (m_az_v3) | 48.7% (taz_v1) |
| params | ~6M | 5.0M |

Two things worth noticing:

- **The batch-size row is CLAUDE.md rule 13 turned into code.** The megaminx script
  still defaults to 4096/512 -- the values whose 4x gradient steps memorised the
  policy (top-1 99% by ep 99) and destroyed V calibration on the AZ v4 relaunch.
  Tetraminx bakes in the good 8192/1024 launch values and cites the rule in a comment.
- **Symmetry augmentation on the policy is tetraminx-only.** On megaminx, symmetry
  augmentation of the V head was separately REJECTED (m31); it was kept for the
  Q-shortlister. Tetraminx applies it to the AZ policy labels.

### Megaminx needed three attempts; tetraminx skipped to the winner

| model | dataset | bench | verdict |
|---|---|---|---|
| m_az_v1 | path states only, joint CE + MSE | 0/10 | REJECTED -- V calibrated only on path states; the beam expands 24 children, 23 unseen |
| m_az_v2 | + 23 siblings/state labelled by teacher V | 0/10 | REJECTED -- target conflict between path `remaining_len` and sibling `V_teacher` |
| m_az_v3 | Bellman V recipe + policy CE | 10/10, total 943 | **ACCEPTED** -- value calibration via Bellman is the load-bearing piece |

Tetraminx went straight to the v3 recipe.

---

## 3. Usage -- where they diverge

### Megaminx: policy head extracted and searched with

`scripts/103_export_az_pi_only.py` pulls the policy head into a standalone
`ResMLPDistance(output_dim=24)` -> `models/m_az_v4_pi_only.pt`. Then, in `03_solve.py`:

| mechanism | flag | verdict |
|---|---|---|
| memoryless per-step policy penalty | `--policy-model --lambda-policy` | **REGRESSED** |
| **PHS cumulative path score** `V(child) + w * sum_{t<=d}(-log pi(a_t\|s_t))` | `--phs-cumulative` | **VALIDATED, not deployed** -- at w=0.03, -67 moves over 50 pids (-1.34/pid, difficulty-monotonic), but beats the merged best on only 3/50 pids (-13); min-merge already absorbs the diversity |
| subgoal proposer (V-gating every k steps) | `scripts/100_subgoal_ab.py` | FALSIFIED -- k=2 solves 2/6 and is +80 moves; width does not rescue it |
| AZ v5 geodesic-imitation policy head as shortlister | `scripts/130`/`131` | FAILED the recall-by-depth gate (0.05-0.5 vs 0.99 target) |

The distinction that mattered: **memoryless local penalty regresses, cumulative
path-history score helps.** Same head, same weight scale -- only the accumulator differs.

Caveat recorded at the time: under `--sym-ensemble` the AZ v4 pi sees rotated
(non-rot-augmented) states on 3 of 4 rotations, so it is partly OOD there.

### Tetraminx: policy head discarded at export

`scripts/22_export_az_v_only.py` keeps `embedding./input_stack./res_blocks.` and maps
`value_head -> head`; **`policy_head` and `log_Z` are dropped**. The resulting
`taz_v1_v_only.pt` is the scorer for every TPU beam (`v8val` 1M, `frames_b8M_f4` 8M x 4,
`wide_b33M` 32M).

HANDOFF is explicit: *"The policy head was NOT wired into search (untested here;
historically regresses)"*, and the launch plan said *"Do NOT wire the policy head in as
a child-orderer without a separate A/B -- that is the configuration that has repeatedly
regressed (megaminx qshort, all-neighbor Q head)."*

**Do not misread `30_solve.py --policy-checkpoint`**: that flag exists, but it is fed the
**frontier-regret** policy head (`models/frontier_verified_policy_head_v0/`) for greedy
rollout steps with confidence-gap recovery. It is a different model, not the AZ pi.

Where the tetraminx AZ model did pay off is the **value** head, and as a diversity
source rather than a better scorer:

| | total (13 common pids) |
|---|---|
| tv0 ep24 | 437 |
| taz_v1 | 434 |
| **min-merge of the two** | **421** |

Alone it is a wash (-3, noise); min-merged it is worth **-1.00 move/pid**.

---

## 4. The shared conclusion

Both puzzles converge on the same finding from opposite directions:
**V's per-child ordering is already near-optimal at 6M, so no policy head beats it as a
child-orderer or shortlister.** Megaminx established this the expensive way
(qshort regression, all-neighbor Q head, frontier-regret, subgoal, geodesic-imitation --
all neutral or worse); tetraminx inherited the conclusion and spent the GPU on solving
instead.

The only formulation of the policy head that measurably helped anywhere is the
**cumulative** PHS term on megaminx -- and even that lost to min-merge diversity at
deploy time.

---

## 5. File pointers

```
megaminx/scripts/67_build_az_dataset.py      policy labels from our own submission
megaminx/scripts/69_build_az_dataset_v2.py   + off-path siblings (value-side only)
megaminx/scripts/71_train_az_v3.py           the trainer
megaminx/scripts/95_export_az_v_only.py      V-only export
megaminx/scripts/103_export_az_pi_only.py    pi-only export  <-- no tetraminx equivalent
megaminx/scripts/03_solve.py                 --policy-model / --lambda-policy / --phs-cumulative
megaminx/scripts/100_subgoal_ab.py           pi as subgoal proposer (falsified)

tetraminx/scripts/20_build_az_dataset.py     floor labels, x48 sym/antisym augmented
tetraminx/scripts/21_train_az.py             the port
tetraminx/scripts/22_export_az_v_only.py     V-only export; policy_head + log_Z DROPPED
tetraminx/scripts/30_solve.py                --policy-checkpoint = frontier-regret head, NOT AZ pi
```

Related: `megaminx/EXPERIMENTS.md` (2026-05-10/11 AZ v1/v2/v3, 2026-05-24/25 PHS,
2026-07-21 geodesic pi), `tetraminx/HANDOFF.md` (AZ dual-head verdict; TPU scorer table),
memories `az-tb-dual-head-findings`, `az-v4-breakthrough`,
`phs-cumulative-validated-marginal`, `allneighbor-qhead-rejected`.
