# IHES Picture Cube — 96 distance-preserving symmetries

The 3x3x3 picture cube (72 facelets, 18 moves `f/r/d` x layers `0/1/2` x {CW,CCW})
has a symmetry group you can exploit for free in training and search.

## The number is 96

| | rotations | + mirrors | + inverse antisymmetry |
|---|---|---|---|
| **distinct on real states** | 24 | **48** | **96** |

- **48** = the full octahedral group (24 whole-cube rotations + 24 mirror images).
  Each maps a scrambled state to an equidistant one (`s -> P . s . P^-1`).
- **96** = those 48 plus the **inverse antisymmetry** `s -> P . s^-1 . P^-1`
  (a state and its group-inverse have the same optimal solution length, because
  every move's inverse is also a move).

> Note: enumerating *all* facelet permutations that conjugate the move set to
> itself gives **1152 = 48 x 24**. The factor 24 is the centralizer — relabelings
> that commute with every move, so they map every reachable state to *itself*.
> They are duplicates in effect and are dropped. (Same structure as the megaminx:
> 720 = 120 x 6, effective 120.)

## Files

| file | what |
|---|---|
| `build_cube_symmetries.py` | enumerate + verify the group, write the arrays below (≈8 s) |
| `use_cube_symmetries.py` | load + apply: training augmentation and search sym-ensemble, with a self-test |
| `puzzle_info.json` | competition puzzle definition (input) |
| `cube_symmetries.npy` | `(48, 72)` the symmetry perms `P` |
| `cube_symmetries_inv.npy` | `(48, 72)` their inverses `P^-1` |
| `cube_symmetry_group.npz` | the 96: `perms`, `perms_inv`, `is_antisymmetry` |
| `cube_symmetries_meta.json` | counts + provenance |

Only `numpy` is required to build; the torch augmenter in `use_*` is optional.

## Run

```bash
python build_cube_symmetries.py --puzzle-info puzzle_info.json --out-dir .
python use_cube_symmetries.py   --data-dir .  --puzzle-info puzzle_info.json   # self-test
```

## Use 1 — training augmentation

For each `(state, depth)` sample, apply a random one of the 96; the depth label
is unchanged, giving 96x coverage for free.

```python
from use_cube_symmetries import load_group, augment_numpy
import numpy as np
group = load_group(".")
aug = augment_numpy(states, group, np.random.default_rng(0))   # states: (B,72) int
```

`augment_torch(states, group_t, generator)` is a drop-in for a 24-rotation
augmenter but over the full 96.

## Use 2 — search sym-ensemble

Solve several relabeled copies of a scramble, keep the shortest, map each
solution back with the move-relabel `sigma` (where `P^-1 . g_m . P = g_sigma(m)`):

```python
from use_cube_symmetries import load_group, apply_symmetry, move_relabel
group = load_group(".")
P, Pinv = group["sym"][i], group["sym_inv"][i]
s_rot = apply_symmetry(s, P, Pinv)                  # transform the scramble
sigma = move_relabel(P, Pinv, generators)           # move-name -> move-name
# after solving s_rot to a move list Q':  [sigma[m] for m in Q'] solves the original s
```

## Verified invariants

`build_*` checks group closure, identity presence, and that the 24 rotations are
a subgroup. `use_*` checks the 96 act as 96 distinct maps on a real scramble and
that a relabeled solution actually solves the transformed scramble.
