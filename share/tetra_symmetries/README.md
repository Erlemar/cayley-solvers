# Professor Tetraminx — 48 distance-preserving symmetry frames

For the [CayleyPy Professor Tetraminx](https://www.kaggle.com/competitions/cayley-py-professor-tetraminx-solve-optimally)
puzzle: 88 facelets, 24 generators (4 vertex axes `D/F/BL/BR` x layers `2/3/4` x
two directions), every generator of order 3.

Conjugating a state by one of these relabelings gives a **different state at the
same distance from solved**. That is free leverage in two places: training
augmentation, and running the same search in several frames and keeping the best.

## The number is 48

| | rotations | + mirrors | + inverse antisymmetry |
|---|---|---|---|
| **distinct on real states** | 12 | **24** | **48** |

- **24** = the full tetrahedral group Td. 12 rotations (even permutations of the
  4 vertex axes, move direction preserved) + 12 mirrors (odd permutations, move
  direction flipped).
- **48** = those 24 plus the **inverse antisymmetry** `s -> s^-1`. A state and its
  group inverse have the same optimal solution length, because every generator's
  inverse is also a generator — so solve `s^-1`, then reverse the path and invert
  each move.

> Enumerating *all* facelet permutations that conjugate the move set to itself
> gives **15,552 = 24 x 648**. The 648 is the centralizer — relabelings that
> commute with every move and therefore map every reachable state to *itself*.
> They are duplicates in effect and are dropped. Same structure as the IHES cube
> (1152 = 48 x 24) and the megaminx (720 = 120 x 6).

## Files

| file | what |
|---|---|
| `build_tetra_symmetries.py` | enumerate + verify the group, write the arrays below (<1 s) |
| `use_tetra_symmetries.py` | load + apply, for augmentation and search frames; has a self-test |
| `puzzle_info.json` | competition puzzle definition (input) |
| `tetra_symmetries.npy` | `(24, 88)` int16 — the relabelings `P` |
| `tetra_symmetries_inv.npy` | `(24, 88)` int16 — their inverses `P^-1` |
| `tetra_move_relabel.npy` | `(24, 24)` int16 — `sigma`, the induced move permutation |
| `tetra_symmetries_meta.json` | counts, move order, provenance |

Everything is regenerable from `puzzle_info.json` alone:

```bash
python build_tetra_symmetries.py     # rewrites the .npy / .json next to itself
python use_tetra_symmetries.py       # self-test
```

Expected self-test output:

```
loaded 24 spatial symmetries (48 frames with antisymmetry)
OK: 480 frame round-trips + 10 x 48 augmentation variants verified
```

## Conventions

Matching the competition's `puzzle_info.json`:

```
apply(s, g)[i] = s[g[i]]         states and generators compose as functions
conj(s)        = P_inv[s[P]]
```

The load-bearing identity, which is what makes frames usable:

```
apply(conj(s), m) = conj(apply(s, sigma(m)))
```

so a path `m_1..m_L` solving `conj(s)` maps to `sigma(m_1)..sigma(m_L)` solving
`s`. To push an existing solution *into* a frame, use `sigma^-1`.

## Use in search (sym-ensemble)

```python
from use_tetra_symmetries import load_group, load_puzzle, frame_list, to_frame, from_frame

g, pz = load_group(), load_puzzle()
best = None
for (k, inverted) in frame_list(4):            # 4 frames, see note below
    u = to_frame(state, k, inverted, g)        # solve THIS
    path = my_solver(u)                        # frame-local move indices
    orig = from_frame(path, k, inverted, g, pz["inv_idx"])
    if best is None or len(orig) < len(best):
        best = orig
```

**Use 4 frames, not 48.** Measured on this puzzle: `sym-frames 4` was worth
**−3.2 moves/pid** over a single frame, and `sym-frames 8` reproduced the 4-frame
result *byte for byte* — same totals, same per-pid lengths, same winning frames.
The productive set is `{identity, identity-inverse, k=1, k=1-inverse}`, which is
exactly what `frame_list(4)` returns. Frames 5–8 added nothing at 4x the wall.

**Inverse frames carry most of the wins.** On a 5-pid test, 4 of 5 pids were won
by a non-identity frame and 3 of those by an *inverse* frame (one pid went
38 → 31). The same pattern shows up on the IHES cube. If you only run two frames,
run identity and identity-inverse.

## Use in training (augmentation)

```python
from use_tetra_symmetries import load_group, load_puzzle, augment_path

g, pz = load_group(), load_puzzle()
variants = augment_path(state, solving_path_indices, g, pz)   # 48 (state, path) pairs
```

Every variant is replayed and asserted to reach solved before it is returned, so
a bad relabeling cannot silently poison a dataset. Applied to a 29,622-move
solution set this turns 29,622 samples into 1,421,856.

## Verification

`build_tetra_symmetries.py` checks, at build time:

- the conjugation identity `P^-1 . g_sigma(m) . P == g_m` for all 24 moves x 24 symmetries
- 480 conjugate-then-solve-by-relabel round-trips on random scrambles
- 20 inverse-antisymmetry round-trips
- the 24 relabelings are distinct and the first is the identity

`use_tetra_symmetries.py` independently re-checks all 48 frames round-trip in
both directions, plus the augmentation path.

## License / provenance

Derived only from the competition's public `puzzle_info.json`. Free to use.
