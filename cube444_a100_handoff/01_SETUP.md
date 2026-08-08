# 01 — Setup and correctness verification

## Layout

Unpack so the tree looks like this. The scripts resolve `PROJECT = parents[2]`, i.e. two
levels above `scripts/`, so keep the `cube444/` nesting.

```
<root>/
  src/cayley/          <- from code/src/cayley
  cube444/
    src/cube444/       <- from code/src/cube444
    scripts/           <- from code/scripts
    configs/           <- from code/configs
    data/              <- from code/data
    models/            <- created by training
    logs/
```

Concretely:

```bash
mkdir -p ~/cayley/cube444/{scripts,configs,data,models,logs} ~/cayley/src ~/cayley/cube444/src
cp -r code/src/cayley      ~/cayley/src/
cp -r code/src/cube444     ~/cayley/cube444/src/
cp    code/scripts/*.py    ~/cayley/cube444/scripts/
cp    code/configs/*.yaml  ~/cayley/cube444/configs/
cp    code/data/*          ~/cayley/cube444/data/
```

Every script does:

```python
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "cube444" / "src"))
```

## Environment

A100 (40 or 80 GB). Python 3.11+ is fine; the origin machine ran 3.14.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or cu124
pip install numpy pyyaml pandas tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available(),
           torch.cuda.get_device_name(0))"
```

No JAX needed — that was only for the TPU beam, which is not part of this package. The
PyTorch beam in `src/cayley/khoruzhii_search.py` is what you will use.

**ASCII-only output and `encoding="utf-8"` on every `open()`.** The origin machine was
Windows/cp932 and the codebase is written to that constraint; keeping it costs nothing and
means code can go back.

## Verify the port BEFORE training anything

```bash
cd ~/cayley
python cube444/scripts/test_symmetry.py     # the port gate -- must print SYMMETRY TABLES OK
python cube444/scripts/20_label_fix.py --self-test
```

Expected output of the first (verified from a clean tree):

```
generator inverse pairs: OK
rotation group order: 24  (expect 24)
sym(solved, R_k) == solved: 24/24
sym(apply(s,g),R) == apply(sym(s,R), relabel[k,g]): 500/500
round-trip (solve frame k -> translate -> solves original): 500/500
SYMMETRY TABLES OK
```

(`00_verify.py` and `73_verify_submission.py` are **submission** verifiers — they take
`--submission <csv>` and replay paths against `test.csv`. They are not the port gate; use
them after a solve, not before training.)

This checks the things that fail silently rather than loudly:

* **The symmetry identity.** On a colour cube a rotation is *not* just a permutation of
  slots — the colours move too. The correct form is

  ```
  sym(s, R) = color_map_R[ s[ rotation_R ] ]
  ```

  Applying only `s[rotation_R]` produces a state that looks plausible, verifies as a
  legal colouring, and is **wrong**. It will silently degrade every symmetry-frame beam.
  `rotations_24.npy` and `color_maps_24.npy` are shipped in `data/`.
* **Move application convention**: `apply(s, m) = s[gen[m]]`.
* **`num_classes = 6`.** See gotcha #1 in `06_GOTCHAS.md`.

If `test_symmetry.py` does not print `SYMMETRY TABLES OK`, stop. Nothing downstream is
meaningful.

## Sanity numbers for this puzzle

Have these in mind; they are how you tell a broken run from a slow one.

| quantity | value | how it was obtained |
|---|---|---|
| state size | 96 stickers | `puzzle_info.json` |
| colours | 6 (x16 each) | — |
| generators | 24 | `puzzle_info.json` |
| branching factor | **19.18** | measured, dead stable with depth |
| state count | 1.78e47 | counting |
| counting lower bound | **36.8 moves** for a random state | log_19.18(1.78e47) |
| best known mean | 44.79 (best public file) | 46,718 / 1043 |
| random walks mix at | length ~26-30 | measured |

That last line matters: **~1000 of the 1043 test pids are statistically uniform-random
deep states.** The "difficulty ladder" implied by pid order is only real for the first
~40. Do not build a curriculum around pid index.

## Data you must build (not shipped — too large)

`solved_ball_d6.npz` (603 MB) and `outer_ball_d8.npz` (774 MB) were deliberately left out.
Rebuild the one you need with `01_build_bfs.py` — see `02_DATA.md`. Everything else
(`puzzle_info.json`, `test.csv`, the 24 rotations and colour maps) is shipped.
