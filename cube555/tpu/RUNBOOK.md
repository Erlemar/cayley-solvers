# cube555 TPU beam — build, verify, push, run

JAX SPMD beam search for the CayleyPy 5x5x5 picture cube, sharded across all 8 cores
of a Kaggle v5e-8, scored by the 30-wide `ResMLPQ` Q head.

Ported from `tetraminx/kaggle_notebooks/tpu_beam_tetraminx/`. The search engine is
that kernel's; the transformer is gone (cube555 never trained one), the state dtype
changed, and four search defaults differ because cube555 measured them differently.

    jax_model.py                JAX ResMLPQ + PyTorch->JAX loader  (no PieceTransformer)
    jax_beam_spmd_v_only.py     the SPMD beam engine   (uint8 states, PACK_SIZE 160)
    make_artifacts.py           builds + verifies the Kaggle dataset
    build_notebook.py           generates the .ipynb and its kernel metadata
    test_parity.py              JAX forward == PyTorch forward
    test_beam_cpu.py            the kernel, 8 simulated CPU devices, 4 cases
    test_notebook_cpu.py        the NOTEBOOK's own cells, end to end
    kaggle_dataset/             staged artifacts (477 MB)

---

## 1. The two changes that matter

**State dtype: `int8` -> `uint8`.** cube555 sticker classes run 0..149; `int8` holds
-128..127, so 128..149 wrap negative. Measured on a real batch: **5,632 of 38,400
values (14.7%)**. The state stays injective (150 < 256) so hashing, dedup, routing and
walkback all keep working on wrapped values — the casualty is the embedding lookup.
How loudly that fails is version-dependent (`.astype(int8)` wraps *silently* on
numpy 2.4.4/jax 0.10; `np.asarray(list, int8)` range-checks), which is exactly why it
is pinned rather than left to crash. Every state array routes through `STATE_DTYPE`
in one place; move/rank arrays stay signed because they use `-1` sentinels.

A **matched negative control** was run: reverting the engine to `int8` fails on this
stack. Do not remove that pin because "it seems to work".

**`PACK_SIZE: 96 -> 160`.** A bucket record is state (150) + parent_local (4) +
move (1) + bf16 score (2) = 157 bytes.

Two things needed no change: `BPTR_MOVE_BITS = 5` already covers 30 generators (max
32), and the `all_to_all` bucket was already `uint8`.

## 2. Search defaults that differ from the tetraminx kernel

| knob | tetraminx | cube555 | why |
|---|---|---|---|
| `B_GLOBAL` | 16M | **2^21** | width is measured NON-MONOTONIC here: 2^20 2/6, 2^21 6/6, 2^22 3/6, 2^23 0/1 |
| `QV_CONSISTENCY` | 0.3 | **0.0** | rejected twice on cube555: 3/3 -> 2/3 solved, and +8.0 mean on 5 matched pids |
| `HISTORY_DEPTH` | 1 | **4** | -14% path length; saturates at 4 (all 30 generators are odd -> parity-bipartite) |
| frame policy | first-of | **best-of** | the source project lists best-of-frames as untested and "strictly better" |

**The width default is the one to re-measure first.** That sweep ran on
`q555_20k_deployed.pt`, which the same document measures as the *worst of six*
checkpoints, and its stated mechanism ("a wider beam admits candidates the scorer
cannot rank") is about scorer quality. With `q555_2k_BEST` the optimum may move.
Run 2^20 / 2^21 / 2^22 on one pid set, one invocation, nothing else varying.

## 3. Build

    .venv/Scripts/python.exe cube555/tpu/make_artifacts.py
    .venv/Scripts/python.exe cube555/tpu/build_notebook.py

`make_artifacts.py` needs `anchors_d5.pt` in the handoff tree — it is deliberately
not shipped there and rebuilds in ~72 s:

    .venv/Scripts/python.exe cube555/scripts/11_build_anchors.py --depth 5

It emits the two `.py` files the kernel imports at runtime, `puzzle_info.json`, three
symmetry tables, the Zobrist-hashed `d<=5` endgame table, and four optimizer-stripped
checkpoints (`q555_pretrained` drops 297 MB -> 99 MB). It verifies before writing:
frame round trip on 8 real pids x 6 frames x 2 directions, and endgame lookups.

**The relabel direction is the silent trap.** The handoff ships two 30-wide move maps
differing only by direction. `sym_move_relabel_inv_48.npy` is the one a solver needs
(frame -> original); it is staged as `cube555_move_relabel_inv.npy`, named after what
it does. Using the other gives paths of the right length that solve the *conjugated*
state and not the original — indistinguishable from "frames don't help".

## 4. Verify — all three, before spending a TPU session

    .venv/Scripts/python.exe cube555/tpu/test_parity.py         # model
    .venv/Scripts/python.exe cube555/tpu/test_beam_cpu.py       # kernel
    .venv/Scripts/python.exe cube555/tpu/test_notebook_cpu.py   # notebook

Last full run, 2026-08-23, all `VERDICT : PASS`:

- **parity** — 24,757,807 params both sides; Q agrees to 2.7e-4 max in fp32; V agrees;
  `apply` and `apply_qv` bit-identical; argmin 256/256; full 30-action ordering
  identical on 255/256. In bf16 (the kernel dtype) argmin agrees 223/256, which is
  ordinary bf16 tie-breaking and matches the source project's documented 0-2 move
  drift.
- **kernel** — 4 cases on 8 simulated CPU devices with the real weights and the real
  10.7M-state table: non-streaming, `PARENT_CHUNK` streaming, a symmetry frame
  round-tripped back to the original state, and no-endgame. All found optimal-length
  paths (depth-9 scramble -> 9 moves) and all replayed.
- **notebook** — the generated cells executed against a staged `/kaggle/input`:
  3 pids x 2 frames, 6/6 solved and verified, per-pid-min submission written and
  independently replayed 3/3.

CPU emulation validates **plumbing only** — not TPU numerics, collectives or
throughput.

## 5. Push

Both are live (2026-08-23, private):

- dataset — <https://www.kaggle.com/datasets/artgor/cube555-tpu-artifacts>
- kernel  — <https://www.kaggle.com/code/artgor/cayleypy-cube555-tpu-beam-q>

**Push from PowerShell with Windows paths.** From Bash, MINGW mangles the `-p` path:
`-p /c/Users/...` staged the upload under `C_/Users/...` and the create exited 0 having
uploaded nothing but the first file. Each PowerShell call also needs the token
re-exported — Bash inherits an `export` across calls, PowerShell does not.

    $env:KAGGLE_API_TOKEN="..."; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    & .venv\Scripts\kaggle.exe datasets create  -p C:\Users\...\cube555\tpu\kaggle_dataset
    & .venv\Scripts\kaggle.exe datasets version -p C:\Users\...\cube555\tpu\kaggle_dataset -m "..."
    & .venv\Scripts\kaggle.exe kernels  push    -p C:\Users\...\cube555\tpu\push

`kernels push` needs a directory containing a file named exactly
`kernel-metadata.json`, so `push/` holds that plus a copy of the notebook;
`build_notebook.py` writes the canonical pair one level up.

The kernel title must slugify to its id or the push 400s with "title does not resolve
to specified id". One batch TPU session per account — a push fails while another
version is queued or running, and CLI 2.2.0 exposes no cancel.

Generate a variant without hand-editing cells:

    python cube555/tpu/build_notebook.py --pids 1034,1020        # TPU smoke
    python cube555/tpu/build_notebook.py --b-global 4194304      # a width-sweep arm

## 6. Run

Defaults solve pids 1034..1001 deepest-first at 2^21 x 2 frames. **Run deepest-first**:
pids 35..1034 are random walks whose baseline is length `pid-34`, so a deep pid saves
hundreds of moves while the 35 Santa pids (0..34) save **zero** — ~177 against their
93.6 baseline. Zero Santa pids belong in a merge.

Set `BASELINE_CSV` to a strong CSV placed in the asset dataset so the emitted
`submission.csv` is a per-pid min on top of real work rather than on top of the
competition sample.

Sizing: a child block is `B_LOCAL x 30 x 150` bytes — 2.1x the tetraminx block, so
`PARENT_CHUNK` becomes necessary at a *lower* width than in the sibling kernel. At
2^21 it is 1.2 GB/rank and no chunking is needed.

## 7. Known-unverified

- **No TPU run yet.** Everything above is CPU. The first session should be a 1-2 pid
  smoke at the shipped defaults before any long queue.
- `BLEND_CHECKPOINTS` is wired and untested on cube555. The measured multi-checkpoint
  result is a per-pid **min over separate runs** (24/24 vs 18/24 for the best single
  arm), which is not the same thing as averaging scores inside one step. Prefer
  separate arms plus a merge.
- The endgame membership test on device compares a 64-bit Zobrist hash only. At 10.7M
  entries and ~1e11 lookups a false positive is expected ~0.06 times per long
  campaign, and one killed a 29-hour PyTorch run. The host-side splice re-derives the
  descent and **fails soft** — it drops that frame and records
  `endgame_false_positive`, rather than raising. Pass `--endgame-depth 4` to
  `make_artifacts.py` to cut both the table and the exposure 24x.
