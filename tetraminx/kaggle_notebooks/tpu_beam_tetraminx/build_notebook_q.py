"""Build the Q-head Kaggle TPU notebook -- the CURRENT best tetraminx recipe.

DERIVED from build_notebook.py by string-patching its cell constants rather than
forked, so a fix to the shared kernel/report/solve code lands in both notebooks. The
only things this file changes are the model stack and four search settings.

What differs from the V-only notebook, and the measurement behind each (all on the
15-pid stratified set unless stated, 2026-08-02/03):

  MODEL   PieceTransformer Q head (mx_tfaz_ep1500.pt), not the AZ ResMLP V.
          ep1500 is the best checkpoint measured: 424 vs ep1200 426, ep1300 428,
          ep1400 431. Scores all 24 children from ONE forward on the parent
          (~22x cheaper per beam step than per-child V), which is what buys the width.

  BLEND   + ResMLP+AZ at 0.8/0.2. The ResMLP is ~1/17 of the transformer's forward
          cost, so the blend runs at ~1.07x, not 2x. Measured 435 -> 433 at 1M.
          The two heads share a distance scale almost exactly (std ratio 1.000),
          which is what makes raw averaging legitimate.

  QV      qv_consistency 0.3: score = Q + lam*|Q - (V(s)-1)|, both heads from ONE
          trunk pass, so it is free. Measured 435 -> 431 with history on, and it
          ADDS to history_depth rather than duplicating it (~90% additive).

  FRAMES  [(0, False), (1, True)] -- k0f + k1i, TWO frames not four. Replaying the
          16M x4 run's per-frame logs: k1i alone 405, k0i 408, k0f 411, k1f 413;
          best PAIR 400, best TRIPLE 400, all four 400. Frames SATURATE AT 2 -- the
          3rd and 4th bought exactly zero. The winning pair mixes both axes (one
          forward + one inverse, different rotations); taking the two best single
          frames (k1i+k0i) would be the wrong heuristic.

  WIDTH   16M. The 1-frame width curve is 1M 432 -> 4M 419 -> 8M 417, i.e. exhausted
          past 4M, but width still compounds with frames: 16M x2 is the config that
          produced -14 vs the standing best on 15 pids (8 wins, 0 losses) and
          -1.75/pid on the long tail.

  HISTORY 1, unchanged -- but note h4 measured BYTE-IDENTICAL to h1 and h0 is only
          +1, so do not raise it expecting gains.

Kaggle runs v5e-8, roughly half a v6e-8, so budget ~2x the GCP wall: about 25 min
per pid per frame at 16M, i.e. ~10 pids per 9-hour session at two frames. Lower
B_GLOBAL to 8M to roughly double the pid count if breadth matters more than depth.

    .venv/Scripts/python.exe tetraminx/kaggle_notebooks/tpu_beam_tetraminx/build_notebook_q.py
"""
from __future__ import annotations

import json
from pathlib import Path

import build_notebook as base

HERE = Path(__file__).resolve().parent
SLUG = "cayleypy-tetraminx-tpu-beam-q"

MD_INTRO = r"""# Professor Tetraminx beam search on Kaggle TPU -- Q head + cross-arch blend

Solves [CayleyPy Professor Tetraminx](https://www.kaggle.com/competitions/cayley-py-professor-tetraminx-solve-optimally)
with one **shared beam sharded across all 8 TPU cores**, scored by an
**all-neighbours Q head** instead of a per-child value function.

**Why a Q head.** A V model must expand every child and score it: `B x 24` forwards
per step. A Q head with `output_dim = 24` scores all 24 children from **one forward on
the parent**. Measured end-to-end at B=65,536: 578.7 ms -> 33.7 ms (**17.2x**). On this
puzzle width is the lever, so a cheaper step converts directly into a wider beam.

Note this kernel still MATERIALIZES and HASHES all 24 children per parent for owner
routing before its per-owner top-K -- the Q head saves the model forward, not the
expansion. (The PyTorch searcher additionally does "progressive top-k", scoring first
and hashing only the top candidates, which takes the same benchmark to 26.4 ms /
21.9x; that is not ported here.) Because children are fully materialized, the
`B_local x 24 x 88` array is the binding memory constraint -- 4.4 GB/rank at 16M.

**The objective that makes it work.** Walk `k` steps from solved, pick a pivot `p`,
and label exactly two of the 24 actions: `Q(s, undo) = p-1`, `Q(s, next) = p+1`. The
two labels always differ by exactly 2 with zero conditional variance, so the loss
**cannot** be reduced by flattening the gap between a good and a bad child -- which is
precisely what an MSE-on-walk-depth V loss does at depth (its measured gap decays
1.85 -> 0.44 from pivot band 1-4 to 30-40).

**Three inference additions**, each measured on the 15-pid stratified set:

| addition | effect | cost |
|---|---|---|
| cross-architecture blend (transformer 0.8 + ResMLP 0.2) | 435 -> 433 | ~1.07x |
| `QV_CONSISTENCY` -- `Q + lam*|Q - (V(s)-1)|`, both heads one trunk pass | 435 -> 431 | free |
| two frames `k0f + k1i` instead of one | 405 -> 400 (at 16M) | 2x |

**Frames saturate at two.** Replaying a 16M four-frame run per frame: any single frame
405-413, the best *pair* 400, the best *triple* 400, all four 400. The third and fourth
frames contribute nothing, so this notebook ships two. The winning pair mixes rotation
*and* inversion -- two inverse frames are more correlated than one of each.

**Exact endgame.** The goal test is "inside the BFS d<=6 table", not "solved", so the
beam stops ~6 steps early -- exactly where it is narrowest -- and the tail is the
table's optimal descent. Every emitted path is replayed against the ORIGINAL test state
before being recorded.

**Kaggle sizing.** v5e-8 is roughly half a v6e-8: ~25 min/pid/frame at 16M, so about
10 pids per 9-hour session at two frames. Drop `B_GLOBAL` to 8M for roughly double the
breadth.
"""


def patched_config() -> str:
    """base.CONFIG with the model stack and the four measured settings swapped in."""
    cfg = base.CONFIG
    old_model = '''V_CHECKPOINT = "taz_v1_v_only.pt"
HIDDEN_DIMS  = (2048, 512)         # must match the checkpoint
NUM_RES_BLOCKS = 2'''
    new_model = '''# --- model stack (this notebook's whole point) --------------------------------
# PieceTransformer Q head. ep1500 is the best checkpoint measured on the 15-pid
# sweep: 424, vs ep1200 426 / ep1300 428 / ep1400 431.
Q_CHECKPOINT = "mx_tfaz_ep1500.pt"
PIECE_LAYOUT = "piece_layout.json"
# Cross-architecture blend. The ResMLP is ~1/17 of the transformer's forward cost,
# so this is ~1.07x, not 2x. Set BLEND_CHECKPOINT = None to disable.
BLEND_CHECKPOINT = "mx_resmlp_az.pt"
BLEND_WEIGHTS    = (0.8, 0.2)      # (transformer, resmlp); probe plateau is 0.70-0.95
HIDDEN_DIMS      = (2048, 512)     # blend member (ResMLP) shape
NUM_RES_BLOCKS   = 2
# score = Q + lam*|Q - (V(s)-1)|, both heads from ONE trunk pass, so this is free.
# Needs an az_head checkpoint. 0.0 disables. Measured -4 and ADDITIVE with history.
QV_CONSISTENCY = 0.3'''
    assert old_model in cfg, "build_notebook.CONFIG model block moved"
    cfg = cfg.replace(old_model, new_model)

    # q_mode REQUIRES pack_v_score: a Q model returns 24 action scores, not a state
    # value, so it cannot rescore a bare state on the receive side -- the send-side
    # score must travel with the candidate.
    cfg = cfg.replace("PACK_V_SCORE = False", "PACK_V_SCORE = True   # REQUIRED by q_mode")

    old_frames = "FRAMES = [(0, True)]"
    new_frames = '''# TWO frames, k0f + k1i. Frames saturate at 2: replaying a 16M x4 run per frame gave
# single 405/408/411/413, best PAIR 400, best TRIPLE 400, all four 400. The pair must
# mix BOTH axes -- the two best single frames (k1i+k0i) are more correlated.
FRAMES = [(0, False), (1, True)]'''
    assert old_frames in cfg, "build_notebook.CONFIG FRAMES line moved"
    cfg = cfg.replace(old_frames, new_frames)

    # The base ships 48M (a GCP-only width that needs PARENT_CHUNK). Assert the anchor
    # rather than replace-and-hope: a silently-unmatched replace here would ship a
    # 48M notebook that OOMs on the first step, and the earlier version of this file
    # did exactly that by assuming an "8_388_608" literal that was never there.
    old_b = "B_GLOBAL   = 48 * 1024 * 1024"
    assert old_b in cfg, "build_notebook.CONFIG B_GLOBAL line moved"
    cfg = cfg.replace(
        old_b,
        "B_GLOBAL   = 16 * 1024 * 1024   # 16M: fits a 16 GB core unchunked "
        "(4.2 GB of children/rank)")
    cfg = cfg.replace("PIDS       = [990, 991, 992, 993]",
                      "PIDS       = [990, 991, 992, 993]   # ~10 pids fit a 9h v5e-8 session at 16M x2")
    return cfg


def patched_load() -> str:
    """base.LOAD with the transformer + blend loader in place of the ResMLP V loader."""
    load = base.LOAD
    load = load.replace(
        "from jax_model import load_params_from_pt, num_params",
        "from jax_model import (load_params_from_pt, load_piece_transformer_params_from_pt,\n"
        "                       make_blend, has_value_head, num_params)")
    old = '''v_params = load_params_from_pt(dataset_root / V_CHECKPOINT,
                              hidden_dims=HIDDEN_DIMS, num_res_blocks=NUM_RES_BLOCKS)'''
    new = '''v_params = load_piece_transformer_params_from_pt(
    dataset_root / Q_CHECKPOINT, layout_path=dataset_root / PIECE_LAYOUT)
if BLEND_CHECKPOINT:
    _rm = load_params_from_pt(dataset_root / BLEND_CHECKPOINT,
                              hidden_dims=HIDDEN_DIMS, num_res_blocks=NUM_RES_BLOCKS)
    v_params = make_blend([v_params, _rm], list(BLEND_WEIGHTS))
    print(f"blend: transformer {BLEND_WEIGHTS[0]:.2f} + resmlp {BLEND_WEIGHTS[1]:.2f}")
if QV_CONSISTENCY and not has_value_head(v_params):
    raise SystemExit("QV_CONSISTENCY needs an az_head checkpoint (no value_head.* found)")
_head = v_params["members"][0] if "members" in v_params else v_params
Q_MODE = int(_head["head_w"].shape[-1]) == 24
print(f"q_mode {Q_MODE}  qv_consistency {QV_CONSISTENCY}")
assert Q_MODE, "this notebook is the Q-head recipe; head width must be 24"'''
    assert old in load, "build_notebook.LOAD model-load block moved"
    load = load.replace(old, new)
    load = load.replace('_ck = dataset_root / V_CHECKPOINT', '_ck = dataset_root / Q_CHECKPOINT')
    load = load.replace('print(f"V checkpoint: {V_CHECKPOINT}',
                        'print(f"Q checkpoint: {Q_CHECKPOINT}')
    return load


def patched_solve() -> str:
    """base.SOLVE with q_mode and the consistency lambda threaded into solve()."""
    solve = base.SOLVE
    old = "                pack_v_score=PACK_V_SCORE, progress_every=PROGRESS_EVERY,"
    new = ("                pack_v_score=PACK_V_SCORE, progress_every=PROGRESS_EVERY,\n"
           "                q_mode=Q_MODE, qv_consistency=QV_CONSISTENCY,")
    assert old in solve, "build_notebook.SOLVE call site moved"
    return solve.replace(old, new)


def main() -> int:
    nb = {
        "cells": [base.cell(MD_INTRO, "markdown"), base.cell(base.SETUP),
                  base.cell(patched_config()), base.cell(patched_load()),
                  base.cell(patched_solve()), base.cell(base.REPORT)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = HERE / f"{SLUG}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = {
        "id": f"artgor/{SLUG}",
        # Kaggle requires the title to slugify to `id` -- keep them in step (CLAUDE.md 7d).
        "title": "CayleyPy Tetraminx TPU beam q",
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "dataset_sources": ["artgor/tetraminx-tpu-artifacts"],
        "competition_sources": ["cayley-py-professor-tetraminx-solve-optimally"],
        "kernel_sources": [],
    }
    (HERE / f"kernel-metadata-{SLUG}.json").write_text(json.dumps(meta, indent=2),
                                                       encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {HERE / f'kernel-metadata-{SLUG}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
