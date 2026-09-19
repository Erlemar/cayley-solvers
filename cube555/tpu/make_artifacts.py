"""Build the `artgor/cube555-tpu-artifacts` Kaggle dataset from the handoff tree.

    .venv/Scripts/python.exe cube555/tpu/make_artifacts.py
    .venv/Scripts/python.exe cube555/tpu/make_artifacts.py --endgame-depth 4 --no-models

Emits into cube555/tpu/kaggle_dataset/:

    jax_model.py                    the JAX ResMLPQ port  (the kernel imports these
    jax_beam_spmd_v_only.py         the SPMD beam engine    two from dataset_root)
    puzzle_info.json                150 stickers, 30 generators
    cube555_sym.npy        (48,150) M      -- slot map
    cube555_sym_inv.npy    (48,150) Minv   -- its inverse
    cube555_move_relabel_inv.npy (48,30)   -- FRAME move -> ORIGINAL move
    bfs_endgame.npz                 exact d<=D ball as Zobrist hashes + depths
    q555_*.pt                       inference-only checkpoints (optimizer stripped)
    dataset-metadata.json

TWO THINGS THIS FILE EXISTS TO GET RIGHT.

1. WHICH RELABEL TABLE. The handoff ships four symmetry arrays and two of them are
   30-wide move maps that differ only by direction. `sym_move_relabel_48.npy` pushes
   an ORIGINAL path INTO a frame; `sym_move_relabel_inv_48.npy` brings a FRAME path
   BACK OUT, which is what a solver needs (`30_solve.py:312`). Picking the wrong one
   is silent: the beam still solves the conjugated state, the translated path still
   has the right length, and it simply does not solve the original -- indistinguishable
   from "frames do not help". So the file is named `_inv` after what it does, and
   `--verify` round-trips it on real pids before writing.

2. ZOBRIST, NOT THE MULTIPLY-HASH. The beam engine's endgame membership is
   `XOR_i ztab[i][state[i]]`, while the PyTorch solver uses `sum_i state[i]*hv[i]`.
   The table has to be built with the hash the consumer uses.

The false-positive budget, since the engine tests the HASH only: at 10.7M entries in
a 64-bit space a single lookup false-positives at ~5.8e-13, so ~1e11 lookups over a
long campaign expect ~0.06 of them. That is small but not zero -- one killed a 29-hour
PyTorch run -- which is why the notebook's splice is wrapped and non-fatal rather than
trusted. Passing --endgame-depth 4 cuts both the table (446,403) and the exposure 24x
at the cost of stopping one step later.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_HANDOFF = Path(
    "C:/Users/and-l/cayley/cube555_pull/cube555_handoff_2026_08_22/cube555")
CHECKPOINTS = ["q555_2k_BEST.pt", "q555_6k.pt", "q555_20k_deployed.pt",
               "q555_pretrained.pt"]


def strip_checkpoint(src: Path, dst: Path) -> tuple[int, int]:
    """Copy a checkpoint keeping only what inference needs.

    q555_pretrained.pt is 297 MB of which ~200 MB is optimizer + scheduler state;
    the other three are already lean. Keeping `config` costs nothing and is the only
    record of the recipe that produced the weights.
    """
    import torch

    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    keep = {k: ck[k] for k in ("model", "model_config", "config", "epoch",
                               "bellman_step", "init") if k in ck}
    torch.save(keep, str(dst))
    return src.stat().st_size, dst.stat().st_size


def build_endgame(anchors_pt: Path, out_npz: Path, move_names: list[str],
                  state_size: int, num_classes: int, seed: int = 555) -> dict:
    """anchors_dD.pt -> Zobrist-hashed, hash-sorted membership table."""
    import torch

    blob = torch.load(str(anchors_pt), map_location="cpu", weights_only=False)
    states = blob["states"].numpy()                      # (N, 150) uint8
    depths = blob["depth"].numpy().astype(np.int8)       # (N,)
    n = states.shape[0]
    max_depth = int(depths.max())
    assert states.shape[1] == state_size, states.shape
    assert states.max() < num_classes and states.min() >= 0

    rng = np.random.default_rng(seed)
    ztab = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                        size=(state_size, num_classes), dtype=np.int64)

    h = np.zeros(n, dtype=np.int64)
    step = 1 << 21
    for lo in range(0, n, step):                          # chunked: 10.7M x 150 gathers
        hi = min(lo + step, n)
        acc = np.zeros(hi - lo, dtype=np.int64)
        blk = states[lo:hi]
        for i in range(state_size):
            acc ^= ztab[i][blk[:, i]]
        h[lo:hi] = acc

    order = np.argsort(h, kind="stable")
    h, depths = h[order], depths[order]

    dup = int((np.diff(h) == 0).sum())
    print(f"  endgame d<={max_depth}: {n:,} states, {dup} internal hash collision(s)")
    if dup:
        print("  *** collisions inside the table: a lookup can return the WRONG depth. "
              "Rebuild with a different --zobrist-seed. ***")

    np.savez_compressed(out_npz, hashes=h, depths=depths, ztab=ztab,
                        move_names=np.asarray(move_names), max_depth=np.int64(max_depth))
    return {"n": n, "max_depth": max_depth, "dup": dup, "ztab": ztab,
            "hashes": h, "depths": depths}


def verify(handoff: Path, out: Path, eg: dict | None) -> int:
    """Round-trip the frame tables and spot-check the endgame on real pids."""
    info = json.loads((out / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"])
    G = np.stack([np.asarray(info["generators"][k], dtype=np.int64) for k in names])
    central = np.asarray(info["central_state"], dtype=np.int64)
    name_i = {n: i for i, n in enumerate(names)}
    INV = np.asarray([name_i[n[1:] if n.startswith("-") else "-" + n] for n in names])

    SYM = np.load(out / "cube555_sym.npy")
    SYM_INV = np.load(out / "cube555_sym_inv.npy")
    RELABEL = np.load(out / "cube555_move_relabel_inv.npy")

    tests, paths = {}, {}
    with open(handoff / "data" / "test.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tests[int(r["initial_state_id"])] = np.asarray(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
    sub = handoff / "submissions" / "cube555_submission_187780.csv"
    with open(sub, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            paths[int(r["initial_state_id"])] = r["path"]

    def to_frame(s0, k, inverted):
        t = np.argsort(s0) if inverted else s0
        return SYM_INV[k][t[SYM[k]]]

    def from_frame(path_idx, k, inverted):
        q = [int(RELABEL[k][m]) for m in path_idx]
        return [int(INV[m]) for m in reversed(q)] if inverted else q

    def replay(s0, idx):
        cur = s0
        for m in idx:
            cur = cur[G[m]]
        return cur

    # A known solution of s, pushed into frame k (and optionally inverted), must come
    # back out through from_frame() as a solution of s again. This is the exact
    # transform the kernel performs, run backwards on ground truth.
    ok = bad = 0
    for pid in [1034, 1000, 900, 700, 500, 300, 100, 50]:
        base = [name_i[m] for m in paths[pid].split(".") if m]
        s0 = tests[pid]
        assert np.array_equal(replay(s0, base), central), f"pid {pid} reference broken"
        for k in [0, 7, 19, 33, 41, 47]:
            for inverted in (False, True):
                u = to_frame(s0, k, inverted)
                # Build the frame-coordinate path the beam would have to find...
                fwd = base if not inverted else [int(INV[m]) for m in reversed(base)]
                inv_rl = np.argsort(RELABEL[k])          # ORIGINAL move -> FRAME move
                fpath = [int(inv_rl[m]) for m in fwd]
                if not np.array_equal(replay(u, fpath), central):
                    bad += 1
                    continue
                # ...then check the kernel's own translation undoes it.
                back = from_frame(fpath, k, inverted)
                ok += 1 if np.array_equal(replay(s0, back), central) else 0
                bad += 0 if np.array_equal(replay(s0, back), central) else 1
    print(f"  frame round trip (8 pids x 6 frames x 2 directions): {ok} ok, {bad} bad")

    n_fail = 1 if bad else 0
    if eg is not None:
        # The table must contain the solved state at depth 0 and every depth-1 child.
        def zhash(st):
            acc = np.zeros(st.shape[0], dtype=np.int64)
            for i in range(st.shape[1]):
                acc ^= eg["ztab"][i][st[:, i]]
            return acc

        def lookup(st):
            h = zhash(st)
            pos = np.clip(np.searchsorted(eg["hashes"], h), 0, eg["hashes"].size - 1)
            out = np.full(st.shape[0], -1, dtype=np.int64)
            hit = eg["hashes"][pos] == h
            out[hit] = eg["depths"][pos[hit]]
            return out

        d0 = int(lookup(central[None, :].astype(np.uint8))[0])
        kids = central[G].astype(np.uint8)
        dk = lookup(kids)
        # and a deep random state must MISS -- a table that hits everything is broken
        rng = np.random.default_rng(0)
        deep = central.copy()
        for _ in range(60):
            deep = deep[G[rng.integers(0, len(G))]]
        dmiss = int(lookup(deep[None, :].astype(np.uint8))[0])
        good = (d0 == 0) and bool((dk == 1).all()) and dmiss == -1
        print(f"  endgame lookup: solved->{d0}  30 children->{sorted(set(dk.tolist()))}"
              f"  depth-60 state->{dmiss}  {'OK' if good else 'FAIL'}")
        n_fail += 0 if good else 1
    return n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    ap.add_argument("--out", type=Path, default=HERE / "kaggle_dataset")
    ap.add_argument("--endgame-depth", type=int, default=5)
    ap.add_argument("--zobrist-seed", type=int, default=555)
    ap.add_argument("--no-models", action="store_true", help="skip the .pt copies")
    ap.add_argument("--no-endgame", action="store_true")
    args = ap.parse_args()

    hs, out = args.handoff, args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"handoff : {hs}\nout     : {out}")

    # ---- code the kernel imports from dataset_root ----
    for f in ("jax_model.py", "jax_beam_spmd_v_only.py"):
        shutil.copy2(HERE / f, out / f)
        print(f"  code    {f}")

    # ---- puzzle ----
    shutil.copy2(hs / "data" / "puzzle_info.json", out / "puzzle_info.json")
    info = json.loads((out / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"])
    state_size, n_gen = len(info["central_state"]), len(names)
    num_classes = int(max(info["central_state"])) + 1
    print(f"  puzzle  state_size={state_size} n_gen={n_gen} num_classes={num_classes}")
    assert (state_size, n_gen, num_classes) == (150, 30, 150)

    # ---- symmetry: rename to what the notebook reads, keeping direction explicit ----
    for src, dst in [("sym_slots_48.npy", "cube555_sym.npy"),
                     ("sym_slots_inv_48.npy", "cube555_sym_inv.npy"),
                     ("sym_move_relabel_inv_48.npy", "cube555_move_relabel_inv.npy")]:
        a = np.load(hs / "data" / src).astype(np.int64)
        np.save(out / dst, a)
        print(f"  sym     {dst}  {a.shape}  <- {src}")

    # ---- endgame ----
    eg = None
    if not args.no_endgame:
        anchors = hs / "data" / f"anchors_d{args.endgame_depth}.pt"
        if not anchors.exists():
            print(f"  [skip] {anchors.name} not built -- run "
                  f"`11_build_anchors.py --depth {args.endgame_depth}` first")
        else:
            eg = build_endgame(anchors, out / "bfs_endgame.npz", names,
                               state_size, num_classes, seed=args.zobrist_seed)
            print(f"  endgame bfs_endgame.npz "
                  f"({(out / 'bfs_endgame.npz').stat().st_size / 1e6:.0f} MB)")

    # ---- checkpoints ----
    if not args.no_models:
        for name in CHECKPOINTS:
            src = hs / "models" / name
            if not src.exists():
                print(f"  [skip] {name}")
                continue
            a, b = strip_checkpoint(src, out / name)
            print(f"  model   {name}  {a / 1e6:.0f} MB -> {b / 1e6:.0f} MB")

    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "cube555 tpu artifacts",
        "id": "artgor/cube555-tpu-artifacts",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    print("\nverifying:")
    n_fail = verify(hs, out, eg)
    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"\n{len(list(out.iterdir()))} files, {total / 1e6:.0f} MB")
    print("VERDICT :", "PASS" if n_fail == 0 else f"FAIL ({n_fail})")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())
