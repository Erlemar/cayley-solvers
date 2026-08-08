"""CPU shape-validation for the tetraminx TPU beam kernel (88 facelets, 24 moves).

Kaggle TPU time is scarce and the JAX-on-TPU notes are explicit that CPU emulation
does NOT reproduce TPU-specific bugs -- but it DOES catch the whole class of
shape/packing errors that a 72->88, 18->24 port can introduce (bucket record
layout, backpointer bit budget, padding detection, path reconstruction).  Run
this before spending a Kaggle session.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    .venv/Scripts/python.exe tetraminx/kaggle_notebooks/tpu_beam_tetraminx/cpu_smoke.py \
        --checkpoint tetraminx/models/<ckpt>.pt --scramble 8 --b-global 8192
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
# x64 must be on BEFORE jax is imported -- the kernel hashes states into int64.
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from jax_beam_spmd_v_only import beam_solve_v_only_spmd_packed, make_mesh  # noqa: E402
from jax_model import load_params_from_pt, num_params  # noqa: E402


def make_hash_vec(state_size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, int(1e15), size=state_size, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--scramble", type=int, default=8, help="random scramble length")
    ap.add_argument("--b-global", type=int, default=8192)
    ap.add_argument("--num-steps", type=int, default=25)
    ap.add_argument("--history-depth", type=int, default=0)
    ap.add_argument("--no-backtrack", action="store_true")
    ap.add_argument("--parent-chunk", type=int, default=None,
                    help="select the STREAMING step body (required above ~16M on "
                         "real hardware). Must divide B_local. Run the smoke with "
                         "and without it: the two bodies are separate code paths "
                         "and only drifted apart because nothing exercised this one.")
    ap.add_argument("--endgame", type=Path, default=None,
                    help="BFS endgame npz; exercises the table goal test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    devices = jax.devices()
    print(f"jax {jax.__version__}, {len(devices)} devices: {devices[0].platform}")

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    move_names = list(info["generators"].keys())
    solved = tuple(info["central_state"])
    n_gen, state_size = len(move_names), len(solved)
    print(f"N_GEN={n_gen}  STATE_SIZE={state_size}")
    assert (n_gen, state_size) == (24, 88), "this smoke is for the tetraminx port"

    all_moves_np = np.array([info["generators"][n] for n in move_names], dtype=np.int32)
    all_moves = jnp.asarray(all_moves_np)
    V0 = jnp.asarray(np.array(solved, dtype=np.int8))
    hash_vec = jnp.asarray(make_hash_vec(state_size, seed=0))
    owner_rng = np.random.default_rng(12345)
    owner_hash_vec = jnp.asarray(owner_rng.integers(
        0, np.iinfo(np.uint32).max, size=state_size, dtype=np.uint32))

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    v_params = load_params_from_pt(args.checkpoint, hidden_dims=hidden_dims,
                                  num_res_blocks=args.num_res_blocks)
    print(f"V params: {num_params(v_params):,}")

    rng = np.random.default_rng(args.seed)
    state = np.array(solved, dtype=np.int64)
    word = rng.integers(0, n_gen, size=args.scramble)
    for m in word:
        state = state[all_moves_np[m]]
    print(f"scramble: {args.scramble} moves, {int((state != np.arange(state_size)).sum())} "
          f"facelets displaced")

    eg_hashes = eg_ztab = None
    if args.endgame:
        _eg = np.load(args.endgame, allow_pickle=True)
        eg_hashes = jnp.asarray(_eg["hashes"])
        eg_ztab = jnp.asarray(_eg["ztab"])
        print(f"endgame table: {_eg['hashes'].size:,} states <= d{int(_eg['max_depth'])}")

    mesh = make_mesh(devices)
    world = len(devices)
    b_local = args.b_global // world
    print(f"step body: {'STREAMING (parent_chunk=%d)' % args.parent_chunk if args.parent_chunk else 'non-streaming'}")
    r = beam_solve_v_only_spmd_packed(
        list(state), v_params, all_moves, V0, hash_vec, mesh,
        B_local=b_local, K_per_peer=b_local // world,
        n_gen=n_gen, state_size=state_size,
        num_steps=args.num_steps, dtype=jnp.float32,
        internal_bs=min(4096, b_local),   # must divide/fit B_local
        parent_chunk=args.parent_chunk,
        owner_hash_vec=owner_hash_vec, history_depth=args.history_depth,
        eg_hashes=eg_hashes, eg_ztab=eg_ztab,
        inv_move_tbl=(jnp.asarray(np.array([move_names.index(m[1:] if m.startswith("-") else "-"+m)
                                            for m in move_names], dtype=np.int32))
                      if args.no_backtrack else None),
        tree_path=str(HERE / "_cpu_smoke_tree.u32"),
    )
    print(f"found={r['found']}  path_len={r.get('path_len')}  wall={r['wall_s']:.1f}s")
    if not r["found"]:
        print("RESULT: NOT FOUND (widen --b-global or shorten --scramble)")
        return 1

    path_idx = list(r["path_idx"])
    cur = np.array(state, dtype=np.int64)
    for mi in path_idx:
        cur = cur[all_moves_np[mi]]

    # With an endgame table the beam stops on a TABLE node, not the goal, so the
    # raw path is a prefix and replay MUST fail without the host-side tail splice.
    # (Omitting this made a correct run look like a failure on 2026-08-02.)
    if args.endgame is not None and not np.array_equal(cur, np.array(solved, dtype=np.int64)):
        _eg = np.load(args.endgame, allow_pickle=True)
        EGH, EGD, EGZ = _eg["hashes"], _eg["depths"], _eg["ztab"]
        def eg_lookup(sts):
            sts = np.atleast_2d(sts)
            h = np.zeros(sts.shape[0], dtype=np.int64)
            for i in range(sts.shape[1]):
                h ^= EGZ[i][sts[:, i]]
            pos = np.clip(np.searchsorted(EGH, h), 0, EGH.size - 1)
            out = np.full(sts.shape[0], -1, dtype=np.int64)
            hit = EGH[pos] == h
            out[hit] = EGD[pos[hit]]
            return out
        d = int(eg_lookup(cur[None, :])[0])
        print(f"beam stopped inside the table at depth {d}; splicing optimal tail")
        assert d >= 0, "beam reported found but the state is not in the table"
        while d > 0:
            kids = cur[all_moves_np]
            nxt = int(np.nonzero(eg_lookup(kids) == d - 1)[0][0])
            path_idx.append(nxt)
            cur = kids[nxt]
            d -= 1
    ok = np.array_equal(cur, np.array(solved, dtype=np.int64))
    r = dict(r); r["path_idx"] = path_idx
    print(f"path replay solves: {ok}  "
          f"({'.'.join(move_names[m] for m in r['path_idx'])})")
    print("RESULT: PASS" if ok else "RESULT: FAIL -- path does not solve")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
