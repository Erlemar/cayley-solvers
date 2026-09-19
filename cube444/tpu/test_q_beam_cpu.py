"""End-to-end CPU smoke test for the Q-native SPMD beam.

Runs the real 8-way sharded kernel on 8 simulated CPU devices with the real s3 weights,
on a shallow scramble that a tiny beam can actually solve, and REPLAYS the returned path
to confirm it reaches solved.

What this does and does not prove:
  * proves  -- score alignment (Q[parent, move] really is the score of child parent*24+move),
               owner routing, per-owner top-K, all_to_all, dedup, packed backptr walkback,
               and path reconstruction all still agree after the Q-native change.
  * proves  -- the blend runs inside the sharded step.
  * does NOT prove anything about TPU numerics or throughput. Per [[jax_tpu_gotchas]],
    CPU emulation validates plumbing only; bf16 on CPU accumulates in bf16 while the TPU
    MXU accumulates in fp32, so the CPU run is pessimistic about ordering drift.

Usage:
  .venv/Scripts/python.exe cube444/tpu/test_q_beam_cpu.py [--scramble 8] [--b-global 512]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Must precede the jax import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BUNDLE = PROJECT / "cube444" / "kaggle_inference" / "cube444_inference" / "solver"
sys.path.insert(0, str(HERE))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import jax_endgame as eg  # noqa: E402
import jax_q_models as jqm  # noqa: E402
from jax_beam_spmd_v_only import beam_solve_v_only_spmd_packed, make_mesh  # noqa: E402


def load_puzzle():
    import torch

    sys.path.insert(0, str(BUNDLE))
    from pilgrim.utils import generate_inverse_moves, parse_generator_spec

    spec = json.loads((BUNDLE / "generators" / "p002.json").read_text(encoding="utf-8"))
    moves, move_names = parse_generator_spec(spec)
    all_moves = np.asarray(moves, dtype=np.int64)
    inverse_moves = np.asarray(generate_inverse_moves(move_names), dtype=np.int64)
    v0 = torch.load(BUNDLE / "targets" / "p002-t000.pt",
                    map_location="cpu", weights_only=False).numpy().astype(np.int64)
    return all_moves, inverse_moves, v0, move_names


def scramble(v0, all_moves, inverse_moves, depth, seed=0):
    rng = np.random.default_rng(seed)
    state, last = v0.copy(), -1
    for _ in range(depth):
        while True:
            m = int(rng.integers(len(all_moves)))
            if last < 0 or m != inverse_moves[last]:
                break
        state, last = state[all_moves[m]], m
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scramble", type=int, default=8)
    ap.add_argument("--b-global", type=int, default=512)
    ap.add_argument("--alpha", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--mlp-weight", type=float, default=0.4)
    ap.add_argument("--internal-bs", type=int, default=0,
                    help="model chunk inside a parent chunk. Set this BELOW "
                         "--parent-chunk to exercise multi-chunk scoring: a test with "
                         "internal_bs >= parent_chunk makes the chunking a no-op and "
                         "cannot catch an unchunked forward (which is how a 47 GB HBM "
                         "OOM reached the TPU).")
    ap.add_argument("--parent-chunk", type=int, default=0,
                    help="0 = non-streaming body; >0 uses the streaming body and must "
                         "divide b_local")
    ap.add_argument("--tail-depth", type=int, default=0,
                    help="exact endgame depth; 0 = off. Use 4 or 5 on CPU (depth 6 is "
                         "67M entries and is a TPU-scale table)")
    args = ap.parse_args()

    devices = jax.devices()
    print(f"devices: {len(devices)} x {devices[0].platform}")
    all_moves, inverse_moves, v0, move_names = load_puzzle()
    state_size, n_gen = v0.shape[0], all_moves.shape[0]

    tr = jqm.load_piece_transformer(BUNDLE / "models" / "s3" / "model.pth")
    mlp = (jqm.load_pair_qmlp(BUNDLE / "models" / "mlp_x16" / "model.pth")
           if args.mlp_weight > 0 else None)
    tr = jax.tree_util.tree_map(jnp.asarray, tr)
    if mlp is not None:
        mlp = jax.tree_util.tree_map(jnp.asarray, mlp)

    def q_score_fn(states):
        # fp32 on CPU: this test is about plumbing, not about bf16 tolerance.
        return jqm.apply_blend(tr, mlp, states, args.mlp_weight, jnp.float32)

    n_dev = len(devices)
    b_local = args.b_global // n_dev
    k_per_peer = (args.alpha * b_local) // n_dev
    print(f"B_global={args.b_global} b_local={b_local} k_per_peer={k_per_peer} "
          f"scramble={args.scramble} mlp_weight={args.mlp_weight}")

    init = scramble(v0, all_moves, inverse_moves, args.scramble, seed=0)
    mesh = make_mesh(devices)
    # backend="numpy" -- must match what the beam hashes with, or the tail table
    # silently never hits (see jax_endgame.make_hash_vec).
    hash_vec_np = eg.make_hash_vec(state_size, 0, backend="numpy")
    hash_vec = jnp.asarray(hash_vec_np)

    tail_hashes = tail_codes = tail_zobrist = None
    zob_np = eg.make_zobrist(state_size, 6, 0)
    if args.tail_depth > 0:
        tail_hashes_np, tail_codes = eg.build_tail_table(
            all_moves, inverse_moves, v0.astype(np.uint8), hash_vec_np,
            args.tail_depth, zobrist=zob_np)
        tail_hashes = jnp.asarray(tail_hashes_np)
        tail_zobrist = jnp.asarray(zob_np)
        print(f"tail table: {len(tail_codes):,} states at depth {args.tail_depth}")
    owner_hash_vec = jnp.asarray(np.random.default_rng(12345).integers(
        0, np.iinfo(np.uint32).max, size=state_size, dtype=np.uint32))

    with tempfile.TemporaryDirectory() as tmp:
        result = beam_solve_v_only_spmd_packed(
            [int(x) for x in init],
            None,                                   # v_params unused on the Q path
            jnp.asarray(all_moves),
            jnp.asarray(v0),
            hash_vec,
            mesh,
            B_local=b_local,
            K_per_peer=k_per_peer,
            n_gen=n_gen,
            state_size=state_size,
            num_steps=args.num_steps,
            dtype=jnp.float32,
            internal_bs=(args.internal_bs or min(64, b_local)),
            tree_path=str(Path(tmp) / "tree.mmap"),
            pack_v_score=True,                      # required by the Q path
            owner_hash_vec=owner_hash_vec,
            q_score_fn=q_score_fn,
            tail_hashes=tail_hashes,
            tail_zobrist=tail_zobrist,
            parent_chunk=(args.parent_chunk or None),
        )

    print(f"result: found={result.get('found')} path_len={result.get('path_len')} "
          f"found_step={result.get('found_step')} wall={result.get('wall_s', 0):.1f}s")
    if not result.get("found"):
        print("\nQ-BEAM SMOKE FAIL (no solution at this width -- raise --b-global "
              "or lower --scramble)")
        return 1

    path = list(result["path_idx"])
    # With the endgame on, the beam stops at a state inside the tail ball rather than at
    # solved. Decode the stored optimal tail on the host and append it.
    if tail_hashes is not None:
        cur = init.copy()
        for m in path:
            cur = cur[all_moves[m]]
        if not np.array_equal(cur, v0):
            h = int(eg.zobrist_hash(cur.reshape(1, -1).astype(np.int64), zob_np)[0])
            idx = int(np.searchsorted(tail_hashes_np, h))
            if idx >= len(tail_hashes_np) or int(tail_hashes_np[idx]) != h:
                print("\nQ-BEAM SMOKE FAIL (beam stopped at a state not in the tail "
                      "table -- hash_vec mismatch?)")
                return 1
            tail = eg.decode_code(int(tail_codes[idx]), n_gen + 1)
            print(f"  endgame fired: beam path {len(path)} + exact tail {len(tail)}")
            path += tail

    cur = init.copy()
    for m in path:
        cur = cur[all_moves[m]]
    solved = bool(np.array_equal(cur, v0))
    print(f"replay: {len(path)} moves -> solved={solved}")
    print("  moves:", ".".join(move_names[m] for m in path))
    if not solved:
        print("\nQ-BEAM SMOKE FAIL (path does not replay to solved)")
        return 1
    if len(path) < args.scramble:
        print(f"  (found a {len(path)}-move solution for an {args.scramble}-move "
              f"scramble -- shorter is legal, the scramble need not be geodesic)")

    print("\nQ-BEAM SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
