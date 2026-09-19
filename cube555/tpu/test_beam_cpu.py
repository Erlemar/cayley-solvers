"""End-to-end CPU smoke for the cube555 Q-native SPMD beam.

Runs the REAL 8-way sharded kernel on 8 simulated CPU devices, with the real
q555 weights and the real d<=5 endgame table, on scrambles shallow enough for a
tiny beam -- then replays every returned path against the ORIGINAL state.

WHAT THIS PROVES
  * uint8 state carry survives the whole round trip. This is the port's one
    dangerous change: classes 128..149 wrap negative under the engine's original
    int8. Whether that fails loudly is version-dependent -- `.astype(int8)` wraps
    silently on numpy 2.4.4/jax 0.10 while `np.asarray(list, int8)` range-checks --
    so the guarantee has to come from an end-to-end replay, not from trusting a
    crash. Case A asserts the scramble actually contains values >= 128, otherwise
    the test is vacuous. The matched negative control (engine reverted to int8)
    was run on 2026-08-23 and does fail on this stack.
  * PACK_SIZE=160 bucket layout: state(150) + parent_local(4) + move(1) + bf16
    score(2) all land inside the record, and the receive side unpacks the same
    bytes the send side wrote.
  * q_mode score alignment -- that Q[parent, move] really is the score of child
    `parent * 30 + move` after the 30-wide head swap.
  * owner routing, per-owner top-K, all_to_all, in-rank dedup, packed uint32
    backpointer walkback, endgame membership + host-side tail splice.
  * the symmetry frame transform the notebook uses, round-tripped on a real pid.
  * both step bodies: non-streaming and PARENT_CHUNK streaming.

WHAT IT DOES NOT PROVE
  Nothing about TPU numerics, collectives or throughput. Per [[jax_tpu_gotchas]]
  CPU emulation validates plumbing only; bf16 on CPU accumulates in bf16 while the
  TPU MXU accumulates in fp32, so this runs fp32 and is pessimistic about ordering.

Usage:
  .venv/Scripts/python.exe cube555/tpu/test_beam_cpu.py
  .venv/Scripts/python.exe cube555/tpu/test_beam_cpu.py --b-global 8192 --scramble 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "kaggle_dataset"
sys.path.insert(0, str(ASSETS))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jax_beam_spmd_v_only import beam_solve_v_only_spmd_packed, make_mesh  # noqa: E402
from jax_model import apply as model_apply  # noqa: E402
from jax_model import load_params_from_pt, num_params  # noqa: E402

HANDOFF = Path("C:/Users/and-l/cayley/cube555_pull/cube555_handoff_2026_08_22/cube555")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ASSETS / "q555_2k_BEST.pt")
    ap.add_argument("--b-global", type=int, default=4096)
    ap.add_argument("--scramble", type=int, default=9)
    ap.add_argument("--num-steps", type=int, default=24)
    args = ap.parse_args()

    devices = jax.devices()
    world = len(devices)
    print(f"jax {jax.__version__}  {world} simulated devices  x64={jax.config.jax_enable_x64}")
    assert world == 8, "expected 8 simulated devices"

    info = json.loads((ASSETS / "puzzle_info.json").read_text(encoding="utf-8"))
    move_names = list(info["generators"])
    all_moves_np = np.asarray([info["generators"][m] for m in move_names], dtype=np.int32)
    solved_np = np.asarray(info["central_state"], dtype=np.int64)
    n_gen, state_size = len(move_names), len(solved_np)
    all_moves = jnp.asarray(all_moves_np)
    V0 = jnp.asarray(solved_np.astype(np.uint8))
    inv_idx = np.asarray([move_names.index(m[1:] if m.startswith("-") else "-" + m)
                          for m in move_names], dtype=np.int32)

    params = load_params_from_pt(args.checkpoint)
    print(f"model: {args.checkpoint.name}  {num_params(params):,} params  "
          f"head width {params['head_w'].shape[-1]}")
    assert int(params["head_w"].shape[-1]) == n_gen, "q_mode needs a 30-wide head"

    _eg = np.load(ASSETS / "bfs_endgame.npz", allow_pickle=True)
    EGH, EGD, EGZ = _eg["hashes"], _eg["depths"], _eg["ztab"]
    eg_hashes, eg_ztab = jnp.asarray(EGH), jnp.asarray(EGZ)
    print(f"endgame: {EGH.size:,} states <= d{int(_eg['max_depth'])}")

    SYM = np.load(ASSETS / "cube555_sym.npy")
    SYM_INV = np.load(ASSETS / "cube555_sym_inv.npy")
    RELABEL = np.load(ASSETS / "cube555_move_relabel_inv.npy")

    rng_h = np.random.default_rng(0)
    hash_vec = jnp.asarray(rng_h.integers(0, int(1e15), size=state_size, dtype=np.int64))
    owner_hash_vec = jnp.asarray(np.random.default_rng(12345).integers(
        0, np.iinfo(np.uint32).max, size=state_size, dtype=np.uint32))
    mesh = make_mesh(devices)

    b_local = args.b_global // world
    inv_tbl = jnp.asarray(inv_idx)

    def eg_lookup(sts):
        sts = np.atleast_2d(np.asarray(sts, dtype=np.int64))
        h = np.zeros(sts.shape[0], dtype=np.int64)
        for i in range(sts.shape[1]):
            h ^= EGZ[i][sts[:, i]]
        pos = np.clip(np.searchsorted(EGH, h), 0, EGH.size - 1)
        out = np.full(sts.shape[0], -1, dtype=np.int64)
        hit = EGH[pos] == h
        out[hit] = EGD[pos[hit]]
        return out

    def replay(s0, idx):
        cur = np.asarray(s0, dtype=np.int64)
        for m in idx:
            cur = cur[all_moves_np[m]]
        return cur

    def splice(u, path_idx):
        """Host-side optimal tail from wherever the beam stopped inside the ball."""
        cur = replay(u, path_idx)
        if np.array_equal(cur, solved_np):
            return list(path_idx), True
        d = int(eg_lookup(cur[None, :])[0])
        if d < 0:                       # hash false positive -- must be non-fatal
            return list(path_idx), False
        out = list(path_idx)
        while d > 0:
            kids = cur[all_moves_np]
            hit = np.nonzero(eg_lookup(kids) == d - 1)[0]
            if hit.size == 0:
                return out, False
            out.append(int(hit[0]))
            cur = kids[int(hit[0])]
            d -= 1
        return out, np.array_equal(cur, solved_np)

    def scramble(depth, seed):
        r = np.random.default_rng(seed)
        s, last = solved_np.copy(), -1
        for _ in range(depth):
            while True:
                m = int(r.integers(0, n_gen))
                if last < 0 or m != inv_idx[last]:
                    break
            s = s[all_moves_np[m]]
            last = m
        return s

    def run(label, start, parent_chunk, history_depth, no_backtrack, expect_uint8=False):
        if expect_uint8:
            hi = int((np.asarray(start) >= 128).sum())
            print(f"  [{label}] start has {hi} sticker values >= 128 "
                  f"(these are what int8 would wrap)")
            assert hi > 0, "vacuous uint8 test: no class >= 128 in the start state"
        ibs = min(256, b_local)
        r = beam_solve_v_only_spmd_packed(
            list(np.asarray(start, dtype=np.int64)), params, all_moves, V0, hash_vec,
            mesh, B_local=b_local, K_per_peer=(2 * b_local) // world,
            n_gen=n_gen, state_size=state_size, num_steps=args.num_steps,
            dtype=jnp.float32, internal_bs=ibs, parent_chunk=parent_chunk,
            owner_hash_vec=owner_hash_vec,
            history_depth=history_depth, history_exact=True,
            eg_hashes=eg_hashes, eg_ztab=eg_ztab,
            inv_move_tbl=inv_tbl if no_backtrack else None,
            q_mode=True, pack_v_score=True,
            tree_path=str(HERE / f"_cpu_tree_{label}.u32"),
        )
        return r

    cases, fails = [], 0

    # ---- A: non-streaming, the uint8 gate --------------------------------------
    s_a = scramble(args.scramble, 1)
    print("\nA  non-streaming | q_mode | endgame | history=4 | no-backtrack")
    r = run("A", s_a, None, 4, True, expect_uint8=True)
    if r["found"]:
        path, ok = splice(s_a, list(r["path_idx"]))
        print(f"  found in {len(r['path_idx'])} beam steps -> {len(path)} moves  "
              f"replay={ok}  wall={r['wall_s']:.1f}s")
        fails += 0 if ok else 1
        cases.append(("A", ok, len(path)))
    else:
        print(f"  NOT FOUND ({r['wall_s']:.1f}s)")
        fails += 1
        cases.append(("A", False, None))

    # ---- B: streaming body, same puzzle ----------------------------------------
    pc = b_local // 2
    print(f"\nB  STREAMING (parent_chunk={pc}) | q_mode | endgame | history=4")
    r = run("B", s_a, pc, 4, True)
    if r["found"]:
        path, ok = splice(s_a, list(r["path_idx"]))
        print(f"  found in {len(r['path_idx'])} beam steps -> {len(path)} moves  "
              f"replay={ok}  wall={r['wall_s']:.1f}s")
        fails += 0 if ok else 1
        cases.append(("B", ok, len(path)))
    else:
        print(f"  NOT FOUND ({r['wall_s']:.1f}s)")
        fails += 1
        cases.append(("B", False, None))

    # ---- C: symmetry frame, full round trip to the ORIGINAL state --------------
    # The kernel never solves `s`; it solves conj(s) and translates back. A wrong
    # relabel direction still returns a valid-length path for the frame and simply
    # fails to solve the original, which is what this case catches.
    k, inverted = 7, True
    s_c = scramble(args.scramble, 2)
    t = np.argsort(s_c) if inverted else s_c
    u = SYM_INV[k][t[SYM[k]]]
    print(f"\nC  frame k={k} inverted={inverted} | streaming | round trip to original")
    r = run("C", u, pc, 4, True)
    if r["found"]:
        fpath, ok_frame = splice(u, list(r["path_idx"]))
        q = [int(RELABEL[k][m]) for m in fpath]
        orig = [int(inv_idx[m]) for m in reversed(q)] if inverted else q
        ok = bool(np.array_equal(replay(s_c, orig), solved_np))
        print(f"  frame path {len(fpath)} moves (solves conj: {ok_frame})")
        print(f"  translated  {len(orig)} moves  solves ORIGINAL: {ok}  "
              f"wall={r['wall_s']:.1f}s")
        fails += 0 if (ok and ok_frame) else 1
        cases.append(("C", ok and ok_frame, len(orig)))
    else:
        print(f"  NOT FOUND ({r['wall_s']:.1f}s)")
        fails += 1
        cases.append(("C", False, None))

    # ---- D: no endgame table (solve to the exact goal) -------------------------
    print("\nD  no endgame table | q_mode | history=0 (goal is the solved state alone)")
    s_d = scramble(4, 3)
    r = beam_solve_v_only_spmd_packed(
        list(np.asarray(s_d, dtype=np.int64)), params, all_moves, V0, hash_vec, mesh,
        B_local=b_local, K_per_peer=(2 * b_local) // world,
        n_gen=n_gen, state_size=state_size, num_steps=12, dtype=jnp.float32,
        internal_bs=min(256, b_local), parent_chunk=None,
        owner_hash_vec=owner_hash_vec, history_depth=0,
        q_mode=True, pack_v_score=True,
        tree_path=str(HERE / "_cpu_tree_D.u32"))
    if r["found"]:
        ok = bool(np.array_equal(replay(s_d, list(r["path_idx"])), solved_np))
        print(f"  {len(r['path_idx'])} moves  replay={ok}  wall={r['wall_s']:.1f}s")
        fails += 0 if ok else 1
        cases.append(("D", ok, len(r["path_idx"])))
    else:
        print(f"  NOT FOUND ({r['wall_s']:.1f}s)")
        fails += 1
        cases.append(("D", False, None))

    for f in HERE.glob("_cpu_tree_*.u32"):
        f.unlink(missing_ok=True)

    print("\n" + "-" * 62)
    for name, ok, n in cases:
        print(f"  case {name}: {'PASS' if ok else 'FAIL'}"
              + (f"  ({n} moves)" if n is not None else ""))
    print("VERDICT :", "PASS" if fails == 0 else f"FAIL ({fails} case(s))")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
