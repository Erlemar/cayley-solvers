"""Standalone tetraminx beam driver for a GCP v6e TPU VM.

Same kernel and same frame logic as the Kaggle notebook and
`tetraminx/scripts/30_solve.py`, packaged as a CLI so two models can be A/B'd at
identical settings.

    ~/tpu-env/bin/python gcp_beam_tetraminx.py \
        --checkpoint tv0_bellman.pt --tag resmlp \
        --b-global 1048576 --frames 4 --pids 159,449,... \
        --out /mnt/data/out/resmlp.json

Notes
-----
* BPTR fields stay 24/3/5. With 4 ranks, 3 rank bits is valid (4 <= 8) and 24
  parent-local bits cover B_local up to 16.7M vs our 262K, so no kernel edit is
  needed for a 4-chip run (the skill's 25/2/5 split only buys headroom we do not
  use). The kernel still asserts world_size <= 1<<BPTR_RANK_BITS.
* JAX persistent compilation cache is ON. The beam rebuilds its step_fn closure
  per call, so the in-memory jit cache misses every time; without the on-disk
  cache a fast-solving puzzle spends most of its wall recompiling.
* ALPHA=2 (receive-side oversampling). At ALPHA=1 each owner receives exactly
  B_LOCAL candidates and its top-k is a no-op -- that configuration found 0/8 on
  Kaggle while ALPHA=2 found 8/8.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from jax_beam_spmd_v_only import beam_solve_v_only_spmd_packed, make_mesh  # noqa: E402
from jax_model import (load_params_from_pt, load_piece_transformer_params_from_pt,  # noqa: E402
                       make_blend, has_value_head,
                       num_params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="label written into each record")
    ap.add_argument("--data-dir", type=Path, default=HERE)
    ap.add_argument("--hidden-dims", default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--b-global", type=int, default=1_048_576)
    ap.add_argument("--num-steps", type=int, default=60)
    ap.add_argument("--internal-bs", type=int, default=16384)
    ap.add_argument("--parent-chunk", type=int, default=None)
    ap.add_argument("--alpha", type=int, default=2)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--frame-spec", default=None,
                    help="explicit frames, e.g. '0i' or '0i,1i' or '0f,0i' -- k index "
                         "then f/i for forward/inverse. Overrides --frames, whose "
                         "positional scheme (f//2, f%%2) makes the FORWARD frame the "
                         "only choice at --frames 1. Measured over 171 pids: the k0 "
                         "INVERSE frame alone returns 4.54 moves/TPU-hour against 1.84 "
                         "for all four and 1.99 for k0 forward alone -- the inverse "
                         "antisymmetry carries the win, not the spatial rotation, and "
                         "frames 3-4 add only 0.75 moves/hour. Prefer breadth (more "
                         "pids at 0i) over depth (more frames per pid).")
    ap.add_argument("--no-backtrack", action="store_true",
                    help="ban the child that undoes the parent's incoming move "
                         "(free: one integer compare, no hashing)")
    ap.add_argument("--history-bitmask", action="store_true",
                    help="approximate bitmask membership (~12%% faster, ~3%% false "
                         "positives that drop good candidates). Default is the "
                         "exact binary search, which all published width "
                         "datapoints were measured with.")
    ap.add_argument("--q-mode", choices=("auto", "on", "off"), default="auto",
                    help="all-neighbours Q head: score children from one forward on "
                         "the parents. auto = on when the head width equals n_gen.")
    ap.add_argument("--pre-topk-mult", type=int, default=0,
                    help="progressive top-k: materialize only this multiple of B_local "
                         "children instead of all n_gen (q_mode only). 0 = off. Must be "
                         "at least alpha (send buckets need alpha*B_local candidates); "
                         "4 gives 2x headroom at 4/24 = 17%% of the memory.")
    ap.add_argument("--qv-consistency", type=float, default=0.0,
                    help="score = Q + lam*|Q - (V(s)-1)| using the AZ value head read "
                         "from the SAME trunk pass (free). Needs a q_mode checkpoint "
                         "trained with az_head=true. Measured -4 on the PyTorch beam, "
                         "ADDITIVE with history_depth.")
    ap.add_argument("--blend", nargs="+", default=None,
                    help="extra checkpoints to average with --checkpoint at every step "
                         "(output-space ensemble; one forward per member)")
    ap.add_argument("--blend-weights", nargs="+", type=float, default=None,
                    help="per-member weights, --checkpoint first (default: uniform)")
    ap.add_argument("--history-depth", type=int, default=0,
                    help="sharded cross-layer dedup: drop candidates this rank "
                         "owned in the last N steps (0 = off)")
    ap.add_argument("--pids", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tree-dir", default="/dev/shm/trees")
    ap.add_argument("--endgame", type=Path, default=None,
                    help="BFS d<=6 table npz; the beam stops on any node inside it "
                         "and the optimal tail is spliced host-side")
    args = ap.parse_args()

    devices = jax.devices()
    world = len(devices)
    print(f"jax {jax.__version__}  {world} devices: {devices[0].device_kind}", flush=True)

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    move_names = list(info["generators"].keys())
    solved = np.array(info["central_state"], dtype=np.int64)
    n_gen, state_size = len(move_names), len(solved)
    assert (n_gen, state_size) == (24, 88)
    all_moves_np = np.array([info["generators"][m] for m in move_names], dtype=np.int32)
    all_moves = jnp.asarray(all_moves_np)
    V0 = jnp.asarray(solved.astype(np.int8))
    inv_idx = np.array([move_names.index(m[1:] if m.startswith("-") else "-" + m)
                        for m in move_names])
    inv_move_tbl = jnp.asarray(inv_idx.astype(np.int32))

    SYM = np.load(args.data_dir / "tetra_symmetries.npy").astype(np.int64)
    SYM_INV = np.load(args.data_dir / "tetra_symmetries_inv.npy").astype(np.int64)
    RELABEL = np.load(args.data_dir / "tetra_move_relabel.npy").astype(np.int64)

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states = {int(r["initial_state_id"]):
                  np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                  for r in csv.DictReader(f)}

    # Architecture is read from the checkpoint, not the CLI: a PieceTransformer
    # checkpoint carries arch='transformer' in its model_config and needs the folded
    # value-table loader plus the piece layout. apply() then dispatches on key
    # presence, so the beam kernel is untouched.
    import torch as _torch

    def _load_one(path):
        _ck = _torch.load(path, map_location="cpu", weights_only=False)
        arch = str(_ck.get("model_config", {}).get("arch", "resmlp"))
        del _ck
        if arch == "transformer":
            return arch, load_piece_transformer_params_from_pt(
                path, layout_path=args.data_dir / "piece_layout.json")
        return arch, load_params_from_pt(
            path, hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")),
            num_res_blocks=args.num_res_blocks)

    _arch, v_params = _load_one(args.checkpoint)
    print(f"[{args.tag}] arch: {_arch}", flush=True)
    if args.blend:
        # Cross-architecture output-space ensemble. The ResMLP is ~1/17 of the
        # transformer's forward cost, so a transformer+ResMLP blend is ~1.07x, unlike
        # a blend of transformers. Weights default to uniform; 0.8/0.2 toward the
        # transformer is what measured -2 on the PyTorch beam.
        members = [v_params]
        for extra in args.blend:
            a_i, p_i = _load_one(extra)
            print(f"[{args.tag}] blend member: {Path(extra).name} arch={a_i}", flush=True)
            members.append(p_i)
        weights = args.blend_weights if args.blend_weights else None
        if weights is not None and len(weights) != len(members):
            raise SystemExit(f"--blend-weights has {len(weights)} entries "
                             f"for {len(members)} members")
        v_params = make_blend(members, weights)
        print(f"[{args.tag}] blend: {len(members)} models, weights "
              + ", ".join(f"{w:.3f}" for w in v_params["weights"]), flush=True)
    if args.qv_consistency != 0.0 and not has_value_head(v_params):
        raise SystemExit("--qv-consistency needs a checkpoint trained with az_head=true "
                         "(no value_head.* in the state dict)")
    _head_src = v_params["members"][0] if "members" in v_params else v_params
    head_width = int(_head_src["head_w"].shape[-1])
    if args.q_mode == "auto":
        q_mode = head_width == n_gen
    else:
        q_mode = args.q_mode == "on"
    if q_mode and head_width != n_gen:
        raise SystemExit(f"--q-mode on needs a head of width n_gen={n_gen}, got {head_width}")
    if not q_mode and head_width != 1:
        raise SystemExit(f"scalar-V mode needs head width 1, got {head_width}; pass --q-mode on")
    # Identify the CHECKPOINT, not just its shape. Param count alone is ambiguous:
    # an AZ V-only export and the pure ResMLP V are both 4,996,481 with head width 1,
    # so a finished run could not be attributed to one or the other after its VM was
    # deleted (2026-07-29). Path + size + digest survives in the log.
    try:
        _ck = Path(args.checkpoint)
        _sz = _ck.stat().st_size
        _h = hashlib.sha256(_ck.read_bytes()).hexdigest()[:12]
    except Exception:                                   # never fail a run over logging
        _sz, _h = -1, "unknown"
    print(f"[{args.tag}] checkpoint: {args.checkpoint} ({_sz:,} B, sha256:{_h})", flush=True)
    print(f"[{args.tag}] params: {num_params(v_params):,} | head width {head_width} | "
          f"scorer: {'Q (1 forward per parent)' if q_mode else 'V (1 per child)'}", flush=True)

    hash_vec = jnp.asarray(np.random.default_rng(0).integers(
        0, int(1e15), size=state_size, dtype=np.int64))
    owner_hash_vec = jnp.asarray(np.random.default_rng(12345).integers(
        0, np.iinfo(np.uint32).max, size=state_size, dtype=np.uint32))
    mesh = make_mesh(devices)

    # ---- exact endgame table ------------------------------------------------
    eg_hashes = eg_ztab = None
    EG = None
    eg_path = args.endgame or (args.data_dir / "bfs_endgame.npz")
    if Path(eg_path).exists():
        _eg = np.load(eg_path, allow_pickle=True)
        EG = {"hashes": _eg["hashes"], "depths": _eg["depths"], "ztab": _eg["ztab"],
              "max_depth": int(_eg["max_depth"])}
        eg_hashes = jnp.asarray(EG["hashes"])
        eg_ztab = jnp.asarray(EG["ztab"])
        print(f"endgame table: {EG['hashes'].size:,} states <= d{EG['max_depth']}",
              flush=True)
    else:
        print(f"[warn] no endgame table at {eg_path} -- solving to the exact goal",
              flush=True)

    def eg_lookup(states):
        st = np.atleast_2d(states)
        h = np.zeros(st.shape[0], dtype=np.int64)
        for i in range(st.shape[1]):
            h ^= EG["ztab"][i][st[:, i]]
        pos = np.clip(np.searchsorted(EG["hashes"], h), 0, EG["hashes"].size - 1)
        out = np.full(st.shape[0], -1, dtype=np.int64)
        hit = EG["hashes"][pos] == h
        out[hit] = EG["depths"][pos[hit]]
        return out

    def eg_descend(state):
        """Optimal move list from a table state to solved (frame coordinates)."""
        d = int(eg_lookup(state[None, :])[0])
        assert d >= 0, "descend called on a state outside the table"
        out, cur = [], state
        while d > 0:
            children = cur[all_moves_np]
            cd = eg_lookup(children)
            nxt = int(np.nonzero(cd == d - 1)[0][0])
            out.append(nxt)
            cur = children[nxt]
            d -= 1
        return out

    b_local = args.b_global // world
    k_per_peer = (args.alpha * b_local) // world
    print(f"B_GLOBAL {args.b_global:,}  B_LOCAL {b_local:,}  K_PER_PEER {k_per_peer:,} "
          f"alpha={args.alpha} history_depth={args.history_depth} "
          f"no_backtrack={args.no_backtrack}", flush=True)

    if args.frame_spec:
        frames = []
        for tok in args.frame_spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok[-1] not in "fi":
                raise SystemExit(f"--frame-spec token {tok!r} must end in 'f' or 'i'")
            frames.append((int(tok[:-1]) % SYM.shape[0], tok[-1] == "i"))
        if not frames:
            raise SystemExit("--frame-spec parsed to no frames")
    else:
        frames = [(f // 2 % SYM.shape[0], f % 2 == 1) for f in range(args.frames)]
    pids = [int(x) for x in args.pids.split(",")]
    os.makedirs(args.tree_dir, exist_ok=True)

    def to_frame(s0, k, inverted):
        t = np.argsort(s0) if inverted else s0
        return SYM_INV[k][t[SYM[k]]]

    def from_frame(path_idx, k, inverted):
        q = [int(RELABEL[k][m]) for m in path_idx]
        if inverted:
            q = [int(inv_idx[m]) for m in reversed(q)]
        return q

    def replay(s0, path_idx):
        cur = s0
        for m in path_idx:
            cur = cur[all_moves_np[m]]
        return cur

    results = []
    for pid in pids:
        s0 = states[pid]
        for (k, inverted) in frames:
            u = to_frame(s0, k, inverted)
            # Re-create per frame, not once at startup. On 2026-07-29 a 100-pid run
            # lost 99 pids because systemd-tmpfiles-clean removed /dev/shm/trees
            # mid-run: the in-flight frame survived (unlink does not close an open
            # memmap) but every later frame died with FileNotFoundError trying to
            # CREATE its tree file in a directory that no longer existed. makedirs
            # is idempotent and costs nothing next to a ~400 s beam.
            os.makedirs(args.tree_dir, exist_ok=True)
            tree = f"{args.tree_dir}/t_{args.tag}_{pid}_{k}_{int(inverted)}.u32"
            t0 = time.time()
            try:
                r = beam_solve_v_only_spmd_packed(
                    list(u), v_params, all_moves, V0, hash_vec, mesh,
                    B_local=b_local, K_per_peer=k_per_peer,
                    n_gen=n_gen, state_size=state_size,
                    num_steps=args.num_steps, dtype=jnp.bfloat16, q_mode=q_mode,
                    qv_consistency=args.qv_consistency,
                    pre_topk_mult=args.pre_topk_mult,
                    internal_bs=args.internal_bs, tree_path=tree,
                    parent_chunk=args.parent_chunk, owner_hash_vec=owner_hash_vec,
                    history_depth=args.history_depth,
                    history_exact=(not args.history_bitmask),
                    eg_hashes=eg_hashes, eg_ztab=eg_ztab,
                    inv_move_tbl=(inv_move_tbl if args.no_backtrack else None))
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({"tag": args.tag, "pid": pid, "sym": k,
                                "inverted": inverted, "found": False,
                                "error": str(e), "wall_s": time.time() - t0})
                continue
            rec = {"tag": args.tag, "pid": pid, "sym": k, "inverted": inverted,
                   "found": bool(r["found"]), "wall_s": r["wall_s"]}
            if r["found"]:
                frame_path = list(r["path_idx"])
                if EG is not None:
                    # The beam may have stopped on a TABLE node rather than the
                    # goal; splice the table's optimal tail. Verified below by
                    # replay, which also catches any hash false-positive.
                    reached = replay(u, frame_path)
                    if not np.array_equal(reached, solved):
                        frame_path = frame_path + eg_descend(reached)
                orig = from_frame(frame_path, k, inverted)
                ok = bool(np.array_equal(replay(s0, orig), solved))
                rec.update({"path_len": len(orig), "verify_ok": ok,
                            "path": ".".join(move_names[m] for m in orig)})
                print(f"[{args.tag}] pid {pid} k={k} inv={int(inverted)}: "
                      f"{len(orig)} moves verify={ok} {r['wall_s']:.0f}s", flush=True)
            else:
                # min-V per step separates "descending but out of steps" from
                # "not descending at all" -- the only way to tell a width/steps
                # problem from a search-quality problem.
                traj = r.get("min_v_trajectory_rank0") or []
                rec["min_v_trajectory"] = [float(v) for v in traj]
                if traj:
                    print(f"[{args.tag}]   min V/step: "
                          + " ".join(f"{v:.1f}" for v in traj), flush=True)
                print(f"[{args.tag}] pid {pid} k={k} inv={int(inverted)}: NOT FOUND "
                      f"{r['wall_s']:.0f}s", flush=True)
            results.append(rec)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1)

    best = {}
    for r in results:
        if r.get("found") and r.get("verify_ok"):
            best[r["pid"]] = min(best.get(r["pid"], 10**9), r["path_len"])
    print(f"\n[{args.tag}] best per pid: {best}", flush=True)
    # Report mean alongside the total: totals are only comparable across runs that
    # covered the SAME pid set, and the mean makes a set change obvious immediately.
    # `nz` excludes trivially-short pids (pid 0 is a 1-move puzzle) which otherwise
    # drag the mean well below the real per-puzzle cost.
    _tot = sum(best.values())
    _nz = [v for v in best.values() if v > 1]
    _mean = _tot / len(best) if best else 0.0
    _mean_nz = sum(_nz) / len(_nz) if _nz else 0.0
    print(
        f"[{args.tag}] total over {len(best)} solved: {_tot}"
        f"  avg {_mean:.2f}  avg>1 {_mean_nz:.2f} (n={len(_nz)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
