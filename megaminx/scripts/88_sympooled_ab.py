"""Equal-wall A/B: sequential sym-K ensemble vs sym-pooled beam (direction 3C).

Arms (same V model, same rotations, same total candidate budget per pid):
  seq  — production-style sym-ensemble: K independent beam searches of width B,
         one per rotated copy, best translated path wins. (No NISS — matches
         the Rule-11 production recipe.)
  pool — ONE pooled beam of total width K*B seeded with all K rotated copies
         at once (SymPooledSolver); width reallocates across rotation frames
         every step, subject to a per-root floor quota.
  Optional `--pool-inverse` adds the K rotated copies of the INVERTED scramble
  as extra roots at the SAME total width (2K roots sharing K*B slots) — the
  sym x inverse grid at zero extra wall.

Both arms post-process with `full_post_process` and verify against the
original state, so reported lengths are submission-grade.

Examples (local 4090):
  # smoke + K=1 self-check on two easy pids, tiny beam
  .venv/Scripts/python.exe megaminx/scripts/88_sympooled_ab.py \
      --checkpoint megaminx/models/m_az_v4_v_only.pt \
      --pids 150,350 --k-sym 2 --beam 2048 --self-test

  # the real A/B on a hard slice
  .venv/Scripts/python.exe megaminx/scripts/88_sympooled_ab.py \
      --checkpoint megaminx/models/m_az_v4_v_only.pt \
      --pids 991,992,993,994,995 --k-sym 4 --beam 16384 --bf16 \
      --out-json megaminx/results/sympooled_ab_hard5.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "beam_lab"))

from beam_search import KhoruzhiiSearchConfig, KhoruzhiiSolver, setup_model_for_inference
from beam_search_sympooled import SymPooledSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_test_states, verify_path
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def _apply_rotation_to_state(state, R, R_inv):
    """Compute R . state . R_inv (same convention as 03_solve.py)."""
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))


def _compute_conjugation_map(R, R_inv, generators_dict, gen_names):
    """Map rotated-frame generator names back to original-frame names
    (R_inv . g . R must itself be a generator). Same as 03_solve.py."""
    n = len(R)
    perm_to_name = {tuple(g): nm for nm, g in generators_dict.items()}
    out = {}
    for nm in gen_names:
        g = generators_dict[nm]
        conj = tuple(R_inv[g[R[i]]] for i in range(n))
        if conj not in perm_to_name:
            raise ValueError(f"rotation is not a symmetry: R_inv * g_{nm} * R is not a generator")
        out[nm] = perm_to_name[conj]
    return out


def _select_rotations(puzzle, rotations_path: Path, k: int, sym_seed: int):
    """Identity first + (k-1) seeded picks — identical policy to 03_solve.py,
    so the A/B compares the exact rotation set production would use."""
    rot_arr = np.load(rotations_path)
    assert rot_arr.shape[1] == len(puzzle.solved_state), "rotation size mismatch"
    identity = np.arange(rot_arr.shape[1], dtype=rot_arr.dtype)
    identity_idx = None
    for i in range(rot_arr.shape[0]):
        if np.array_equal(rot_arr[i], identity):
            identity_idx = i
            break
    assert identity_idx is not None, "rotations file missing identity row"
    rng = np.random.default_rng(sym_seed)
    other = [i for i in range(rot_arr.shape[0]) if i != identity_idx]
    chosen = [identity_idx]
    if k > 1:
        chosen += list(rng.choice(other, size=k - 1, replace=False).tolist())
    out = []
    for idx in chosen:
        R = tuple(int(x) for x in rot_arr[idx])
        R_inv = tuple(int(x) for x in np.argsort(rot_arr[idx]))
        cm = _compute_conjugation_map(R, R_inv, puzzle.generators, puzzle.move_names)
        out.append({"idx": int(idx), "R": R, "R_inv": R_inv, "conj": cm})
    return out


def _translate_and_verify(puzzle, state, raw_names, conj_map, is_inverse):
    """raw_names solves the (rotated, maybe inverted) root. Return a verified
    path for the ORIGINAL state, post-processed, or None."""
    p = [conj_map[m] for m in raw_names]
    if is_inverse:
        p = puzzle.invert_path(p)
    p = full_post_process(p, puzzle=puzzle)
    return p if verify_path(puzzle, state, p).ok else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT / "models" / "m_az_v4_v_only.pt")
    ap.add_argument("--pids", type=str, required=True,
                    help="comma-separated pids to A/B")
    ap.add_argument("--k-sym", type=int, default=4)
    ap.add_argument("--beam", type=int, default=16384,
                    help="per-rotation width B for the seq arm; pool uses k*B total")
    ap.add_argument("--total-width", type=int, default=None,
                    help="override the pooled total width (default k_sym * beam)")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--floor-frac", type=float, default=0.5,
                    help="per-root floor quota as a fraction of the equal share")
    ap.add_argument("--alloc", choices=["progress", "global"], default="progress",
                    help="pooled width allocator: progress (v1, frame-bias-free) "
                         "or global (v0 raw cross-frame top-B, for comparison)")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="softmax temperature on the progress signal (V units)")
    ap.add_argument("--ema", type=float, default=0.7,
                    help="EMA smoothing of the progress signal")
    ap.add_argument("--sym-rotations", type=Path, default=PROJECT / "data" / "rotations.npy")
    ap.add_argument("--sym-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="solver hash seed")
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--pool-inverse", action="store_true",
                    help="add K inverse roots to the pool at the same total width")
    ap.add_argument("--arms", type=str, default="seq,pool",
                    help="comma subset of seq,pool")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="also assert K=1 pooled == plain solver on the first pid")
    args = ap.parse_args()

    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    total_width = args.total_width or args.k_sym * args.beam

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    setup_model_for_inference(model)

    rotations = _select_rotations(puzzle, args.sym_rotations, args.k_sym, args.sym_seed)
    print(f"model={args.checkpoint.name} dtype={dtype} device={args.device}")
    print(f"k_sym={args.k_sym} (rot idxs {[r['idx'] for r in rotations]})  "
          f"beam(seq)={args.beam}  total(pool)={total_width}  "
          f"floor_frac={args.floor_frac}  pool_inverse={args.pool_inverse}")

    seq_solver = KhoruzhiiSolver(
        puzzle, model, device=args.device,
        internal_batch_size=args.internal_batch_size,
        random_seed=args.seed, profile=False,
    )
    pool_solver = SymPooledSolver(
        puzzle, model, device=args.device,
        internal_batch_size=args.internal_batch_size,
        random_seed=args.seed, profile=True,
    )

    if args.self_test:
        _run_self_test(puzzle, seq_solver, pool_solver, states[pids[0]], args)

    results = []
    for pid in pids:
        state = states[pid]
        row: dict = {"pid": pid}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if "seq" in arms:
            t0 = time.time()
            per_rot = []
            for rot in rotations:
                s_rot = _apply_rotation_to_state(state, rot["R"], rot["R_inv"])
                cfg = KhoruzhiiSearchConfig(
                    beam_width=args.beam, num_steps=args.max_steps,
                    num_attempts=args.num_attempts,
                )
                found, _, raw, _prof = seq_solver.solve(s_rot, cfg)
                if not found:
                    per_rot.append(None)
                    continue
                p = _translate_and_verify(puzzle, state, raw, rot["conj"], False)
                per_rot.append(len(p) if p is not None else None)
            seq_wall = time.time() - t0
            solved_lens = [x for x in per_rot if x is not None]
            row["seq"] = {
                "per_rot": per_rot,
                "best": min(solved_lens) if solved_lens else None,
                "wall_s": round(seq_wall, 1),
            }

        if "pool" in arms:
            t0 = time.time()
            roots = []
            meta = []  # (conj_map, is_inverse) per root
            for rot in rotations:
                roots.append(_apply_rotation_to_state(state, rot["R"], rot["R_inv"]))
                meta.append((rot["conj"], False))
            if args.pool_inverse:
                inv_state = puzzle.invert_state(state)
                for rot in rotations:
                    roots.append(_apply_rotation_to_state(inv_state, rot["R"], rot["R_inv"]))
                    meta.append((rot["conj"], True))
            cfg = KhoruzhiiSearchConfig(
                beam_width=total_width, num_steps=args.max_steps,
                num_attempts=args.num_attempts,
            )
            found, _, raw, root_id, prof = pool_solver.solve_pooled(
                roots, cfg, floor_frac=args.floor_frac,
                alloc=args.alloc, tau=args.tau, ema=args.ema,
            )
            pool_wall = time.time() - t0
            pool_len = None
            if found:
                cm, is_inv = meta[root_id]
                p = _translate_and_verify(puzzle, state, raw, cm, is_inv)
                pool_len = len(p) if p is not None else None
            # Occupancy trace: first/mid/last step root shares show reallocation.
            occ = prof.per_step_root_counts
            occ_summary = {}
            if occ:
                occ_summary = {
                    "step0": occ[0],
                    "mid": occ[len(occ) // 2],
                    "last": occ[-1],
                }
            row["pool"] = {
                "len": pool_len,
                "root_id": root_id if found else -1,
                "root_is_inverse": (meta[root_id][1] if found else None),
                "n_roots": len(roots),
                "n_roots_effective": prof.n_roots_effective,
                "wall_s": round(pool_wall, 1),
                "n_steps": prof.n_steps,
                "occupancy": occ_summary,
            }
            row["pool_occupancy_full"] = occ

        s = row.get("seq", {})
        p = row.get("pool", {})
        delta = None
        if s.get("best") is not None and p.get("len") is not None:
            delta = p["len"] - s["best"]
        print(f"pid={pid:4d}  seq_best={s.get('best')} ({s.get('wall_s')}s, "
              f"per_rot={s.get('per_rot')})  "
              f"pool={p.get('len')} ({p.get('wall_s')}s, root={p.get('root_id')}"
              f"{'/inv' if p.get('root_is_inverse') else ''})  "
              f"delta={delta}", flush=True)
        row["delta_pool_minus_seq"] = delta
        results.append(row)

    both = [r for r in results
            if r.get("seq", {}).get("best") is not None
            and r.get("pool", {}).get("len") is not None]
    if both:
        seq_total = sum(r["seq"]["best"] for r in both)
        pool_total = sum(r["pool"]["len"] for r in both)
        seq_wall = sum(r["seq"]["wall_s"] for r in both)
        pool_wall = sum(r["pool"]["wall_s"] for r in both)
        wins = sum(1 for r in both if r["pool"]["len"] < r["seq"]["best"])
        ties = sum(1 for r in both if r["pool"]["len"] == r["seq"]["best"])
        print(f"\nsummary over {len(both)} pids (both arms solved):")
        print(f"  seq  total={seq_total}  wall={seq_wall:.0f}s")
        print(f"  pool total={pool_total}  wall={pool_wall:.0f}s")
        print(f"  pool-seq = {pool_total - seq_total:+d}  "
              f"(pool wins {wins}, ties {ties}, losses {len(both) - wins - ties})")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                       "results": results}, f, indent=1)
        print(f"wrote {args.out_json}")
    return 0


def _run_self_test(puzzle, seq_solver, pool_solver, state, args):
    """K=1 pooled must behave exactly like the plain solver (same hash seed,
    floor disabled by n_roots=1): same found flag and same path length."""
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)
    f1, l1, n1, _ = seq_solver.solve(state, cfg)
    f2, l2, n2, root_id, _ = pool_solver.solve_pooled([state], cfg, floor_frac=args.floor_frac)
    assert f1 == f2, f"self-test: found mismatch {f1} vs {f2}"
    if f1:
        assert root_id == 0, f"self-test: root_id {root_id} != 0"
        assert l1 == l2 and n1 == n2, (
            f"self-test: K=1 pooled diverged from plain solver "
            f"(len {l1} vs {l2})"
        )
        ok = verify_path(puzzle, state, n2).ok
        assert ok, "self-test: pooled path failed verification"
    print(f"self-test PASSED (K=1 pooled == plain solver; found={f1}, len={l1})")


if __name__ == "__main__":
    sys.exit(main())
