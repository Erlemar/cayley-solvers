"""Direction-1 falsifier: learned bidirectional meet-in-the-middle.

Big-swings report Direction 1. Thesis: grow a beam from the scramble `s` AND a
beam from solved `e`; splice them where their frontiers near-meet, scoring the
residual with the EXISTING V evaluated on the group-composition residual
`z = b^-1 . f` (which lands in V's calibrated, low-variance near-solved band).
Zero new training: the backward beam is the same forward solver run on `s^-1`
(see beam_search_bidir.py for the relabelling that makes this exact).

What this script measures, per hard pid, at MATCHED-ish node budget:
  L_uni   unidirectional beam (width B, single pass) path length (or "unsolved")
  L_floor current best-CSV path length (reference; also sets the depth budget)
  L_bd    best VERIFIED bidirectional splice length found via near-meeting
and the meeting diagnostics that make a null interpretable:
  - D1 agreement-vs-distance: how many of the 120 positions a length-k residual
    disturbs -> tells us whether coordinate-hash (LSH) can have any recall.
  - M0 exact (residual-0) meetings, FULL recall over the pooled frontiers
    (confirms / denies the classic MITM collision-sparsity at depth).
  - M-LSH coordinate-hash near-meetings -> V(z) filter -> exact residual close.
  - meeting counts + the (d_f, d_b) split of the best meeting (real balanced
    meeting vs "forward almost solved anyway").

FALSIFIED if, on the hard tail, no verified splice beats L_uni (and L_floor),
i.e. closeable near-meetings never appear before the forward beam would have
solved. The first fallback before declaring defeat is front-to-front guidance
(needs a metric embedding) -- the --phi-check flag reports whether V's
penultimate layer is already a usable metric for that (cheap go/no-go).

Example (local 4090):
  .venv/Scripts/python.exe megaminx/scripts/91_bidir_probe.py \
      --checkpoint megaminx/models/m_az_v4_v_only.pt \
      --n-hard 20 --beam 16384 --bf16 --phi-check \
      --out-json megaminx/results/bidir_probe_hard20.json

Smoke (correctness — the spliced path MUST verify):
  .venv/Scripts/python.exe megaminx/scripts/91_bidir_probe.py \
      --checkpoint megaminx/models/m_az_v4_v_only.pt \
      --pids 150,350 --beam 4096 --bf16
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

from beam_search import KhoruzhiiSearchConfig, setup_model_for_inference  # noqa: E402
from beam_search_bidir import BidirSolver  # noqa: E402
from cayley.search import load_model_checkpoint  # noqa: E402
from cayley.verify import load_submission, load_test_states, verify_path  # noqa: E402
from megaminx.puzzle import Megaminx  # noqa: E402


# --------------------------------------------------------------------------- D1
def diag_agreement_vs_distance(puzzle: Megaminx, rng: np.random.Generator,
                               max_k: int = 14, n_samples: int = 400) -> dict:
    """From solved, take length-k non-backtracking random walks; report the mean
    number of the 120 positions that DIFFER from solved (= positions a length<=k
    residual disturbs). Coordinate-hash recall for residual k ~ (agree/120)^P."""
    solved = puzzle.solved_state
    names = list(puzzle.move_names)
    inv = {n: puzzle.inverse_name(n) for n in names}
    out = {}
    for k in range(1, max_k + 1):
        diffs = []
        for _ in range(n_samples):
            cur = solved
            last = None
            for _step in range(k):
                while True:
                    m = names[rng.integers(len(names))]
                    if last is None or m != inv[last]:
                        break
                cur = puzzle.apply_move(cur, m)
                last = m
            diffs.append(sum(1 for a, b in zip(cur, solved) if a != b))
        d = float(np.mean(diffs))
        out[k] = {"changed_mean": round(d, 1), "agree_frac": round((120 - d) / 120, 3)}
    return out


# ---------------------------------------------------------------- pooling / LSH
def pool_frontiers(fr: dict, map_state: np.ndarray | None = None):
    """Concatenate per-layer frontiers into (states, depth, layer, pos).

    `map_state`: if given (the scramble s as int array), the stored states are
    backward b_tilde and the actual node is b = s[b_tilde]; we return the actual
    nodes. depth = layer+1.
    """
    states, depth, layer, pos = [], [], [], []
    for j, f in enumerate(fr["frontier"]):
        n = f.shape[0]
        nodes = map_state[f] if map_state is not None else f
        states.append(nodes)
        depth.append(np.full(n, j + 1, dtype=np.int32))
        layer.append(np.full(n, j, dtype=np.int32))
        pos.append(np.arange(n, dtype=np.int32))
    return (np.concatenate(states), np.concatenate(depth),
            np.concatenate(layer), np.concatenate(pos))


def lsh_candidate_pairs(F: np.ndarray, B: np.ndarray, rng: np.random.Generator,
                        n_proj: int, proj_size: int, max_per_b: int,
                        max_cand: int) -> np.ndarray:
    """Coordinate-hash candidate (fi, bi) pairs: nodes agreeing on a random
    subset of positions bucket together. Multiple projections for recall.
    Returns unique pairs as int64 (M, 2)."""
    S = F.shape[1]
    Nb = B.shape[0]
    seen: set[int] = set()
    pairs_fi: list[int] = []
    pairs_bi: list[int] = []
    for _p in range(n_proj):
        proj = rng.choice(S, size=proj_size, replace=False)
        w = rng.integers(1, 2**31, size=proj_size).astype(np.int64)
        kf = (F[:, proj].astype(np.int64) * w).sum(1)
        kb = (B[:, proj].astype(np.int64) * w).sum(1)
        order = np.argsort(kf, kind="stable")
        kf_sorted = kf[order]
        lo = np.searchsorted(kf_sorted, kb, side="left")
        hi = np.searchsorted(kf_sorted, kb, side="right")
        has = np.nonzero(hi > lo)[0]
        for bi in has:
            cand_f = order[lo[bi]:hi[bi]][:max_per_b]
            for fi in cand_f:
                key = int(fi) * Nb + int(bi)
                if key not in seen:
                    seen.add(key)
                    pairs_fi.append(int(fi))
                    pairs_bi.append(int(bi))
            if len(seen) >= max_cand:
                break
        if len(seen) >= max_cand:
            break
    if not pairs_fi:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([np.asarray(pairs_fi), np.asarray(pairs_bi)], axis=1)


def residual_states(Bnodes: np.ndarray, Fnodes: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """z = b^-1 . f for each (fi,bi) pair. z[i] = argsort(b)[f]. V(z) ~ |w_r|."""
    fi = pairs[:, 0]
    bi = pairs[:, 1]
    Binv = np.argsort(Bnodes[bi], axis=1)  # row-wise inverse perms
    return np.take_along_axis(Binv, Fnodes[fi], axis=1)


def brute_min_residual(Fpool, Bpool, solver, rng, n_sample: int, chunk: int = 16,
                       topk: int = 64):
    """UNBIASED closest-approach between the frontiers: sample n_sample nodes per
    side, score V(z) over ALL sample pairs, return the lowest-V(z) pairs (global
    pool indices) + the min. This is what tells us whether the frontiers ever
    come close WITHOUT the coordinate-hash recall bias.

    Returns (best_pairs (K,2 global idx), best_vz (K,), min_vz float).
    """
    Nf = Fpool[0].shape[0]; Nb = Bpool[0].shape[0]
    nf = min(n_sample, Nf); nb = min(n_sample, Nb)
    fsel = rng.choice(Nf, nf, replace=False) if Nf > nf else np.arange(Nf)
    bsel = rng.choice(Nb, nb, replace=False) if Nb > nb else np.arange(Nb)
    Fb = Fpool[0][fsel]                  # (nf,S)
    Bb = Bpool[0][bsel]                  # (nb,S)
    Binv = np.argsort(Bb, axis=1)        # (nb,S) inverse perms
    # running top-K lowest V(z) across all (f in sample) x (b in sample)
    best_v = np.full(topk, np.inf, dtype=np.float32)
    best_f = np.full(topk, -1, dtype=np.int64)   # local f index
    best_b = np.full(topk, -1, dtype=np.int64)   # local b index
    for c0 in range(0, nf, chunk):
        Fc = Fb[c0:c0 + chunk]                       # (C,S)
        C = Fc.shape[0]
        # z[c,bj,k] = Binv[bj, Fc[c,k]]
        z = Binv[:, Fc]                              # (nb, C, S)
        z = np.transpose(z, (1, 0, 2)).reshape(C * nb, -1)
        vz = solver.eval_V(z)                        # (C*nb,)
        # fold into running top-K
        cand_v = np.concatenate([best_v, vz])
        f_local = np.repeat(np.arange(c0, c0 + C), nb)
        b_local = np.tile(np.arange(nb), C)
        cand_f = np.concatenate([best_f, f_local])
        cand_b = np.concatenate([best_b, b_local])
        keep = np.argsort(cand_v)[:topk]
        best_v, best_f, best_b = cand_v[keep], cand_f[keep], cand_b[keep]
    valid = best_f >= 0
    best_pairs = np.stack([fsel[best_f[valid]], bsel[best_b[valid]]], axis=1)
    return best_pairs, best_v[valid], float(best_v.min())


# ----------------------------------------------------------------------- M0/M-LSH
def exact_meetings(Fpool, Bpool):
    """Residual-0 meetings (FULL recall): forward state == backward node.
    Returns list of (fi, bi) global-pool indices."""
    fstates, _fd, _fl, _fp = Fpool
    bstates, _bd, _bl, _bp = Bpool
    fmap: dict[bytes, int] = {}
    for i in range(fstates.shape[0]):
        kb = fstates[i].tobytes()
        if kb not in fmap:  # keep first (any depth; we min over total later)
            fmap[kb] = i
    hits = []
    for j in range(bstates.shape[0]):
        fi = fmap.get(bstates[j].tobytes())
        if fi is not None:
            hits.append((fi, j))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "m_az_v4_v_only.pt")
    ap.add_argument("--best-csv", type=Path,
                    default=PROJECT / "submissions" / "merge_v14_plus_min_count_v4.csv")
    ap.add_argument("--pids", type=str, default=None, help="comma list; overrides --n-hard")
    ap.add_argument("--n-hard", type=int, default=20, help="N longest-path pids from best-csv")
    ap.add_argument("--beam", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=160, help="cap for the uni baseline")
    ap.add_argument("--depth-frac", type=float, default=0.55,
                    help="run each side to ceil(depth_frac * L_floor) layers")
    ap.add_argument("--pool-cap", type=int, default=250000,
                    help="subsample each side's LSH pool to this many nodes")
    ap.add_argument("--n-proj", type=int, default=32)
    ap.add_argument("--proj-size", type=int, default=10)
    ap.add_argument("--max-per-b", type=int, default=4)
    ap.add_argument("--max-cand", type=int, default=400000)
    ap.add_argument("--brute-sample", type=int, default=1500,
                    help="unbiased closest-approach: sample this many nodes/side, score all pairs")
    ap.add_argument("--v-keep", type=float, default=9.0, help="close residuals with V(z) <= this")
    ap.add_argument("--k-close", type=int, default=96, help="exact-close the K lowest-V(z) candidates")
    ap.add_argument("--close-beam", type=int, default=8192)
    ap.add_argument("--close-steps", type=int, default=20)
    ap.add_argument("--phi-check", action="store_true",
                    help="report whether ||phi(f)-phi(b)|| tracks d(f,b) (front-to-front go/no-go)")
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    best = load_submission(args.best_csv)
    floor_len = {pid: len(p) for pid, p in best.items()}

    if args.pids:
        pids = [int(x) for x in args.pids.split(",") if x.strip()]
    else:
        pids = [pid for pid, _ in sorted(floor_len.items(), key=lambda kv: -kv[1])][:args.n_hard]

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    setup_model_for_inference(model)
    solver = BidirSolver(puzzle, model, device=args.device,
                         internal_batch_size=args.internal_batch_size,
                         random_seed=args.seed, profile=False)

    print(f"model={args.checkpoint.name} dtype={dtype} device={args.device} beam={args.beam}")
    print(f"pids ({len(pids)}): {pids}")

    d1 = diag_agreement_vs_distance(puzzle, rng)
    print("\n[D1] residual-length -> changed positions (of 120) / agreement fraction")
    for k, v in d1.items():
        print(f"   k={k:2d}  changed~{v['changed_mean']:5.1f}  agree={v['agree_frac']:.3f}")
    print("   (coordinate-hash recall for residual k ~ agree_frac**proj_size)\n")

    results = []
    for pid in pids:
        s = states[pid]
        s_arr = np.array(s, dtype=np.int8)
        L_floor = floor_len.get(pid)
        D_max = int(np.ceil(args.depth_frac * L_floor)) if L_floor else args.max_steps // 2
        row = {"pid": pid, "L_floor": L_floor, "D_max": D_max}
        t0 = time.time()

        # --- unidirectional baseline (matched width, single pass)
        f_u, L_uni, path_u, _ = solver.solve(
            s, KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps,
                                     num_attempts=1))
        if f_u and not verify_path(puzzle, s, path_u).ok:
            f_u = False
        row["L_uni"] = L_uni if f_u else None

        # --- frontiers: forward on s, backward on s^-1
        fwd = solver.beam_frontiers(s, args.beam, D_max)
        s_inv = puzzle.invert_state(s)
        bwd = solver.beam_frontiers(s_inv, args.beam, D_max)
        Fpool = pool_frontiers(fwd)                  # actual forward states f
        Bpool = pool_frontiers(bwd, map_state=s_arr)  # actual backward nodes b = s[b_tilde]
        row["n_fwd"] = int(Fpool[0].shape[0])
        row["n_bwd"] = int(Bpool[0].shape[0])

        # --- M0 exact meetings (full recall)
        m0 = exact_meetings(Fpool, Bpool)
        row["n_exact_meet"] = len(m0)

        best_total = None
        best_info = None

        def consider(fi, bi, w_r_names, resid_len):
            nonlocal best_total, best_info
            d_f = int(Fpool[1][fi]); d_b = int(Bpool[1][bi])
            total = d_f + resid_len + d_b
            if best_total is not None and total >= best_total:
                return
            # reconstruct full path and VERIFY against original s
            wf = solver.reconstruct_words(fwd["tree_move"], fwd["tree_idx"],
                                          int(Fpool[2][fi]), [int(Fpool[3][fi])])[0]
            wb = solver.reconstruct_words(bwd["tree_move"], bwd["tree_idx"],
                                          int(Bpool[2][bi]), [int(Bpool[3][bi])])[0]
            w_f = [puzzle.move_names[m] for m in wf]
            w_b = [puzzle.move_names[m] for m in wb]
            full = w_f + list(w_r_names) + puzzle.invert_path(w_b)
            if verify_path(puzzle, s, full).ok:
                best_total = total
                best_info = {"total": total, "d_f": d_f, "d_b": d_b,
                             "resid": resid_len, "len": len(full)}

        # exact-meeting splices (residual 0)
        for fi, bi in m0:
            consider(fi, bi, [], 0)

        # --- M-LSH near-meetings -> V(z) filter -> exact close
        # subsample pools for the (Python-side) LSH join
        def subsample(pool, cap):
            n = pool[0].shape[0]
            if n <= cap:
                return pool, np.arange(n)
            idx = rng.choice(n, size=cap, replace=False)
            return tuple(a[idx] for a in pool), idx

        (Fs, Fd, Fl, Fp), fidx = subsample(Fpool, args.pool_cap)
        (Bs, Bd, Bl, Bp), bidx = subsample(Bpool, args.pool_cap)
        pairs = lsh_candidate_pairs(Fs, Bs, rng, args.n_proj, args.proj_size,
                                    args.max_per_b, args.max_cand)
        row["n_lsh_cand"] = int(pairs.shape[0])
        lsh_min_vz = None
        n_closed = 0
        if pairs.shape[0] > 0:
            z = residual_states(Bs, Fs, pairs)            # (M, S)
            vz = solver.eval_V(z)                          # (M,)
            lsh_min_vz = float(vz.min())
            order = np.argsort(vz)
            keep = order[(vz[order] <= args.v_keep)][:args.k_close]
            for k in keep:
                fi_l, bi_l = int(pairs[k, 0]), int(pairs[k, 1])
                fi_g, bi_g = int(fidx[fi_l]), int(bidx[bi_l])
                # potential best: skip if even V(z) can't beat current best_total
                d_f = int(Fpool[1][fi_g]); d_b = int(Bpool[1][bi_g])
                if best_total is not None and d_f + d_b + int(round(vz[k])) >= best_total + 6:
                    continue
                zc = z[k]
                fc, lc, wr, _ = solver.solve(
                    tuple(int(x) for x in zc),
                    KhoruzhiiSearchConfig(beam_width=args.close_beam,
                                          num_steps=args.close_steps, num_attempts=1))
                if not fc:
                    continue
                # sanity: applying w_r to f must reach b
                f_state = tuple(int(x) for x in Fpool[0][fi_g])
                if puzzle.apply_path(f_state, wr) != tuple(int(x) for x in Bpool[0][bi_g]):
                    continue
                n_closed += 1
                consider(fi_g, bi_g, wr, lc)
        row["lsh_min_vz"] = round(lsh_min_vz, 2) if lsh_min_vz is not None else None

        # --- UNBIASED closest-approach (brute-force sampled) + close its best pairs
        brute_min_vz = None
        if args.brute_sample > 0 and Fpool[0].shape[0] and Bpool[0].shape[0]:
            bpairs, bvz, brute_min_vz = brute_min_residual(
                Fpool, Bpool, solver, rng, args.brute_sample)
            # report the split of the unbiased-closest pair
            if bpairs.shape[0]:
                fi0, bi0 = int(bpairs[0, 0]), int(bpairs[0, 1])
                row["brute_closest_split"] = [int(Fpool[1][fi0]), round(float(bvz[0]), 1),
                                              int(Bpool[1][bi0])]
            # try to close the lowest-V(z) brute pairs that could beat current best
            for k in range(min(args.k_close, bpairs.shape[0])):
                if bvz[k] > args.v_keep:
                    break
                fi_g, bi_g = int(bpairs[k, 0]), int(bpairs[k, 1])
                d_f = int(Fpool[1][fi_g]); d_b = int(Bpool[1][bi_g])
                if best_total is not None and d_f + d_b + int(round(bvz[k])) >= best_total:
                    continue
                zc = residual_states(Bpool[0], Fpool[0],
                                     np.array([[fi_g, bi_g]], dtype=np.int64))[0]
                fc, lc, wr, _ = solver.solve(
                    tuple(int(x) for x in zc),
                    KhoruzhiiSearchConfig(beam_width=args.close_beam,
                                          num_steps=args.close_steps, num_attempts=1))
                if not fc:
                    continue
                if puzzle.apply_path(tuple(int(x) for x in Fpool[0][fi_g]), wr) != \
                        tuple(int(x) for x in Bpool[0][bi_g]):
                    continue
                n_closed += 1
                consider(fi_g, bi_g, wr, lc)
        row["brute_min_vz"] = round(brute_min_vz, 2) if brute_min_vz is not None else None
        row["n_closed"] = n_closed
        row["best_meeting"] = best_info
        row["L_bd"] = best_total

        # --- phi feasibility (front-to-front go/no-go): does ||phi(f)-phi(b)|| ~ d(f,b)?
        if args.phi_check and pairs.shape[0] > 0:
            ns = min(2000, pairs.shape[0])
            sel = rng.choice(pairs.shape[0], size=ns, replace=False)
            zc = residual_states(Bs, Fs, pairs[sel])
            vz_s = solver.eval_V(zc)
            phf = solver.penultimate_embed(Fs[pairs[sel, 0]])
            phb = solver.penultimate_embed(Bs[pairs[sel, 1]])
            if phf is not None and phb is not None:
                dist = np.linalg.norm(phf - phb, axis=1)
                # correlation of embedding-distance with the V(z) proxy of d(f,b)
                cc = float(np.corrcoef(dist, vz_s)[0, 1])
                row["phi_corr_vz"] = round(cc, 3)
            else:
                row["phi_corr_vz"] = None

        row["wall_s"] = round(time.time() - t0, 1)
        bm = best_info
        print(f"pid={pid:4d}  L_floor={L_floor}  L_uni={row['L_uni']}  "
              f"exact_meet={len(m0)}  brute_minV(z)={row['brute_min_vz']}  "
              f"lsh_minV(z)={row['lsh_min_vz']}  closed={n_closed}  "
              f"L_bd={best_total}"
              + (f" (d_f={bm['d_f']},resid={bm['resid']},d_b={bm['d_b']})" if bm else "")
              + (f"  phi_corr={row.get('phi_corr_vz')}" if args.phi_check else "")
              + f"  [{row['wall_s']}s]", flush=True)
        results.append(row)

    # ---- summary
    solved_uni = [r for r in results if r["L_uni"] is not None and r["L_bd"] is not None]
    print(f"\n==== summary over {len(results)} pids ====")
    if solved_uni:
        wins = sum(1 for r in solved_uni if r["L_bd"] < r["L_uni"])
        ties = sum(1 for r in solved_uni if r["L_bd"] == r["L_uni"])
        du = sum(r["L_uni"] for r in solved_uni)
        db = sum(r["L_bd"] for r in solved_uni)
        print(f"  both uni+bd solved: {len(solved_uni)}  "
              f"sum L_uni={du}  sum L_bd={db}  delta={db - du:+d}  "
              f"(bd wins {wins}, ties {ties}, losses {len(solved_uni) - wins - ties})")
    vs_floor = [r for r in results if r["L_bd"] is not None and r["L_floor"] is not None]
    if vs_floor:
        fw = sum(1 for r in vs_floor if r["L_bd"] < r["L_floor"])
        print(f"  bd beats L_floor on {fw}/{len(vs_floor)} pids")
    n_any_meet = sum(1 for r in results if (r["n_exact_meet"] or r["n_closed"]))
    print(f"  pids with ANY verified meeting: {n_any_meet}/{len(results)}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                       "d1_agreement": d1, "results": results}, f, indent=1)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
