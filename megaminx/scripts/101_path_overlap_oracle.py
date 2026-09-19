"""Cheap headroom oracles: Dir-6 (path recombination) + Dir-10 (portfolio ceiling).

Both are pure offline analyses over existing submission CSVs. Nothing is solved
and no model is loaded. The point is to decide whether two expensive research
families deserve any engineering, BEFORE building them.

Dir-6 -- zero-bridge recombination ceiling
------------------------------------------
The deep-research reports rank "population path search / route relinking" as
"highest practical upside". Its cheapest gate: do two independent solutions for
the SAME pid ever pass through the exact same intermediate 120-perm state? If
so, we can stitch prefix(reach-h-fastest) + suffix(leave-h-fastest) at zero
bridge cost. We build the per-pid UNION GRAPH of all pooled solution paths
(nodes = distinct states, directed unit-weight edges = consecutive states in any
path) and BFS from scramble -> solved. That shortest path is the exact ceiling
of zero-bridge recombination (it allows multiple crossovers, strictly more than
a single splice). Headroom = best_single_path - union_shortest_path, summed over
pids. If ~0, the whole route-relinking / path-space family is dead for us
(state sharing is measure-zero away from solved); script 82 already failed the
*bridged* form, this settles the *un*-bridged upper bound.

Dir-10 -- portfolio / routing ceiling
-------------------------------------
Per-pid min over geometrically distinct config CSVs (= what min-merge realizes),
its total vs the current submitted best, and per-source unique-win attribution.
Tells us if a learned router over our existing search modes has any headroom
beyond the free post-hoc min-merge. Reported with an ours-only vs
with-community split (submission policy: community paths are diagnostic only).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/101_path_overlap_oracle.py
    # or override the pool:
    ... --csvs a.csv,b.csv,... --max-path-len 200 --out-json results/overlap_oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.verify import load_submission, load_test_states
from megaminx.puzzle import Megaminx

# Curated pool of genuinely distinct full-1001 solver geometries. Merges are
# included only to carry community / cross-source distinct paths; dedupe by
# move-string collapses redundant copies. Tag = "ours" | "comm".
DEFAULT_POOL = [
    ("m_az_v4_prod_1001.csv", "ours"),            # AZ v4 V, sym4+niss, GPU
    ("m_dd_v0_50ep_prod_1001.csv", "ours"),       # m_dd_v0 V (different model)
    ("merge_v12_az_v4_plus_our.csv", "ours"),     # our standalone best (no community)
    ("tpu_v16_full1001.csv", "ours"),             # TPU K=4 B=131k
    ("tpu_v17_full.csv", "ours"),                 # TPU K=8 B=262k
    ("tpu_v19a_fix.csv", "ours"),                 # TPU K=4 B=1M
    ("merge_tpu_v19b_standalone.csv", "ours"),    # TPU K=4 B=1M (v19b)
    ("phase_b_plus198.csv", "ours"),              # m05 fresh + sym rescue stack
    ("m42_lowerq25_full1001.csv", "ours"),        # distributional V (low q25)
    ("merge_v14_plus_min_count_v4.csv", "comm"),  # current best (incl community v4)
    ("merge_v13_az_v4_plus_community.csv", "comm"),  # community 76,251 paths
    ("merge_v16_public73731_plus_bridge.csv", "comm"),  # public 73,731 + bridge
]

CURRENT_BEST = 75200


def _hash_state(state) -> bytes:
    return np.asarray(state, dtype=np.int8).tobytes()


def _replay_states(initial_state, path, puzzle):
    """List of state-hashes along the path, length len(path)+1. None if the
    path does not end at the solved state (defensive: drop corrupt paths)."""
    cur = tuple(initial_state)
    hashes = [_hash_state(cur)]
    for move in path:
        cur = puzzle.apply_move(cur, move)
        hashes.append(_hash_state(cur))
    return hashes


def union_shortest(initial_h: bytes, solved_h: bytes, paths_hashes: list[list[bytes]]) -> int:
    """BFS shortest scramble->solved over the union DAG of all path edges."""
    adj: dict[bytes, set[bytes]] = {}
    for hs in paths_hashes:
        for a, b in zip(hs[:-1], hs[1:]):
            adj.setdefault(a, set()).add(b)
    # BFS (unit weights).
    dist = {initial_h: 0}
    dq = deque([initial_h])
    while dq:
        u = dq.popleft()
        if u == solved_h:
            return dist[u]
        for v in adj.get(u, ()):  # type: ignore[arg-type]
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist.get(solved_h, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", type=str, default=None,
                    help="Comma-separated CSV paths (override the curated pool).")
    ap.add_argument("--submissions-dir", type=Path, default=PROJECT / "submissions")
    ap.add_argument("--test-csv", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--max-path-len", type=int, default=200,
                    help="Drop paths longer than this (fallback filter).")
    ap.add_argument("--top-report", type=int, default=25)
    ap.add_argument("--out-json", type=Path, default=PROJECT / "results" / "overlap_oracle.json")
    args = ap.parse_args()

    puzzle = Megaminx.load(args.puzzle_info)
    solved_h = _hash_state(puzzle.solved_state)
    states = load_test_states(args.test_csv)

    if args.csvs:
        pool = [(Path(p.strip()).name, "ours") for p in args.csvs.split(",") if p.strip()]
        pool_dir = Path(args.csvs.split(",")[0]).parent if "/" in args.csvs or "\\" in args.csvs else args.submissions_dir
    else:
        pool = DEFAULT_POOL
        pool_dir = args.submissions_dir

    # Load each source: pid -> path.
    sources: list[tuple[str, str, dict[int, list[str]]]] = []
    for name, tag in pool:
        fp = (pool_dir / name) if not Path(name).is_absolute() else Path(name)
        if not fp.exists():
            print(f"  WARNING missing, skipping: {fp}", flush=True)
            continue
        sub = load_submission(fp)
        sources.append((name, tag, sub))
        print(f"  loaded {name:42s} [{tag}] pids={len(sub)} total={sum(len(p) for p in sub.values())}", flush=True)
    if not sources:
        print("no sources loaded", flush=True)
        return 1
    print(f"\npool: {len(sources)} sources\n", flush=True)

    from collections import Counter

    # ---- single validation + replay pass: keep only valid solving paths ----
    t0 = time.time()
    valid_by_pid: dict[int, list[tuple[str, str, int, list[bytes]]]] = {}
    dropped_paths = 0
    for name, tag, sub in sources:
        for pid, p in sub.items():
            if pid not in states:
                continue
            if len(p) == 0 or len(p) > args.max_path_len:
                if len(p) == 0:
                    dropped_paths += 1
                continue
            hs = _replay_states(states[pid], p, puzzle)
            if hs[-1] != solved_h:
                dropped_paths += 1
                continue
            valid_by_pid.setdefault(pid, []).append((name, tag, len(p), hs))
    print(f"  validation pass done ({time.time()-t0:.0f}s), dropped {dropped_paths} empty/non-solving",
          flush=True)

    # ---- Dir-10: portfolio ceiling + attribution (valid paths only) ----
    minmerge_total = 0
    minmerge_total_ours = 0
    sole_wins = {name: 0 for name, _, _ in sources}
    sole_wins_ours = {name: 0 for name, _, _ in sources if _tag_of(sources, name) == "ours"}
    covered = 0
    covered_ours = 0
    for pid, entries in valid_by_pid.items():
        covered += 1
        m = min(L for _, _, L, _ in entries)
        minmerge_total += m
        ach = [name for name, _, L, _ in entries if L == m]
        if len(ach) == 1:
            sole_wins[ach[0]] += 1
        ours = [(name, L) for name, tag, L, _ in entries if tag == "ours"]
        if ours:
            covered_ours += 1
            mo = min(L for _, L in ours)
            minmerge_total_ours += mo
            ach_o = [name for name, L in ours if L == mo]
            if len(ach_o) == 1:
                sole_wins_ours[ach_o[0]] += 1

    # ---- Dir-6: zero-bridge recombination ceiling (union-graph shortest path) ----
    n_eligible = 0          # pids with >=2 distinct valid paths
    n_recomb_pids = 0       # pids where union path < best single
    total_best_single = 0
    total_dag_best = 0
    headroom_total = 0
    per_pid_wins: list[tuple[int, int, int, int]] = []  # (pid, best_single, dag_best, saved)
    shared_interior_hist: dict[int, int] = {}
    for pid, entries in valid_by_pid.items():
        # Dedupe identical paths (same state-hash sequence).
        seen: set[bytes] = set()
        paths_hashes: list[list[bytes]] = []
        lens: list[int] = []
        for _, _, L, hs in entries:
            key = b"".join(hs)
            if key in seen:
                continue
            seen.add(key)
            paths_hashes.append(hs)
            lens.append(L)
        if len(paths_hashes) < 2:
            continue
        n_eligible += 1
        initial_h = paths_hashes[0][0]
        best_single = min(lens)
        dag_best = union_shortest(initial_h, solved_h, paths_hashes)
        if dag_best < 0:
            dag_best = best_single
        total_best_single += best_single
        total_dag_best += dag_best
        saved = best_single - dag_best
        if saved > 0:
            n_recomb_pids += 1
            headroom_total += saved
            per_pid_wins.append((pid, best_single, dag_best, saved))
        cnt: Counter = Counter()
        for hs in paths_hashes:
            for h in set(hs[1:]):  # exclude shared scramble at position 0
                cnt[h] += 1
        n_shared = sum(1 for h, c in cnt.items() if c >= 2 and h != solved_h)
        shared_interior_hist[n_shared] = shared_interior_hist.get(n_shared, 0) + 1

    per_pid_wins.sort(key=lambda r: -r[3])

    # ---- Report ----
    print("\n" + "=" * 72)
    print("Dir-10  PORTFOLIO / ROUTING CEILING (per-pid min over distinct configs)")
    print("=" * 72)
    print(f"  pids covered:                {covered}")
    print(f"  min-merge total (all srcs):  {minmerge_total:,}   vs current best {CURRENT_BEST:,}  "
          f"(gap {CURRENT_BEST - minmerge_total:+,})")
    print(f"  min-merge total (ours only): {minmerge_total_ours:,}  over {covered_ours} pids")
    print("\n  per-source SOLE wins (pid where this source is the unique min):")
    print("    [with community pool]")
    for name, _, _ in sources:
        print(f"      {name:44s} {sole_wins[name]:5d}")
    print("    [ours-only pool]")
    for name in sole_wins_ours:
        print(f"      {name:44s} {sole_wins_ours[name]:5d}")

    print("\n" + "=" * 72)
    print("Dir-6  ZERO-BRIDGE RECOMBINATION CEILING (union-graph shortest path)")
    print("=" * 72)
    print(f"  eligible pids (>=2 distinct valid paths): {n_eligible}")
    print(f"  dropped non-solving paths:                {dropped_paths}")
    print(f"  pids with ANY recombination win:          {n_recomb_pids}")
    print(f"  sum best-single  over eligible:           {total_best_single:,}")
    print(f"  sum union-best   over eligible:           {total_dag_best:,}")
    print(f"  >>> TOTAL recombination headroom:         {headroom_total:,} moves "
          f"({100.0*headroom_total/max(total_best_single,1):.3f}% of eligible)")
    print("\n  interior-shared-state histogram (#pids by # states shared across >=2 paths):")
    for k in sorted(shared_interior_hist):
        if k <= 10 or shared_interior_hist[k] > 0:
            bar = "#" * min(60, shared_interior_hist[k])
            print(f"      {k:4d} shared : {shared_interior_hist[k]:5d}  {bar}")
        if k == 10:
            rest = sum(v for kk, v in shared_interior_hist.items() if kk > 10)
            print(f"      >10  shared : {rest:5d}")
            break
    print(f"\n  top {args.top_report} recombination wins (pid: best_single -> union_best  saved):")
    for pid, bs, db, sv in per_pid_wins[: args.top_report]:
        print(f"      pid {pid:4d}: {bs:4d} -> {db:4d}   ({sv:+d})")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "pool": [(n, t) for n, t, _ in sources],
            "dir10": {
                "covered": covered,
                "minmerge_total": minmerge_total,
                "minmerge_total_ours": minmerge_total_ours,
                "current_best": CURRENT_BEST,
                "sole_wins": sole_wins,
                "sole_wins_ours": sole_wins_ours,
            },
            "dir6": {
                "n_eligible": n_eligible,
                "n_recomb_pids": n_recomb_pids,
                "total_best_single": total_best_single,
                "total_dag_best": total_dag_best,
                "headroom_total": headroom_total,
                "top_wins": per_pid_wins[:100],
                "shared_interior_hist": shared_interior_hist,
            },
        }, f, indent=2)
    print(f"\nwrote {args.out_json}")
    return 0


def _tag_of(sources, name):
    for n, t, _ in sources:
        if n == name:
            return t
    return "ours"


if __name__ == "__main__":
    sys.exit(main())
