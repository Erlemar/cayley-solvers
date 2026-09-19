"""Deep BFS builder (d7 / d8): hash-only endgame table + full-state anchors.

Same sweep and the same Zobrist table as 03_build_bfs.py (seed 0 -> hashes are
directly comparable), but restructured for scale.  Three differences, each
forced by a measured blow-up in the d6 script:

  1. States at the FINAL depth are never materialized.  d7 has ~407M states,
     which is 35.8 GB as uint8 (d8 would be 545 GB).  The endgame table only
     needs their hashes, and nothing expands the last level, so we keep hashes
     and drop states there.
  2. The final table is assembled by scattering depths into the already-sorted
     hash union, instead of argsort over ~435M int64.  argsort would cost a
     3.5 GB index array plus a full gather; searchsorted-scatter costs one
     int64 temp per level.
  3. Child expansion + hashing + the seen-filter run in a fork-based process
     pool.  On Linux the frontier and the seen set are inherited copy-on-write,
     so workers read them with no IPC and no duplication; only the surviving
     fresh hashes travel back.

    python3 tetraminx/scripts/03b_build_bfs_deep.py --max-depth 7 --workers 48

Level sizes (measured through d6): 1 / 24 / 408 / 6592 / 105136 / 1659416 /
26008172, branching ~15.67, so d7 ~ 407M (cumulative ~435M, 3.9 GB of table).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]

# set in the parent before the pool forks; children inherit copy-on-write
_G: dict = {}


def zobrist_table(state_size: int, n_values: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**63 - 1, size=(state_size, n_values), dtype=np.int64)


def hash_states(states: np.ndarray, ztab: np.ndarray) -> np.ndarray:
    """XOR-fold Zobrist hash; states is (N, S) of small ints."""
    out = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[1]):
        out ^= ztab[i][states[:, i]]
    return out


def _expand_chunk(task):
    """Expand frontier[lo:hi], hash, drop in-chunk dups and anything already seen.

    Returns (fresh_hashes, states_or_None).  States come back only when this
    level is kept in full (anchors) or when the caller asked for a sample.
    """
    lo, hi, want_states, n_sample, seed = task
    frontier = _G["frontier"]
    seen = _G["seen"]
    ztab = _G["ztab"]
    gens_flat = _G["gens_flat"]
    n = _G["n"]

    block = frontier[lo:hi]
    children = block[:, gens_flat].reshape(-1, n)
    h = hash_states(children, ztab)

    uniq_h, first_idx = np.unique(h, return_index=True)
    del h
    pos = np.searchsorted(seen, uniq_h)
    np.clip(pos, 0, seen.size - 1, out=pos)
    fresh = seen[pos] != uniq_h
    keep_h = uniq_h[fresh]
    keep_idx = first_idx[fresh]

    out_states = None
    if keep_h.size:
        if want_states:
            out_states = children[keep_idx].astype(np.uint8)
        elif n_sample > 0:
            k = min(n_sample, keep_h.size)
            sel = np.random.default_rng(seed).choice(keep_h.size, size=k, replace=False)
            out_states = children[keep_idx[sel]].astype(np.uint8)
            return keep_h, out_states, keep_h[sel]
    return keep_h, out_states, None


def _run_pool(fn, tasks, workers):
    """Map fn over tasks via a fork pool, falling back to serial.

    Fork is what makes the globals in _G free to read (copy-on-write); `spawn`
    would re-pickle the frontier for every task, which is worse than serial. So
    on a platform without fork (Windows) we just run serially -- fine for the
    small-depth builds used to test this script; the real d7/d8 runs are Linux.
    """
    if len(tasks) == 1 or workers <= 1:
        return [fn(t) for t in tasks]
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return [fn(t) for t in tasks]
    with ctx.Pool(processes=min(workers, len(tasks))) as pool:
        return pool.map(fn, tasks, chunksize=1)


def _bake_chunk(task):
    """Exact distances for all 24 children of anchor_states[lo:hi]."""
    lo, hi = task
    states = _G["anchor_states"][lo:hi]
    ztab, gens_flat, n = _G["ztab"], _G["gens_flat"], _G["n"]
    hashes, depths = _G["tbl_h"], _G["tbl_d"]
    children = states[:, gens_flat].reshape(-1, n)
    h = hash_states(children, ztab)
    pos = np.searchsorted(hashes, h)
    np.clip(pos, 0, hashes.size - 1, out=pos)
    hit = hashes[pos] == h
    out = np.full(h.size, -1, dtype=np.int8)
    out[hit] = depths[pos[hit]]
    return out.reshape(hi - lo, -1)


def bake_targets(a_states, hashes, depths, ztab, gens_flat, n, workers, chunk,
                 miss_depth=None):
    """(N, 24) int8 of exact child distances -- what the Q anchors actually need.

    Baking means the table is a BUILD-time artifact only: a d8 table would be 54 GB
    of hashes, fitting neither a 23 GB card nor a 31 GB training host, while a baked
    50M-anchor file is ~5.6 GB and needs no lookup at all.

    `miss_depth` is what makes the NEXT level free. The generator set is closed
    under inverse (checked: 0 generators lack their inverse), so the Cayley graph is
    undirected and every child of a depth-d state is at d-1, d or d+1. Hence for an
    anchor at depth exactly `max_depth`, a child absent from the d<=max_depth table
    cannot be anything but `max_depth + 1`. Passing miss_depth=max_depth+1 therefore
    yields EXACT 24-way targets one level deeper than the table -- no deeper build.
    With miss_depth=None any miss is a bug and asserts.
    """
    _G["anchor_states"] = a_states
    _G["tbl_h"], _G["tbl_d"] = hashes, depths
    bounds = [(lo, min(lo + chunk, a_states.shape[0]))
              for lo in range(0, a_states.shape[0], chunk)]
    t0 = time.time()
    parts = _run_pool(_bake_chunk, bounds, workers)
    q = np.concatenate(parts)
    del parts
    n_miss = int((q < 0).sum())
    if miss_depth is None:
        assert n_miss == 0, (
            f"{n_miss} anchor children missing from the table -- anchor depth must be "
            f"<= max_depth - 1 for every one of the 24 children to be present")
    else:
        q[q < 0] = miss_depth
    print(f"baked 24-way targets for {q.shape[0]:,} anchors in {time.time() - t0:.1f}s"
          + (f" ({n_miss:,} children resolved to d={miss_depth} by the undirected-graph "
             f"argument)" if miss_depth is not None else ""), flush=True)
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "puzzle_info.json")
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--anchor-depth", type=int, default=None,
                    help="keep every state up to this depth as anchors "
                         "(default max_depth-1: exactly the states whose 24 "
                         "children are all inside the table)")
    ap.add_argument("--sample-final", type=int, default=2_000_000,
                    help="random states sampled at the final depth (states at "
                         "that depth are otherwise never materialized)")
    ap.add_argument("--chunk", type=int, default=500_000,
                    help="frontier states per task (memory knob: chunk*24*88 B "
                         "of children live per worker)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--hash-out", type=Path, required=True)
    ap.add_argument("--torch-out", type=Path, default=None)
    ap.add_argument("--states-out", type=Path, default=None)
    ap.add_argument("--anchor-sample", type=int, default=0,
                    help="cap anchor states kept per FULL level (0 = keep all). "
                         "At d8 the d7 level alone is 405.6M states = 35.7 GB, so "
                         "the anchor copy has to be a sample; the frontier used for "
                         "expansion is unaffected.")
    ap.add_argument("--bake-anchor-depth", type=int, default=None,
                    help="bake anchors up to this depth (default max_depth-1). Set it "
                         "EQUAL to max_depth to get exact targets one level deeper "
                         "than the table for free -- see bake_targets' miss_depth.")
    ap.add_argument("--bake-out", type=Path, default=None,
                    help="write anchors with their 24 exact child distances baked in. "
                         "This is what training actually needs, and it removes the "
                         "table from the train-time footprint entirely (a d8 table is "
                         "54 GB of hashes; a baked 50M-anchor file is ~5.6 GB).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    anchor_depth = (args.max_depth - 1) if args.anchor_depth is None else args.anchor_depth

    info = json.loads(args.puzzle_info.read_text(encoding="utf-8"))
    move_names = list(info["generators"].keys())
    gens = np.array([info["generators"][nm] for nm in move_names], dtype=np.int64)
    solved = np.array(info["central_state"], dtype=np.int64)
    n = len(solved)

    _G["ztab"] = zobrist_table(n, n, seed=args.seed)
    _G["gens_flat"] = gens.reshape(-1)
    _G["n"] = n

    frontier = solved.astype(np.uint8)[None, :]
    seen_sorted = hash_states(frontier, _G["ztab"])

    level_hashes = [seen_sorted.copy()]
    level_sizes = [1]
    anchor_states = [frontier.copy()]
    anchor_depths = [np.zeros(1, dtype=np.int8)]

    print(f"state_size={n} n_gen={gens.shape[0]} workers={args.workers} "
          f"max_depth={args.max_depth} anchor_depth={anchor_depth}", flush=True)

    t0 = time.time()
    for depth in range(1, args.max_depth + 1):
        is_final = depth == args.max_depth
        want_states = depth <= anchor_depth
        # the last level is never expanded, so its states are pure cost
        keep_frontier = not is_final

        _G["frontier"] = frontier
        _G["seen"] = seen_sorted

        bounds = [(lo, min(lo + args.chunk, frontier.shape[0]))
                  for lo in range(0, frontier.shape[0], args.chunk)]
        n_chunks = len(bounds)
        per_chunk_sample = 0
        if is_final and not want_states and args.sample_final > 0:
            per_chunk_sample = max(1, -(-args.sample_final // n_chunks))
        need_states = want_states or keep_frontier or per_chunk_sample > 0
        tasks = [(lo, hi, want_states or keep_frontier, per_chunk_sample,
                  args.seed + 1000 * depth + i)
                 for i, (lo, hi) in enumerate(bounds)]

        t_lvl = time.time()
        results = _run_pool(_expand_chunk, tasks, args.workers)

        h_parts = [r[0] for r in results if r[0].size]
        s_parts = [r[1] for r in results if r[1] is not None and r[1].size]
        sh_parts = [r[2] for r in results if r[2] is not None and r[2].size]
        del results

        new_h = np.concatenate(h_parts) if h_parts else np.zeros(0, np.int64)
        del h_parts
        if need_states and s_parts:
            new_s = np.concatenate(s_parts)
            del s_parts
        else:
            new_s = np.zeros((0, n), np.uint8)

        if per_chunk_sample > 0:
            # sampled states carry their own hashes; dedup them against each other
            samp_h = np.concatenate(sh_parts) if sh_parts else np.zeros(0, np.int64)
            _, s_idx = np.unique(samp_h, return_index=True)
            new_s = new_s[s_idx]
            del samp_h
            new_h = np.unique(new_h)
        elif new_s.shape[0]:
            new_h, idx = np.unique(new_h, return_index=True)
            new_s = new_s[idx]
            del idx
        else:
            new_h = np.unique(new_h)

        level_hashes.append(new_h)
        level_sizes.append(int(new_h.size))

        if want_states:
            if 0 < args.anchor_sample < new_s.shape[0]:
                sel = rng.choice(new_s.shape[0], size=args.anchor_sample, replace=False)
                sel.sort()
                anchor_states.append(new_s[sel])
                anchor_depths.append(np.full(sel.size, depth, dtype=np.int8))
            else:
                # aliasing the frontier is fine (it is never mutated in place);
                # copying it would double a 35.7 GB level at d8
                anchor_states.append(new_s)
                anchor_depths.append(np.full(new_s.shape[0], depth, dtype=np.int8))
        elif per_chunk_sample > 0 and new_s.shape[0]:
            anchor_states.append(new_s)
            anchor_depths.append(np.full(new_s.shape[0], depth, dtype=np.int8))

        frontier = new_s if keep_frontier else np.zeros((0, n), np.uint8)
        if not is_final:
            # nothing consults the seen set after the last level
            seen_sorted = np.sort(np.concatenate([seen_sorted, new_h]))
        cum = sum(level_sizes)
        print(f"depth {depth}: {new_h.size:,} new  (cum {cum:,})  "
              f"chunks={n_chunks}  {time.time() - t_lvl:.1f}s lvl / "
              f"{time.time() - t0:.1f}s total", flush=True)

    # ---- assemble the table without argsort over the full union ----------
    total = sum(level_sizes)
    print(f"assembling table: {total:,} hashes", flush=True)
    hashes = np.concatenate(level_hashes)
    hashes.sort()
    depths = np.empty(hashes.size, dtype=np.int8)
    for depth, lh in enumerate(level_hashes):
        if lh.size:
            depths[np.searchsorted(hashes, lh)] = depth
    del level_hashes
    assert hashes.size == total, "level/table size mismatch"
    d_counts = np.bincount(depths.astype(np.int64), minlength=args.max_depth + 1)
    assert list(d_counts[:args.max_depth + 1]) == level_sizes, \
        f"depth scatter mismatch: {list(d_counts)} vs {level_sizes}"

    args.hash_out.parent.mkdir(parents=True, exist_ok=True)
    t_s = time.time()
    np.savez(args.hash_out, hashes=hashes, depths=depths, ztab=_G["ztab"],
             move_names=np.array(move_names), max_depth=args.max_depth)
    print(f"endgame table: {hashes.size:,} states <= d{args.max_depth} -> "
          f"{args.hash_out} ({time.time() - t_s:.1f}s)", flush=True)

    a_states = np.concatenate(anchor_states)
    a_depths = np.concatenate(anchor_depths)
    anchor_states.clear()

    if args.bake_out is not None:
        bad = args.max_depth - 1 if args.bake_anchor_depth is None else args.bake_anchor_depth
        assert bad <= args.max_depth, "cannot bake anchors deeper than the table"
        miss = (args.max_depth + 1) if bad == args.max_depth else None
        keep = a_depths <= bad
        b_states, b_depths = a_states[keep], a_depths[keep]
        q = bake_targets(b_states, hashes, depths, _G["ztab"], _G["gens_flat"], n,
                         args.workers, max(1, args.chunk // 2), miss_depth=miss)
        import torch
        torch.save({"states": torch.from_numpy(b_states.astype(np.int8)),
                    "q_targets": torch.from_numpy(q),
                    "depths": torch.from_numpy(b_depths.astype(np.int8)),
                    "move_names": move_names,
                    "table_depth": args.max_depth},
                   args.bake_out)
        print(f"baked anchors -> {args.bake_out} "
              f"({b_states.shape[0]:,} states at d<={bad})", flush=True)
        del q, b_states, b_depths
    del hashes, depths
    if args.states_out is not None:
        np.savez(args.states_out, states=a_states, depths=a_depths,
                 move_names=np.array(move_names))
    if args.torch_out is not None:
        import torch
        torch.save({"states": torch.from_numpy(a_states.astype(np.int8)),
                    "distances": torch.from_numpy(a_depths.astype(np.int8)),
                    "move_names": move_names,
                    "level_sizes": level_sizes},
                   args.torch_out)
        print(f"bellman anchors -> {args.torch_out}", flush=True)

    print(f"level sizes: {level_sizes}", flush=True)
    print(f"anchors: {a_states.shape[0]:,} states "
          f"({dict(zip(*[x.tolist() for x in np.unique(a_depths, return_counts=True)]))})",
          flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
