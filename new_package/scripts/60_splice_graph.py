"""Cross-trajectory splicing: shortest path over a graph of every waypoint we have.

THE ONLY LEVER AIMED AT SLACK THAT PROVABLY EXISTS. Measured 2026-08-04: all 44,390
windows of length <=6 in the 28,456 submission are EXACTLY geodesic (true distance ==
window length). So no local rewrite of any radius can fire -- the path is locally
perfect. Yet it is not globally optimal, which means the slack is STRUCTURAL: a route
that is best between each of its own consecutive waypoints, but not the best route.
Changing that needs a different route, which is what this does.

THE GRAPH. Per pid, vertices are STATES drawn from every trajectory we own for that pid
(the corpus keeps long/alternative solutions precisely because they contribute waypoints).
Edges:
  * trajectory steps                    weight 1
  * exact bridges  x -> y               weight d, whenever d(x^-1 y) <= 6
  * endgame descent  x -> solved        weight d, whenever x is inside the d<=6 ball
Dijkstra from the scramble to solved then returns the best splice, which may leave
trajectory A, cross to B, and finish through the table.

WHY BRIDGES <=6 ARE ESSENTIALLY FREE. A bridge is a single d6 hash lookup, not a search.
The competitor's crossover only tried radius 1/2/3 corridors; going to 6 costs nothing
extra because the table is already resident. Longer bridges (7-10, via a B5 x B5 join)
are ~1.8M probes each and are gated behind a budget test, since a bridge only helps if
  cost_from_start(x) + d + cost_to_end(y) < incumbent
and that bound is usually <=0, killing the pair before any join is attempted.

Every spliced path is replayed against the original scramble before being emitted.

    python tetraminx/scripts/60_splice_graph.py \
        --target tetraminx/submissions/submission_28456.csv \
        --corpus tetraminx/submissions tetraminx/results \
        --out tetraminx/submissions/spliced.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import heapq
import io
import json
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "tetraminx" / "data"


class Pz:
    def __init__(self, d: Path):
        info = json.loads((d / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        self.n_gen, self.S = self.gen.shape
        self.ident = np.arange(self.S, dtype=np.int64)
        self.idx = {n: i for i, n in enumerate(self.names)}
        z = np.load(d / "bfs_endgame.npz")
        self.H = z["hashes"]; self.DEP = z["depths"].astype(np.int64); self.ZT = z["ztab"].astype(np.int64)

    def hash_rows(self, st: np.ndarray) -> np.ndarray:
        h = np.zeros(st.shape[0], dtype=np.int64)
        for i in range(self.S):
            h ^= self.ZT[i][st[:, i]]
        return h

    def depth_of(self, st: np.ndarray) -> np.ndarray:
        h = self.hash_rows(st)
        pos = np.searchsorted(self.H, h)
        np.clip(pos, 0, len(self.H) - 1, out=pos)
        return np.where(self.H[pos] == h, self.DEP[pos], -1)

    def descend(self, state: np.ndarray):
        d = int(self.depth_of(state[None, :])[0])
        assert d >= 0
        out, cur = [], state.copy()
        while d > 0:
            kids = cur[self.gen]
            dk = self.depth_of(kids)
            nxt = int(np.argmax(dk == d - 1))
            out.append(nxt); cur = kids[nxt]; d -= 1
        return out


def load_csv(p: Path):
    out = {}
    with io.open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            s = (r.get("path") or "").strip()
            if s:
                out[int(r["initial_state_id"])] = s.split(".")
    return out


def load_json(p: Path):
    out = {}
    try:
        recs = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(recs, list):
        return out
    for r in recs:
        if isinstance(r, dict) and r.get("found") and r.get("verify_ok") and r.get("path"):
            out.setdefault(int(r["pid"]), []).append(r["path"].split("."))
    return out


def gather(dirs, pz: Pz):
    """pid -> list of distinct move-index paths (ALL of them, long ones included)."""
    per = {}
    for d in dirs:
        d = Path(d)
        files = ([d] if d.is_file() else
                 [Path(x) for x in glob.glob(str(d / "**" / "*.csv"), recursive=True)] +
                 [Path(x) for x in glob.glob(str(d / "**" / "*.json"), recursive=True)])
        for f in files:
            if f.suffix == ".csv":
                got = {k: [v] for k, v in load_csv(f).items()}
            else:
                got = load_json(f)
            for pid, plist in got.items():
                for mv in plist:
                    try:
                        t = tuple(pz.idx[m] for m in mv)
                    except KeyError:
                        break
                    per.setdefault(pid, set()).add(t)
    return {k: [list(t) for t in v] for k, v in per.items()}


def splice_pid(pz: Pz, start: np.ndarray, trajs, incumbent: int, max_bridge: int = 6):
    """Dijkstra over waypoints of every trajectory. Returns (word, length) or None."""
    # --- vertices: distinct states across all trajectories
    states, index = [], {}
    traj_nodes = []
    for w in trajs:
        cur = start.copy()
        nodes = []
        key = cur.astype(np.uint8).tobytes()
        if key not in index:
            index[key] = len(states); states.append(cur.copy())
        nodes.append(index[key])
        for m in w:
            cur = cur[pz.gen[m]]
            key = cur.astype(np.uint8).tobytes()
            if key not in index:
                index[key] = len(states); states.append(cur.copy())
            nodes.append(index[key])
        traj_nodes.append((nodes, w))
    ST = np.array(states, dtype=np.int64)
    n = ST.shape[0]

    # --- edges: trajectory steps
    adj = [[] for _ in range(n)]
    for nodes, w in traj_nodes:
        for a in range(len(w)):
            adj[nodes[a]].append((nodes[a + 1], 1, [w[a]]))

    # --- endgame descents (state -> SOLVED sentinel n)
    dep = pz.depth_of(ST)
    for i in np.nonzero(dep >= 0)[0]:
        adj[int(i)].append((n, int(dep[i]), None))     # word filled in lazily

    # --- exact bridges x -> y, gated by a budget so we never test hopeless pairs.
    # cost_from_start / cost_to_end come from the trajectories themselves.
    INF = 10**9
    cfs = np.full(n, INF); cte = np.full(n, INF)
    for nodes, w in traj_nodes:
        for a, node in enumerate(nodes):
            cfs[node] = min(cfs[node], a)
            cte[node] = min(cte[node], len(w) - a)
    cand = [i for i in range(n) if cfs[i] < INF]
    XINV = np.argsort(ST, axis=1)
    bridges = 0
    for i in cand:
        budget = incumbent - cfs[i] - cte
        live = np.nonzero((budget > 0) & (np.arange(n) != i))[0]
        if live.size == 0:
            continue
        elems = XINV[i][ST[live]]                      # element x^-1 y for each y
        d = pz.depth_of(elems)
        ok = np.nonzero((d >= 0) & (d < np.minimum(budget[live], max_bridge + 1)))[0]
        for t in ok:
            j = int(live[t])
            adj[i].append((j, int(d[t]), ("BRIDGE", i, j)))
            bridges += 1

    # --- Dijkstra
    dist = [INF] * (n + 1); prev = [None] * (n + 1)
    dist[0] = 0
    pq = [(0, 0)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u] or u == n:
            continue
        for v, wgt, tag in adj[u]:
            if du + wgt < dist[v]:
                dist[v] = du + wgt; prev[v] = (u, tag)
                heapq.heappush(pq, (dist[v], v))
    if dist[n] >= incumbent:
        return None, bridges

    # --- reconstruct
    chain, v = [], n
    while v != 0:
        u, tag = prev[v]
        chain.append((u, v, tag)); v = u
    chain.reverse()
    word = []
    for u, v, tag in chain:
        if v == n:
            word += pz.descend(ST[u])
        elif tag and tag[0] == "BRIDGE":
            elem = XINV[u][ST[v]]
            word += [int(pz.gen.shape[0]) and m for m in _bridge_word(pz, elem)]
        else:
            word += tag
    return word, bridges


def _bridge_word(pz: Pz, elem: np.ndarray):
    """Word b with perm(b) == elem, via the table. descend() gives the word for elem^-1,
    so reverse+invert (same trap as 42_window_reduce / 59_mitm)."""
    d = pz.descend(elem)
    inv = np.zeros(pz.n_gen, dtype=np.int64)
    for i in range(pz.n_gen):
        for j in range(pz.n_gen):
            if np.array_equal(pz.gen[i][pz.gen[j]], pz.ident):
                inv[i] = j; break
    return [int(inv[m]) for m in reversed(d)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--test-csv", type=Path, default=DATA / "test.csv")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-bridge", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pz = Pz(DATA)
    starts = {}
    with io.open(args.test_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            starts[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
    tgt = {p: [pz.idx[m] for m in mv] for p, mv in load_csv(args.target).items()}
    print(f"target {args.target.name}: {sum(map(len, tgt.values())):,} moves")

    per = gather(args.corpus, pz)
    print(f"corpus: trajectories for {len(per)} pids, "
          f"{sum(len(v) for v in per.values()):,} distinct paths")

    pids = sorted(tgt)
    if args.limit:
        pids = pids[:args.limit]
    saved, t0, nb = 0, time.time(), 0
    for n, pid in enumerate(pids):
        trajs = per.get(pid, [])
        if tgt[pid] not in trajs:
            trajs = trajs + [tgt[pid]]
        word, br = splice_pid(pz, starts[pid], trajs, len(tgt[pid]), args.max_bridge)
        nb += br
        if word is not None:
            s = starts[pid].copy()
            for m in word:
                s = s[pz.gen[m]]
            if np.array_equal(s, pz.ident) and len(word) < len(tgt[pid]):
                print(f"  pid {pid}: {len(tgt[pid])} -> {len(word)}", flush=True)
                saved += len(tgt[pid]) - len(word)
                tgt[pid] = word
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(pids)} pids, saved {saved}, "
                  f"{nb:,} bridges, {time.time()-t0:.0f}s", flush=True)
    print(f"\nsaved {saved} moves; total {sum(map(len, tgt.values())):,}")
    if args.out and saved:
        with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["initial_state_id", "path"])
            for pid in sorted(tgt):
                wr.writerow([pid, ".".join(pz.names[m] for m in tgt[pid])])
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
