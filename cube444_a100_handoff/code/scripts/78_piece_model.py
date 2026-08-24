"""3x3x3 piece model derived from the 4x4x4 colour cube -- phase 2 of two-phase.

WHY THIS EXISTS. Post-processing the best public file is exhausted at -8 (EXPERIMENTS.md
2026-08-05): every window <= 10 is geodesic and our beam matches but never beats it. The
418 moves needed to pass Rokicki must come from new search, and the measured structure
points at one place: the file is a merge of two solver families, and its reduction+3x3x3
family (374 pids, mean 45.76) is 1.5 moves/pid WORSE than its neural family (669 pids,
44.25). Those 374 are the hard tail. A multi-reduction two-phase attack needs a real
QTM-optimal 3x3x3 solver for phase 2, and that needs a piece model.

WHAT IS DERIVED, AND FROM WHAT. Nothing is hardcoded about cube geometry. Everything
comes out of the 24 move permutations in puzzle_info.json:

  * A layer turn moves a piece entirely or not at all, so the set of generators that move
    a facelet is constant across that facelet's piece. Partitioning the 96 facelets by
    that MOVE SIGNATURE yields exactly 8 triples (corners), 24 pairs (wings) and 24
    singletons (centres) -- no geometry needed.
  * Under OUTER layers only, the two wings of one edge move together, so grouping wings
    by their outer-only signature yields the 12 edges (4 facelets each).
  * Centres never leave their face under outer moves and the 4 centres of a face are
    colour-identical, so they are invisible in colour space and are dropped entirely.

A colouring pins the pieces because a corner's 3 facelets carry 3 distinct colours and an
edge's carry 2 -- that is exactly what makes a COLOUR cube readable as a 3x3x3 once it is
reduced (wings paired, centres uniform).

VALIDATION IS AGAINST EXTERNAL GROUND TRUTH, not self-consistency: BFS over the derived
piece representation must reproduce the published 3x3x3 QTM level counts
1, 12, 114, 1068, 10011, 93840, 878880, 8221632, 76843595.

    python cube444/scripts/78_piece_model.py --validate --depth 7
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cube444_window_reduce", Path(__file__).with_name("70_window_reduce.py"))
wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wr)

OUTER = ("f0", "f3", "r0", "r3", "d0", "d3")
# published 3x3x3 QTM counts; the derived model must reproduce these exactly
QTM_LEVELS = [1, 12, 114, 1068, 10011, 93840, 878880, 8221632, 76843595]


def base(name: str) -> str:
    return name[1:] if name.startswith("-") else name


class PieceModel:
    def __init__(self, data_dir: Path):
        info = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        self.solved = np.array(info["central_state"], dtype=np.int64)
        self.S = self.gen.shape[1]
        self.idx = {n: i for i, n in enumerate(self.names)}

        # ---- move signature: which generators move each facelet
        sig = defaultdict(list)
        for f in range(self.S):
            key = tuple(sorted(base(self.names[m]) for m in range(len(self.names))
                               if self.gen[m][f] != f))
            sig[key].append(f)
        groups = list(sig.values())
        self.corners = sorted([g for g in groups if len(g) == 3])
        self.wings = sorted([g for g in groups if len(g) == 2])
        self.centres = sorted([g for g in groups if len(g) == 1])
        assert len(self.corners) == 8, f"expected 8 corners, got {len(self.corners)}"
        assert len(self.wings) == 24, f"expected 24 wings, got {len(self.wings)}"
        assert len(self.centres) == 24, f"expected 24 centres, got {len(self.centres)}"

        # ---- edges: wings sharing an OUTER-only signature move together
        esig = defaultdict(list)
        for w in self.wings:
            key = tuple(sorted(base(self.names[m]) for m in range(len(self.names))
                               if self.gen[m][w[0]] != w[0] and base(self.names[m]) in OUTER))
            esig[key].append(w)
        self.edges = []
        for key, ws in sorted(esig.items()):
            assert len(ws) == 2, f"edge class {key} has {len(ws)} wings, expected 2"
            self.edges.append(sorted(ws[0] + ws[1]))
        assert len(self.edges) == 12, f"expected 12 edges, got {len(self.edges)}"

        # ---- solved colour signatures identify pieces
        self.corner_id = {}
        for i, c in enumerate(self.corners):
            self.corner_id[frozenset(int(self.solved[f]) for f in c)] = i
        self.edge_id = {}
        for i, e in enumerate(self.edges):
            self.edge_id[frozenset(int(self.solved[f]) for f in e)] = i
        assert len(self.corner_id) == 8, "corner colour sets are not distinct"
        assert len(self.edge_id) == 12, "edge colour sets are not distinct"

        # reference facelet per slot fixes the orientation origin
        self.corner_ref = [c[0] for c in self.corners]
        self.corner_slot_faces = [[int(self.solved[f]) for f in c] for c in self.corners]
        self.edge_ref = [e[0] for e in self.edges]
        self.edge_slot_faces = [[int(self.solved[f]) for f in e] for e in self.edges]

    # ---- state -> (corner perm/ori, edge perm/ori)
    def extract(self, state: np.ndarray):
        cp = np.zeros(8, dtype=np.int8)
        co = np.zeros(8, dtype=np.int8)
        for s, c in enumerate(self.corners):
            cols = [int(state[f]) for f in c]
            pid = self.corner_id.get(frozenset(cols))
            if pid is None:
                return None                       # not a legal reduced corner
            cp[s] = pid
            home = [int(self.solved[f]) for f in self.corners[pid]]
            co[s] = home.index(cols[0])           # where the slot's ref colour sits
        ep = np.zeros(12, dtype=np.int8)
        eo = np.zeros(12, dtype=np.int8)
        for s, e in enumerate(self.edges):
            cols = [int(state[f]) for f in e]
            if len(set(cols)) != 2:
                return None                       # wings not paired -- cube not reduced
            pid = self.edge_id.get(frozenset(cols))
            if pid is None:
                return None
            ep[s] = pid
            home = [int(self.solved[f]) for f in self.edges[pid]]
            eo[s] = 0 if cols[0] == home[0] else 1
        return cp, co, ep, eo

    def is_reduced(self, state: np.ndarray) -> bool:
        """Centres uniform per face AND every edge's 4 facelets show exactly 2 colours."""
        for c in self.corners:
            if len({int(state[f]) for f in c}) != 3:
                return False
        for e in self.edges:
            if len({int(state[f]) for f in e}) != 2:
                return False
        # centres: the 4 centres of a face must share a colour
        byface = defaultdict(list)
        for c in self.centres:
            byface[c[0] // 16].append(int(state[c[0]]))
        return all(len(set(v)) == 1 for v in byface.values())


def validate(pm: PieceModel, depth: int) -> bool:
    """BFS over the piece representation; level sizes must equal the published counts."""
    cube = wr.Cube(PROJECT / "cube444" / "data", "cpu")
    keep = [i for i, n in enumerate(pm.names) if base(n) in OUTER]
    print(f"outer generators: {[pm.names[i] for i in keep]}")

    def key(st):
        r = pm.extract(st)
        assert r is not None, "solved-descendant state failed extraction"
        return b"".join(x.tobytes() for x in r)

    frontier = {key(pm.solved): pm.solved}
    seen = set(frontier)
    ok = True
    for d in range(1, depth + 1):
        nxt = {}
        for st in frontier.values():
            for m in keep:
                ch = st[pm.gen[m]]
                k = key(ch)
                if k not in seen:
                    seen.add(k)
                    nxt[k] = ch
        exp = QTM_LEVELS[d] if d < len(QTM_LEVELS) else None
        flag = "OK" if exp is None or len(nxt) == exp else "MISMATCH"
        if exp is not None and len(nxt) != exp:
            ok = False
        print(f"  depth {d}: {len(nxt):,}"
              + (f"  expected {exp:,}  {flag}" if exp is not None else ""), flush=True)
        frontier = nxt
        if not ok:
            break
    print(f"piece-model validation: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "cube444" / "data")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--check-file", type=Path,
                    help="report how many paths reach a REDUCED state at their 3x3x3 boundary")
    args = ap.parse_args()

    pm = PieceModel(args.data_dir)
    print(f"derived: {len(pm.corners)} corners, {len(pm.wings)} wings -> "
          f"{len(pm.edges)} edges, {len(pm.centres)} centres")

    if args.check_file:
        cube = wr.Cube(args.data_dir, "cpu")
        tests = wr.load_tests(args.data_dir)
        rows = wr.load_paths(cube, args.check_file)
        red = notred = 0
        lens = []
        for pid, w in rows.items():
            t = 0
            while t < len(w) and base(cube.names[w[len(w) - 1 - t]]) in OUTER:
                t += 1
            if t < 15:
                continue
            st = cube.path_states_np(tests[pid], w)[len(w) - t].astype(np.int64)
            if pm.is_reduced(st):
                red += 1
                lens.append(t)
            else:
                notred += 1
        print(f"3x3x3 boundary states: {red} reduced, {notred} NOT reduced")
        if lens:
            print(f"  their phase-2 lengths: mean {np.mean(lens):.2f}, "
                  f"min {min(lens)}, max {max(lens)}")

    if args.validate:
        return 0 if validate(pm, args.depth) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
