"""Build static graph features for the megaminx graph transformer.

Outputs `megaminx/data/graph_features.pt` with per-position features and
pairwise relation/distance tables. These features are static (don't depend
on state values, only on positions and the generator group), so they're
computed once and consumed as buffers by `GraphTransformerV`.

Schema (all tensors keyed by position id 0..119):

  state_size        : int = 120
  n_pieces          : int = 50    (20 corner pieces + 30 edge pieces)
  n_piece_slots     : int = 3     (3 slots per corner, 2 per edge + padding)
  n_faces           : int = 12
  n_relations       : int = 6
  n_dist_buckets    : int = 9     (0..8)
  piece_type        : (120,)      0=corner, 1=edge
  piece_id          : (120,)      0..49 (corners 0..19, edges 20..49)
  piece_slot        : (120,)      0..2 (corner) or 0..1 (edge); padding=0
  face_set          : (120, 3)    face IDs (0..11); -1 if no third face (edges)
  relation_id       : (120, 120)  see below
  dist_bucket       : (120, 120)  min(shortest-path, 8)
  edge_index        : (2, E)      generator-graph edges (undirected, deduped)
  edge_type         : (E,)        face_id of a generator that creates this edge

Relation IDs (precedence: lower-numbered wins when multiple apply):
  0 unrelated
  1 same_position           (i==j)
  2 generator_edge          (adjacent in some forward 5-cycle, symmetric)
  3 same_piece              (corner-mates or edge-mates, not relation 2)
  4 same_cycle_stride2      (2-stride in some 5-cycle, not relation 2 or 3)
  5 same_face               (in any common face's cycle, not above)

Run:
    .venv/Scripts/python.exe megaminx/scripts/72_build_graph_features.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def _cycles(perm: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return non-trivial cycles of a permutation."""
    seen = set()
    out = []
    for i in range(len(perm)):
        if i in seen or perm[i] == i:
            seen.add(i)
            continue
        cyc = []
        j = i
        while j not in seen:
            seen.add(j)
            cyc.append(j)
            j = perm[j]
        if len(cyc) > 1:
            out.append(tuple(cyc))
    return out


def build_graph_features(puzzle: Megaminx) -> dict:
    state_size = len(puzzle.solved_state)
    forward_faces = [n for n in puzzle.move_names if not n.startswith("-")]
    n_faces = len(forward_faces)
    face_to_id = {f: i for i, f in enumerate(forward_faces)}

    # Cycle decomposition of each forward generator.
    # `face_cycles[face_id]` = list of 5 cycles (each a 5-tuple of positions).
    face_cycles: list[list[tuple[int, ...]]] = [
        _cycles(puzzle.generators[f]) for f in forward_faces
    ]
    for f, cs in zip(forward_faces, face_cycles):
        assert len(cs) == 5 and all(len(c) == 5 for c in cs), (
            f"unexpected cycle structure for {f}: {[len(c) for c in cs]}"
        )

    # ---------------------------------------------------------------
    # Per-sticker face memberships.
    # For each position p, collect (face_id, cycle_idx) pairs.
    # Edges live in cycles 3-4 of exactly 2 faces.
    # Corners live in cycles 0-2 of exactly 3 faces.
    # ---------------------------------------------------------------
    membership: dict[int, list[tuple[int, int]]] = {i: [] for i in range(state_size)}
    for fid, cs in enumerate(face_cycles):
        for ci, cyc in enumerate(cs):
            for pos in cyc:
                membership[pos].append((fid, ci))

    piece_type = torch.full((state_size,), -1, dtype=torch.long)
    face_set = torch.full((state_size, 3), -1, dtype=torch.long)
    for pos, mems in membership.items():
        faces_here = sorted({f for f, _ in mems})
        cycle_idxs = [c for _, c in mems]
        if len(faces_here) == 3 and all(0 <= c <= 2 for c in cycle_idxs):
            piece_type[pos] = 0  # corner
            face_set[pos, 0] = faces_here[0]
            face_set[pos, 1] = faces_here[1]
            face_set[pos, 2] = faces_here[2]
        elif len(faces_here) == 2 and all(c in (3, 4) for c in cycle_idxs):
            piece_type[pos] = 1  # edge
            face_set[pos, 0] = faces_here[0]
            face_set[pos, 1] = faces_here[1]
            # face_set[pos, 2] stays -1
        else:
            raise AssertionError(
                f"position {pos}: unexpected membership pattern {mems}"
            )

    assert (piece_type >= 0).all(), "some positions unassigned"

    # ---------------------------------------------------------------
    # Piece identification.
    # Corner piece := sorted triple of face IDs (20 unique triples).
    # Edge piece   := sorted pair  of face IDs (30 unique pairs).
    # piece_id 0..19 = corners, 20..49 = edges.
    # ---------------------------------------------------------------
    corner_keys: list[tuple[int, int, int]] = []
    edge_keys: list[tuple[int, int]] = []
    for pos in range(state_size):
        fs = tuple(int(x) for x in face_set[pos].tolist() if x >= 0)
        if piece_type[pos].item() == 0:
            assert len(fs) == 3
            corner_keys.append(fs)
        else:
            assert len(fs) == 2
            edge_keys.append(fs)
    corner_unique = sorted(set(corner_keys))
    edge_unique = sorted(set(edge_keys))
    assert len(corner_unique) == 20, f"expected 20 corners, got {len(corner_unique)}"
    assert len(edge_unique) == 30, f"expected 30 edges, got {len(edge_unique)}"
    corner_to_pid = {k: i for i, k in enumerate(corner_unique)}
    edge_to_pid = {k: 20 + i for i, k in enumerate(edge_unique)}

    piece_id = torch.full((state_size,), -1, dtype=torch.long)
    piece_slot = torch.full((state_size,), 0, dtype=torch.long)
    # Within a piece, slot = position of this sticker's face within the piece's sorted face tuple.
    # For corners (3 stickers, 3 faces): slot ∈ {0,1,2}.
    # For edges   (2 stickers, 2 faces): slot ∈ {0,1}.
    # We assign by: this sticker's "primary face" = the face among its face_set whose
    # cycle index here is the lowest (the most "specific" face in some sense, but
    # really we need one canonical face per sticker). For a corner sticker at
    # (faceA, cycle a), (faceB, cycle b), (faceC, cycle c), the slot is the
    # index of the chosen face in the sorted triple. We choose the face whose
    # cycle here equals the position within the cycle when listed in order — but
    # an equivalent simple rule is: the face whose cycle_idx for this position
    # equals (some canonical mapping). Pragmatic choice: slot = the rank of this
    # sticker among its piece-mates, in increasing position order.
    piece_mates: dict[int, list[int]] = {}
    for pos in range(state_size):
        fs = tuple(int(x) for x in face_set[pos].tolist() if x >= 0)
        if piece_type[pos].item() == 0:
            pid = corner_to_pid[fs]
        else:
            pid = edge_to_pid[fs]
        piece_id[pos] = pid
        piece_mates.setdefault(pid, []).append(pos)
    for pid, positions in piece_mates.items():
        positions.sort()
        for slot, pos in enumerate(positions):
            piece_slot[pos] = slot

    # ---------------------------------------------------------------
    # Generator-graph edges (undirected, deduped).
    # For each forward generator g, the 5 cycles produce 5 edges each (the
    # adjacency around the cycle). Inverse generators give the same set.
    # ---------------------------------------------------------------
    edge_pairs: dict[tuple[int, int], int] = {}  # (i,j) sorted -> face_id witness
    for fid, cs in enumerate(face_cycles):
        for cyc in cs:
            L = len(cyc)
            for k in range(L):
                a = cyc[k]
                b = cyc[(k + 1) % L]
                key = (min(a, b), max(a, b))
                edge_pairs.setdefault(key, fid)
    edge_list = sorted(edge_pairs.keys())
    edge_index = torch.tensor(
        [[a for a, _ in edge_list], [b for _, b in edge_list]], dtype=torch.long
    )
    edge_type = torch.tensor([edge_pairs[(a, b)] for (a, b) in edge_list], dtype=torch.long)
    print(f"  generator-graph: {edge_index.size(1)} undirected edges")

    # ---------------------------------------------------------------
    # Shortest-path distance buckets.
    # ---------------------------------------------------------------
    INF = 10**6
    dist = [[INF] * state_size for _ in range(state_size)]
    for i in range(state_size):
        dist[i][i] = 0
    for a, b in edge_list:
        dist[a][b] = 1
        dist[b][a] = 1
    # BFS from each node (state_size BFS calls; faster than Floyd-Warshall here)
    adj_list: list[list[int]] = [[] for _ in range(state_size)]
    for a, b in edge_list:
        adj_list[a].append(b)
        adj_list[b].append(a)
    from collections import deque

    for src in range(state_size):
        d_src = dist[src]
        q = deque([src])
        d_src[src] = 0
        seen = {src}
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if v not in seen:
                    seen.add(v)
                    d_src[v] = d_src[u] + 1
                    q.append(v)
    # Sanity: graph must be connected.
    max_dist = max(d for row in dist for d in row if d < INF)
    assert max_dist < INF, "graph not connected"
    print(f"  max shortest-path distance: {max_dist}")
    dist_bucket = torch.tensor(
        [[min(dist[i][j], 8) for j in range(state_size)] for i in range(state_size)],
        dtype=torch.long,
    )

    # ---------------------------------------------------------------
    # Relation ID matrix.
    # Precedence (low wins): same_position(1), generator_edge(2), same_piece(3),
    # same_cycle_stride2(4), same_face(5), unrelated(0).
    # ---------------------------------------------------------------
    relation_id = torch.zeros((state_size, state_size), dtype=torch.long)
    # 5: same_face (default for any pair touched by a common face)
    face_members: list[set[int]] = [set() for _ in range(n_faces)]
    for pos, mems in membership.items():
        for fid, _ in mems:
            face_members[fid].add(pos)
    for fid in range(n_faces):
        members = list(face_members[fid])
        for ii in members:
            for jj in members:
                if ii != jj and relation_id[ii, jj].item() == 0:
                    relation_id[ii, jj] = 5
    # 4: same_cycle_stride2 (overwrites 5)
    for fid, cs in enumerate(face_cycles):
        for cyc in cs:
            L = len(cyc)
            for k in range(L):
                a = cyc[k]
                b = cyc[(k + 2) % L]
                if relation_id[a, b].item() != 0:
                    relation_id[a, b] = 4
                    relation_id[b, a] = 4
    # 3: same_piece (overwrites 4,5)
    for pid, positions in piece_mates.items():
        for ii in positions:
            for jj in positions:
                if ii != jj:
                    relation_id[ii, jj] = 3
    # 2: generator_edge (overwrites 3,4,5)
    for a, b in edge_list:
        relation_id[a, b] = 2
        relation_id[b, a] = 2
    # 1: same_position (diagonal)
    for i in range(state_size):
        relation_id[i, i] = 1

    # Quick stats
    from collections import Counter

    c = Counter(int(v) for row in relation_id.tolist() for v in row)
    print(f"  relation_id histogram: {dict(sorted(c.items()))}")

    # Checksum
    raw = json.dumps(
        {
            "solved_state": list(puzzle.solved_state),
            "generators": {n: list(g) for n, g in puzzle.generators.items()},
        },
        sort_keys=True,
    )
    puzzle_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    return {
        "state_size": state_size,
        "n_classes": state_size,
        "n_pieces": 50,
        "n_piece_slots": 3,
        "n_faces": n_faces,
        "n_relations": 6,
        "n_dist_buckets": 9,
        "n_generators": len(puzzle.move_names),
        "piece_type": piece_type,
        "piece_id": piece_id,
        "piece_slot": piece_slot,
        "face_set": face_set,
        "relation_id": relation_id,
        "dist_bucket": dist_bucket,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "forward_face_names": forward_faces,
        "puzzle_info_hash": puzzle_hash,
    }


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"puzzle: state_size={len(puzzle.solved_state)} "
          f"generators={len(puzzle.generators)}")
    features = build_graph_features(puzzle)
    out = PROJECT / "data" / "graph_features.pt"
    torch.save(features, out)
    print(f"saved {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
