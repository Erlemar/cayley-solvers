"""Build static tables for the Bipartite Slot-Sticker Graph Transformer (doc 3.2).

Token layout (T = 264):
    [0, 120)    slot nodes      (token i  == slot i)
    [120, 240)  sticker nodes   (token 120+j == sticker j)
    [240, 264)  action nodes    (token 240+a == action a)

State enters the model NOT through these static tables but through per-batch
node features (a slot knows the sticker it holds, a sticker knows the slot it
occupies). Everything here is state-independent: node identity features, the
relation-type matrix, distance buckets, and the sparse "local" attention mask.

Because the megaminx solved state is the identity permutation (solved_state[i]==i,
verified), a sticker's home slot is itself, so sticker nodes reuse the SAME static
metadata (face/piece/piece_slot) as slot nodes. The slot/sticker distinction is
carried by the node-type embedding plus the dynamic content/location features.

Relation codes (deliberately aligned with 72_build_graph_features so the 120x120
slot block can be copied verbatim):
    0 NONE         (no structural edge; only global attention sees these pairs)
    1 SELF         (diagonal)
    2 GEN_EDGE     (adjacent on a generator 5-cycle)        [slot-slot, sticker-sticker]
    3 SAME_PIECE   (corner/edge mates)                       [slot-slot, sticker-sticker]
    4 STRIDE2      (2-stride on a 5-cycle)                   [slot-slot, sticker-sticker]
    5 SAME_FACE    (share a face, none of the above)         [slot-slot, sticker-sticker]
    6 HOME         (slot i <-> sticker i, home correspondence)
    7 AFFECTS      (action a <-> slot it permutes; ~25 each)
    8 INV          (action a <-> its inverse action)

Local (sparse MPNN) edges = relations {SELF, GEN_EDGE, SAME_PIECE, HOME, AFFECTS,
INV}. STRIDE2 and SAME_FACE are intentionally left to the global-attention path
(they are dense and medium-range).

Run:
    .venv/Scripts/python.exe megaminx/scripts/74_build_bipartite_features.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable
from megaminx.puzzle import Megaminx

REL_NONE, REL_SELF, REL_GEN_EDGE, REL_SAME_PIECE = 0, 1, 2, 3
REL_STRIDE2, REL_SAME_FACE, REL_HOME, REL_AFFECTS, REL_INV = 4, 5, 6, 7, 8
N_REL = 9
LOCAL_RELATIONS = (REL_SELF, REL_GEN_EDGE, REL_SAME_PIECE, REL_HOME, REL_AFFECTS, REL_INV)


def build_bipartite_features(puzzle: Megaminx, gf: dict) -> dict:
    n_slots = int(gf["state_size"])               # 120
    n_stickers = n_slots                            # solved is identity -> sticker home = slot
    n_actions = len(puzzle.move_names)              # 24
    T = n_slots + n_stickers + n_actions            # 264
    n_faces = int(gf["n_faces"])
    n_pieces = int(gf["n_pieces"])
    n_piece_slots = int(gf["n_piece_slots"])
    n_dist = int(gf["n_dist_buckets"])              # 9 buckets (0..8)
    NA_DIST = n_dist                                 # extra bucket for non-(slot/sticker) pairs

    assert list(puzzle.solved_state) == list(range(n_slots)), \
        "builder assumes identity solved state (sticker home slot == sticker id)"

    SLOT, STK, ACT = slice(0, n_slots), slice(n_slots, 2 * n_slots), slice(2 * n_slots, T)
    act0 = 2 * n_slots  # first action token index

    gens = GeneratorTable.from_puzzle(puzzle).perms  # (n_actions, n_slots) int64 numpy
    arange = np.arange(n_slots)

    # ---- per-token static identity features (sentinels for non-applicable types) ----
    token_type = torch.empty(T, dtype=torch.long)
    token_type[SLOT] = 0
    token_type[STK] = 1
    token_type[ACT] = 2

    SENT_PIECE, SENT_ACTION = n_pieces, n_actions
    piece_id_full = torch.full((T,), SENT_PIECE, dtype=torch.long)
    piece_id_full[SLOT] = gf["piece_id"].long()
    piece_id_full[STK] = gf["piece_id"].long()

    piece_type_full = torch.full((T,), 2, dtype=torch.long)  # action -> 2
    piece_type_full[SLOT] = gf["piece_type"].long()
    piece_type_full[STK] = gf["piece_type"].long()

    piece_slot_full = torch.zeros(T, dtype=torch.long)
    piece_slot_full[SLOT] = gf["piece_slot"].long()
    piece_slot_full[STK] = gf["piece_slot"].long()

    # face_idx_full[t] = up to 3 face ids, +1 so that "none"(-1) maps to 0.
    face_idx_full = torch.zeros((T, 3), dtype=torch.long)
    face_idx_full[SLOT] = (gf["face_set"].long() + 1)
    face_idx_full[STK] = (gf["face_set"].long() + 1)
    for a in range(n_actions):
        face_idx_full[act0 + a, 0] = (a // 2) + 1   # forward_face index +1

    action_id_full = torch.full((T,), SENT_ACTION, dtype=torch.long)
    action_dir_full = torch.full((T,), 2, dtype=torch.long)
    action_inv_full = torch.full((T,), SENT_ACTION, dtype=torch.long)
    for a in range(n_actions):
        action_id_full[act0 + a] = a
        action_dir_full[act0 + a] = a % 2           # 0 = CW/forward, 1 = CCW/inverse
        action_inv_full[act0 + a] = a ^ 1           # interleaved fwd/inv -> XOR 1

    # ---- relation matrix (T, T) ----
    rel = torch.zeros((T, T), dtype=torch.long)
    gf_rel = gf["relation_id"].long()               # (120,120), codes already aligned 0..5
    rel[SLOT, SLOT] = gf_rel
    rel[STK, STK] = gf_rel
    # HOME: slot i <-> sticker i
    idx = torch.arange(n_slots)
    rel[idx, n_slots + idx] = REL_HOME
    rel[n_slots + idx, idx] = REL_HOME
    # AFFECTS: action a <-> slots it permutes
    affects_rows = []
    for a in range(n_actions):
        affected = np.nonzero(gens[a] != arange)[0]
        assert affected.size == 25, f"action {a} affects {affected.size} slots (expected 25)"
        affects_rows.append(torch.from_numpy(affected.astype(np.int64)))
        ta = act0 + a
        rel[ta, affected] = REL_AFFECTS
        rel[affected, ta] = REL_AFFECTS
    action_affects = torch.stack(affects_rows, dim=0)  # (24, 25)
    # INV: action a <-> a^1
    for a in range(n_actions):
        rel[act0 + a, act0 + (a ^ 1)] = REL_INV
    # SELF on the full diagonal (overrides everything)
    di = torch.arange(T)
    rel[di, di] = REL_SELF

    # ---- distance buckets (T, T) ----
    dist = torch.full((T, T), NA_DIST, dtype=torch.long)
    gf_dist = gf["dist_bucket"].long()
    dist[SLOT, SLOT] = gf_dist
    dist[STK, STK] = gf_dist

    # ---- local sparse-attention mask (bool, True = attend) ----
    local_mask = torch.zeros((T, T), dtype=torch.bool)
    for r in LOCAL_RELATIONS:
        local_mask |= (rel == r)

    # ---- sanity checks ----
    assert torch.equal(rel, rel.t()), "relation matrix must be symmetric"
    assert torch.equal(local_mask, local_mask.t()), "local mask must be symmetric"
    assert int((rel == REL_HOME).sum()) == 2 * n_slots
    assert int((rel == REL_AFFECTS).sum()) == 2 * n_actions * 25
    assert int((rel == REL_INV).sum()) == n_actions          # each a points to a^1 (once)
    assert bool(local_mask[di, di].all()), "self must be local"
    # every action node attends (locally) to exactly 25 slots + itself + its inverse
    for a in range(n_actions):
        deg = int(local_mask[act0 + a].sum())
        assert deg == 25 + 1 + 1, f"action {a} local degree {deg} (expected 27)"

    raw = json.dumps(
        {"solved_state": list(puzzle.solved_state),
         "generators": {n: list(g) for n, g in puzzle.generators.items()}},
        sort_keys=True,
    )
    puzzle_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    return {
        "n_slots": n_slots, "n_stickers": n_stickers, "n_actions": n_actions, "n_tokens": T,
        "n_relations_full": N_REL, "n_dist_buckets_full": n_dist + 1,
        "n_faces": n_faces, "n_pieces": n_pieces, "n_piece_slots": n_piece_slots,
        "sent_piece": SENT_PIECE, "sent_action": SENT_ACTION,
        "token_type": token_type,
        "piece_id_full": piece_id_full, "piece_type_full": piece_type_full,
        "piece_slot_full": piece_slot_full, "face_idx_full": face_idx_full,
        "action_id_full": action_id_full, "action_dir_full": action_dir_full,
        "action_inv_full": action_inv_full,
        "relation_full": rel, "dist_bucket_full": dist, "local_mask": local_mask,
        "action_affects": action_affects,
        "slot_range": (0, n_slots), "sticker_range": (n_slots, 2 * n_slots),
        "action_range": (2 * n_slots, T),
        "puzzle_info_hash": puzzle_hash,
    }


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    gf = torch.load(PROJECT / "data" / "graph_features.pt", map_location="cpu", weights_only=False)
    feats = build_bipartite_features(puzzle, gf)
    out = PROJECT / "data" / "bipartite_features.pt"
    torch.save(feats, out)
    print(f"saved {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"  T={feats['n_tokens']}  relations={feats['n_relations_full']}  "
          f"dist_buckets={feats['n_dist_buckets_full']}")
    from collections import Counter
    c = Counter(int(v) for row in feats["relation_full"].tolist() for v in row)
    print(f"  relation histogram: {dict(sorted(c.items()))}")
    print(f"  local edges (incl self): {int(feats['local_mask'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
