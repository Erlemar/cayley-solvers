"""Lossless 17-byte piece codec for the IHES picture cube (packs 72 facelets into 130 bits).

Why: a 256M-state TPU beam cannot hold 72 bytes per state. The cube444 256M kernel solved the
same problem with a piece codec (96 stickers -> 29 bytes). IHES packs better: every one of the
26 piece slots has exactly 24 possible occupants, so 5 bits each = 130 bits = 17 bytes, a 4.2x
reduction.

  slot type   slots   occupants                          bits
  corner      8       8 corner pieces x 3 rotations = 24  5
  edge        12      12 edge pieces x 2 flips     = 24   5
  centre      6       6 centre pieces x 4 rotations = 24  5

Each slot's code indexes a sorted list of the value-tuples that can legally sit in it. A move
permutes slots and relabels codes: code_child[s] = MOVE_CODES[m, s, code_parent[SOURCE[m, s]]],
so a child is built from packed codes alone -- no 72-byte state is ever materialised.

Writes data/ihes_piece_codec.npz with:
  slot_positions (26,4) int32, slot_len (26,) int32   facelet positions per slot
  options       (26,24,4) int8                        value tuples, sorted, per slot
  encode_lut    (26, 72) int32                        first-facelet value -> code (exact: the
                                                      first value identifies the occupant)
  source        (18,26) int32                         child slot s reads parent slot source[m,s]
  move_codes    (18,26,24) uint8                      code relabelling under each move
  decode_values (72,24) int8                          position -> value for each code of its slot
  byte_lo/shift_lo/hi_bits (26,) int32                tight 17-byte bit layout (a field may
                                                      straddle two bytes)

    python scripts/97_build_ihes_piece_codec.py [--out data/ihes_piece_codec.npz]
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def build(puzzle_path: Path, layout_path: Path):
    spec = json.loads(puzzle_path.read_text(encoding="utf-8"))
    solved = np.asarray(spec["central_state"], dtype=np.int64)
    names = list(spec["generators"])
    gens = np.stack([np.asarray(spec["generators"][n], dtype=np.int64) for n in names])
    n_move, n_pos = gens.shape
    assert n_pos == solved.size == 72 and n_move == 18
    assert len(set(solved.tolist())) == 72, "IHES stickers must all be distinct"
    for m in range(n_move):
        assert sorted(gens[m].tolist()) == list(range(n_pos))

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    slots = [sorted(int(x) for x, keep in zip(pos, mask) if keep)
             for pos, mask in zip(layout["piece_positions"], layout["piece_mask"])]
    assert len(slots) == 26 and sorted(p for s in slots for p in s) == list(range(72))
    slot_len = np.asarray([len(s) for s in slots], dtype=np.int32)
    owner = np.empty(n_pos, dtype=np.int32)
    for i, s in enumerate(slots):
        owner[s] = i

    # child[i] = parent[gens[m][i]], so the child's slot s reads the parent positions gens[m][s].
    # Those must be exactly one parent slot, which makes the codec closed under moves.
    source = np.empty((n_move, 26), dtype=np.int32)
    src_order = np.zeros((n_move, 26, 4), dtype=np.int32)
    for m in range(n_move):
        for s, pos in enumerate(slots):
            src_pos = gens[m][pos]
            s0 = int(owner[src_pos[0]])
            assert sorted(src_pos.tolist()) == slots[s0], "a move split a piece"
            source[m, s] = s0
            for j, p in enumerate(src_pos):
                src_order[m, s, j] = slots[s0].index(int(p))

    # Every ordered placement a piece can reach; each one is one legal content of some slot.
    options: list[set] = [set() for _ in slots]
    for home in slots:
        vals = solved[home]
        start = tuple(home)
        seen = {start}
        queue = deque([start])
        while queue:
            where = np.asarray(queue.popleft(), dtype=np.int64)
            s = int(owner[where[0]])
            content = np.zeros(4, dtype=np.int64)
            order = [slots[s].index(int(p)) for p in where]
            content[order] = vals
            options[s].add(tuple(content.tolist()))
            for m in range(n_move):
                nxt = tuple(int(np.flatnonzero(gens[m] == p)[0]) for p in where)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    counts = np.asarray([len(o) for o in options])
    assert (counts == 24).all(), f"expected 24 options per slot, got {sorted(set(counts.tolist()))}"

    opts = np.zeros((26, 24, 4), dtype=np.int8)
    code_of: list[dict] = []
    for s, o in enumerate(options):
        table = {vals: i for i, vals in enumerate(sorted(o))}
        code_of.append(table)
        for vals, code in table.items():
            opts[s, code] = np.asarray(vals, dtype=np.int8)

    # The first facelet value of a slot identifies its occupant uniquely (all 72 stickers differ).
    encode_lut = np.full((26, 72), -1, dtype=np.int32)
    for s, table in enumerate(code_of):
        for vals, code in table.items():
            encode_lut[s, vals[0]] = code
        assert (encode_lut[s] >= 0).sum() == 24

    move_codes = np.zeros((n_move, 26, 24), dtype=np.uint8)
    for m in range(n_move):
        for s in range(26):
            s0 = int(source[m, s])
            for code in range(24):
                src_vals = opts[s0, code]
                content = np.zeros(4, dtype=np.int64)
                for j in range(int(slot_len[s])):
                    content[j] = src_vals[src_order[m, s, j]]
                move_codes[m, s, code] = code_of[s][tuple(content.tolist())]

    decode_values = np.zeros((72, 24), dtype=np.int8)
    for pos in range(72):
        s = int(owner[pos])
        j = slots[s].index(pos)
        decode_values[pos] = opts[s, :, j]

    slot_positions = np.zeros((26, 4), dtype=np.int32)
    for s, pos in enumerate(slots):
        slot_positions[s, :len(pos)] = pos
    # Tight little-endian bit stream: field s occupies bits [5s, 5s+5) of 130 bits = 17 bytes.
    # A field may straddle two bytes, so keep the low byte, its shift, and the spill width.
    bit = np.arange(26, dtype=np.int32) * 5
    byte_lo, shift_lo = bit // 8, bit % 8
    lo_bits = np.minimum(8 - shift_lo, 5)
    hi_bits = 5 - lo_bits
    assert int((byte_lo + (hi_bits > 0)).max()) == 16, "expected exactly 17 bytes"
    return dict(solved=solved.astype(np.int8), gens=gens.astype(np.int32),
                move_names=np.asarray(names), slot_positions=slot_positions, slot_len=slot_len,
                owner=owner, options=opts, encode_lut=encode_lut, source=source,
                move_codes=move_codes, decode_values=decode_values,
                byte_lo=byte_lo.astype(np.int32), shift_lo=shift_lo.astype(np.int32),
                lo_bits=lo_bits.astype(np.int32), hi_bits=hi_bits.astype(np.int32),
                n_bytes=np.int32(17))


def encode(states: np.ndarray, t: dict) -> np.ndarray:
    """(N,72) int -> (N,26) uint8 codes."""
    first = states[:, t["slot_positions"][:, 0]]
    return t["encode_lut"][np.arange(26)[None, :], first].astype(np.uint8)


def decode(codes: np.ndarray, t: dict) -> np.ndarray:
    """(N,26) codes -> (N,72) int8 facelets."""
    owner = t["owner"]
    return t["decode_values"][np.arange(72)[None, :], codes[:, owner]]


def apply_move(codes: np.ndarray, m: int, t: dict) -> np.ndarray:
    src = codes[:, t["source"][m]]
    return t["move_codes"][m][np.arange(26)[None, :], src]


def pack(codes: np.ndarray, t: dict) -> np.ndarray:
    """(N,26) codes -> (N,17) uint8, the 130-bit little-endian stream."""
    out = np.zeros((codes.shape[0], int(t["n_bytes"])), dtype=np.uint8)
    for s in range(26):
        c = codes[:, s].astype(np.uint16)
        lo, sh, hb = int(t["byte_lo"][s]), int(t["shift_lo"][s]), int(t["hi_bits"][s])
        out[:, lo] |= ((c << sh) & 0xFF).astype(np.uint8)
        if hb:
            out[:, lo + 1] |= (c >> (8 - sh)).astype(np.uint8)
    return out


def unpack(packed: np.ndarray, t: dict) -> np.ndarray:
    """(N,17) uint8 -> (N,26) codes."""
    out = np.zeros((packed.shape[0], 26), dtype=np.uint8)
    for s in range(26):
        lo, sh, hb = int(t["byte_lo"][s]), int(t["shift_lo"][s]), int(t["hi_bits"][s])
        v = packed[:, lo].astype(np.uint16) >> sh
        if hb:
            v |= packed[:, lo + 1].astype(np.uint16) << (8 - sh)
        out[:, s] = (v & 31).astype(np.uint8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and verify the IHES 17-byte piece codec.")
    ap.add_argument("--puzzle", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--layout", type=Path, default=PROJECT / "data" / "ihes_piece_layout.json")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "ihes_piece_codec.npz")
    ap.add_argument("--n-check", type=int, default=2000)
    ap.add_argument("--walk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = build(args.puzzle, args.layout)
    gens, solved = t["gens"], t["solved"].astype(np.int64)
    rng = np.random.default_rng(args.seed)
    states = np.repeat(solved[None, :], args.n_check, axis=0)
    for _ in range(args.walk):
        mv = rng.integers(0, gens.shape[0], size=args.n_check)
        states = np.take_along_axis(states, gens[mv], axis=1)

    codes = encode(states, t)
    assert np.array_equal(decode(codes, t).astype(np.int64), states), "round trip failed"
    assert np.array_equal(unpack(pack(codes, t), t), codes), "bit packing failed"
    for m in range(gens.shape[0]):
        ref = np.take_along_axis(states, np.repeat(gens[m][None, :], args.n_check, 0), axis=1)
        got = decode(apply_move(codes, m, t), t).astype(np.int64)
        assert np.array_equal(got, ref), f"move {m} on codes disagrees with facelets"
    sc = encode(solved[None, :], t)
    assert np.array_equal(decode(sc, t)[0].astype(np.int64), solved)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{k: v for k, v in t.items() if k != "move_names"},
                        move_names=np.asarray(t["move_names"]))
    per_state = 26 * 5
    print(f"verified on {args.n_check:,} random states (walk {args.walk}) and all "
          f"{gens.shape[0]} moves")
    print(f"slots 26 (sizes {np.bincount(t['slot_len'])[2:].tolist()} for 2/3/4 facelets), "
          f"24 options each -> {per_state} bits = {per_state / 8:.2f} bytes/state "
          f"(raw 72 bytes, {72 * 8 / per_state:.2f}x smaller)")
    for width in (2 ** 27, 2 ** 28, 2 ** 29):
        per_chip = width // 8
        print(f"  B={width:>12,}: per chip {per_chip:>11,} states = "
              f"{per_chip * 17 / 2**30:5.2f} GiB packed (17 B) vs "
              f"{per_chip * 72 / 2**30:5.2f} GiB raw; int64 hashes "
              f"{per_chip * 8 / 2**30:5.2f} GiB; host ancestry {width * 4 / 2**30:5.2f} GiB/depth")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
