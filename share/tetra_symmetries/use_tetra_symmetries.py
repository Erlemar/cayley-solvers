"""Use the 48 Professor Tetraminx symmetry frames (built by build_tetra_symmetries.py).

Two things you can do with them:

  TRAINING AUGMENTATION -- conjugating a state by a symmetry gives a different
  state at the SAME distance from solved, so one solved path yields 48 labelled
  paths for free (24 spatial x forward/inverse).

  SEARCH SYM-ENSEMBLE -- solve the same puzzle in several frames and keep the
  shortest. The beam is deterministic given a state, so a different frame is a
  genuinely different search trajectory, not a re-roll.

Conventions (matching the competition's puzzle_info.json):
    apply(s, g)[i] = s[g[i]]        states and generators compose as functions
    conj(s)        = P_inv[s[P]]
    a path m_1..m_L solving conj(s) maps to sigma(m_1)..sigma(m_L) solving s
    inverse antisymmetry: solve s^-1, then REVERSE the path and invert each move

Self-test:
    python use_tetra_symmetries.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_group(data_dir=HERE) -> dict:
    """Load the symmetry bundle. Returns dict with P, P_inv, sigma, meta."""
    d = Path(data_dir)
    meta = json.loads((d / "tetra_symmetries_meta.json").read_text(encoding="utf-8"))
    return {
        "P": np.load(d / "tetra_symmetries.npy").astype(np.int64),        # (24, 88)
        "P_inv": np.load(d / "tetra_symmetries_inv.npy").astype(np.int64),
        "sigma": np.load(d / "tetra_move_relabel.npy").astype(np.int64),  # (24, 24)
        "meta": meta,
        "move_names": meta["move_names"],
    }


def load_puzzle(data_dir=HERE) -> dict:
    info = json.loads((Path(data_dir) / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"].keys())
    return {
        "move_names": names,
        "gens": np.array([info["generators"][n] for n in names], dtype=np.int64),
        "solved": np.array(info["central_state"], dtype=np.int64),
        "inv_idx": np.array([names.index(n[1:] if n.startswith("-") else "-" + n)
                             for n in names]),
    }


def invert_state(s: np.ndarray) -> np.ndarray:
    """Group inverse of a state permutation."""
    out = np.empty_like(s)
    out[s] = np.arange(s.shape[0], dtype=s.dtype)
    return out


def apply_symmetry(s: np.ndarray, P: np.ndarray, P_inv: np.ndarray) -> np.ndarray:
    """conj(s) = P^-1 . s . P -- same distance from solved as s."""
    return P_inv[s[P]]


# --------------------------------------------------------------------------- #
# search frames
# --------------------------------------------------------------------------- #
def frame_list(n_frames: int, n_syms: int = 24) -> list[tuple[int, bool]]:
    """(sym_index, use_inverse) pairs, interleaved so small ensembles still
    include the inverse -- which empirically carries most of the wins."""
    return [(f // 2 % n_syms, f % 2 == 1) for f in range(n_frames)]


def to_frame(s: np.ndarray, k: int, inverted: bool, group: dict) -> np.ndarray:
    """Transform a puzzle state into frame (k, inverted). Solve THIS."""
    t = invert_state(s) if inverted else s
    return apply_symmetry(t, group["P"][k], group["P_inv"][k])


def from_frame(path_idx, k: int, inverted: bool, group: dict, inv_idx) -> list[int]:
    """Map a solution found in frame (k, inverted) back to the original puzzle."""
    q = [int(group["sigma"][k][m]) for m in path_idx]
    if inverted:
        q = [int(inv_idx[m]) for m in reversed(q)]
    return q


# --------------------------------------------------------------------------- #
# training augmentation
# --------------------------------------------------------------------------- #
def augment_path(state: np.ndarray, path_idx, group: dict, puzzle: dict,
                 n_syms: int = 24, include_inverse: bool = True):
    """One (state, solving path) -> up to 48 equivalent (state, path) pairs.

    Every variant is replayed and asserted to solve, so a bad relabeling can
    never silently poison a dataset.
    """
    gens, solved, inv_idx = puzzle["gens"], puzzle["solved"], puzzle["inv_idx"]

    def replay(s0, path):
        cur = s0
        for m in path:
            cur = cur[gens[m]]
        return cur

    base = [(state, list(path_idx))]
    if include_inverse:
        base.append((invert_state(state),
                     [int(inv_idx[m]) for m in reversed(path_idx)]))
    out = []
    for k in range(n_syms):
        sig_inv = np.argsort(group["sigma"][k])   # push a solution INTO frame k
        for s0, pth in base:
            if k == 0:
                vs, vp = s0, pth
            else:
                vs = apply_symmetry(s0, group["P"][k], group["P_inv"][k])
                vp = [int(sig_inv[m]) for m in pth]
            assert np.array_equal(replay(vs, vp), solved), "variant does not solve"
            out.append((vs, vp))
    return out


def _selftest() -> int:
    g, pz = load_group(), load_puzzle()
    gens, solved, inv_idx = pz["gens"], pz["solved"], pz["inv_idx"]
    print(f"loaded {g['P'].shape[0]} spatial symmetries "
          f"({g['meta']['with_inverse_antisymmetry']} frames with antisymmetry)")

    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(10):
        word = rng.integers(0, len(pz["move_names"]), size=20)
        s = solved.copy()
        for m in word:
            s = s[gens[m]]
        sol = [int(inv_idx[m]) for m in word[::-1]]          # solves s

        # every frame must round-trip
        for (k, inv) in frame_list(48):
            u = to_frame(s, k, inv, g)
            sig_inv = np.argsort(g["sigma"][k])
            fp = [int(sig_inv[m]) for m in sol]
            if inv:
                fp = [int(inv_idx[m]) for m in reversed(fp)]
            cur = u.copy()
            for m in fp:
                cur = cur[gens[m]]
            assert np.array_equal(cur, solved), f"frame ({k},{inv}) does not solve"
            back = from_frame(fp, k, inv, g, inv_idx)
            cur = s.copy()
            for m in back:
                cur = cur[gens[m]]
            assert np.array_equal(cur, solved), f"frame ({k},{inv}) map-back failed"
            n_ok += 1

        aug = augment_path(s, sol, g, pz)
        assert len(aug) == 48
    print(f"OK: {n_ok} frame round-trips + 10 x 48 augmentation variants verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
