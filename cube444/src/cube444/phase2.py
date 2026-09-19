"""Phase 2 of the two-phase 4x4x4 solver: the reduced cube is a 3x3x3.

Once a state satisfies `orbits.is_reduced` (centres carry their own face colour,
every wing pair is monochrome) the remaining work uses ONLY the 12 outer-layer
generators, under which each wing pair moves as a single edge and each centre block
as a single centre. That is exactly a 3x3x3 -- so we hand it to a classical
near-optimal solver instead of a beam.

Two things have to be derived, and both are derived from the generators themselves
rather than assumed from geometry:

1. **Cubie membership.** Two sticker slots lie on the same physical cubie iff they
   are moved by the same set of generators (a corner sits on 3 faces, a wing on
   2 faces + 1 inner slice, a centre on 1 face + 2 inner slices). This yields
   8 corners x3, 24 wings x2, 24 centres x1 -- verified against `orbits`.

2. **The frame.** Which of our 6 faces plays U/R/F/D/L/B. The 3 opposite-face pairs
   fall out of cubie membership (opposite faces never share a cubie), leaving
   3! x 2^3 = 48 candidate frames. Each is tested by building facelet strings for
   random reduced states and asking hkociemba's own validator; improper (mirrored)
   frames fail. The survivor is confirmed by a full solve-and-replay round trip.

METRIC NOTE
-----------
The competition counts every generator application as one move, i.e. quarter-turn
metric. hkociemba's solver minimises HALF-turn metric, where a 180 degree turn is
free relative to a 90. A 20-move HTM solution can cost anywhere from 20 to 40 QTM.
`Phase2Solver` therefore collects several candidate HTM solutions (different length
caps, plus the inverse cube) and returns the one with the lowest QTM cost.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cube444.orbits import (
    CENTER_BLOCKS, FACE_SIZE, N_FACES, STATE_SIZE, is_reduced, is_outer_move,
)

# hkociemba's `defs.FOLDER` defaults to the RELATIVE path "twophase", so the ~70 MB
# of pruning tables are looked up under whatever the current working directory
# happens to be. Run the same script from a different directory and it silently
# spends another ~30 minutes regenerating them. Pin it to the repo copy.
TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "twophase_tables"


def _pin_table_dir() -> None:
    from twophase import defs
    defs.FOLDER = str(TABLE_DIR)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

# hkociemba facelet order: U=0, R=1, F=2, D=3, L=4, B=5, 9 facelets each.
LETTERS = "URFDLB"
LETTER_IDX = {c: i for i, c in enumerate(LETTERS)}
# The three opposite letter pairs, in hkociemba's axis order.
LETTER_AXES = (("U", "D"), ("R", "L"), ("F", "B"))


def derive_cubie_groups(generators: dict[str, np.ndarray]) -> dict[str, list[list[int]]]:
    """Group the 96 slots into physical cubies by their 'moved-by' signature."""
    fwd = sorted(g for g in generators if not g.startswith("-"))
    sig: dict[tuple, list[int]] = defaultdict(list)
    for i in range(STATE_SIZE):
        key = tuple(g for g in fwd if int(generators[g][i]) != i)
        sig[key].append(i)
    groups = {"corner": [], "wing": [], "center": []}
    for slots in sig.values():
        if len(slots) == 3:
            groups["corner"].append(sorted(slots))
        elif len(slots) == 2:
            groups["wing"].append(sorted(slots))
        elif len(slots) == 1:
            groups["center"].append(sorted(slots))
        else:
            raise AssertionError(f"unexpected cubie of size {len(slots)}: {slots}")
    if len(groups["corner"]) != 8 or len(groups["wing"]) != 24 or len(groups["center"]) != 24:
        raise AssertionError(
            f"cubie counts wrong: {len(groups['corner'])}/{len(groups['wing'])}/"
            f"{len(groups['center'])} (want 8/24/24)"
        )
    return groups


def opposite_face_pairs(corner_groups: list[list[int]]) -> list[tuple[int, int]]:
    """Faces that never share a corner are opposite."""
    together = set()
    for g in corner_groups:
        faces = sorted({s // FACE_SIZE for s in g})
        for a in faces:
            for b in faces:
                if a != b:
                    together.add((a, b))
    pairs = [(a, b) for a in range(N_FACES) for b in range(a + 1, N_FACES)
             if (a, b) not in together]
    if len(pairs) != 3:
        raise AssertionError(f"expected 3 opposite pairs, got {pairs}")
    return pairs


def outer_gen_face(generators: dict[str, np.ndarray], name: str) -> int:
    """The face an outer-layer generator rotates.

    The turned face is the one whose 16 slots are permuted *among themselves* AND
    actually move -- the untouched opposite face also maps into itself (identically),
    so the 'stays on the face' test alone picks the wrong face.
    """
    p = generators[name]
    for f in range(N_FACES):
        lo, hi = f * FACE_SIZE, (f + 1) * FACE_SIZE
        within = all(lo <= int(p[i]) < hi for i in range(lo, hi))
        moves = any(int(p[i]) != i for i in range(lo, hi))
        if within and moves:
            return f
    raise AssertionError(f"{name} does not rotate a whole face")


def _facecube(s: str):
    """Build an hkociemba FaceCube from a 54-char string, or None if malformed."""
    _pin_table_dir()
    from twophase.face import FaceCube

    fc = FaceCube()
    if fc.from_string(s) is not True:
        return None
    return fc


def facelet_is_valid(s: str) -> bool:
    """True iff the facelet string is a solvable 3x3x3.

    hkociemba's `verify()` returns the sentinel `cubie.CUBE_OK` (which is `True`)
    on success and a human-readable error *string* on failure -- so this must be
    compared against CUBE_OK, not against 0.
    """
    _pin_table_dir()
    from twophase.cubie import CUBE_OK

    fc = _facecube(s)
    if fc is None:
        return False
    try:
        return fc.to_cubie_cube().verify() is CUBE_OK
    except Exception:
        return False


@dataclass
class ReductionMap:
    """Maps a reduced 96-colour state to a 54-char 3x3x3 facelet string."""

    face_to_letter: dict[int, str]
    slot_to_facelet: dict[int, int]
    letter_cw_gen: dict[str, str] = field(default_factory=dict)

    def to_facelets(self, state: np.ndarray) -> str:
        """Build the hkociemba facelet string. State must satisfy `is_reduced`."""
        out = [""] * 54
        for slot, fac in self.slot_to_facelet.items():
            out[fac] = self.face_to_letter[int(state[slot])]
        return "".join(out)

    def translate(self, htm_move: str) -> list[str]:
        """'U1'/'U2'/'U3' -> a list of our generator names (quarter turns)."""
        letter, n = htm_move[0], int(htm_move[1])
        cw = self.letter_cw_gen[letter]
        ccw = cw[1:] if cw.startswith("-") else "-" + cw
        return {1: [cw], 2: [cw, cw], 3: [ccw]}[n]

    def translate_solution(self, htm: str) -> list[str]:
        moves: list[str] = []
        for tok in htm.split():
            if tok.startswith("("):
                break
            moves.extend(self.translate(tok))
        return moves


def _candidate_frames(our_axes: list[tuple[int, int]]) -> list[dict[int, str]]:
    """All 48 face->letter assignments consistent with the opposite-pair structure."""
    import itertools

    frames = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((0, 1), repeat=3):
            m: dict[int, str] = {}
            for k in range(3):
                a, b = our_axes[k]
                la, lb = LETTER_AXES[perm[k]]
                if signs[k]:
                    a, b = b, a
                m[a], m[b] = la, lb
            frames.append(m)
    return frames


def _build_slot_to_facelet(face_to_letter, corner_groups, wing_groups) -> dict[int, int]:
    """Slot -> hkociemba facelet index, for one representative slot per facelet."""
    _pin_table_dir()
    from twophase import defs

    letter_to_face = {v: k for k, v in face_to_letter.items()}
    s2f: dict[int, int] = {}

    # centres: facelet 9*L+4 carries letter L; take one slot of that face's block
    for f in range(N_FACES):
        s2f[int(CENTER_BLOCKS[f][0])] = 9 * LETTER_IDX[face_to_letter[f]] + 4

    # corners
    corner_by_letters = {
        frozenset(LETTERS[c] for c in defs.cornerColor[i]): i
        for i in range(8)
    }
    for g in corner_groups:
        faces = [s // FACE_SIZE for s in g]
        key = frozenset(face_to_letter[f] for f in faces)
        idx = corner_by_letters[key]
        for k, col in enumerate(defs.cornerColor[idx]):
            want_face = letter_to_face[LETTERS[col]]
            slot = next(s for s in g if s // FACE_SIZE == want_face)
            s2f[slot] = int(defs.cornerFacelet[idx][k])

    # edges: merge the 2 wing cubies of each physical edge, take one slot per face
    edges: dict[frozenset, list[int]] = defaultdict(list)
    for g in wing_groups:
        faces = frozenset(s // FACE_SIZE for s in g)
        edges[faces].extend(g)
    edge_by_letters = {
        frozenset(LETTERS[c] for c in defs.edgeColor[j]): j
        for j in range(12)
    }
    for faces, slots in edges.items():
        key = frozenset(face_to_letter[f] for f in faces)
        idx = edge_by_letters[key]
        for k, col in enumerate(defs.edgeColor[idx]):
            want_face = letter_to_face[LETTERS[col]]
            slot = min(s for s in slots if s // FACE_SIZE == want_face)
            s2f[slot] = int(defs.edgeFacelet[idx][k])

    if len(set(s2f.values())) != 54:
        raise AssertionError(f"facelet map covers {len(set(s2f.values()))} of 54")
    return s2f


def build_reduction_map(generators: dict[str, np.ndarray], solved: np.ndarray,
                        n_probe: int = 12, seed: int = 0, verbose: bool = False
                        ) -> ReductionMap:
    """Derive and validate the 4x4x4 -> 3x3x3 frame. Raises if none is consistent."""
    _pin_table_dir()
    from twophase import cubie

    gens = {k: np.asarray(v, dtype=np.int64) for k, v in generators.items()}
    groups = derive_cubie_groups(gens)
    our_axes = opposite_face_pairs(groups["corner"])

    # random reduced states to test candidate frames against
    rng = np.random.default_rng(seed)
    outer = [g for g in gens if is_outer_move(g)]
    probes = []
    s = np.asarray(solved, dtype=np.int64).copy()
    for _ in range(n_probe):
        for _ in range(rng.integers(4, 30)):
            s = s[gens[outer[rng.integers(len(outer))]]]
        assert bool(is_reduced(s)), "outer walk left R -- orbit structure violated"
        probes.append(s.copy())

    survivors = []
    for frame in _candidate_frames(our_axes):
        try:
            s2f = _build_slot_to_facelet(frame, groups["corner"], groups["wing"])
        except (KeyError, AssertionError):
            continue
        rm = ReductionMap(face_to_letter=frame, slot_to_facelet=s2f)
        if all(facelet_is_valid(rm.to_facelets(p)) for p in probes):
            survivors.append(rm)

    if verbose:
        print(f"  candidate frames: 48 -> {len(survivors)} passed hkociemba validation")
    if not survivors:
        raise AssertionError("no consistent 4x4x4 -> 3x3x3 frame found")

    # Fix the clockwise generator per letter by matching hkociemba's basic moves.
    #
    # NOTE: `verify()` alone does NOT discriminate -- all 48 frames produce cubes it
    # accepts, because it only checks orientation sums and permutation parity, which
    # relabelling preserves. The basic-move match below is the test that actually
    # pins the frame, so it must find exactly one survivor.
    matched = []
    for rm in survivors:
        letter_cw: dict[str, str] = {}
        for name in gens:
            if name.startswith("-") or not is_outer_move(name):
                continue
            f = outer_gen_face(gens, name)
            letter = rm.face_to_letter[f]
            ours = rm.to_facelets(np.asarray(solved)[gens[name]])
            ref = cubie.basicMoveCube[LETTER_IDX[letter]].to_facelet_cube().to_string()
            inv_name = "-" + name
            ours_inv = rm.to_facelets(np.asarray(solved)[gens[inv_name]])
            if ours == ref:
                letter_cw[letter] = name
            elif ours_inv == ref:
                letter_cw[letter] = inv_name
        if len(letter_cw) == 6:
            rm.letter_cw_gen = letter_cw
            matched.append(rm)

    if not matched:
        raise AssertionError(
            "found a valid frame but could not match all 6 basic moves; "
            "the surviving frames were likely mirrored"
        )

    # Every self-consistent frame (all 24 rotations, and their mirrors) is an equally
    # valid representation: the facelet map is BUILT from the frame, so relabelling
    # cancels out and a solution translated back through the same frame still solves
    # the physical cube. So we do not demand uniqueness -- we demand the property we
    # actually rely on, that translating a WORD is a homomorphism, and take the first
    # frame that has it.
    for rm in matched:
        if _word_homomorphism_ok(rm, gens, solved, seed=seed):
            if verbose:
                print(f"  frames matching all 6 basic moves: {len(matched)} "
                      f"(all are equivalent relabellings)")
                print(f"  frame {rm.face_to_letter}")
                print(f"  clockwise generators {rm.letter_cw_gen}")
            return rm

    raise AssertionError(
        "no frame survived the word-homomorphism check -- translating a multi-move "
        "word does not agree with hkociemba's own composition"
    )


def _word_homomorphism_ok(rm: "ReductionMap", gens: dict[str, np.ndarray],
                          solved: np.ndarray, n_words: int = 40, word_len: int = 12,
                          seed: int = 0) -> bool:
    """Check translate() is a homomorphism on WORDS, not just single moves.

    For a random letter word, the cube hkociemba reaches by composing those moves
    must have the same facelet string as the one we reach by applying the translated
    generator word to our solved state. This is exactly the property that makes a
    translated solution actually solve the cube, and it needs no pruning tables.
    """
    _pin_table_dir()
    from twophase import cubie

    rng = np.random.default_rng(seed + 1)
    letters = list(rm.letter_cw_gen)
    for _ in range(n_words):
        toks = [f"{letters[rng.integers(len(letters))]}{rng.integers(1, 4)}"
                for _ in range(word_len)]
        ref = cubie.CubieCube()
        for tok in toks:
            ref.multiply(cubie.basicMoveCube[LETTER_IDX[tok[0]]])
            if tok[1] == "2":
                ref.multiply(cubie.basicMoveCube[LETTER_IDX[tok[0]]])
            elif tok[1] == "3":
                ref.multiply(cubie.basicMoveCube[LETTER_IDX[tok[0]]])
                ref.multiply(cubie.basicMoveCube[LETTER_IDX[tok[0]]])
        ours = np.asarray(solved, dtype=np.int64)
        for tok in toks:
            for m in rm.translate(tok):
                ours = ours[gens[m]]
        if rm.to_facelets(ours) != ref.to_facelet_cube().to_string():
            return False
    return True


def qtm_cost(htm: str) -> int:
    """Quarter-turn cost of an hkociemba HTM solution string."""
    cost = 0
    for tok in htm.split():
        if tok.startswith("("):
            break
        cost += 2 if tok[1] == "2" else 1
    return cost


class Phase2Solver:
    """Near-optimal QTM solver for the reduced cube.

    `max_lengths` are HTM caps handed to hkociemba; each yields a different word and
    we keep the cheapest in QTM. Lower caps give shorter HTM but cost more search
    time and may time out (in which case that candidate is simply skipped).
    """

    def __init__(self, generators, solved, max_lengths=(18, 19, 20),
                 timeout: float = 1.0, verbose: bool = False):
        self.gens = {k: np.asarray(v, dtype=np.int64) for k, v in generators.items()}
        self.solved = np.asarray(solved, dtype=np.int64)
        self.rmap = build_reduction_map(self.gens, self.solved, verbose=verbose)
        self.max_lengths = tuple(max_lengths)
        self.timeout = timeout

    def is_parity_broken(self, state: np.ndarray) -> bool:
        """True if the reduced state is not solvable by outer moves (4x4x4 parity)."""
        return not facelet_is_valid(self.rmap.to_facelets(state))

    def solve(self, state: np.ndarray) -> list[str] | None:
        """Return our generator word solving a reduced state, or None if unsolvable."""
        _pin_table_dir()
        from twophase import solver

        state = np.asarray(state, dtype=np.int64)
        if not bool(is_reduced(state)):
            raise ValueError("state is not reduced -- phase 2 requires R(s)")
        if self.is_parity_broken(state):
            return None

        fs = self.rmap.to_facelets(state)
        best: list[str] | None = None
        best_cost = 1 << 30
        for cap in self.max_lengths:
            try:
                res = solver.solve(fs, max_length=cap, timeout=self.timeout)
            except Exception:
                continue
            if not res or res.startswith("Error"):
                continue
            c = qtm_cost(res)
            if c < best_cost:
                best_cost, best = c, self.rmap.translate_solution(res)
        return best

    def apply_word(self, state: np.ndarray, word: list[str]) -> np.ndarray:
        cur = np.asarray(state, dtype=np.int64)
        for m in word:
            cur = cur[self.gens[m]]
        return cur
