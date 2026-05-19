"""Build action-relabel tables for symmetry-aware policy/Q training.

For each rotation R, computes the integer table

    action_relabel[R, a] = the action index a' such that
                          apply(R*s*R_inv, a') == R*apply(s, a)*R_inv

Mirrors the conjugation logic in `megaminx/scripts/03_solve.py:_compute_conjugation_map`.

Outputs:
    megaminx/data/action_relabel.pt        (rotations.npy / 360 elements)
    megaminx/data/action_relabel_720.pt    (rotations_720.npy / 720 elements)

Round-trip unit test inside this script verifies correctness.

Run:
    .venv/Scripts/python.exe megaminx/scripts/73_build_action_relabel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def _invert_perm(p: np.ndarray) -> np.ndarray:
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    return inv


def build_action_relabel(puzzle: Megaminx, rotations: np.ndarray) -> torch.Tensor:
    """Returns LongTensor of shape (N_rotations, n_generators)."""
    n_rot = rotations.shape[0]
    n_gen = len(puzzle.move_names)
    state_size = len(puzzle.solved_state)
    assert rotations.shape[1] == state_size

    perm_to_idx: dict[tuple[int, ...], int] = {
        tuple(puzzle.generators[name]): idx for idx, name in enumerate(puzzle.move_names)
    }

    relabel = torch.full((n_rot, n_gen), -1, dtype=torch.long)
    for r in range(n_rot):
        R = rotations[r].astype(np.int64)
        R_inv = _invert_perm(R)
        for a, name in enumerate(puzzle.move_names):
            g = np.array(puzzle.generators[name], dtype=np.int64)
            # action_relabel[r, a] = a' such that:
            #     apply(R s R_inv, a') == R apply(s, a) R_inv
            # Derivation: gen_{a'} = R · gen_a · R_inv as a permutation.
            # In index form: gen_{a'}[i] = R[gen_a[R_inv[i]]].
            conj = R[g[R_inv]]
            key = tuple(conj.tolist())
            if key not in perm_to_idx:
                raise ValueError(
                    f"rotation {r}: R * {name} * R_inv is not a generator"
                )
            relabel[r, a] = perm_to_idx[key]
    assert (relabel >= 0).all()
    return relabel


def _apply_rotation_to_state(state: tuple[int, ...], R: np.ndarray, R_inv: np.ndarray) -> tuple[int, ...]:
    """Compute R · state · R_inv per the 03_solve.py convention.

    out[i] = R[state[R_inv[i]]]
    """
    return tuple(int(R[state[int(R_inv[i])]]) for i in range(len(state)))


def _apply_gen(state: tuple[int, ...], gen: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(state[g] for g in gen)


def round_trip_test(puzzle: Megaminx, rotations: np.ndarray, relabel: torch.Tensor,
                    n_samples: int = 200, seed: int = 0) -> None:
    """For random states + random rotations + random actions, verify:
        apply(rot(s), relabel[r, a]) == rot(apply(s, a))
    """
    rng = np.random.default_rng(seed)
    state_size = len(puzzle.solved_state)
    n_rot = rotations.shape[0]
    n_gen = len(puzzle.move_names)
    n_pass = 0
    for _ in range(n_samples):
        # Random state: apply random walk of 10 moves from solved
        s = puzzle.solved_state
        n_steps = int(rng.integers(0, 20))
        for _ in range(n_steps):
            mv = puzzle.move_names[int(rng.integers(0, n_gen))]
            s = puzzle.apply_move(s, mv)
        # Random rotation + random action
        r = int(rng.integers(0, n_rot))
        a = int(rng.integers(0, n_gen))
        R = rotations[r].astype(np.int64)
        R_inv = _invert_perm(R)
        s_rot = _apply_rotation_to_state(s, R, R_inv)
        # LHS: apply rotated action to rotated state
        a_prime = int(relabel[r, a].item())
        lhs = _apply_gen(s_rot, puzzle.generators[puzzle.move_names[a_prime]])
        # RHS: apply original action, then rotate
        s_after = _apply_gen(s, puzzle.generators[puzzle.move_names[a]])
        rhs = _apply_rotation_to_state(s_after, R, R_inv)
        if lhs != rhs:
            raise AssertionError(
                f"round-trip failed: rot={r}, action={a} ({puzzle.move_names[a]} -> "
                f"{puzzle.move_names[a_prime]}), state_len={len(s)}"
            )
        n_pass += 1
    print(f"  round-trip OK on {n_pass}/{n_samples} samples")


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")

    for name in ("rotations.npy", "rotations_720.npy"):
        path = PROJECT / "data" / name
        if not path.exists():
            print(f"skipping {name}: not found")
            continue
        rotations = np.load(path)
        print(f"loading {name}: shape={rotations.shape}")
        relabel = build_action_relabel(puzzle, rotations)
        round_trip_test(puzzle, rotations, relabel)
        out = path.with_name(name.replace("rotations", "action_relabel").replace(".npy", ".pt"))
        torch.save(
            {
                "relabel": relabel,
                "n_rotations": int(rotations.shape[0]),
                "n_generators": len(puzzle.move_names),
                "move_names": list(puzzle.move_names),
                "source_rotations": name,
            },
            out,
        )
        print(f"  saved {out}  ({out.stat().st_size / 1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
