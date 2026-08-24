"""Export one local exact-sticker 666 state to KMCoders' stdin format.

This is a diagnostic bridge to the public Santa 2023 classical cube solver.  It
keeps our competition data immutable and emits only the 18 positive generators;
the Rust program constructs their inverses itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    rows = list(puzzle.iter_test_states(args.data_dir / "test.csv"))
    if not 0 <= args.index < len(rows):
        raise ValueError(f"index must be in 0..{len(rows) - 1}")
    state_id, state = rows[args.index]
    positive_moves = tuple(name for name in puzzle.move_names if not name.startswith("-"))
    expected = tuple(f"{axis}{layer}" for axis in "frd" for layer in range(6))
    if positive_moves != expected:
        raise ValueError(
            "KMCoders' cube coordinate code requires generator order "
            f"{expected}, got {positive_moves}"
        )

    tokens: list[str] = [str(state_id), "cube_6/6/6", str(puzzle.size), str(len(positive_moves)), "0"]
    for name in positive_moves:
        tokens.append(name)
        tokens.extend(str(value) for value in puzzle.generators[name])
    tokens.extend(str(value) for value in puzzle.solved_state)
    tokens.extend(str(value) for value in state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(" ".join(tokens) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"exported state {state_id} ({puzzle.size} stickers, {len(positive_moves)} moves) to {args.out}")


if __name__ == "__main__":
    main()
