"""Measure projected reachability after locking cluster subsets.

This is the algebraic gate that decides whether the ladder may use a strict
single-cluster stage or must solve a coupled set.  It reports the group induced
on every remaining cluster by macros that fix the requested locked subset.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402

ALT24_ORDER = math.factorial(24) // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--puzzle-info",
        type=Path,
        default=PROJECT / "cayley-py-666-cube" / "puzzle_info.json",
    )
    parser.add_argument(
        "--action-library",
        type=Path,
        default=(
            PROJECT
            / "cube666"
            / "training"
            / "kmc_macro_teacher_allruns_sym8_v3"
            / "action_library.json"
        ),
    )
    parser.add_argument("--max-locked", type=int, default=2)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "stabilizer_reachability_v1.json",
    )
    return parser.parse_args()


def projected_order(effects: np.ndarray) -> tuple[int, int]:
    unique = {row.tobytes(): row for row in effects}
    if not unique:
        return 1, 0
    identity = Permutation(list(range(24)), size=24)
    generators = [identity]
    group = PermutationGroup(generators)
    order = 1
    for row in unique.values():
        permutation = Permutation(row.tolist(), size=24)
        if group.contains(permutation):
            continue
        generators.append(permutation)
        group = PermutationGroup(generators)
        order = int(group.order())
        if order == ALT24_ORDER:
            break
        if order > ALT24_ORDER or ALT24_ORDER % order:
            raise ValueError(f"unexpected projected group order {order}")
    return order, len(unique)


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.puzzle_info)
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )
    identity = np.arange(24, dtype=np.uint8)
    active = np.any(table.effects != identity[None, None, :], axis=2)

    rows = []
    for locked_count in range(args.max_locked + 1):
        for locked in itertools.combinations(range(6), locked_count):
            eligible = np.ones(table.action_count, dtype=np.bool_)
            if locked:
                eligible &= ~active[:, locked].any(axis=1)
            for target in range(6):
                if target in locked:
                    continue
                selected = eligible & active[:, target]
                order, unique = projected_order(table.effects[selected, target])
                row = {
                    "deficit_bits_from_A24": round(math.log2(ALT24_ORDER / order), 6),
                    "full_A24": order == ALT24_ORDER,
                    "locked": list(locked),
                    "projected_group_order": order,
                    "raw_actions": int(selected.sum()),
                    "target": target,
                    "unique_projected_actions": unique,
                }
                rows.append(row)
                print(
                    f"locked={locked!s:9} target={target} raw={row['raw_actions']:5} "
                    f"unique={unique:5} full={row['full_A24']} "
                    f"deficit_bits={row['deficit_bits_from_A24']:8.3f}",
                    flush=True,
                )

    payload = {
        "action_library": str(args.action_library.resolve()),
        "action_library_digest": table.digest,
        "a24_order": ALT24_ORDER,
        "format_version": 1,
        "max_locked": args.max_locked,
        "rows": rows,
        "strict_singleton_transitions_full": sum(
            row["full_A24"] and len(row["locked"]) > 0 for row in rows
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
