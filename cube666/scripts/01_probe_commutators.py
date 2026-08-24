"""Measure short inner-slice commutators on the real 666 move definition."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macros import (  # noqa: E402
    enumerate_basic_corner_fixing_commutators,
    enumerate_basic_inner_commutators,
    enumerate_conjugated_macros,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "basic_commutators.json",
    )
    parser.add_argument(
        "--conjugator-depth",
        type=int,
        default=1,
        help="close base effects under setup conjugators through this depth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    inner_only = enumerate_basic_inner_commutators(puzzle.generators, decomposition)
    macros = enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition)
    expanded_macros = enumerate_conjugated_macros(
        macros,
        puzzle.generators,
        decomposition,
        max_conjugator_depth=args.conjugator_depth,
    )
    pure = tuple(macro for macro in macros if macro.pure_three_cycles)
    wing_macros = tuple(
        macro for macro in macros if macro.cluster_cycles[4] or macro.cluster_cycles[5]
    )

    cycle_count_distribution = Counter(macro.total_three_cycles for macro in pure)
    activity_distribution = Counter(macro.active_cluster_mask for macro in pure)
    all_activity_distribution = Counter(macro.active_cluster_mask for macro in macros)
    cycle_length_profiles = Counter(
        "/".join(
            ",".join(str(len(cycle)) for cycle in cycles) or "-"
            for cycles in macro.cluster_cycles
        )
        for macro in macros
    )
    unique_cycles_by_cluster = []
    for cluster_index in range(len(decomposition.physical_clusters)):
        cycles = {
            cycle
            for macro in pure
            for cycle in macro.cluster_cycles[cluster_index]
        }
        unique_cycles_by_cluster.append(len(cycles))

    expanded_cycles_by_cluster = []
    expanded_cycle_lengths_by_cluster = []
    for cluster_index in range(len(decomposition.physical_clusters)):
        cycles = {
            cycle
            for macro in expanded_macros
            for cycle in macro.cluster_cycles[cluster_index]
        }
        expanded_cycles_by_cluster.append(len(cycles))
        expanded_cycle_lengths_by_cluster.append(
            dict(sorted(Counter(len(cycle) for cycle in cycles).items()))
        )

    exemplars = []
    for macro in sorted(pure, key=lambda item: (-item.total_three_cycles, item.path))[:20]:
        exemplars.append(
            {
                "path": list(macro.path),
                "parallel_three_cycles": macro.total_three_cycles,
                "active_cluster_mask": macro.active_cluster_mask,
                "cycle_counts_by_cluster": [len(cycles) for cycles in macro.cluster_cycles],
            }
        )

    wing_exemplars = []
    for macro in sorted(
        wing_macros,
        key=lambda item: (-item.total_three_cycles, item.path),
    )[:20]:
        wing_exemplars.append(
            {
                "path": list(macro.path),
                "three_cycles": macro.total_three_cycles,
                "active_cluster_mask": macro.active_cluster_mask,
                "cycle_lengths_by_cluster": [
                    [len(cycle) for cycle in cycles]
                    for cycles in macro.cluster_cycles
                ],
            }
        )

    report = {
        "ordered_perpendicular_candidates": sum(
            left.lstrip("-")[0] != right.lstrip("-")[0]
            and (
                left in decomposition.inner_move_names
                or right in decomposition.inner_move_names
            )
            for left in puzzle.move_names
            for right in puzzle.move_names
        ),
        "inner_inner_unique_effects": len(inner_only),
        "unique_nonidentity_effects": len(macros),
        "algebraic_three_cycle_unit_distribution": dict(
            sorted(Counter(macro.unrestricted_three_cycle_units for macro in macros).items())
        ),
        "pure_three_cycle_effects": len(pure),
        "wing_moving_effects": len(wing_macros),
        "pure_three_cycle_wing_effects": sum(macro.pure_three_cycles for macro in wing_macros),
        "parallel_three_cycle_distribution": dict(sorted(cycle_count_distribution.items())),
        "active_cluster_mask_distribution": dict(sorted(activity_distribution.items())),
        "all_effect_active_cluster_mask_distribution": dict(sorted(all_activity_distribution.items())),
        "all_effect_cycle_length_profiles": dict(sorted(cycle_length_profiles.items())),
        "unique_directed_cycles_by_cluster": unique_cycles_by_cluster,
        "conjugator_depth": args.conjugator_depth,
        "expanded_unique_effects": len(expanded_macros),
        "expanded_unique_cycles_by_cluster": expanded_cycles_by_cluster,
        "expanded_unique_cycle_lengths_by_cluster": expanded_cycle_lengths_by_cluster,
        "exemplars": exemplars,
        "wing_exemplars": wing_exemplars,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.json_out)


if __name__ == "__main__":
    main()
