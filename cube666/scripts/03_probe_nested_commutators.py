"""Probe [A, C B C^-1] finishing macros on the real 666 definition."""

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
    build_isolated_three_cycle_library,
    enumerate_nested_corner_fixing_commutators,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--inner-conjugator-depth", type=int, default=1)
    parser.add_argument("--finisher-conjugator-depth", type=int, default=1)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "nested_commutators.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    macros = enumerate_nested_corner_fixing_commutators(
        puzzle.generators,
        decomposition,
        inner_conjugator_depth=args.inner_conjugator_depth,
    )
    single_three_cycles = tuple(
        macro
        for macro in macros
        if macro.total_nontrivial_cycles == 1 and macro.total_three_cycles == 1
    )
    pure_three = tuple(macro for macro in macros if macro.pure_three_cycles)
    library = build_isolated_three_cycle_library(
        single_three_cycles,
        puzzle.generators,
        decomposition,
        max_conjugator_depth=args.finisher_conjugator_depth,
    )

    single_coverage = []
    for cluster_index in range(len(decomposition.physical_clusters)):
        cycles = {
            macro.cluster_cycles[cluster_index][0]
            for macro in single_three_cycles
            if macro.cluster_cycles[cluster_index]
        }
        single_coverage.append(len(cycles))
    library_coverage = [0] * len(decomposition.physical_clusters)
    library_path_lengths: Counter[int] = Counter()
    for (cluster_index, _), macro in library.items():
        library_coverage[cluster_index] += 1
        library_path_lengths[len(macro.path)] += 1

    profiles = Counter(
        "/".join(
            ",".join(str(len(cycle)) for cycle in cycles) or "-"
            for cycles in macro.cluster_cycles
        )
        for macro in macros
    )
    path_lengths = Counter(len(macro.path) for macro in macros)
    unit_distribution = Counter(macro.unrestricted_three_cycle_units for macro in macros)
    exemplars = [
        {
            "path": list(macro.path),
            "cluster": next(
                index for index, cycles in enumerate(macro.cluster_cycles) if cycles
            ),
            "cycle": list(next(cycles[0] for cycles in macro.cluster_cycles if cycles)),
        }
        for macro in sorted(single_three_cycles, key=lambda item: (len(item.path), item.path))[:20]
    ]

    report = {
        "inner_conjugator_depth": args.inner_conjugator_depth,
        "unique_effects": len(macros),
        "path_length_distribution": dict(sorted(path_lengths.items())),
        "algebraic_three_cycle_unit_distribution": dict(sorted(unit_distribution.items())),
        "pure_three_cycle_effects": len(pure_three),
        "isolated_three_cycle_effects": len(single_three_cycles),
        "isolated_directed_cycle_coverage_by_cluster": single_coverage,
        "finisher_conjugator_depth": args.finisher_conjugator_depth,
        "finisher_library_size": len(library),
        "finisher_directed_cycle_coverage_by_cluster": library_coverage,
        "finisher_path_length_distribution": dict(sorted(library_path_lengths.items())),
        "cycle_length_profiles": dict(sorted(profiles.items())),
        "isolated_exemplars": exemplars,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.json_out)


if __name__ == "__main__":
    main()
