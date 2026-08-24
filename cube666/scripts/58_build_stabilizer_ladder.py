"""Build a verified six-stage stabilizer ladder from the KMC macro vocabulary.

The flat macro policy has to rank almost 28,000 actions over six coupled
24-piece clusters.  This script instead finds a cluster order for which every
stage has macros that fix all earlier clusters pointwise at macro boundaries.
For each stage it deduplicates actions by their active-cluster effect and then
extracts a short, inverse-closed generating basis for A24.

No learned quantity is used here.  Every retained word is replayed through the
exact 216-sticker puzzle and every locked-cluster claim is checked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, permutation_parity  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macros import inverse_permutation  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_ORDER = (0, 4, 5, 2, 3, 1)
IDENTITY24 = tuple(range(24))
ALT24_ORDER = math.factorial(24) // 2


@dataclass(frozen=True)
class Candidate:
    source_action: int
    effect: tuple[int, ...]
    path: tuple[str, ...]


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
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "stabilizer_ladder_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "stabilizer_ladder_v1.json",
    )
    parser.add_argument(
        "--order",
        type=int,
        nargs=6,
        default=DEFAULT_ORDER,
        metavar=("C0", "C1", "C2", "C3", "C4", "C5"),
    )
    return parser.parse_args()


def permutation_group_order(effects: Sequence[tuple[int, ...]]) -> int:
    generators = [Permutation(list(IDENTITY24), size=24)]
    group = PermutationGroup(generators)
    order = 1
    for effect in effects:
        permutation = Permutation(list(effect), size=24)
        if group.contains(permutation):
            continue
        generators.append(permutation)
        group = PermutationGroup(generators)
        order = int(group.order())
        if order == ALT24_ORDER:
            break
        if order > ALT24_ORDER or ALT24_ORDER % order:
            raise ValueError(f"unexpected projected group order {order}")
    return order


def deduplicated_candidates(
    active_cluster: int,
    locked_clusters: tuple[int, ...],
    active_masks: np.ndarray,
    effects: np.ndarray,
    paths: Sequence[tuple[str, ...]],
) -> list[Candidate]:
    eligible = active_masks[:, active_cluster].copy()
    if locked_clusters:
        eligible &= ~active_masks[:, locked_clusters].any(axis=1)

    shortest: dict[bytes, Candidate] = {}
    for action in np.flatnonzero(eligible):
        effect = tuple(int(value) for value in effects[action, active_cluster])
        path = tuple(paths[int(action)])
        candidate = Candidate(int(action), effect, path)
        key = bytes(effect)
        old = shortest.get(key)
        if old is None or (len(path), path, int(action)) < (
            len(old.path),
            old.path,
            old.source_action,
        ):
            shortest[key] = candidate
    return sorted(
        shortest.values(),
        key=lambda candidate: (len(candidate.path), candidate.path, candidate.source_action),
    )


def inverse_closed_basis(
    candidates: Sequence[Candidate],
    *,
    target_order: int,
) -> tuple[list[Candidate], list[int]]:
    by_effect = {bytes(candidate.effect): candidate for candidate in candidates}
    selected: dict[bytes, Candidate] = {}
    order_trajectory = [1]
    group = PermutationGroup([Permutation(list(IDENTITY24), size=24)])

    for candidate in candidates:
        permutation = Permutation(list(candidate.effect), size=24)
        if group.contains(permutation):
            continue
        inverse = tuple(inverse_permutation(candidate.effect))
        inverse_candidate = by_effect.get(bytes(inverse))
        if inverse_candidate is None:
            raise ValueError(
                f"projected effect for source action {candidate.source_action} has no inverse"
            )
        selected[bytes(candidate.effect)] = candidate
        selected[bytes(inverse_candidate.effect)] = inverse_candidate
        group = PermutationGroup(
            [Permutation(list(item.effect), size=24) for item in selected.values()]
        )
        current_order = int(group.order())
        order_trajectory.append(current_order)
        if current_order == target_order:
            break
        if current_order > target_order or target_order % current_order:
            raise ValueError(f"unexpected projected subgroup order {current_order}")

    final = sorted(
        selected.values(),
        key=lambda candidate: (len(candidate.path), candidate.path, candidate.source_action),
    )
    final_order = permutation_group_order([candidate.effect for candidate in final])
    if final_order != target_order:
        raise ValueError(
            f"basis generates order {final_order}, expected {target_order}"
        )
    return final, order_trajectory


def main() -> None:
    args = parse_args()
    order = tuple(int(cluster) for cluster in args.order)
    if set(order) != set(range(6)):
        raise ValueError("--order must be a permutation of 0..5")

    puzzle = Cube666Puzzle.load(args.puzzle_info)
    decomposition = build_decomposition(puzzle.generators)
    macros, table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )
    identity = np.arange(24, dtype=np.uint8)
    active_masks = np.any(table.effects != identity[None, None, :], axis=2)

    stages = []
    for stage_index, active_cluster in enumerate(order):
        locked = order[:stage_index]
        candidates = deduplicated_candidates(
            active_cluster,
            locked,
            active_masks,
            table.effects,
            table.paths,
        )
        if not candidates:
            raise ValueError(
                f"stage {stage_index} has no actions for active cluster {active_cluster}"
            )
        if any(permutation_parity(candidate.effect) for candidate in candidates):
            raise ValueError(f"stage {stage_index} contains an odd active-cluster action")
        candidate_group_order = permutation_group_order(
            [candidate.effect for candidate in candidates]
        )
        if stage_index < len(order) - 1 and candidate_group_order != ALT24_ORDER:
            raise ValueError(
                f"nonterminal stage {stage_index} reaches order {candidate_group_order}, "
                f"expected full A24={ALT24_ORDER}"
            )
        basis, order_trajectory = inverse_closed_basis(
            candidates,
            target_order=candidate_group_order,
        )

        # Recheck the exact full-cluster effects loaded from replayed primitive words.
        for candidate in basis:
            macro = macros[candidate.source_action]
            for locked_cluster in locked:
                if macro.cluster_permutations[locked_cluster] != IDENTITY24:
                    raise AssertionError(
                        f"stage {stage_index} action {candidate.source_action} moves "
                        f"locked cluster {locked_cluster}"
                    )
            if macro.cluster_permutations[active_cluster] == IDENTITY24:
                raise AssertionError("retained action is identity on its active cluster")

        raw_eligible = active_masks[:, active_cluster].copy()
        if locked:
            raw_eligible &= ~active_masks[:, locked].any(axis=1)
        stage = {
            "active_cluster": active_cluster,
            "basis": [
                {
                    "active_effect": list(candidate.effect),
                    "path": list(candidate.path),
                    "primitive_length": len(candidate.path),
                    "source_action": candidate.source_action,
                }
                for candidate in basis
            ],
            "basis_action_count": len(basis),
            "basis_primitive_length_max": max(len(candidate.path) for candidate in basis),
            "basis_primitive_length_mean": round(
                float(np.mean([len(candidate.path) for candidate in basis])), 6
            ),
            "deduplicated_active_effects": len(candidates),
            "locked_clusters": list(locked),
            "projected_deficit_bits_from_A24": round(
                math.log2(ALT24_ORDER / candidate_group_order), 6
            ),
            "projected_group_order": candidate_group_order,
            "projected_group_order_trajectory": order_trajectory,
            "raw_eligible_actions": int(raw_eligible.sum()),
            "stage_index": stage_index,
        }
        stages.append(stage)
        print(
            f"stage={stage_index} active={active_cluster} locked={locked} "
            f"raw={stage['raw_eligible_actions']} dedup={len(candidates)} "
            f"basis={len(basis)} mean_len={stage['basis_primitive_length_mean']}"
        )

    payload = {
        "action_library": str(args.action_library.resolve()),
        "action_library_digest": table.digest,
        "cluster_names": list(decomposition.cluster_names),
        "format_version": 1,
        "order": list(order),
        "projected_target_group": "A24",
        "projected_target_group_order": ALT24_ORDER,
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = {
        "artifact": str(args.output.resolve()),
        "basis_actions_total": sum(stage["basis_action_count"] for stage in stages),
        "format_version": 1,
        "gate_a_passed": True,
        "order": list(order),
        "stages": [
            {key: value for key, value in stage.items() if key != "basis"}
            for stage in stages
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
