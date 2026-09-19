"""Classical-search building blocks for the CayleyPy 6x6x6 puzzle."""

from cube666.classical import (
    Cube666Decomposition,
    ResidualReport,
    apply_path,
    build_decomposition,
    cluster_permutation,
    minimum_unrestricted_three_cycles,
    parity_repair_path,
    permutation_parity,
    residual_report,
)
from cube666.puzzle import Cube666Puzzle
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver
from cube666.macros import (
    MacroEffect,
    GreedyMacroResult,
    InsertedFinisherResult,
    build_isolated_three_cycle_library,
    greedy_macro_reduce,
    finish_with_inserted_three_cycles,
    load_three_cycle_library,
    save_three_cycle_library,
    enumerate_basic_corner_fixing_commutators,
    enumerate_basic_inner_commutators,
    enumerate_conjugated_macros,
    enumerate_nested_corner_fixing_commutators,
)

__all__ = [
    "Cube666Decomposition",
    "Cube666Puzzle",
    "CornerCoordinateSystem",
    "ExactCornerSolver",
    "MacroEffect",
    "GreedyMacroResult",
    "InsertedFinisherResult",
    "build_isolated_three_cycle_library",
    "greedy_macro_reduce",
    "finish_with_inserted_three_cycles",
    "load_three_cycle_library",
    "save_three_cycle_library",
    "enumerate_basic_corner_fixing_commutators",
    "enumerate_basic_inner_commutators",
    "enumerate_conjugated_macros",
    "enumerate_nested_corner_fixing_commutators",
    "ResidualReport",
    "apply_path",
    "build_decomposition",
    "cluster_permutation",
    "minimum_unrestricted_three_cycles",
    "parity_repair_path",
    "permutation_parity",
    "residual_report",
]
