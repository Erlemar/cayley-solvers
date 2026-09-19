"""Fast integrity and one-step neural-search smoke test for the portable bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import load_macro_action_library, validate_factorized_action_table  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def main() -> None:
    data = ROOT / "cayley-py-666-cube"
    puzzle = Cube666Puzzle.load(data / "puzzle_info.json")
    puzzle.verify_inverse_pairs()
    decomposition = build_decomposition(puzzle.generators)
    orbit_count = (
        1 + len(decomposition.center_orbits) + 2 * len(decomposition.wing_pairs)
    )
    if puzzle.size != 216 or orbit_count != 9:
        raise AssertionError("unexpected CUBE666 structure")

    actions = ROOT / "cube666" / "training" / "finisher_policy_v1" / "action_library.json"
    _, table = load_macro_action_library(actions, puzzle.generators, decomposition)
    block_size = validate_factorized_action_table(table)
    checkpoint_path = (
        ROOT / "models" / "cube666_finisher_factorized_pro6000_v1" / "checkpoint.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest:
        raise AssertionError("checkpoint/action digest mismatch")
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()
    outward_action = 0
    one_step_state = table.apply(identity, outward_action)
    result = learned_macro_beam_search(
        one_step_state,
        model,
        table,
        beam_width=64,
        branch_width=32,
        max_steps=4,
        policy_nll_weight=0.02,
        model_batch_size=64,
    )
    if not result.solved:
        raise AssertionError("neural beam did not solve the one-action smoke state")
    replay = one_step_state.copy()
    for action in result.actions:
        replay = table.apply(replay, action)
    if not np.array_equal(replay, identity):
        raise AssertionError("cluster replay failed")

    kmc_root = ROOT / "external" / "third_party" / "kmcoders_santa2023" / "solution"
    kmc_windows = kmc_root / "target" / "release" / "solve_cube_beam.exe"
    kmc_source = kmc_root / "src" / "bin" / "solve_cube_beam.rs"
    if not kmc_windows.is_file() or not kmc_source.is_file():
        raise AssertionError("KMC binary/source missing")

    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "factorized_block_size": block_size,
        "kmc_windows_binary": True,
        "macro_steps": result.macro_steps,
        "move_count": len(puzzle.move_names),
        "orbits": orbit_count,
        "python": sys.version.split()[0],
        "puzzle_size": puzzle.size,
        "torch": torch.__version__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
