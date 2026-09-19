"""Append Walton effects absent from the checkpoint-compatible KMC action library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    load_macro_action_library,
    save_macro_action_library,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument(
        "--kmc-library",
        type=Path,
        default=PROJECT
        / "cube666"
        / "training"
        / "kmc_macro_teacher_v1"
        / "action_library.json",
    )
    parser.add_argument(
        "--walton-library",
        type=Path,
        default=PROJECT
        / "cube666"
        / "training"
        / "walton_macro4"
        / "action_library.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "training" / "kmc_walton_macro4",
    )
    parser.add_argument("--expected-novel", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    kmc_macros, kmc_table = load_macro_action_library(
        args.kmc_library,
        puzzle.generators,
        decomposition,
    )
    walton_macros, walton_table = load_macro_action_library(
        args.walton_library,
        puzzle.generators,
        decomposition,
    )

    kmc_effects = {effect.tobytes() for effect in kmc_table.effects}
    novel = tuple(
        (walton_index, macro)
        for walton_index, (macro, effect) in enumerate(
            zip(walton_macros, walton_table.effects, strict=True)
        )
        if effect.tobytes() not in kmc_effects
    )
    if len(novel) != args.expected_novel:
        raise ValueError(
            f"expected {args.expected_novel} Walton effects absent from KMC; found {len(novel)}"
        )

    combined_macros = kmc_macros + tuple(macro for _, macro in novel)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "action_library.json"
    combined_table = save_macro_action_library(output, combined_macros)
    reloaded_macros, reloaded_table = load_macro_action_library(
        output,
        puzzle.generators,
        decomposition,
    )
    if reloaded_table.digest != combined_table.digest:
        raise AssertionError("combined action-library digest changed after reload")
    if not np.array_equal(
        reloaded_table.effects[: kmc_table.action_count],
        kmc_table.effects,
    ):
        raise AssertionError("combined library did not preserve the KMC model indices")
    if reloaded_table.paths[: kmc_table.action_count] != kmc_table.paths:
        raise AssertionError("combined library changed a checkpoint-compatible KMC path")
    if len(reloaded_macros) != kmc_table.action_count + len(novel):
        raise AssertionError("combined action count is inconsistent")

    manifest = {
        "combined_action_count": combined_table.action_count,
        "combined_action_digest": combined_table.digest,
        "format_version": 1,
        "kmc_action_count": kmc_table.action_count,
        "kmc_action_digest": kmc_table.digest,
        "kmc_indices_preserved": True,
        "novel_action_count": len(novel),
        "novel_actions": [
            {
                "combined_index": kmc_table.action_count + offset,
                "path": list(macro.path),
                "walton_index": walton_index,
            }
            for offset, (walton_index, macro) in enumerate(novel)
        ],
        "source_libraries": {
            "kmc": str(args.kmc_library.relative_to(PROJECT)),
            "walton": str(args.walton_library.relative_to(PROJECT)),
        },
        "walton_action_count": walton_table.action_count,
        "walton_action_digest": walton_table.digest,
    }
    manifest_path = args.out_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
