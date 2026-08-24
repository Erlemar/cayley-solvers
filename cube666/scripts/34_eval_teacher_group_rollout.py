"""Roll out a macro model from held-out teacher-group boundary states."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import MacroTeacherDataset, load_macro_action_library  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument(
        "--state-teacher-dir",
        type=Path,
        help="optional different teacher providing rollout states and PID groups",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pids", default="200,210,220,230,240")
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--branch-width", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--policy-nll-weight", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    pids = tuple(int(value) for value in args.pids.split(",") if value.strip())
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.teacher_dir / "action_library.json", puzzle.generators, decomposition
    )
    state_teacher_dir = args.state_teacher_dir or args.teacher_dir
    teacher = MacroTeacherDataset.load(state_teacher_dir / "teacher.npz")
    groups = np.load(state_teacher_dir / "source_state_ids.npy", allow_pickle=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest:
        raise ValueError("checkpoint and action library digests differ")
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows: list[dict[str, object]] = []
    for pid in pids:
        candidates = np.flatnonzero(groups == pid)
        if not len(candidates):
            rows.append({"pid": pid, "available": False, "solved": False})
            continue
        position = int(
            candidates[np.argmax(teacher.search_value_targets[candidates])]
        )
        case_started = time.perf_counter()
        beam = learned_macro_beam_search(
            teacher.states[position],
            model,
            table,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
            policy_nll_weight=args.policy_nll_weight,
            exact_cost_weight=None,
            model_batch_size=2048,
            fallback_exact_cost=True,
        )
        row = {
            "available": True,
            "elapsed_seconds": round(time.perf_counter() - case_started, 4),
            "final_residual": beam.final_exact_cost,
            "macro_steps": len(beam.actions),
            "pid": pid,
            "primitive_moves": sum(len(table.paths[action]) for action in beam.actions),
            "solved": beam.solved,
            "teacher_search_value": float(teacher.search_value_targets[position]),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    available = [row for row in rows if row.get("available")]
    report = {
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "mean_final_residual": statistics.fmean(
            int(row["final_residual"]) for row in available
        ),
        "policy_nll_weight": args.policy_nll_weight,
        "rows": rows,
        "solved": sum(bool(row["solved"]) for row in available),
        "states": len(available),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
