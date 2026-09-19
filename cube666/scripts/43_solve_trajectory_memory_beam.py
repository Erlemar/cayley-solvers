"""Solve cube666 puzzles by beam search over a saved trajectory-memory model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--branch-width", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--compare-submission", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_submission(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"initial_state_id", "path"}:
        raise ValueError(f"{path}: expected initial_state_id,path")
    return {
        int(row["initial_state_id"]): tuple(
            token for token in row["path"].split(".") if token
        )
        for row in rows
    }


def main() -> None:
    args = parse_args()
    if args.beam_width <= 0 or args.branch_width <= 0 or args.max_steps <= 0:
        raise ValueError("beam dimensions must be positive")
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    initial_states = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    if manifest.get("kind") != "cube666_trajectory_memory_value_model_v1":
        raise ValueError("unsupported trajectory-memory model")
    model_path = args.model_dir / "model.npz"
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ValueError("trajectory-memory model hash does not match manifest")
    payload = np.load(model_path, allow_pickle=False)
    states = payload["states"]
    pids = payload["pids"]
    distances = payload["distances"]
    action_masks = payload["action_masks"]
    offsets = payload["offsets"]
    move_names = tuple(str(move) for move in manifest["move_names"])
    if move_names != tuple(puzzle.move_names):
        raise ValueError("model move vocabulary differs from puzzle generators")
    if len(offsets) != len(initial_states) + 1:
        raise ValueError("model does not have exact PID coverage")
    comparison = load_submission(args.compare_submission) if args.compare_submission else {}

    output_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, int | bool]] = []
    total_moves = 0
    comparison_total = 0
    replay_verified = 0
    for pid in sorted(initial_states):
        start = int(offsets[pid])
        stop = int(offsets[pid + 1])
        if stop <= start or not np.all(pids[start:stop] == pid):
            raise ValueError(f"PID {pid}: corrupt model offset")
        local_states = states[start:stop]
        local_distances = distances[start:stop]
        local_masks = action_masks[start:stop]
        state_index = {
            state.tobytes(): index for index, state in enumerate(local_states)
        }
        initial = initial_states[pid]
        # A* beam rank is exact path cost plus the learned memory value.  Every
        # stored action is exact-replayed; beam width controls tie exploration.
        beam: list[tuple[tuple[int, ...], tuple[int, ...]]] = [(initial, ())]
        expanded = 0
        generated = 0
        solution: tuple[int, ...] | None = None
        for _ in range(args.max_steps + 1):
            solved = [path for state, path in beam if state == puzzle.solved_state]
            if solved:
                solution = min(solved, key=lambda path: (len(path), path))
                break
            candidates: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
            for state, path in beam:
                position = state_index.get(np.asarray(state, dtype=np.uint8).tobytes())
                if position is None:
                    continue
                mask = int(local_masks[position])
                actions = [action for action in range(len(move_names)) if mask >> action & 1]
                actions = actions[: args.branch_width]
                expanded += 1
                for action in actions:
                    next_state = puzzle.apply_move(state, move_names[action])
                    next_position = state_index.get(
                        np.asarray(next_state, dtype=np.uint8).tobytes()
                    )
                    if next_position is None:
                        raise AssertionError(f"PID {pid}: model action left its state graph")
                    next_path = path + (action,)
                    rank = len(next_path) + int(local_distances[next_position])
                    candidates.append((rank, int(local_distances[next_position]), next_state, next_path))
                    generated += 1
            if not candidates:
                break
            candidates.sort(key=lambda item: (item[0], item[1], item[3]))
            next_beam: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
            seen: set[tuple[int, ...]] = set()
            for _, _, state, path in candidates:
                if state in seen:
                    continue
                seen.add(state)
                next_beam.append((state, path))
                if len(next_beam) == args.beam_width:
                    break
            beam = next_beam
        if solution is None:
            raise RuntimeError(f"PID {pid}: beam failed to solve within model graph")
        path = tuple(move_names[action] for action in solution)
        if puzzle.apply_path(initial, path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: model path failed exact replay")
        replay_verified += 1
        total_moves += len(path)
        comparison_moves = len(comparison[pid]) if comparison else len(path)
        comparison_total += comparison_moves
        report_rows.append(
            {
                "expanded_states": expanded,
                "generated_states": generated,
                "moves": len(path),
                "pid": pid,
                "replay_verified": True,
                "saving_vs_comparison": comparison_moves - len(path),
            }
        )
        output_rows.append(
            {"initial_state_id": str(pid), "path": ".".join(path)}
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.out)
    report = {
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "comparison": str(args.compare_submission) if args.compare_submission else None,
        "comparison_total": comparison_total,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "model": str(args.model_dir),
        "model_sha256": manifest["model_sha256"],
        "output": str(args.out),
        "output_total": total_moves,
        "replay_verified": replay_verified,
        "rows": report_rows,
        "saving_vs_comparison": comparison_total - total_moves,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = args.report_out.with_suffix(args.report_out.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_temporary.replace(args.report_out)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
