"""Apply replay-verified learned short-macro window improvements to a submission."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--action-dir", type=Path, required=True)
    parser.add_argument("--window-metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument(
        "--indices", default="", help="optional comma-separated query-index allowlist"
    )
    parser.add_argument("--minimum-saving", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-report", type=Path)
    return parser.parse_args()


def select_nonoverlapping(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Weighted interval scheduling, maximizing total primitive-move saving."""
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["after_prefix"]),
            int(item["before_prefix"]),
            int(item["index"]),
        ),
    )
    previous: list[int] = []
    for position, item in enumerate(ordered):
        before = int(item["before_prefix"])
        compatible = -1
        for candidate_position in range(position - 1, -1, -1):
            if int(ordered[candidate_position]["after_prefix"]) <= before:
                compatible = candidate_position
                break
        previous.append(compatible)
    best_saving = [0] * (len(ordered) + 1)
    choose = [False] * len(ordered)
    for position, item in enumerate(ordered, start=1):
        with_item = int(item["saving"]) + best_saving[previous[position - 1] + 1]
        without_item = best_saving[position - 1]
        if with_item > without_item:
            best_saving[position] = with_item
            choose[position - 1] = True
        else:
            best_saving[position] = without_item
    selected: list[dict[str, object]] = []
    position = len(ordered) - 1
    while position >= 0:
        with_item = int(ordered[position]["saving"]) + best_saving[previous[position] + 1]
        without_item = best_saving[position]
        if choose[position] and with_item > without_item:
            selected.append(ordered[position])
            position = previous[position]
        else:
            position -= 1
    return sorted(selected, key=lambda item: int(item["before_prefix"]))


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    test_states = {
        int(pid): state for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    metadata = json.loads(args.window_metadata.read_text(encoding="utf-8"))
    windows = {int(item["index"]): item for item in metadata["selected"]}
    allowed_indices = {
        int(token)
        for token in args.indices.replace("_", ",").split(",")
        if token.strip()
    }
    library = json.loads(
        (args.action_dir / "action_library.json").read_text(encoding="utf-8")
    )
    action_paths = [tuple(path) for path in library["paths"]]

    best_by_location: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for report_path in args.report:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report["rows"]:
            if not row.get("solved") or not row.get("replay_verified"):
                continue
            index = int(row["index"])
            if index not in windows:
                continue
            if allowed_indices and index not in allowed_indices:
                continue
            macro_actions = tuple(int(value) for value in row["path"])
            primitive_path = tuple(
                move for action in macro_actions for move in action_paths[action]
            )
            if len(primitive_path) != int(row["primitive_cost"]):
                raise AssertionError(f"window {index}: reported primitive cost disagrees")
            window = windows[index]
            occurrences = window.get("occurrences") or [window]
            for occurrence in occurrences:
                source_moves = int(occurrence["source_moves"])
                saving = source_moves - len(primitive_path)
                if saving < args.minimum_saving:
                    continue
                candidate = {
                    **occurrence,
                    "effect_sha256": window.get("effect_sha256"),
                    "index": index,
                    "macro_actions": list(macro_actions),
                    "primitive_path": list(primitive_path),
                    "replacement_moves": len(primitive_path),
                    "saving": saving,
                    "source_report": str(report_path),
                    "target_actions": window.get("target_actions"),
                    "target_cost": window.get("target_cost"),
                }
                location = (
                    int(candidate["pid"]),
                    int(candidate["row_position"]),
                    int(candidate["before_prefix"]),
                    int(candidate["after_prefix"]),
                )
                incumbent = best_by_location.get(location)
                if incumbent is None or len(primitive_path) < int(incumbent["replacement_moves"]):
                    best_by_location[location] = candidate

    with args.submission.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "initial_state_id" not in fieldnames or "path" not in fieldnames:
        raise ValueError("submission must contain initial_state_id and path columns")
    if len(rows) != len(test_states):
        raise ValueError(f"submission has {len(rows)} rows, expected {len(test_states)}")

    by_pid: dict[int, list[dict[str, object]]] = {}
    for candidate in best_by_location.values():
        by_pid.setdefault(int(candidate["pid"]), []).append(candidate)
    selected_by_pid = {
        pid: select_nonoverlapping(candidates) for pid, candidates in by_pid.items()
    }

    source_score = 0
    output_score = 0
    applied: list[dict[str, object]] = []
    for row_position, row in enumerate(rows):
        pid = int(row["initial_state_id"])
        if pid not in test_states:
            raise ValueError(f"unknown PID {pid}")
        source_path = tuple(token for token in row["path"].split(".") if token)
        if puzzle.apply_path(test_states[pid], source_path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: source path failed replay")
        source_score += len(source_path)
        output_path = list(source_path)
        selected = selected_by_pid.get(pid, [])
        for candidate in reversed(selected):
            if int(candidate["row_position"]) != row_position:
                raise AssertionError(f"window {candidate['index']}: row position disagrees")
            before = int(candidate["before_prefix"])
            after = int(candidate["after_prefix"])
            replacement = list(candidate["primitive_path"])
            output_path[before:after] = replacement
            applied.append({key: value for key, value in candidate.items() if key != "primitive_path"})
        if puzzle.apply_path(test_states[pid], output_path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: rewritten path failed replay")
        output_score += len(output_path)
        row["path"] = ".".join(output_path)

    expected_saving = sum(int(item["saving"]) for item in applied)
    if source_score - output_score != expected_saving:
        raise AssertionError("aggregate score saving disagrees with applied windows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "applied": sorted(applied, key=lambda item: (int(item["pid"]), int(item["before_prefix"]))),
        "applied_windows": len(applied),
        "output": str(args.out),
        "output_score": output_score,
        "replay_verified": len(rows),
        "saving": source_score - output_score,
        "source_score": source_score,
        "submission": str(args.submission),
    }
    report_path = args.out_report or args.out.with_suffix(".json")
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
