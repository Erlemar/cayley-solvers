"""Materialize and replay-check a frozen online ranker versus the fixed proxy."""

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
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--proxy-ranked", type=Path, required=True)
    parser.add_argument("--proxy-disagreements", type=Path, required=True)
    parser.add_argument(
        "--completion-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_completions(
    directories: list[Path],
) -> dict[tuple[int, str], dict[str, object]]:
    completions: dict[tuple[int, str], dict[str, object]] = {}
    for directory in directories:
        for metadata_file in sorted(directory.glob("*.json")):
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            pid = int(metadata["pid"])
            digest = str(metadata["rough_path_digest"])
            path_file = directory / f"{pid:04d}.path.txt"
            if not path_file.exists():
                raise FileNotFoundError(path_file)
            path = parse_path(path_file.read_text(encoding="utf-8"))
            if len(path) != int(metadata["moves"]):
                raise AssertionError(
                    f"PID {pid} {digest[:8]}: metadata/path length mismatch"
                )
            key = (pid, digest)
            candidate = {
                "condition": str(metadata["condition"]),
                "directory": str(directory),
                "moves": len(path),
                "path": path,
            }
            existing = completions.get(key)
            if existing is not None and existing["path"] != path:
                raise AssertionError(f"PID {pid} {digest[:8]}: conflicting completions")
            completions[key] = candidate
    return completions


def main() -> None:
    args = parse_args()
    ranked = json.loads(args.ranked.read_text(encoding="utf-8"))
    proxy_ranked = json.loads(args.proxy_ranked.read_text(encoding="utf-8"))
    proxy_disagreements = json.loads(
        args.proxy_disagreements.read_text(encoding="utf-8")
    )
    completions = load_completions(args.completion_dir)

    proxy_alternatives = {
        int(row["pid"]): row for row in proxy_disagreements["selections"]
    }
    proxy_keys: dict[int, tuple[int, str]] = {}
    proxy_conditions: dict[int, str] = {}
    for row in proxy_ranked["selections"]:
        pid = int(row["pid"])
        source = proxy_alternatives.get(pid, row)
        proxy_keys[pid] = (pid, str(source["rough_path_digest"]))
        proxy_conditions[pid] = str(source["condition"])

    model_rows = {int(row["pid"]): row for row in ranked["selections"]}
    if set(model_rows) != set(proxy_keys):
        raise AssertionError("model and proxy PID sets differ")

    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    states = {
        int(pid): state
        for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    materialized: dict[int, tuple[str, ...]] = {}
    comparisons = []
    missing = []
    for pid in sorted(model_rows):
        model_row = model_rows[pid]
        model_key = (pid, str(model_row["rough_path_digest"]))
        proxy_key = proxy_keys[pid]
        if model_key not in completions:
            missing.append({"pid": pid, "role": "model", "digest": model_key[1]})
            continue
        if proxy_key not in completions:
            missing.append({"pid": pid, "role": "proxy", "digest": proxy_key[1]})
            continue
        model = completions[model_key]
        proxy = completions[proxy_key]
        model_path = tuple(model["path"])
        proxy_path = tuple(proxy["path"])
        if puzzle.apply_path(states[pid], model_path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: model completion failed replay")
        if puzzle.apply_path(states[pid], proxy_path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: proxy completion failed replay")
        materialized[pid] = model_path
        comparisons.append(
            {
                "model_condition": str(model_row["condition"]),
                "model_moves": len(model_path),
                "model_minus_proxy": len(model_path) - len(proxy_path),
                "pid": pid,
                "proxy_condition": proxy_conditions[pid],
                "proxy_moves": len(proxy_path),
                "same_rough_path": model_key == proxy_key,
            }
        )
    if missing:
        raise ValueError(f"missing exact completions: {missing}")

    args.candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.candidate_csv.with_suffix(args.candidate_csv.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        for pid, path in sorted(materialized.items()):
            writer.writerow({"initial_state_id": pid, "path": ".".join(path)})
    temporary.replace(args.candidate_csv)

    model_total = sum(int(row["model_moves"]) for row in comparisons)
    proxy_total = sum(int(row["proxy_moves"]) for row in comparisons)
    report = {
        "candidate_csv": str(args.candidate_csv),
        "comparisons": comparisons,
        "model_better_pids": sum(
            int(row["model_minus_proxy"]) < 0 for row in comparisons
        ),
        "model_minus_proxy": model_total - proxy_total,
        "model_proxy_disagreements": sum(
            not bool(row["same_rough_path"]) for row in comparisons
        ),
        "model_total": model_total,
        "pids": len(comparisons),
        "proxy_better_pids": sum(
            int(row["model_minus_proxy"]) > 0 for row in comparisons
        ),
        "proxy_total": proxy_total,
        "replay_verified_model_paths": len(comparisons),
        "replay_verified_proxy_paths": len(comparisons),
        "ties": sum(int(row["model_minus_proxy"]) == 0 for row in comparisons),
    }
    atomic_write(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
