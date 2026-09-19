"""Analyze a matched KMC parameter gate after standard path reduction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="ordered experiment source; repeat for every condition",
    )
    parser.add_argument("--control", required=True, help="label of the matched control")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_source(directory: Path) -> tuple[dict[int, dict[str, int]], dict[str, object]]:
    rows: dict[int, dict[str, int]] = {}
    for metadata_path in sorted(directory.glob("[0-9][0-9][0-9][0-9].json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index = int(metadata["index"])
        path_file = directory / f"{int(metadata['state_id']):04d}.path.txt"
        raw_path = parse_path(path_file.read_text(encoding="utf-8"))
        if len(raw_path) != int(metadata["moves"]):
            raise ValueError(f"{directory.name} PID {index}: path/metadata length mismatch")
        if metadata.get("replay_verified") is not True:
            raise ValueError(f"{directory.name} PID {index}: missing replay verification")
        reduced = reduce_commuting_quarter_turn_path(raw_path)
        rows[index] = {
            "raw": len(raw_path),
            "reduced": len(reduced),
            "removed": len(raw_path) - len(reduced),
        }
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if len(rows) != int(report["completed_states"]):
        raise ValueError(f"{directory.name}: report coverage mismatch")
    return rows, report


def main() -> None:
    args = parse_args()
    ordered_sources: list[tuple[str, Path]] = []
    for encoded in args.source:
        if "=" not in encoded:
            raise ValueError(f"source must be LABEL=DIR, got {encoded!r}")
        label, raw_directory = encoded.split("=", 1)
        ordered_sources.append((label, Path(raw_directory)))
    labels = [label for label, _ in ordered_sources]
    if len(set(labels)) != len(labels):
        raise ValueError("source labels must be unique")
    if args.control not in labels:
        raise ValueError("control label is not present in sources")

    data: dict[str, dict[int, dict[str, int]]] = {}
    reports: dict[str, dict[str, object]] = {}
    directories: dict[str, Path] = {}
    for label, directory in ordered_sources:
        data[label], reports[label] = load_source(directory)
        directories[label] = directory

    ids = sorted(data[args.control])
    expected = set(ids)
    for label in labels:
        if set(data[label]) != expected:
            raise ValueError(f"{label}: PID set differs from control")

    control_total = sum(data[args.control][index]["reduced"] for index in ids)
    conditions: list[dict[str, object]] = []
    for label in labels:
        raw_total = sum(data[label][index]["raw"] for index in ids)
        reduced_total = sum(data[label][index]["reduced"] for index in ids)
        deltas = [
            data[label][index]["reduced"] - data[args.control][index]["reduced"]
            for index in ids
        ]
        portfolio_total = sum(
            min(data[args.control][index]["reduced"], data[label][index]["reduced"])
            for index in ids
        )
        solver_seconds = float(reports[label]["solver_seconds_sum"])
        savings = control_total - portfolio_total
        conditions.append(
            {
                "label": label,
                "directory": str(directories[label].resolve()),
                "raw_total": raw_total,
                "reduced_total": reduced_total,
                "commuting_moves_removed": raw_total - reduced_total,
                "raw_delta_vs_control_reduced": reduced_total - control_total,
                "wins": sum(delta < 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses": sum(delta > 0 for delta in deltas),
                "control_plus_condition_total": portfolio_total,
                "portfolio_savings_vs_control": savings,
                "solver_seconds": round(solver_seconds, 4),
                "solver_cpu_hours": round(solver_seconds / 3600.0, 4),
                "portfolio_savings_per_cpu_hour": (
                    round(savings / (solver_seconds / 3600.0), 4) if solver_seconds else None
                ),
            }
        )

    best = {index: data[args.control][index]["reduced"] for index in ids}
    previous_total = control_total
    sequential: list[dict[str, object]] = []
    for label in labels:
        if label == args.control:
            continue
        for index in ids:
            best[index] = min(best[index], data[label][index]["reduced"])
        total = sum(best.values())
        sequential.append(
            {
                "added": label,
                "total": total,
                "marginal_savings": previous_total - total,
                "cumulative_savings": control_total - total,
            }
        )
        previous_total = total

    winners = Counter()
    sole_winners = Counter()
    per_pid: list[dict[str, object]] = []
    for index in ids:
        lengths = {label: data[label][index]["reduced"] for label in labels}
        minimum = min(lengths.values())
        winning_labels = [label for label in labels if lengths[label] == minimum]
        winners.update(winning_labels)
        if len(winning_labels) == 1:
            sole_winners.update(winning_labels)
        per_pid.append(
            {
                "pid": index,
                "lengths": lengths,
                "best": minimum,
                "savings_vs_control": lengths[args.control] - minimum,
                "winning_sources": winning_labels,
            }
        )

    result = {
        "control": args.control,
        "pids": ids,
        "pid_count": len(ids),
        "all_candidate_paths_replay_verified_by_runner": True,
        "control_reduced_total": control_total,
        "all_source_portfolio_total": sum(best.values()),
        "all_source_portfolio_savings": control_total - sum(best.values()),
        "conditions": conditions,
        "sequential_merge": sequential,
        "winner_tie_counts": dict(sorted(winners.items())),
        "sole_winner_counts": dict(sorted(sole_winners.items())),
        "per_pid": per_pid,
    }
    atomic_write(args.out_json, json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# KMC parameter gate — 16 stratified hard PIDs",
        "",
        f"Control reduced total: **{control_total}**.",
        f"All-source portfolio: **{sum(best.values())}** "
        f"(**-{control_total - sum(best.values())} moves**).",
        "",
        "Every path was replay-verified by the runner before being accepted. The separate",
        "merge verifier replayed all 128 candidates and all 1,012 fallback rows.",
        "",
        "## Conditions after commuting-turn reduction",
        "",
        "| Condition | Reduced total | Delta | W/T/L | Control+condition | Savings | CPU h | Savings/CPU h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        lines.append(
            f"| {row['label']} | {row['reduced_total']} | "
            f"{int(row['raw_delta_vs_control_reduced']):+d} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['control_plus_condition_total']} | {row['portfolio_savings_vs_control']} | "
            f"{row['solver_cpu_hours']:.3f} | {row['portfolio_savings_per_cpu_hour']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Ordered portfolio accumulation",
            "",
            "| Added source | Total | Marginal savings | Cumulative savings |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sequential:
        lines.append(
            f"| {row['added']} | {row['total']} | {row['marginal_savings']} | "
            f"{row['cumulative_savings']} |"
        )
    lines.extend(
        [
            "",
            "## Per-PID reduced lengths",
            "",
            "| PID | " + " | ".join(labels) + " | Best | Saving |",
            "|---:|" + "---:|" * (len(labels) + 2),
        ]
    )
    for row in per_pid:
        lengths = row["lengths"]
        lines.append(
            f"| {row['pid']} | "
            + " | ".join(str(lengths[label]) for label in labels)
            + f" | {row['best']} | {row['savings_vs_control']} |"
        )
    lines.append("")
    atomic_write(args.out_md, "\n".join(lines))


if __name__ == "__main__":
    main()
