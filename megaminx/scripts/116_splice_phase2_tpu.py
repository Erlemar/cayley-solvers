"""Splice local Phase-1 exports with TPU Phase-2 JSON results.

The Phase-2 TPU runner returns restricted-puzzle move indices. This script maps
those indices back to Phase-2 move names, concatenates them after the Phase-1
full-puzzle path, verifies against the original scramble, and optionally
compares/min-merges against an existing submission.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.verify import load_submission, load_test_states, verify_path, verify_submission
from megaminx.puzzle import Megaminx
from megaminx.two_phase import build_phase2_puzzle


def _load_json_rows(path: Path) -> dict[int, dict]:
    rows = json.load(open(path, encoding="utf-8"))
    return {int(r["pid"]): r for r in rows}


def _write_rows(path: Path, rows: list[tuple[int, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid, moves in rows:
            w.writerow([pid, ".".join(moves)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-json", type=Path, required=True)
    ap.add_argument("--phase2-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="partial CSV containing verified two-phase rows")
    ap.add_argument("--baseline-csv", type=Path, default=None,
                    help="optional submission/floor CSV to compare against")
    ap.add_argument("--merged-out", type=Path, default=None,
                    help="optional full CSV: baseline with shorter candidates substituted")
    args = ap.parse_args()

    full = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    phase2 = build_phase2_puzzle(full)
    states = load_test_states(PROJECT / "data" / "test.csv")

    phase1 = _load_json_rows(args.phase1_json)
    phase2_rows = _load_json_rows(args.phase2_json)

    verified: dict[int, list[str]] = {}
    rejected: list[tuple[int, str]] = []
    for pid in sorted(phase2_rows):
        r2 = phase2_rows[pid]
        if not r2.get("found"):
            rejected.append((pid, "phase2_not_found"))
            continue
        r1 = phase1.get(pid)
        if r1 is None or not r1.get("found"):
            rejected.append((pid, "missing_phase1"))
            continue
        try:
            suffix = [phase2.move_names[int(i)] for i in r2["path_idx"]]
        except Exception as exc:
            rejected.append((pid, f"bad_phase2_path_idx:{exc}"))
            continue
        path = list(r1["path1"]) + suffix
        res = verify_path(full, states[pid], path)
        if not res.ok:
            rejected.append((pid, f"verify_fail:{res.reason}"))
            continue
        verified[pid] = path

    _write_rows(args.out, [(pid, verified[pid]) for pid in sorted(verified)])
    print(f"verified two-phase rows: {len(verified)}/{len(phase2_rows)} -> {args.out}")
    if rejected:
        print(f"rejected rows: {rejected[:20]}")

    if args.baseline_csv is None:
        return 0 if not rejected else 1

    baseline = load_submission(args.baseline_csv)
    wins: list[tuple[int, int, int, int]] = []
    ties: list[tuple[int, int]] = []
    losses: list[tuple[int, int, int, int]] = []
    missing_baseline: list[int] = []
    for pid, path in sorted(verified.items()):
        base = baseline.get(pid)
        if base is None:
            missing_baseline.append(pid)
            continue
        cand_len = len(path)
        base_len = len(base)
        delta = cand_len - base_len
        if delta < 0:
            wins.append((pid, base_len, cand_len, delta))
        elif delta == 0:
            ties.append((pid, cand_len))
        else:
            losses.append((pid, base_len, cand_len, delta))

    print(f"baseline: {args.baseline_csv}")
    print(f"covered pids: {len(verified)}")
    print(f"wins/ties/losses: {len(wins)}/{len(ties)}/{len(losses)}")
    print(f"sum candidate covered: {sum(len(p) for p in verified.values())}")
    print(f"sum baseline covered:  {sum(len(baseline[p]) for p in verified if p in baseline)}")
    print(f"standalone delta on covered: "
          f"{sum(len(verified[p]) - len(baseline[p]) for p in verified if p in baseline):+d}")
    print(f"merge gain on covered: {sum(-d for _pid, _b, _c, d in wins)}")
    if wins:
        print("best wins:", wins[:20])
    if losses:
        print("worst losses:", sorted(losses, key=lambda x: x[3], reverse=True)[:20])
    if missing_baseline:
        print(f"missing baseline pids: {missing_baseline}")

    if args.merged_out is not None:
        rows: list[tuple[int, list[str]]] = []
        for pid in sorted(states):
            base = baseline[pid]
            cand = verified.get(pid)
            rows.append((pid, cand if cand is not None and len(cand) < len(base) else base))
        _write_rows(args.merged_out, rows)
        report = verify_submission(full, PROJECT / "data" / "test.csv", args.merged_out)
        print(f"merged out: {args.merged_out}")
        print(f"merged verify: {report.n_valid}/{report.n_total} total={report.total_moves}")
        return 0 if report.all_valid and not rejected else 1

    return 0 if not rejected else 1


if __name__ == "__main__":
    sys.exit(main())
