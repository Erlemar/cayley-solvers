"""N-way per-pid min over EVERY IHES picture-cube result source, replay-verified.

Rule 26 / 26b in one command, ported from `tetraminx/scripts/90_merge_all.py`.

Discovery is by CONTENT, not by name or location: any CSV whose header carries
`initial_state_id` + `path` is opened, and every row is replayed against the
original `data/test.csv` state before it is allowed to lower the floor. A file
belonging to a different puzzle therefore self-rejects (its move alphabet is not
in the picture cube's generator dict), so scanning unrelated folders is safe.

    python scripts/23_merge_all_ihes.py                    # repo only, report
    python scripts/23_merge_all_ihes.py --roots <dir> ...  # widen the sweep
    python scripts/23_merge_all_ihes.py --write --out submissions/foo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

csv.field_size_limit(10_000_000)

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages",
    ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints",
}


def iter_files(roots: list[Path], max_mb: float) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    limit = int(max_mb * 1024 * 1024)
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            out.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".csv", ".json"):
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            try:
                if path.stat().st_size > limit:
                    continue
                key = str(path.resolve()).lower()
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def looks_like_submission(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.readline(4096)
    except OSError:
        return False
    return "initial_state_id" in head and "path" in head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--roots", type=Path, nargs="*", default=[PROJECT])
    ap.add_argument("--max-mb", type=float, default=64.0)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="attribute the merge against this CSV")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"])
    gi = {n: i for i, n in enumerate(names)}
    gens = np.array([info["generators"][n] for n in names], dtype=np.int64)
    solved = np.array(info.get("central_state") or info.get("solved_state"),
                      dtype=np.int64)

    st0: dict[int, np.ndarray] = {}
    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            st0[int(row["initial_state_id"])] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int64)

    def valid(pid: int, path: str) -> bool:
        state = st0[pid]
        for move in path.split("."):
            idx = gi.get(move)
            if idx is None:
                return False
            state = state[gens[idx]]
        return bool(np.array_equal(state, solved))

    best: dict[int, str] = {}
    src: dict[int, str] = {}
    contributors: dict[str, int] = {}

    def offer(pid: int, path: str, where: str) -> None:
        if not path or pid not in st0:
            return
        length = path.count(".") + 1
        if pid in best and length >= best[pid].count(".") + 1:
            return
        if valid(pid, path):
            best[pid], src[pid] = path, where

    files = iter_files([r.resolve() for r in args.roots], args.max_mb)
    print(f"candidate files: {len(files):,}")
    n_used = 0
    for path in files:
        if not looks_like_submission(path):
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            print(f"  [skip] {path}: {exc}")
            continue
        if not rows or "initial_state_id" not in rows[0] or "path" not in rows[0]:
            continue
        before = sum(1 for pid in best)
        prev = dict(best)
        tag = str(path)
        for row in rows:
            try:
                pid = int(row["initial_state_id"])
            except (TypeError, ValueError):
                continue
            offer(pid, (row.get("path") or "").strip(), tag)
        gained = sum(1 for pid, p in best.items()
                     if prev.get(pid) != p)
        n_used += 1
        if gained:
            contributors[tag] = gained
        if args.verbose or gained:
            print(f"  [{n_used:4d}] {gained:5d} new-best  {path}")
        del before

    total = sum(p.count(".") + 1 for p in best.values())
    print(f"\nsubmission-shaped files read : {n_used:,}")
    print(f"pids covered                 : {len(best)} / {len(st0)}")
    print(f"MERGED TOTAL                 : {total:,}")
    missing = sorted(set(st0) - set(best))
    if missing:
        print(f"MISSING {len(missing)} pids -- NOT submittable: {missing[:10]}")

    if args.baseline is not None:
        base: dict[int, str] = {}
        with open(args.baseline, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                base[int(row["initial_state_id"])] = (row.get("path") or "").strip()
        base_total = sum(p.count(".") + 1 for p in base.values() if p)
        wins = [(pid, base[pid].count(".") + 1, best[pid].count(".") + 1)
                for pid in sorted(best)
                if pid in base and base[pid]
                and best[pid].count(".") < base[pid].count(".")]
        print(f"\nbaseline {args.baseline.name}: {base_total:,}")
        print(f"delta vs baseline           : {total - base_total:+,} over {len(wins)} pids")
        for pid, old, new in wins[:60]:
            print(f"  pid {pid}: {old} -> {new}   [{Path(src[pid]).name}]")

    if contributors:
        print("\ncontributing files (pids where this file held the strict best):")
        for tag, count in sorted(contributors.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  {count:5d}  {tag}")

    if args.write:
        out = args.out or (PROJECT / "submissions" / f"ihes_merged_{total}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["initial_state_id", "path"])
            for pid in sorted(best):
                writer.writerow([pid, best[pid]])
        # Re-read and re-verify what actually landed on disk.
        recheck = 0
        with out.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if valid(int(row["initial_state_id"]), row["path"]):
                    recheck += 1
        print(f"\nwrote {out}  ({recheck}/{len(best)} re-verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
