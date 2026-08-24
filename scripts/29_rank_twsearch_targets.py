"""Rank expensive exact-search targets from replay-verified path-population evidence.

The model is deliberately trained on a matched proxy task.  For each pid already
proved optimal by twsearch, its real floor is a negative example and its observed
``floor + 2`` population (with the floor hidden) is a positive example.  Features
only see paths at or above the candidate length, so the positive rows simulate the
information we would have had immediately before discovering the two-move win.

This cannot prove or apply a saving.  It only changes the order in which the exact
solver spends time; every eventual hit is still replay-checked by the ladder and
harvester.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube  # noqa: E402
from cayley.verify import load_submission, load_test_states  # noqa: E402


def _load_prefix_module():
    path = PROJECT / "scripts/26_prefix_structure.py"
    spec = importlib.util.spec_from_file_location("prefix_structure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def axis(move: str) -> str:
    return move[1] if move.startswith("-") else move[0]


def lcp(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


FEATURE_NAMES = [
    "log_n_at",
    "log_n_longer",
    "fraction_at",
    "log_n_plus2",
    "log_n_plus4plus",
    "at_prefixes1",
    "at_prefixes2",
    "at_prefixes3",
    "longer_mass_prefix1",
    "longer_mass_prefix2",
    "longer_mass_prefix3",
    "longer_mass_suffix1",
    "longer_mass_suffix2",
    "longer_mass_suffix3",
    "mean_lcp_to_longer",
    "mean_lcs_to_longer",
    "same_axis_fraction",
    "inverse_adjacent_fraction",
    "move_entropy",
]


def features(candidates: dict[tuple[str, ...], set[str]], length: int) -> np.ndarray | None:
    at = [p for p in candidates if len(p) == length]
    longer = [p for p in candidates if len(p) > length]
    if not at:
        return None
    visible = at + longer
    n_at, n_long = len(at), len(longer)

    out: list[float] = [
        math.log1p(n_at),
        math.log1p(n_long),
        n_at / len(visible),
        math.log1p(sum(len(p) == length + 2 for p in longer)),
        math.log1p(sum(len(p) >= length + 4 for p in longer)),
    ]
    for k in (1, 2, 3):
        out.append(float(len({p[:k] for p in at})))

    for reverse in (False, True):
        for k in (1, 2, 3):
            ap = {(p[::-1] if reverse else p)[:k] for p in at}
            mass = _mean(
                1.0 if (p[::-1] if reverse else p)[:k] in ap else 0.0
                for p in longer
            )
            out.append(mass)

    if longer:
        out.append(_mean(max(lcp(a, b) for b in longer) for a in at))
        out.append(_mean(max(lcp(a[::-1], b[::-1]) for b in longer) for a in at))
    else:
        out.extend((0.0, 0.0))

    # Structural features of the candidate paths themselves.  These are cheap and
    # avoid depending only on how many near-duplicate CSVs happen to be present.
    denom = max(length - 1, 1)
    out.append(_mean(sum(axis(a) == axis(b) for a, b in zip(p, p[1:])) / denom for p in at))
    out.append(_mean(sum(a == (b[1:] if b.startswith("-") else "-" + b)
                              for a, b in zip(p, p[1:])) / denom for p in at))
    entropies = []
    for p in at:
        count = Counter(p)
        entropies.append(-sum((v / len(p)) * math.log(v / len(p) + 1e-30)
                              for v in count.values()))
    out.append(_mean(entropies))
    assert len(out) == len(FEATURE_NAMES), (len(out), len(FEATURE_NAMES))
    return np.asarray(out, dtype=np.float64)


def proven_pids(journal_roots: list[Path], baseline: dict[int, list[str]]) -> set[int]:
    proven: set[int] = set()
    for root in journal_roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*.jsonl")
        for path in paths:
            try:
                handle = path.open(encoding="utf-8")
            except OSError:
                continue
            with handle:
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    pid = rec.get("pid")
                    if pid not in baseline or rec.get("verdict") != "none" or rec.get("start") != 0:
                        continue
                    window = rec.get("window")
                    if window != baseline[pid]:
                        continue
                    length = len(baseline[pid])
                    if rec.get("window_len") != length or rec.get("max_depth") != length - 2:
                        continue
                    proven.add(pid)
    return proven


def fit_ridge_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 folds: int = 5, ridge: float = 2.0):
    pred = np.zeros(len(y), dtype=np.float64)
    for fold in range(folds):
        test = groups % folds == fold
        train = ~test
        mu = x[train].mean(0)
        sd = x[train].std(0)
        sd[sd < 1e-8] = 1.0
        z = (x[train] - mu) / sd
        design = np.column_stack((np.ones(train.sum()), z))
        reg = np.eye(design.shape[1]) * ridge
        reg[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + reg, design.T @ y[train])
        pred[test] = np.column_stack((np.ones(test.sum()), (x[test] - mu) / sd)) @ coef

    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(y)), (x - mu) / sd))
    reg = np.eye(design.shape[1]) * ridge
    reg[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return pred, mu, sd, coef


def auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(_mean((p > neg).mean() + 0.5 * (p == neg).mean() for p in pos))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--roots", nargs="*", default=["submissions"])
    ap.add_argument("--journals", nargs="*", default=[
        "data/ladder", "data/ladder_fleet", "data/ladder_fleet_current/journals"
    ])
    ap.add_argument("--path-length", type=int, default=22)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    prefix = _load_prefix_module()
    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    states = load_test_states(PROJECT / "data/test.csv")
    files = prefix.discover([PROJECT / p for p in args.roots])
    pop, _ = prefix.load_population(files, states, puzzle)
    baseline = load_submission(args.baseline)
    proven = proven_pids([PROJECT / p for p in args.journals], baseline)
    print(f"population: {sum(map(len, pop.values())):,} verified paths from {len(files)} CSVs")
    print(f"current full-path exact proofs usable as matched controls: {len(proven)}")

    rows_x, rows_y, rows_group = [], [], []
    paired = []
    for pid in sorted(proven):
        cand = pop.get(pid)
        if not cand:
            continue
        floor = len(baseline[pid])
        x0 = features(cand, floor)
        x1 = features(cand, floor + 2)
        if x0 is None or x1 is None:
            continue
        rows_x.extend((x0, x1))
        rows_y.extend((0.0, 1.0))
        rows_group.extend((pid, pid))
        paired.append(pid)
    if len(paired) < 20:
        raise SystemExit(f"only {len(paired)} matched controls; ranking would be too weak")

    x = np.vstack(rows_x)
    y = np.asarray(rows_y)
    groups = np.asarray(rows_group)
    cv, mu, sd, coef = fit_ridge_cv(x, y, groups)
    pair_acc = _mean(cv[2 * i + 1] > cv[2 * i] for i in range(len(paired)))
    print(f"5-fold matched CV: AUC {auc(y, cv):.3f}, paired accuracy {pair_acc:.3f} "
          f"on {len(paired)} pids")

    effects = sorted(zip(FEATURE_NAMES, coef[1:] / sd), key=lambda z: -abs(z[1]))
    print("largest fitted effects (positive => resembles a known reducible +2 incumbent):")
    for name, value in effects[:8]:
        print(f"  {name:26s} {value:+.4f}")

    target = []
    for pid, path in baseline.items():
        if len(path) != args.path_length or pid not in pop:
            continue
        feat = features(pop[pid], args.path_length)
        if feat is None:
            continue
        score = float(np.r_[1.0, (feat - mu) / sd] @ coef)
        target.append((score, pid, feat))
    target.sort(reverse=True)

    print(f"\ntop {min(40, len(target))} of {len(target)} length-{args.path_length} targets:")
    print("  rank   pid   proxy_score  n_at  n_longer  prefix1_mass")
    for rank, (score, pid, feat) in enumerate(target[:40], 1):
        print(f"  {rank:4d} {pid:5d} {score:11.4f} {math.expm1(feat[0]):5.0f} "
              f"{math.expm1(feat[1]):9.0f} {feat[8]:12.3f}")

    report = {
        "method": "matched proven-floor vs hidden-floor+2 ridge ranking",
        "path_length": args.path_length,
        "n_proven": len(proven),
        "n_pairs": len(paired),
        "cv_auc": auc(y, cv),
        "cv_pair_accuracy": pair_acc,
        "features": FEATURE_NAMES,
        "ranking": [{"rank": i + 1, "pid": pid, "score": score,
                     "features": {n: float(v) for n, v in zip(FEATURE_NAMES, feat)}}
                    for i, (score, pid, feat) in enumerate(target)],
    }
    out = args.out or PROJECT / "data" / f"twsearch_rank_L{args.path_length}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
