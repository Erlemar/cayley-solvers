"""Fix the 54-pid IHES checkpoint gate (IHES_TRANSFORMER_PLAN_2026-09-16.md section 3.2 item 6).

Deterministic selection from the 21,870 floor, by incumbent length:
    all 7 L24, 20 L23 (evenly spaced), 10 unproven L22, 10 proven-optimal L22,
    7 proven-optimal L21.
On the 17 proven pids the floor IS the optimum, so excess over floor is exact there.

Proven L22 pids come from the recorded campaigns (EXPERIMENTS.md 2026-08-05: 26, 28, 868,
268, 300; 2026-09-08..11 headings: 32 pids). Proven L21 pids are read from the ladder
journals (`data/ladder*/*.jsonl`, full-path records with verdict "none").

    python scripts/81_build_ihes_gate.py --out data/ihes_gate54.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

PROVEN_L22 = sorted({26, 28, 868, 268, 300,
                     772, 874, 170, 644, 614, 996, 910, 46, 558, 358, 604, 730, 272, 952,
                     548, 510, 638, 916, 244, 642, 690, 684, 816, 534, 336, 634, 850, 554,
                     40, 94, 918, 296})


def spaced(items: list[int], n: int) -> list[int]:
    """n evenly spaced members of a sorted list (deterministic)."""
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the fixed 54-pid IHES gate set.")
    ap.add_argument("--floor", type=Path,
                    default=PROJECT / "submissions" / "ihes_20260912_1410_verified.csv")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "ihes_gate54.json")
    ap.add_argument("--extend-to-108", type=Path, default=None,
                    help="also write a 108-pid selection gate: the 54 above plus 54 more "
                         "(27 L23, 10 unproven L22, 10 proven L22, 7 proven L21), disjoint")
    args = ap.parse_args()

    with open(args.floor, encoding="utf-8", newline="") as f:
        L = {int(r["initial_state_id"]): len(r["path"].split(".")) if r["path"] else 0
             for r in csv.DictReader(f)}
    assert len(L) == 1003, f"floor has {len(L)} rows"
    assert all(L[p] == 22 for p in PROVEN_L22), "a recorded proven-L22 pid is not length 22"

    proven_l21 = set()
    for fn in glob.glob(str(PROJECT / "data" / "ladder*" / "*.jsonl")):
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = r.get("pid")
                if (pid is not None and r.get("verdict") == "none" and r.get("start", 0) == 0
                        and r.get("window_len") == L.get(pid) == 21):
                    proven_l21.add(int(pid))

    by_len = {n: sorted(p for p, v in L.items() if v == n) for n in (21, 22, 23, 24)}
    bands = {
        "L24": by_len[24],
        "L23": spaced(by_len[23], 20),
        "L22_unproven": spaced([p for p in by_len[22] if p not in PROVEN_L22], 10),
        "L22_proven": spaced(PROVEN_L22, 10),
        "L21_proven": spaced(sorted(proven_l21), 7),
    }
    pids = sorted(p for v in bands.values() for p in v)
    assert len(pids) == len(set(pids)) == 54, f"expected 54 distinct pids, got {len(pids)}"
    out = {
        "floor_file": str(args.floor.relative_to(PROJECT)).replace("\\", "/"),
        "floor_total": sum(L[p] for p in pids),
        "bands": bands,
        "pids": pids,
        "floor": {str(p): L[p] for p in pids},
        "n_proven_l21_available": len(proven_l21),
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(pids)} pids, floor total {out['floor_total']}; "
          + ", ".join(f"{k}={len(v)}" for k, v in bands.items()))

    if args.extend_to_108:
        used = set(pids)
        rest = lambda xs: [p for p in xs if p not in used]  # noqa: E731
        extra_bands = {
            "L23_b": spaced(rest(by_len[23]), 27),
            "L22_unproven_b": spaced(rest([p for p in by_len[22] if p not in PROVEN_L22]), 10),
            "L22_proven_b": spaced(rest(PROVEN_L22), 10),
            "L21_proven_b": spaced(rest(sorted(proven_l21)), 7),
        }
        extra = sorted(p for v in extra_bands.values() for p in v)
        assert len(extra) == len(set(extra)) == 54 and not (set(extra) & used), "extra set"
        all_pids = sorted(pids + extra)
        out108 = {
            "floor_file": out["floor_file"],
            "floor_total": sum(L[p] for p in all_pids),
            "bands": {**bands, **extra_bands},
            "pids": all_pids,
            "extra_pids": extra,
            "floor": {str(p): L[p] for p in all_pids},
        }
        args.extend_to_108.write_text(json.dumps(out108, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.extend_to_108}: {len(all_pids)} pids, floor total "
              f"{out108['floor_total']}; extra "
              + ", ".join(f"{k}={len(v)}" for k, v in extra_bands.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
