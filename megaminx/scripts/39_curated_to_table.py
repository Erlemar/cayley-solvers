"""Convert curated_macros_phase2.pkl → commutator_table-format for 27_path_sa.py.

`27_path_sa.py --commutator-table FILE` expects:
  cm["table"]: dict[perm_tuple] -> tuple[gen_idx, ...]

`curated_macros_phase2.pkl` has:
  out["macros"]: list of dicts with `perm` (tuple) and `word_idxs` (tuple).

Just remap into the expected schema.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=Path("megaminx/data/curated_macros_phase2.pkl"))
    ap.add_argument("--out", type=Path,
                    default=Path("megaminx/data/curated_table_phase2.pkl"))
    args = ap.parse_args()

    with open(args.in_path, "rb") as f:
        d = pickle.load(f)
    table = {m["perm"]: tuple(m["word_idxs"]) for m in d["macros"]}
    out = {
        "table": table,
        "metadata": {
            "source": str(args.in_path),
            "n_macros": len(table),
            "format": "commutator-table-compatible",
            "from": "T1.1 Phase 2 curated speedcubing macros",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {args.out} with {len(table)} macros")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
