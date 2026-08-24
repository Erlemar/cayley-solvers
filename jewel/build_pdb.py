from __future__ import annotations

import argparse
import json

from .pdb import build_edge_pdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--pieces", required=True, help="comma-separated edge ids")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pieces = [int(x) for x in args.pieces.split(",")]
    print(json.dumps(build_edge_pdb(args.out, pieces, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
