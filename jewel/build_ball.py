from __future__ import annotations

import argparse
import json

from .ball import build_exact_ball


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_exact_ball(args.out, args.depth, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
