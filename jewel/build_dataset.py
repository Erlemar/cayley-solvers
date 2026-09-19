from __future__ import annotations

import argparse
import json

from .data import build_mixed_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--out", default="jewel/artifacts/train_mixed_v3_official.npz")
    parser.add_argument("--exact-balanced", type=int, default=400_000)
    parser.add_argument("--exact-uniform", type=int, default=300_000)
    parser.add_argument("--demo", type=int, default=300_000)
    parser.add_argument("--demo-weight", type=float, default=0.25)
    parser.add_argument("--public-baseline")
    parser.add_argument("--public-test", default="jewel/data/test.csv")
    parser.add_argument("--public-puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--public-count", type=int, default=0)
    parser.add_argument("--public-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    meta = build_mixed_dataset(
        args.ball,
        args.out,
        n_exact_balanced=args.exact_balanced,
        n_exact_uniform=args.exact_uniform,
        n_demo=args.demo,
        demo_weight=args.demo_weight,
        public_baseline=args.public_baseline,
        public_test=args.public_test,
        public_puzzle_info=args.public_puzzle_info,
        n_public=args.public_count,
        public_weight=args.public_weight,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
