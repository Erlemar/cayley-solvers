"""Focused retries of full alternating perturbations of a half rotation."""
import argparse
import json
from pathlib import Path

import numpy as np

from affine_insertion import balanced_keys
from near_rotation_search import potential
from rotation_bounds import diameter_lower
from wide_search import build_endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--cap', type=int, default=3000000)
    args = parser.parse_args()
    n = args.n
    p = [(i + n // 2) % n for i in range(n)]
    for i in range(0, n, 2):
        p[i], p[i + 1] = p[i + 1], p[i]
    ell = int(potential(np.array(balanced_keys(p), np.int64), n))
    budget = diameter_lower(n)
    if (budget - ell) % 2:
        budget -= 1
    endgame = build_endgame(n, 4, 5000000)
    rows = []
    out = Path(__file__).with_name(f'wide_full_matching_n{n}_retries.json')
    for offset in (0, 2):
        for seed in (0, 7, 13, 29):
            row = solve(p, budget + offset, endgame, cap=args.cap, seed=seed)
            row['seed'] = seed
            rows.append(row)
            out.write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k not in ('word', 'permutation')}), flush=True)
            if row['status'] == 'found':
                return
            if row['status'] == 'exhausted':
                break


if __name__ == '__main__':
    main()
