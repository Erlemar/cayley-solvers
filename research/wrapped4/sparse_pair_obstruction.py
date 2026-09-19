"""Probe a possible obstruction to improving inversion-based interval bounds."""
import argparse
import json
from pathlib import Path
import time

from general_sort import apply_to
from wide_search import build_endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--offset', type=int, default=1)
    args = parser.parse_args()
    assert 0 < args.offset < args.stride
    rows = []
    output = Path(__file__).with_name(f'sparse_pair_s{args.stride}_o{args.offset}.json')
    for pairs in range(2, 6):
        for extra in (0, 4):
            n = max(9, args.stride * pairs + extra)
            p = list(range(n))
            for j in range(pairs):
                left = args.stride * j
                p[left], p[left + args.offset] = p[left + args.offset], p[left]
            radius = 5 if n <= 20 else 4
            game = build_endgame(n, radius, 5000000, False)
            for budget in (3 * pairs - 2, 3 * pairs, 3 * pairs + 2):
                row = solve(p, budget, game, radius, 1000000, seed=7, wrapped=False)
                row.update(pairs=pairs, extra=extra, radius=radius)
                if row['status'] == 'found':
                    assert apply_to(p, row['word']) == list(range(n))
                    assert all(0 <= i <= n - 4 for i, _ in row['word'])
                rows.append(row)
                output.write_text(json.dumps(rows, indent=2))
                print('pairs', pairs, 'n', n, 'budget', budget, row['status'],
                      'length', row.get('length'), 'nodes', row['nodes'], 'seconds', row['seconds'], flush=True)
                if row['status'] == 'found':
                    break


if __name__ == '__main__':
    main()
