"""Search finite bases for a block exchange with one crossed transposition."""
from collections import Counter
import json
from pathlib import Path

from near_rotation_search import build_endgame, solve
from rotation_bounds import parity_rounded_third


def target(a, b, x, y):
    p = list(range(a, a + b)) + list(range(a))
    p[y], p[b + x] = p[b + x], p[y]
    return p


def main():
    rows = []
    endgames = {}
    counts = Counter()
    out = Path(__file__).with_name('twisted_exchange_bases.json')
    for a in (6, 7, 8):
        for b in (6, 7, 8):
            n = a + b
            if n not in endgames:
                endgames[n] = build_endgame(n, 4, 2000000, False)
            local = Counter()
            for x in range(a):
                for y in range(b):
                    p = target(a, b, x, y)
                    budget = parity_rounded_third(a * b - 1)
                    for seed in (19, 7, 0, 13):
                        row = solve(p, budget, endgames[n], 4, 200000, seed=seed, wrapped=False)
                        if row['status'] != 'node_limit':
                            break
                    row.update(a=a, b=b, x=x, y=y)
                    assert row['adjacent_length'] == a * b - 1
                    rows.append(row)
                    counts[row['status']] += 1
                    local[row['status']] += 1
                    if row['status'] != 'found':
                        print(a, b, x, y, row['status'], flush=True)
            out.write_text(json.dumps({'counts': dict(counts), 'rows': rows}, indent=2))
            print('BASE', a, b, dict(local), flush=True)
    print('SUMMARY', dict(counts), flush=True)


if __name__ == '__main__':
    main()
