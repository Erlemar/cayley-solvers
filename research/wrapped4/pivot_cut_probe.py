"""Test direct one-token insertion followed by a linear cut sort."""
from itertools import permutations
import json
from pathlib import Path
import random

import numpy as np
from numba import njit


@njit(cache=True)
def score(p):
    n = len(p)
    best = n * n
    best_x = best_direction = 0
    total = 0
    for x in range(n):
        start = 0
        for i in range(n):
            if p[i] == x:
                start = i
                break
        for direction in (-1, 1):
            travel = ((x - start) if direction > 0 else (start - x)) % n
            q = p.copy()
            pos = start
            for step in range(travel):
                other = (pos + direction) % n
                q[pos], q[other] = q[other], q[pos]
                pos = other
            inv = 0
            for i in range(n - 1):
                first = (q[(x + 1 + i) % n] - x - 1) % n
                for j in range(i + 1, n - 1):
                    second = (q[(x + 1 + j) % n] - x - 1) % n
                    inv += first > second
            value = travel + inv
            total += value
            if value < best:
                best, best_x, best_direction = value, x, direction
    return best, best_x, best_direction, total


def main():
    rows = []
    for n in range(3, 9):
        maximum = -1
        maximum_average_numerator = -1
        witness = None
        for p in permutations(range(n)):
            value, x, direction, total = score(np.array(p, np.int64))
            if value > maximum:
                maximum, witness = int(value), p
            maximum_average_numerator = max(maximum_average_numerator, int(total))
        row = {'n': n, 'max_minimum_pivot_cost': maximum, 'K': n * n // 4,
               'witness': witness, 'maximum_average_cost': maximum_average_numerator / (2 * n),
               'method': 'all permutations'}
        rows.append(row)
        print(json.dumps(row), flush=True)
    rng = random.Random(718)
    for n in (12, 16, 24, 36, 48, 72, 100):
        candidates = [list(range(n))[::-1], [(i + n // 2) % n for i in range(n)],
                      [(i + n // 2 + (-1 if i % 2 else 1)) % n for i in range(n)]]
        for _ in range(100):
            p = list(range(n))
            rng.shuffle(p)
            candidates.append(p)
        best = None
        for p in candidates:
            value, x, direction, total = score(np.array(p, np.int64))
            if best is None or value > best[0]:
                best = (int(value), p)
        row = {'n': n, 'largest_sampled_pivot_cost': best[0], 'K': n * n // 4,
               'witness': best[1], 'method': '103 deterministic/sample permutations'}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(__file__).with_name('pivot_cut_probe.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
