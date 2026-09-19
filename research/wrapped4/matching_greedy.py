"""Explore deterministic three-inversion descents after a fixed preparation."""
import json
from pathlib import Path

import numpy as np
from numba import njit

from affine_insertion import balanced_keys
from general_sort import apply_to
from near_rotation_search import potential


PREFIX = [(3, -1), (5, -1), (7, -1), (0, 1)]


@njit(cache=True)
def greedy(initial, mode):
    w = initial.copy()
    n = len(w)
    ell = potential(w, n)
    out = np.empty((ell // 3 + 1, 2), np.int64)
    count = 0
    cursor = 0
    while ell:
        best = 10**18
        bi = bd = -1
        for i in range(n):
            vals = np.array([w[(i + j) % n] + n * ((i + j) // n) for j in range(4)])
            for d in (-1, 1):
                if d == 1:
                    valid = vals[0] > vals[1] and vals[0] > vals[2] and vals[0] > vals[3]
                    energy = 6 * (vals[0] - i) - 2 * (vals[1] + vals[2] + vals[3] - 3 * i - 6) - 12
                else:
                    valid = vals[3] < vals[0] and vals[3] < vals[1] and vals[3] < vals[2]
                    energy = -6 * (vals[3] - i - 3) + 2 * (vals[0] + vals[1] + vals[2] - 3 * i - 3) - 12
                if not valid:
                    continue
                direction = int(d == 1) if mode % 2 == 0 else int(d == -1)
                family = mode // 2
                if family == 0:
                    score = i * 2 + direction
                elif family == 1:
                    score = (n - 1 - i) * 2 + direction
                elif family == 2:
                    score = ((i - cursor) % n) * 2 + direction
                elif family == 3:
                    score = ((cursor - i) % n) * 2 + direction
                elif family == 4:
                    score = -energy * 2 * n + i * 2 + direction
                elif family == 5:
                    score = energy * 2 * n + i * 2 + direction
                elif family == 6:
                    score = direction * n + i
                else:
                    score = direction * n + n - 1 - i
                if score < best:
                    best, bi, bd = score, i, d
        if bi < 0:
            break
        vals = np.array([w[(bi + j) % n] + n * ((bi + j) // n) for j in range(4)])
        for j in range(4):
            w[(bi + j) % n] = vals[(j + bd) % 4] - n * ((bi + j) // n)
        out[count] = (bi, bd)
        count += 1
        ell -= 3
        cursor = (bi + (1 if mode // 2 == 2 else -1)) % n
    assert ell == potential(w, n)
    return ell, out[:count]


def main():
    rows = []
    for n in (12, 24, 36, 48, 60, 72, 96, 120):
        p = [(i + n // 2 + (1 if i % 2 == 0 else -1)) % n for i in range(n)]
        prepared = apply_to(p, PREFIX)
        lift = np.array(balanced_keys(prepared), np.int64)
        outcomes = []
        for mode in range(16):
            residual, tail = greedy(lift, mode)
            record = {'n': n, 'mode': mode, 'residual_adjacent_length': int(residual),
                      'descents': len(tail)}
            if residual == 0:
                word = PREFIX + tail.tolist()
                assert apply_to(p, word) == list(range(n))
                record['word'] = word
            rows.append(record)
            outcomes.append(int(residual))
        print(n, outcomes, flush=True)
    Path(__file__).with_name('matching_greedy_checks.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
