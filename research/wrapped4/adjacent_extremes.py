"""Enumerate high cyclic-adjacent lengths to study extremal structure."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def enumerate_extremes(n, margin, max_saved):
    p = np.arange(n, dtype=np.int64)
    hist = np.zeros(n * n // 4 + 1, np.int64)
    saved = np.empty((max_saved, n + 1), np.int64)
    count = 0
    while True:
        w = p.copy()
        while True:
            lo = hi = 0
            for i in range(1, n):
                if w[i] - i < w[lo] - lo:
                    lo = i
                if w[i] - i > w[hi] - hi:
                    hi = i
            if w[hi] - hi - (w[lo] - lo) <= n:
                break
            w[hi] -= n
            w[lo] += n
        ell = 0
        for i in range(n):
            for j in range(i + 1, n):
                ell += abs((w[j] - w[i]) // n)
        hist[ell] += 1
        if ell >= n * n // 4 - margin:
            if count < max_saved:
                saved[count, 0] = ell
                saved[count, 1:] = p
            count += 1
        i = n - 2
        while i >= 0 and p[i] >= p[i + 1]:
            i -= 1
        if i < 0:
            break
        j = n - 1
        while p[j] <= p[i]:
            j -= 1
        p[i], p[j] = p[j], p[i]
        a, b = i + 1, n - 1
        while a < b:
            p[a], p[b] = p[b], p[a]
            a += 1
            b -= 1
    return hist, saved[:min(count, max_saved)], count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()
    hist, states, count = enumerate_extremes(args.n, 3, 100000)
    result = {'n': args.n, 'histogram': hist.tolist(),
              'high_length_count': int(count), 'high_states': states.tolist()}
    out = Path(__file__).with_name(f'adjacent_extremes_n{args.n}.json')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'high_states'}), flush=True)
    print('MAX', states[states[:, 0] == args.n * args.n // 4].tolist(), flush=True)


if __name__ == '__main__':
    main()
