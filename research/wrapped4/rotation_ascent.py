"""Construct increasing adjacent-length paths to rotations using arbitrary swaps."""
import json
from pathlib import Path
import random

from check_adjacent_extremal_classification import bfs


def balanced(p):
    n = len(p)
    w = list(p)
    while True:
        low = min(range(n), key=lambda i: w[i] - i)
        high = max(range(n), key=lambda i: w[i] - i)
        if w[high] - high - w[low] + low <= n:
            return w
        w[high] -= n
        w[low] += n


def potential(w):
    n = len(w)
    return sum(abs((w[j] - w[i]) // n) for i in range(n) for j in range(i + 1, n))


def ascend(p, exact=None):
    n = len(p)
    w = balanced(p)
    displacements = [w[i] - i for i in range(n)]
    a = max(max(displacements), min(n // 2, min(displacements) + n))
    assert 0 <= a < n
    f = [value + n - a for value in w]
    initial = length = potential(f)
    q = list(p)
    swaps = []
    increments = []
    if exact is not None:
        assert exact[tuple(q)] == length
    while True:
        interior = [i for i in range(n) if i < f[i] < i + n]
        if not interior:
            break
        i = interior[0]
        first = f[i]
        def at(pos):
            return f[pos % n] + n * (pos // n)
        choices = [j for j in range(i + 1, first + 1) if first < at(j) <= i + n]
        assert choices, ('Pigeonhole lemma failed', p, f, i)
        j = choices[0]
        second = at(j)
        increment = 1 + 2 * sum(first < at(pos) < second for pos in range(i + 1, j))
        f[i] = second
        f[j % n] = first - n * (j // n)
        q[i], q[j % n] = q[j % n], q[i]
        swaps.append((i, j % n))
        increments.append(increment)
        length += increment
        assert all(pos <= f[pos] <= pos + n for pos in range(n))
        assert [(value + a) % n for value in f] == q
        if exact is not None:
            assert length == potential(f) == exact[tuple(q)]
        assert length <= a * (n - a)
    assert q == [(i + a) % n for i in range(n)]
    assert potential(f) == length == a * (n - a)
    assert len(swaps) <= length - initial
    return {'n': n, 'rotation': a, 'adjacent_length': initial,
            'rotation_length': length, 'swaps': swaps, 'increments': increments}


def predicted_next_layer(n):
    result = set()
    for a in {n // 2, (n + 1) // 2}:
        base = [(i + a) % n for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                p = base[:]
                p[i], p[j] = p[j], p[i]
                result.add(tuple(p))
    if n % 2 == 0:
        for a in (n // 2 - 1, n // 2 + 1):
            result.add(tuple((i + a) % n for i in range(n)))
    return result


def main():
    rows = []
    checked = steps = 0
    for n in range(2, 9):
        exact = bfs(n)
        for p in exact:
            row = ascend(p, exact)
            checked += 1
            steps += len(row['swaps'])
        if n >= 4:
            actual = {p for p, d in exact.items() if d == n * n // 4 - 1}
            predicted = predicted_next_layer(n)
            assert actual == predicted
            expected = n * (n - 1) if n % 2 else n * (n - 1) // 2 + 2
            assert len(actual) == expected
            rows.append({'n': n, 'next_layer_size': len(actual)})
    rng = random.Random(20261001)
    larger = 0
    for n in (9, 12, 20, 40, 100):
        for _ in range(10):
            p = list(range(n))
            rng.shuffle(p)
            ascend(p)
            larger += 1
    result = {'complete_bfs_states_checked': checked, 'exact_increments_checked': steps,
              'larger_paths_checked': larger, 'larger_sizes': [9, 12, 20, 40, 100],
              'next_layers': rows,
              'scope': 'Finite independent checks of the all-n proof in ROTATION_ASCENT_LEMMA.md.'}
    Path(__file__).with_name('rotation_ascent_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
