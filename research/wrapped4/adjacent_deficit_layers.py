"""Enumerate complete fixed-deficit layers using the proved finite-core theorem."""
from functools import lru_cache
from itertools import combinations, permutations
import json
from pathlib import Path

from rotation_ascent import potential


@lru_cache(maxsize=None)
def cores(deficit):
    if deficit == 0:
        return (((), 0),)
    result = []
    for t in range(2, 2 * deficit + 1):
        for alpha in permutations(range(t)):
            if any(i == x for i, x in enumerate(alpha)):
                continue
            e = sum(x > i for i, x in enumerate(alpha))
            g = [x if x > i else x + t for i, x in enumerate(alpha)]
            if e * (t - e) - potential(g) == deficit:
                result.append((alpha, e))
    return tuple(result)


def layer(n, deficit):
    target = n * n // 4 - deficit
    result = set()
    if target < 0:
        return result
    for a in range(n):
        gap = a * (n - a) - target
        if gap < 0:
            continue
        for alpha, e in cores(gap):
            t = len(alpha)
            if t > n or e > a or t - e > n - a:
                continue
            for support in combinations(range(n), t):
                p = [(i + a) % n for i in range(n)]
                for i, x in enumerate(alpha):
                    p[support[i]] = (support[x] + a) % n
                result.add(tuple(p))
    return result


def main():
    from collections import Counter
    from check_adjacent_extremal_classification import bfs
    rows = []
    classified_cores = set()
    for t in (4, 5, 6):
        for alpha in permutations(range(t)):
            remaining = set(range(t))
            cycles = []
            while remaining:
                cycle = []
                x = min(remaining)
                while x in remaining:
                    remaining.remove(x)
                    cycle.append(x)
                    x = alpha[x]
                cycles.append(cycle)
            shape = sorted(map(len, cycles))
            include = shape == [4]
            if shape == [2, 3]:
                pair = next(c for c in cycles if len(c) == 2)
                include = (pair[0] - pair[1]) % t in (1, t - 1)
            elif shape == [2, 2, 2]:
                pairs = [sorted(c) for c in cycles]
                include = not any(a < c < b < d or c < a < d < b
                                  for (a, b), (c, d) in combinations(pairs, 2))
            if include:
                classified_cores.add(alpha)
    assert classified_cores == {alpha for alpha, e in cores(3)}
    for n in range(2, 9):
        exact = bfs(n)
        for deficit in range(min(3, n * n // 4) + 1):
            expected = {p for p, d in exact.items() if d == n * n // 4 - deficit}
            actual = layer(n, deficit)
            assert actual == expected
            rows.append({'n': n, 'deficit': deficit, 'states': len(actual), 'method': 'independent BFS'})
    for n in (10, 11):
        source = json.loads(Path(__file__).with_name(f'adjacent_extremes_n{n}.json').read_text())
        for deficit in range(4):
            length = n * n // 4 - deficit
            expected = {tuple(r[1:]) for r in source['high_states'] if r[0] == length}
            assert len(expected) == source['histogram'][length]
            actual = layer(n, deficit)
            assert actual == expected
            rows.append({'n': n, 'deficit': deficit, 'states': len(actual), 'method': 'stored full enumeration'})
    report = {'complete_layer_checks': rows, 'third_core_types_checked': len(classified_cores),
              'core_counts_by_support': {str(d): dict(Counter(len(a) for a, e in cores(d))) for d in range(4)},
              'scope': 'Finite checks of the general enumeration theorem in ROTATION_ASCENT_LEMMA.md.'}
    Path(__file__).with_name('adjacent_deficit_layers_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
