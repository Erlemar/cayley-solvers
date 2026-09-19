"""Independent exact BFS checks for the auxiliary-metric structural theorem."""
from collections import deque
import json
from pathlib import Path


def bfs(n):
    identity = tuple(range(n))
    distances = {identity: 0}
    queue = deque([identity])
    while queue:
        p = queue.popleft()
        length = distances[p]
        for i in range(n):
            j = (i + 1) % n
            q = list(p)
            q[i], q[j] = q[j], q[i]
            q = tuple(q)
            if q not in distances:
                distances[q] = length + 1
                queue.append(q)
    return distances


def main():
    rows = []
    formula_checks = 0
    for n in range(2, 9):
        distances = bfs(n)
        maximum = n * n // 4
        rotations = {tuple((i + a) % n for i in range(n))
                     for a in {n // 2, (n + 1) // 2}}
        diameter = max(distances.values())
        actual = {p for p, d in distances.items() if d == diameter}
        assert diameter == maximum
        assert actual == rotations
        for a in range(n):
            p = [(i + a) % n for i in range(n)]
            for length in range(n):
                q = tuple(p[1:length + 1] + p[:1] + p[length + 1:])
                expected = a * (n - a) - length + 2 * max(0, length - a)
                assert distances[q] == expected, (n, a, length)
                formula_checks += 1
        rows.append({'n': n, 'states': len(distances), 'maximum': maximum,
                     'maximizers': [list(p) for p in sorted(actual)]})
    result = {'scope': 'Independent finite validation; all-n proof is in ADJACENT_EXTREMAL_CLASSIFICATION.md',
              'total_states': sum(r['states'] for r in rows),
              'rotation_run_formula_checks': formula_checks, 'rows': rows}
    Path(__file__).with_name('adjacent_extremal_classification_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
