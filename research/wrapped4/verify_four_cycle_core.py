"""Independent standard-library audit of the central four-cycle-core certificate."""
from collections import Counter
from itertools import combinations, permutations
import json
from pathlib import Path

from verify_two_defect_certificate import build, orbit, replay


def is_four_cycle(alpha):
    visited = set()
    x = 0
    while x not in visited:
        visited.add(x)
        x = alpha[x]
    return x == 0 and len(visited) == 4


def model(p):
    n = len(p)
    results = []
    for a in sorted({n // 2, (n + 1) // 2}):
        sigma = [(x - a) % n for x in p]
        support = [i for i, x in enumerate(sigma) if x != i]
        if len(support) != 4:
            continue
        ranks = {x: i for i, x in enumerate(support)}
        alpha = tuple(ranks[sigma[x]] for x in support)
        if not is_four_cycle(alpha):
            continue
        gaps = [[(support[i] + j) % n for j in range(1, (support[(i + 1) % 4] - support[i]) % n)]
                for i in range(4)]
        results.append((a, alpha, gaps))
    assert len(results) == 1
    return results[0]


def expected_roots():
    result = set()
    raw = 0
    patterns = [p for p in permutations(range(4)) if is_four_cycle(p)]
    assert len(patterns) == 6
    for n in range(12, 24):
        for tail in combinations(range(1, n), 3):
            support = (0,) + tail
            gaps = tuple((support[(i + 1) % 4] - support[i]) % n - 1 for i in range(4))
            size = 4 + sum(g if g < 2 else 2 + (g - 2) % 3 for g in gaps)
            if n - 6 >= max(12, size):
                continue
            for alpha in patterns:
                result.add(min(orbit(build(alpha, gaps, n // 2))))
                raw += 1
    return result, raw


def main():
    folder = Path(__file__).parent
    data = json.loads((folder / 'four_cycle_core_certificate.json').read_text())
    assert data['complete']
    expected, raw = expected_roots()
    roots = set(map(tuple, data['roots']))
    assert len(roots) == len(data['roots']) and roots == expected
    table = {tuple(r['permutation']): r for r in data['nodes']}
    assert len(table) == len(data['nodes'])
    total_moves = edges = branches = 0
    defects = Counter()
    missing_sizes = Counter()
    for p, row in table.items():
        n = len(p)
        a, alpha, gaps = model(p)
        motion, travel, crossing = replay(p, row['word'])
        ell = n * n // 4 - 3
        lower = ell // 3 + ell % 3
        upper = (n * n + 11) // 12 + int(n % 3 == 0)
        assert lower <= len(row['word']) <= upper
        assert (len(row['word']) - ell) % 2 == 0
        total_moves += len(row['word'])
        defects[len(row['word']) - lower] += 1
        active = [i for i, gap in enumerate(gaps) if len(gap) >= 2]
        missing = set()
        chosen = []
        for i in active:
            plus = [p[pos] for pos in gaps[i] if motion[p[pos]] == a and travel[p[pos]] == a]
            minus = [p[pos] for pos in gaps[i] if motion[p[pos]] == a - n and travel[p[pos]] == n - a]
            if plus and minus:
                chosen.extend(((plus[0], 1), (minus[0], -1)))
            else:
                missing.add(i)
                missing_sizes[len(gaps[i])] += 1
        for (x, dx), (y, dy) in combinations(chosen, 2):
            assert crossing[tuple(sorted((x, y))) ] == int(dx != dy)
        needed = set()
        for ix, first in enumerate(active):
            for second in active[ix:]:
                if first not in missing and second not in missing:
                    continue
                sizes = list(map(len, gaps))
                sizes[first] += 3
                sizes[second] += 3
                needed.add(min(orbit(build(alpha, sizes, a + 3))))
        supplied = set(map(tuple, row['children']))
        assert len(supplied) == len(row['children']) and supplied == needed
        assert all(q in table and len(q) == n + 6 for q in needed)
        branches += bool(needed)
        edges += len(needed)
    reached = set()
    pending = list(roots)
    while pending:
        p = pending.pop()
        if p not in reached:
            reached.add(p)
            pending.extend(map(tuple, table[p]['children']))
    assert reached == set(table)
    report = {'complete': True, 'independent_root_models': raw,
              'independent_root_orbits': len(roots), 'nodes_replayed': len(table),
              'moves_replayed': total_moves, 'refinement_nodes': branches,
              'refinement_edges': edges, 'anchored_leaves': len(table) - branches,
              'maximum_base_n': max(map(len, table)), 'excess_over_lower_bound': dict(defects),
              'missing_anchor_gap_sizes': dict(missing_sizes),
              'scope': 'All finite words, anchors, roots, edges, and coverage for central rotation times a 4-cycle.'}
    (folder / 'four_cycle_core_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
