"""Independent standard-library coverage/replay audit of the finite certificate."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path


def orbit(p):
    n = len(p)
    inverse = [0] * n
    for i in range(n):
        inverse[p[i]] = i
    images = set()
    for source in (p, inverse):
        for direction in (-1, 1):
            for offset in range(n):
                images.add(tuple((direction * source[(direction * (i - offset)) % n] + offset) % n
                                 for i in range(n)))
    return images


def build(alpha, gaps, a):
    n = len(alpha) + sum(gaps)
    support = [0]
    for gap in gaps[:-1]:
        support.append(support[-1] + gap + 1)
    sigma = list(range(n))
    for i, x in enumerate(alpha):
        sigma[support[i]] = support[x]
    return tuple((x + a) % n for x in sigma)


def model(p):
    n = len(p)
    possibilities = []
    for a in {n // 2, (n + 1) // 2}:
        sigma = [(x - a) % n for x in p]
        support = [i for i in range(n) if sigma[i] != i]
        if len(support) not in (3, 4):
            continue
        indices = {x: i for i, x in enumerate(support)}
        alpha = tuple(indices[sigma[x]] for x in support)
        if alpha not in ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0)):
            continue
        gaps = [[(support[i] + j) % n for j in range(1, (support[(i + 1) % len(support)] - support[i]) % n)]
                for i in range(len(support))]
        possibilities.append((a, alpha, gaps))
    assert len(possibilities) == 1
    return possibilities[0]


def replay(p, word):
    n = len(p)
    state = list(p)
    motion = [0] * n
    travel = [0] * n
    crossings = Counter()
    for start, sign in word:
        assert 0 <= start < n and sign in (-1, 1)
        indices = [(start + j) % n for j in range(4)]
        values = [state[i] for i in indices]
        if sign == 1:
            long, short = values[0], values[1:]
            rotated = values[1:] + values[:1]
        else:
            long, short = values[3], values[:3]
            rotated = values[-1:] + values[:-1]
        motion[long] += 3 * sign
        travel[long] += 3
        for x in short:
            motion[x] -= sign
            travel[x] += 1
            crossings[tuple(sorted((x, long)))] += 1
        for i, x in zip(indices, rotated):
            state[i] = x
    assert state == list(range(n))
    return motion, travel, crossings


def expected_roots():
    result = set()
    raw = 0
    # Enumerate marked position subsets, independently of the gap-composition generator.
    for n in range(12, 24):
        for t in (3, 4):
            for tail in combinations(range(1, n), t - 1):
                support = (0,) + tail
                gaps = tuple((support[(i + 1) % t] - support[i]) % n - 1 for i in range(t))
                reduced_size = t + sum(g if g < 2 else 2 + (g - 2) % 3 for g in gaps)
                if n - 6 >= max(12, reduced_size):
                    continue
                patterns = ((1, 2, 0), (2, 0, 1)) if t == 3 else ((1, 0, 3, 2), (3, 2, 1, 0))
                for alpha in patterns:
                    result.add(min(orbit(build(alpha, gaps, n // 2))))
                    raw += 1
    return result, raw


def main():
    folder = Path(__file__).parent
    data = json.loads((folder / 'two_defect_inflation_complete.json').read_text())
    roots, raw = expected_roots()
    assert len(data['roots']) == len(roots) and set(map(tuple, data['roots'])) == roots
    table = {tuple(row['permutation']): row for row in data['nodes']}
    assert len(table) == len(data['nodes'])
    edges = []
    total_moves = 0
    leaves = branches = 0
    for p, row in table.items():
        n = len(p)
        a, alpha, gaps = model(p)
        k = a * (n - a) - 2
        assert len(row['word']) == k // 3 + k % 3
        motion, travel, crossings = replay(p, row['word'])
        total_moves += len(row['word'])
        missing = []
        positive_anchors = []
        negative_anchors = []
        active = [i for i, gap in enumerate(gaps) if len(gap) >= 2]
        for i in active:
            plus = [p[pos] for pos in gaps[i] if motion[p[pos]] == a and travel[p[pos]] == a]
            minus = [p[pos] for pos in gaps[i] if motion[p[pos]] == -(n - a) and travel[p[pos]] == n - a]
            if plus and minus:
                positive_anchors.append(plus[0])
                negative_anchors.append(minus[0])
            else:
                missing.append(i)
        for x, y in combinations(positive_anchors, 2):
            assert crossings[tuple(sorted((x, y)))] == 0
        for x, y in combinations(negative_anchors, 2):
            assert crossings[tuple(sorted((x, y)))] == 0
        for x in positive_anchors:
            for y in negative_anchors:
                assert crossings[tuple(sorted((x, y)))] == 1
        expected_children = set()
        for i in active:
            for j in active:
                if i > j or i not in missing and j not in missing:
                    continue
                grown = [len(gap) for gap in gaps]
                grown[i] += 3
                grown[j] += 3
                expected_children.add(min(orbit(build(alpha, grown, a + 3))))
        assert expected_children == set(map(tuple, row['children']))
        assert len(row['children']) == len(expected_children)
        for child in expected_children:
            assert child in table and len(child) == n + 6
            edges.append((p, child))
        branches += bool(missing)
        leaves += not missing
    reached = set(roots)
    while True:
        more = {q for p, q in edges if p in reached} - reached
        if not more:
            break
        reached.update(more)
    assert reached == set(table)
    boundary = json.loads((folder / 'two_defect_n12_boundary.json').read_text())
    assert len(boundary) == 6 and {r['separation'] for r in boundary} == set(range(1, 7))
    for row in boundary:
        p = [(i + 5) % 12 for i in range(12)]
        p[0], p[row['separation']] = p[row['separation']], p[0]
        assert row['a'] == 5 and row['b'] == 7 and row['permutation'] == p
        assert len(row['word']) == 12
        replay(p, row['word'])
    report = {'complete': True, 'independent_raw_models_including_both_noncrossing_pairings': raw,
              'root_orbits': len(roots), 'certificate_nodes': len(table), 'anchored_leaves': leaves,
              'refinement_nodes': branches, 'refinement_edges': len(edges),
              'largest_base_n': max(map(len, table)), 'base_word_moves_replayed': total_moves,
              'n12_boundary_words_replayed': len(boundary),
              'scope': 'Complete positive finite certificate for the all-n deficit-two induction; no negative search claim is used.'}
    (folder / 'two_defect_certificate_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
