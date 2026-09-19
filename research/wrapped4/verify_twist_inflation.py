"""Standard-library replay and winding audit of the new exact-family theorem."""
import json
from pathlib import Path
import random

from rotation_twist_inflation import construct, target
from twist_inflation import inflate


def walk(initial, word):
    p = list(initial)
    n = len(p)
    displacement = [0] * n
    travel = [0] * n
    crossing = {}
    for start, direction in word:
        assert 0 <= start < n and direction in (-1, 1)
        i, j, k, l = [(start + offset) % n for offset in range(4)]
        a, b, c, d = p[i], p[j], p[k], p[l]
        if direction == 1:
            p[i], p[j], p[k], p[l] = b, c, d, a
            pairs = ((a, b), (a, c), (a, d))
            moves = ((a, 3), (b, -1), (c, -1), (d, -1))
        else:
            p[i], p[j], p[k], p[l] = d, a, b, c
            pairs = ((a, d), (b, d), (c, d))
            moves = ((a, 1), (b, 1), (c, 1), (d, -3))
        for x, y in pairs:
            key = tuple(sorted((x, y)))
            crossing[key] = crossing.get(key, 0) + 1
        for x, amount in moves:
            displacement[x] += amount
            travel[x] += abs(amount)
    return p, displacement, travel, crossing


def main():
    folder = Path(__file__).parent
    rows = json.loads((folder / 'rotation_twist_inflation_bases.json').read_text())
    assert {(r['a'], r['b'], r['separation']) for r in rows} == {(7, 7, 2), (5, 5, 1), (5, 5, 2)}
    for row in rows:
        a, b, d = row['a'], row['b'], row['separation']
        p = list(range(a, a + b)) + list(range(a))
        p[0], p[d] = p[d], p[0]
        assert p == row['permutation']
        final, motion, travel, pairs = walk(p, row['word'])
        assert final == list(range(a + b))
        assert len(row['word']) == (a * b - 1) // 3
        u, v = row['lower_strand'], row['upper_strand']
        assert 0 <= u < a and a + d < v < a + b
        assert motion[u] == -b and travel[u] == b
        assert motion[v] == a and travel[v] == a
        assert pairs[u, v] == 1
    random_checks = nonzero_winding = 0
    rng = random.Random(20260919)
    for n in range(5, 15):
        for _ in range(12):
            creation = [(rng.randrange(n), rng.choice((-1, 1))) for _ in range(25)]
            p, _, _, _ = walk(list(range(n)), creation)
            sorting = [(i, -d) for i, d in reversed(creation)]
            final, motion, travel, pairs = walk(p, sorting)
            assert final == list(range(n))
            x = rng.randrange(n)
            winding, rem = divmod(p.index(x) + motion[x] - x, n)
            assert rem == 0
            inflated = inflate(p, sorting, [x])
            actual, _, _, _ = walk(inflated['initial'], inflated['word'])
            origin = (-3 * winding) % (n + 3)
            assert inflated['origin'] == origin
            assert actual == [(i - origin) % (n + 3) for i in range(n + 3)]
            assert len(inflated['word']) == len(sorting) + travel[x]
            random_checks += 1
            nonzero_winding += winding != 0
    total = total_moves = 0
    for a in range(6, 33):
        for b in range(6, 33):
            n = a + b
            for d in range(1, n // 2 + 1):
                word = construct(a, b, d)
                expected = (a * b - 1) // 3 + (a * b - 1) % 3
                assert len(word) == expected
                p = list(range(a, n)) + list(range(a))
                p[0], p[d] = p[d], p[0]
                final, _, _, _ = walk(p, word)
                assert final == list(range(n)), (a, b, d)
                total += 1
                total_moves += len(word)
    report = {'finite_bases_replayed': len(rows), 'random_winding_audits': random_checks,
              'nonzero_winding_audits': nonzero_winding,
              'complete_rotation_transposition_targets_replayed': total,
              'four_cycle_moves_replayed': total_moves,
              'target_scope': 'All 6<=a,b<=32 and every circular transposition separation.',
              'scope': 'Independent checks of the all-n proof in STRAND_INFLATION.md.'}
    (folder / 'twist_inflation_independent_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
