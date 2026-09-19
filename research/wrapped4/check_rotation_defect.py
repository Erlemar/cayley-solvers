"""Independent finite checks of the bounded-affine deletion/defect lemma."""
from itertools import combinations, permutations
import json
from math import comb
from pathlib import Path
import random

from check_adjacent_extremal_classification import bfs
from rotation_ascent import balanced, potential


def delete_endpoint(f, h):
    """Rotate h to zero, delete its residue, and compress both integer axes."""
    n = len(f)
    v = [f[(h + i) % n] + n * ((h + i) // n) - h for i in range(n)]
    assert v[0] in (0, n)
    assert all(v[i] % n for i in range(1, n))
    g = [x - x // n - 1 for x in v[1:]]
    assert all(i <= x <= i + n - 1 for i, x in enumerate(g))
    return g, v[0] == n


def core_defect(sigma):
    active = [i for i, x in enumerate(sigma) if i != x]
    if not active:
        return 0, 0, 0
    ranks = {x: i for i, x in enumerate(active)}
    alpha = [ranks[sigma[x]] for x in active]
    t = len(alpha)
    e = sum(x > i for i, x in enumerate(alpha))
    g = [x if x > i else x + t for i, x in enumerate(alpha)]
    return e * (t - e) - potential(g), e, t


def two_swap_cores(n):
    """Every 3-cycle and both noncrossing matchings on each 4-subset."""
    identity = list(range(n))
    for a, b, c in combinations(range(n), 3):
        for values in ((b, c, a), (c, a, b)):
            p = identity[:]
            p[a], p[b], p[c] = values
            yield tuple(p)
    for a, b, c, d in combinations(range(n), 4):
        for pairs in (((a, b), (c, d)), ((a, d), (b, c))):
            p = identity[:]
            for x, y in pairs:
                p[x], p[y] = p[y], p[x]
            yield tuple(p)


def predicted_second_layer(n):
    result = set()
    for a in {n // 2, (n + 1) // 2}:
        for p in two_swap_cores(n):
            result.add(tuple((x + a) % n for x in p))
    m = n // 2
    if n % 2:
        for a in (m - 1, m + 2):
            result.add(tuple((i + a) % n for i in range(n)))
    else:
        for a in (m - 1, m + 1):
            base = [(i + a) % n for i in range(n)]
            for i, j in combinations(range(n), 2):
                p = base[:]
                p[i], p[j] = p[j], p[i]
                result.add(tuple(p))
    return result


def second_layer_count(n):
    return (4 * comb(n, 3) + 4 * comb(n, 4) + 2 if n % 2
            else 2 * comb(n, 3) + 2 * comb(n, 4) + 2 * comb(n, 2))


def check_deletions(n):
    bounded_count = deletion_count = 0
    for p in permutations(range(n)):
        fixed = [i for i in range(n) if p[i] == i]
        for mask in range(1 << len(fixed)):
            f = [p[i] if p[i] >= i else p[i] + n for i in range(n)]
            for bit, i in enumerate(fixed):
                if mask & (1 << bit):
                    f[i] += n
            rank, rem = divmod(sum(x - i for i, x in enumerate(f)), n)
            assert rem == 0
            defect = rank * (n - rank) - potential(f)
            bounded_count += 1
            for h in fixed:
                g, upper = delete_endpoint(f, h)
                new_rank = rank - int(upper)
                assert sum(x - i for i, x in enumerate(g)) == new_rank * (n - 1)
                assert new_rank * (n - 1 - new_rank) - potential(g) == defect
                assert potential(f) - potential(g) == (n - rank if upper else rank)
                deletion_count += 1
    return bounded_count, deletion_count


def main():
    states = formula_checks = bounded_count = deletion_count = 0
    layers = []
    for n in range(2, 9):
        exact = bfs(n)
        states += len(exact)
        for sigma in exact:
            defect, e, t = core_defect(sigma)
            for a in range(n):
                if e <= a and t - e <= n - a:
                    p = tuple((x + a) % n for x in sigma)
                    assert exact[p] == a * (n - a) - defect, (n, sigma, a)
                    formula_checks += 1
        bc, dc = check_deletions(n)
        bounded_count += bc
        deletion_count += dc
        if n >= 6:
            actual = {p for p, d in exact.items() if d == n * n // 4 - 2}
            assert actual == predicted_second_layer(n)
            assert len(actual) == second_layer_count(n)
            layers.append({'n': n, 'size': len(actual), 'method': 'independent BFS'})
        print(f'n={n}: checked {len(exact)} permutations and {bc} bounded lifts', flush=True)
    for n in (10, 11):
        source = json.loads(Path(__file__).with_name(f'adjacent_extremes_n{n}.json').read_text())
        k = n * n // 4 - 2
        actual = {tuple(row[1:]) for row in source['high_states'] if row[0] == k}
        assert len(actual) == source['histogram'][k]
        assert actual == predicted_second_layer(n)
        assert len(actual) == second_layer_count(n)
        layers.append({'n': n, 'size': len(actual), 'method': 'stored full enumeration'})
    rng = random.Random(20260919)
    larger = 0
    for n in (12, 15, 20, 40, 100):
        for _ in range(100):
            sigma = list(range(n))
            support = sorted(rng.sample(range(n), rng.randint(2, min(8, n))))
            values = support[:]
            rng.shuffle(values)
            for i, x in zip(support, values):
                sigma[i] = x
            defect, e, t = core_defect(sigma)
            a = rng.randint(e, min(n - 1, n - t + e))
            p = [(x + a) % n for x in sigma]
            assert potential(balanced(p)) == a * (n - a) - defect
            larger += 1
    result = {'independent_bfs_states': states, 'core_formulas_checked_against_bfs': formula_checks,
              'bounded_lifts_checked': bounded_count, 'endpoint_deletions_checked': deletion_count,
              'larger_core_checks': larger, 'second_layers': layers,
              'scope': 'Finite verification of all-n results proved in ROTATION_ASCENT_LEMMA.md.'}
    Path(__file__).with_name('rotation_defect_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
