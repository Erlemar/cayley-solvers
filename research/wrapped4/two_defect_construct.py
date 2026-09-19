"""Construct optimal words for central rotations with core defect two."""
from functools import lru_cache
import json
from pathlib import Path

from twist_inflation import inflate
from two_defect_roots import anchors, from_gaps, orbit


@lru_cache(maxsize=1)
def certificate():
    data = json.loads(Path(__file__).with_name('two_defect_inflation_complete.json').read_text())
    return {tuple(row['permutation']): row for row in data['nodes']}


def transport(source, word, target):
    n = len(source)
    inverse = [source.index(i) for i in range(n)]
    for p, w in ((source, word), (inverse, [(i, -d) for i, d in reversed(word)])):
        for sign in (-1, 1):
            for shift in range(n):
                image = tuple((sign * p[(sign * (i - shift)) % n] + shift) % n for i in range(n))
                if image == tuple(target):
                    return [((i + shift) % n, d) if sign == 1
                            else ((shift - i - 3) % n, -d) for i, d in w]
    raise AssertionError('Missing symmetry transport')


def extend(p, word, a, position, color):
    """Inflate a monotone fixed-core strand, absorbing its winding in tau_a."""
    n = len(p)
    sigma = [(x - a) % n for x in p]
    assert sigma[position] == position and color in (-1, 1)
    grown = inflate(p, word, [p[position]])
    new_a = a + (3 if color < 0 else 0)
    def embed(i):
        return i + 3 * (i > position)
    new_sigma = [None] * (n + 3)
    for i, x in enumerate(sigma):
        new_sigma[embed(i)] = embed(x)
    for i in range(position, position + 4):
        new_sigma[i] = i
    expected = [(x + new_a) % (n + 3) for x in new_sigma]
    actual = [(x + grown['origin']) % (n + 3) for x in grown['initial']]
    assert actual == expected
    assert len(grown['word']) == len(word) + (a if color > 0 else n - a)
    return actual, grown['word'], new_a


def construct(alpha, gaps, a=None):
    n = len(alpha) + sum(gaps)
    assert n >= 12
    a = n // 2 if a is None else a
    assert a in {n // 2, (n + 1) // 2}
    desired = from_gaps(alpha, gaps, a)
    small = [g if g < 2 else 2 + (g - 2) % 3 for g in gaps]
    removed = [(g - r) // 3 for g, r in zip(gaps, small)]
    base_n = len(alpha) + sum(small)
    while base_n < 12 or (n - base_n) % 6:
        i = next(i for i, r in enumerate(removed) if r > 0)
        small[i] += 3
        removed[i] -= 1
        base_n += 3
    base_a = a - (n - base_n) // 2
    table = certificate()
    refinements = 0
    while True:
        p = list(from_gaps(alpha, small, base_a))
        key = min(orbit(p))
        assert key in table, (alpha, small, base_a)
        row = table[key]
        word = transport(list(key), row['word'], p)
        info = anchors(p, word)
        available = {r['gap']: [r['positive_position'], r['negative_position']] for r in info['anchors']}
        uncovered = [i for i, amount in enumerate(removed) if amount and i not in available]
        if not uncovered:
            break
        first = uncovered[0]
        removed[first] -= 1
        small[first] += 3
        second = next(i for i, r in enumerate(removed) if r > 0)
        removed[second] -= 1
        small[second] += 3
        base_n += 6
        base_a += 3
        refinements += 1
    count = sum(removed)
    assert count % 2 == 0
    negative = count // 2
    for gap, amount in enumerate(removed):
        for _ in range(amount):
            color = -1 if negative else 1
            position = available[gap][1 if color < 0 else 0]
            p, word, base_a = extend(p, word, base_a, position, color)
            for pair in available.values():
                for i in range(2):
                    pair[i] += 3 * (pair[i] > position)
            negative -= color < 0
    assert tuple(p) == desired and base_a == a
    ell = a * (n - a) - 2
    assert len(word) == ell // 3 + ell % 3
    return word, refinements
