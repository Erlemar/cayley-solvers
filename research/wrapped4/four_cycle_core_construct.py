"""All-size H(n) words for a central rotation followed by an arbitrary 4-cycle."""
from functools import lru_cache
import json
from pathlib import Path

from sparse_gap_roots import anchors
from two_defect_construct import extend, transport
from two_defect_roots import from_gaps, orbit


@lru_cache(maxsize=1)
def certificate():
    data = json.loads(Path(__file__).with_name('four_cycle_core_certificate.json').read_text())
    assert data['complete']
    return {tuple(row['permutation']): row for row in data['nodes']}


def construct(alpha, gaps, a=None):
    n = len(alpha) + sum(gaps)
    assert len(alpha) == 4 and n >= 12
    a = n // 2 if a is None else a
    assert a in {n // 2, (n + 1) // 2}
    desired = from_gaps(alpha, gaps, a)
    small = [g if g < 2 else 2 + (g - 2) % 3 for g in gaps]
    removed = [(g - r) // 3 for g, r in zip(gaps, small)]
    n0 = 4 + sum(small)
    while n0 < 12 or (n - n0) % 6:
        i = next(i for i, r in enumerate(removed) if r)
        small[i] += 3
        removed[i] -= 1
        n0 += 3
    a0 = a - (n - n0) // 2
    table = certificate()
    refinements = 0
    while True:
        p = list(from_gaps(alpha, small, a0))
        key = min(orbit(p))
        row = table[key]
        word = transport(list(key), row['word'], p)
        info = anchors(p, a0, word)
        available = {r['gap']: [r['positive_position'], r['negative_position']] for r in info['anchors']}
        uncovered = [i for i, amount in enumerate(removed) if amount and i not in available]
        if not uncovered:
            break
        i = uncovered[0]
        removed[i] -= 1
        small[i] += 3
        j = next(j for j, r in enumerate(removed) if r)
        removed[j] -= 1
        small[j] += 3
        n0 += 6
        a0 += 3
        refinements += 1
    count = sum(removed)
    assert count % 2 == 0
    negative = count // 2
    for gap, amount in enumerate(removed):
        for _ in range(amount):
            color = -1 if negative else 1
            position = available[gap][1 if color < 0 else 0]
            p, word, a0 = extend(p, word, a0, position, color)
            for pair in available.values():
                for i in range(2):
                    pair[i] += 3 * (pair[i] > position)
            negative -= color < 0
    assert tuple(p) == desired and a0 == a
    assert len(word) <= (n * n + 11) // 12 + int(n % 3 == 0)
    return word, refinements
