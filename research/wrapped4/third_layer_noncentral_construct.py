"""Exact words for every noncentral K-3 family, n>=13."""
from functools import lru_cache
import json
from pathlib import Path

from rotation_twist_inflation import construct as twisted_construct, target as twisted_target
from sparse_gap_roots import anchors
from two_defect_construct import construct as central_construct, extend, transport
from two_defect_roots import from_gaps, orbit


@lru_cache(maxsize=1)
def table():
    rows = json.loads(Path(__file__).with_name('third_layer_noncentral_words.json').read_text())
    return {min(orbit(r['permutation'])): r for r in rows}


def construct(alpha, gaps, a):
    n = len(alpha) + sum(gaps)
    assert n >= 13
    p = from_gaps(alpha, gaps, a)
    if n % 2:
        assert tuple(alpha) == (1, 0)
        assert a in {(n - 3) // 2, (n + 3) // 2}
        if n >= 15:
            d = min(gaps) + 1
            word = twisted_construct(a, n - a, d)
            word = transport(twisted_target(a, n - a, d), word, p)
            route = 'one_transposition_theorem'
        else:
            row = table()[min(orbit(p))]
            word = transport(row['permutation'], row['word'], p)
            route = 'finite_boundary'
    else:
        assert a in {n // 2 - 1, n // 2 + 1}
        assert tuple(alpha) in ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0))
        if n >= 16 and max(gaps) >= 6:
            gap = next(i for i, g in enumerate(gaps) if g >= 6)
            small = list(gaps)
            small[gap] -= 3
            color = -1 if a > n // 2 else 1
            a0 = a - (3 if color < 0 else 0)
            base = list(from_gaps(alpha, small, a0))
            word, _ = central_construct(alpha, small, a0)
            info = next(r for r in anchors(base, a0, word)['anchors'] if r['gap'] == gap)
            position = info['negative_position' if color < 0 else 'positive_position']
            grown, word, aa = extend(base, word, a0, position, color)
            assert tuple(grown) == p and aa == a
            route = 'one_inflation_from_central'
        else:
            row = table()[min(orbit(p))]
            word = transport(row['permutation'], row['word'], p)
            route = 'finite_boundary'
    ell = n * n // 4 - 3
    assert len(word) == ell // 3 + ell % 3
    return word, route
