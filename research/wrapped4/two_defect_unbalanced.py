"""Optimal words for tau_a sigma with defect-two sigma and a,b >= 12."""
from two_defect_construct import construct as central_construct, extend
from two_defect_roots import from_gaps
from sparse_gap_roots import anchors


def construct(alpha, gaps, a):
    """Reduce only the larger rotation block, retaining every growing gap >= 3."""
    n = len(alpha) + sum(gaps)
    b = n - a
    assert min(a, b) >= 12
    assert tuple(alpha) in ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0))
    c = min(a, b)
    residue = (abs(a - b)) % 3
    epsilon = -1 if residue == 2 else residue
    a0, b0 = (c + epsilon, c) if a >= b else (c, c + epsilon)
    base_n = a0 + b0
    small = [g if g < 3 else 3 + (g - 3) % 3 for g in gaps]
    removed = [(g - r) // 3 for g, r in zip(gaps, small)]
    minimal = len(alpha) + sum(small)
    assert minimal <= base_n <= n and (base_n - minimal) % 3 == 0
    for _ in range((base_n - minimal) // 3):
        i = next(i for i, r in enumerate(removed) if r > 0)
        small[i] += 3
        removed[i] -= 1
    p = list(from_gaps(alpha, small, a0))
    word, refinements = central_construct(alpha, small, a0)
    info = anchors(p, a0, word)
    available = {r['gap']: [r['positive_position'], r['negative_position']]
                 for r in info['anchors']}
    assert all(i in available for i, amount in enumerate(removed) if amount)
    color = -1 if a >= b else 1
    growth_steps = sum(removed)
    for gap, amount in enumerate(removed):
        for _ in range(amount):
            position = available[gap][1 if color < 0 else 0]
            p, word, a0 = extend(p, word, a0, position, color)
            for pair in available.values():
                for i in range(2):
                    pair[i] += 3 * (pair[i] > position)
    assert tuple(p) == from_gaps(alpha, gaps, a) and a0 == a
    ell = a * b - 2
    assert len(word) == ell // 3 + ell % 3
    return word, {'central_size': base_n, 'central_refinements': refinements,
                  'one_direction_growth_steps': growth_steps}
