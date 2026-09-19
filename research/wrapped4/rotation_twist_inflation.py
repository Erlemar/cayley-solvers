"""Independent block-size extensions for rotations with short transpositions."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

from twist_inflation import inflate, strand_data


FOLDER = Path(__file__).parent


def target(a, b, separation):
    p = list(range(a, a + b)) + list(range(a))
    p[0], p[separation] = p[separation], p[0]
    return p


def qualifying_pairs(a, b, separation, word):
    displacement, travel, crossings = strand_data(target(a, b, separation), word)
    return [(u, v) for u in range(a) for v in range(a + separation + 1, a + b)
            if displacement[u] == -b and travel[u] == b
            and displacement[v] == a and travel[v] == a
            and crossings.get((u, v), 0) == 1]


def write_bases():
    old = json.loads((FOLDER / 'twisted_exchange_checks.json').read_text())['cyclic_exception_tests']
    small = json.loads((FOLDER / 'twist_inflation_bases.json').read_text())
    rows = []
    for m, d in ((7, 2), (5, 1), (5, 2)):
        p = target(m, m, d)
        source = next(row for row in old + small if row['permutation'] == p)
        pairs = qualifying_pairs(m, m, d, source['word'])
        assert pairs
        u, v = pairs[0]
        rows.append({'a': m, 'b': m, 'separation': d, 'permutation': p,
                     'word': source['word'], 'lower_strand': u, 'upper_strand': v})
    (FOLDER / 'rotation_twist_inflation_bases.json').write_text(json.dumps(rows, indent=2))
    return rows


def build(a, b, separation, table):
    assert a % 3 == b % 3 in (1, 2)
    base = 7 if a % 3 == 1 else 5
    assert a >= base and b >= base
    row = table[base, separation]
    aa = bb = base
    word = row['word']
    u, v = row['lower_strand'], row['upper_strand']
    for increase_first, count in ((True, (a - base) // 3), (False, (b - base) // 3)):
        for _ in range(count):
            selected = u if increase_first else v
            expanded = inflate(target(aa, bb, separation), word, [selected])
            cost = bb if increase_first else aa
            aa += 3 * int(increase_first)
            bb += 3 * int(not increase_first)
            assert expanded['initial'] == target(aa, bb, separation)
            assert expanded['origin'] == 0
            assert len(expanded['word']) == len(word) + cost
            word = expanded['word']
            u, v = expanded['first'][u], expanded['first'][v]
    assert len(word) == (a * b - 1) // 3
    displacement, travel, crossings = strand_data(target(a, b, separation), word)
    assert displacement[u] == -b and travel[u] == b
    assert displacement[v] == a and travel[v] == a
    assert crossings[u, v] == 1
    return word


@lru_cache(maxsize=1)
def base_tables():
    interval = json.loads((FOLDER / 'twisted_exchange_certificates.json').read_text())
    inflated = json.loads((FOLDER / 'rotation_twist_inflation_bases.json').read_text())
    return ({(r['a'], r['b'], r['x'], r['y']): r for r in interval},
            {(r['a'], r['separation']): r for r in inflated})


def construct(a, b, separation):
    """Optimal sorting word for tau_a followed by (0,separation), a,b>=6."""
    from twisted_exchange_check import build as interval_build, build_from_base
    n = a + b
    assert a >= 6 and b >= 6 and 1 <= separation <= n // 2
    if a > b:
        return [((separation - i - 3) % n, -direction)
                for i, direction in construct(b, a, separation)]
    interval, inflated = base_tables()
    same = a % 3 == b % 3 and a % 3 != 0
    if same and (separation == 2 or separation == 1 and a % 3 == 2):
        return build(a, b, separation, inflated)
    y = b - separation
    if same:
        core = 6 + a % 3
        y0 = min(y, core - 1)
        y0 -= (y0 - y) % 3
        bad = {5} if core == 7 else {6, 7}
        if y0 in bad:
            y0 -= 3
        assert max(0, core - separation) <= y0 <= min(y, core - 1)
        word = build_from_base(a, b, 0, y, core, core, 0, y0, interval)
    else:
        word = interval_build(a, b, 0, y, interval)
    # The interval constructor maps identity to the target with endpoints y,b.
    # Invert the word to sort, then rotate these endpoints to 0,separation.
    return [((i - y) % n, -direction) for i, direction in reversed(word)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-block', type=int, default=40)
    args = parser.parse_args()
    rows = write_bases()
    table = {(row['a'], row['separation']): row for row in rows}
    checked = 0
    max_moves = 0
    for a in range(5, args.max_block + 1):
        for b in range(5, args.max_block + 1):
            if a % 3 != b % 3 or a % 3 == 0:
                continue
            for d in ((2,) if a % 3 == 1 else (1, 2)):
                word = build(a, b, d, table)
                checked += 1
                max_moves = max(max_moves, len(word))
    result = {'finite_bases': len(rows), 'independently_extended_targets': checked,
              'block_sizes': f'5..{args.max_block}', 'largest_word_length': max_moves,
              'scope': 'Finite checks of two separate all-size strand-inflation identities.'}
    (FOLDER / 'rotation_twist_inflation_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
