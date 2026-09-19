"""Verify and extend the finite twisted block-exchange certificates."""
import argparse
import json
from pathlib import Path

from general_sort import apply_to
from rotation_bounds import parity_rounded_third


FOLDER = Path(__file__).parent


def target(a, b, x, y):
    p = list(range(a, a + b)) + list(range(a))
    p[y], p[b + x] = p[b + x], p[y]
    return p


def complete_bases():
    from near_rotation_search import build_endgame, solve
    data = json.loads((FOLDER / 'twisted_exchange_bases.json').read_text())
    rows = []
    endgames = {}
    for row in data['rows']:
        if row['status'] != 'found':
            n = row['n']
            if n not in endgames:
                endgames[n] = build_endgame(n, 4, 2000000, False)
            found = solve(row['permutation'], row['budget'] + 2, endgames[n], 4,
                          1000000, seed=19, wrapped=False)
            assert found['status'] == 'found'
            row = {**found, **{key: row[key] for key in ('a', 'b', 'x', 'y')},
                   'extra_over_adjacent_lower': 2}
        else:
            row['extra_over_adjacent_lower'] = 0
        rows.append(row)
    (FOLDER / 'twisted_exchange_certificates.json').write_text(json.dumps(rows, indent=2))
    return rows


def build(a, b, x, y, table, start=0):
    """Create the twisted exchange from the identity, using interval moves."""
    if a >= 9:
        if x >= 3:
            word = build(a - 3, b, x - 3, y, table, start + 3)
            word += [(start + j, -1) for j in range(b)]
            return word
        word = [(start + a - 3 + j, -1) for j in range(b)]
        return word + build(a - 3, b, x, y, table, start)
    if b >= 9:
        if y >= 3:
            word = [(start + j, 1) for j in range(a - 1, -1, -1)]
            return word + build(a, b - 3, x, y - 3, table, start + 3)
        word = build(a, b - 3, x, y, table, start)
        return word + [(start + b - 3 + j, 1) for j in range(a - 1, -1, -1)]
    row = table[a, b, x, y]
    return [(start + i, -d) for i, d in reversed(row['word'])]


def build_from_base(a, b, x, y, a0, b0, x0, y0, table):
    row = table[a0, b0, x0, y0]
    word = [(i, -d) for i, d in reversed(row['word'])]
    left_a = (x - x0) // 3
    right_a = (a - a0) // 3 - left_a
    left_b = (y - y0) // 3
    right_b = (b - b0) // 3 - left_b
    assert x - x0 == 3 * left_a and y - y0 == 3 * left_b
    assert a - a0 == 3 * (left_a + right_a)
    assert b - b0 == 3 * (left_b + right_b)
    assert min(left_a, right_a, left_b, right_b) >= 0
    aa, bb = a0, b0
    for _ in range(left_a):
        word = [(i + 3, d) for i, d in word] + [(j, -1) for j in range(bb)]
        aa += 3
    for _ in range(right_a):
        word = [(aa + j, -1) for j in range(bb)] + word
        aa += 3
    for _ in range(left_b):
        word = [(j, 1) for j in range(aa - 1, -1, -1)] + [(i + 3, d) for i, d in word]
        bb += 3
    for _ in range(right_b):
        word += [(bb + j, 1) for j in range(aa - 1, -1, -1)]
        bb += 3
    assert aa == a and bb == b
    return word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-certificates', action='store_true')
    parser.add_argument('--cyclic-checks', action='store_true')
    args = parser.parse_args()
    rows = (complete_bases() if args.build_certificates else
            json.loads((FOLDER / 'twisted_exchange_certificates.json').read_text()))
    table = {(r['a'], r['b'], r['x'], r['y']): r for r in rows}
    assert set(table) == {(a, b, x, y) for a in (6, 7, 8) for b in (6, 7, 8)
                          for x in range(a) for y in range(b)}
    sharp_bases = 0
    for row in rows:
        a, b, x, y = (row[key] for key in ('a', 'b', 'x', 'y'))
        p = target(a, b, x, y)
        assert p == row['permutation']
        # Replay independently from the search and packed state implementation.
        state = p[:]
        for i, direction in row['word']:
            assert 0 <= i <= a + b - 4 and direction in (-1, 1)
            window = state[i:i + 4]
            state[i:i + 4] = (window[1:] + window[:1] if direction == 1
                              else window[-1:] + window[:-1])
        assert state == list(range(a + b))
        lower = parity_rounded_third(a * b - 1)
        assert len(row['word']) <= lower + 2
        if a % 3 != b % 3 or a % 3 == 0:
            assert len(row['word']) == lower
            sharp_bases += 1
    count = 0
    for a in range(6, 41):
        for b in range(6, 41):
            pairs = {(0, 0), (a - 1, b - 1), (0, b - 1), (a - 1, 0),
                     (a // 2, b // 2), (a // 3, 2 * b // 3)}
            for x, y in pairs:
                word = build(a, b, x, y, table)
                assert apply_to(list(range(a + b)), word) == target(a, b, x, y)
                lower = parity_rounded_third(a * b - 1)
                assert len(word) <= lower + 2
                if a % 3 != b % 3 or a % 3 == 0:
                    assert len(word) == lower
                count += 1
    separated = 0
    for m in range(6, 41):
        core = 6 + m % 3
        # The only non-sharp bases with x=0 are these explicitly covered cases.
        bad_y = {5} if core == 7 else {6, 7} if core == 8 else set()
        for y0 in range(core):
            assert (len(table[core, core, 0, y0]['word']) >
                    parity_rounded_third(core * core - 1)) == (y0 in bad_y)
        for distance in range(3, m + 1):
            y = m - distance
            y0 = min(y, core - 1)
            y0 -= (y0 - y) % 3
            if y0 in bad_y:
                y0 -= 3
            assert max(0, core - distance) <= y0 <= min(y, core - 1)
            word = build_from_base(m, m, 0, y, core, core, 0, y0, table)
            assert len(word) == parity_rounded_third(m * m - 1)
            assert apply_to(list(range(2 * m)), word) == target(m, m, 0, y)
            separated += 1
    # The exceptional interval bases may improve when the whole circle is available.
    exceptional = []
    if args.cyclic_checks:
        from near_rotation_search import build_endgame, solve
    for n in ((14, 16) if args.cyclic_checks else ()):
        endgame = build_endgame(n, 4, 2000000)
        for distance in (1, 2):
            p = [(i + n // 2) % n for i in range(n)]
            p[0], p[distance] = p[distance], p[0]
            budget = parity_rounded_third(n * n // 4 - 1)
            for seed in (19, 7, 0, 13):
                row = solve(p, budget, endgame, 4, 1000000, seed=seed)
                if row['status'] != 'node_limit':
                    break
            exceptional.append(row)
            print('CYCLIC', n, distance, row['status'], row.get('length'), flush=True)
    result = {'base_identities': len(rows), 'sharp_bases_for_theorem': sharp_bases,
              'extended_words_replayed': count, 'tested_block_sizes': '6..40',
              'separated_half_rotation_words_replayed': separated}
    if exceptional:
        result['cyclic_exception_tests'] = exceptional
    elif (FOLDER / 'twisted_exchange_checks.json').exists():
        earlier = json.loads((FOLDER / 'twisted_exchange_checks.json').read_text())
        if 'cyclic_exception_tests' in earlier:
            for row in earlier['cyclic_exception_tests']:
                assert apply_to(row['permutation'], row['word']) == list(range(row['n']))
            result['cyclic_exception_tests'] = earlier['cyclic_exception_tests']
    (FOLDER / 'twisted_exchange_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'cyclic_exception_tests'}), flush=True)


if __name__ == '__main__':
    main()
