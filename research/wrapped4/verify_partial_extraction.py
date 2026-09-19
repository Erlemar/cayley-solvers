"""Independent replay and finite-tree coverage for extraction with a disordered tail."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import permutations, product
import json
from pathlib import Path


def inversions(p):
    return sum(x > y for i, x in enumerate(p) for y in p[i + 1:])


def check(path):
    data = json.loads(Path(path).read_text())
    coefficient = Fraction(data['coefficient'])
    table = {(r['k'], tuple(r['pattern'])): r for r in data['records']}
    assert len(table) == len(data['records'])
    def pattern(positions):
        k = len(positions)
        p = [None] * (max(positions) + 1)
        for label, position in enumerate(positions):
            p[position] = label
        filler = k
        for i, value in enumerate(p):
            if value is None:
                p[i] = filler
                filler += 1
        return tuple(p)
    pending = set()
    for gaps in product(range(3), repeat=3):
        for order in permutations(range(3)):
            positions = [0] * 3
            cursor = 0
            for gap, label in zip(gaps, order):
                cursor += gap
                positions[label] = cursor
                cursor += 1
            pending.add(pattern(positions))
    checked = words = moves = disordered = 0
    charges = Counter()
    maximum_width = max(r['width'] for r in data['records'])
    maximum_selected = max(r['k'] for r in data['records'])
    for k in range(3, maximum_selected + 1):
        rows = {p: row for (kk, p), row in table.items() if kk == k}
        assert set(rows) == pending, (k, len(rows), len(pending))
        pending = set()
        for p, row in rows.items():
            assert sorted(p) == list(range(len(p)))
            selected = [p.index(i) for i in range(k)]
            sorted_positions = sorted(selected)
            assert sorted_positions[0] <= 2
            assert all(b - a - 1 <= 2 for a, b in zip(sorted_positions, sorted_positions[1:]))
            checked += 1
            if row['status'] != 'found':
                for pos in [i for i, x in enumerate(p) if x >= k] + list(range(len(p), len(p) + 3)):
                    pending.add(pattern(selected + [pos]))
                continue
            width = row['width']
            assert width >= max(k, len(p))
            before = list(p) + list(range(len(p), width))
            after = before[:]
            for start, sign in row['word']:
                assert 0 <= start <= width - 4 and sign in (-1, 1)
                block = after[start:start + 4]
                after[start:start + 4] = block[1:] + block[:1] if sign == 1 else block[-1:] + block[:-1]
            assert after[:k] == list(range(k))
            assert sorted(after[k:]) == list(range(k, width))
            tail = inversions(after[k:])
            charge = 3 * len(row['word']) + tail - inversions(before)
            assert charge <= coefficient * k
            assert row['endpoint'] == after and row['tail_inversions'] == tail and row['charge'] == charge
            charges[str(Fraction(charge, k))] += 1
            words += 1
            moves += len(row['word'])
            disordered += tail > 0
    assert bool(data['complete']) == (not pending)
    return {'complete': not pending, 'coefficient': str(coefficient), 'cases_checked': checked,
            'words_replayed': words, 'moves_replayed': moves, 'disordered_tail_words': disordered,
            'maximum_width': maximum_width, 'maximum_selected': maximum_selected,
            'uncovered_next_frontier': len(pending), 'charges_per_selected': dict(charges),
            'scope': 'Positive extraction identities and full refinement coverage; incomplete trees do not prove an all-n bound.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    path = Path(args.file)
    if not path.is_absolute():
        path = Path(__file__).with_name(args.file)
    result = check(path)
    path.with_name(path.stem + '_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
