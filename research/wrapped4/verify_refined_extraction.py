"""Independently audit a complete or partial stable-extraction tree.

A nonempty frontier explicitly means this is NOT an all-n certificate.
"""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path


def encode(positions):
    k = len(positions)
    result = [-1] * (max(positions) + 1)
    for label, pos in enumerate(positions):
        result[pos] = label
    filler = k
    for pos in range(len(result)):
        if result[pos] == -1:
            result[pos] = filler
            filler += 1
    return tuple(result)


def check(path):
    data = json.loads(path.read_text())
    coefficient = Fraction(data['defect_per_extracted_token'])
    table = {(row['k'], tuple(row['pattern'])): row for row in data['records']}
    assert len(table) == len(data['records'])
    pending = set()
    for gaps in itertools.product(range(3), repeat=3):
        for order in itertools.permutations(range(3)):
            positions = [0] * 3
            pos = 0
            for label, gap in zip(order, gaps):
                pos += gap
                positions[label] = pos
                pos += 1
            pending.add(encode(positions))
    stages = []
    checked = words = 0
    for k in range(3, max(row['k'] for row in data['records']) + 1):
        actual = {p: row for (kk, p), row in table.items() if kk == k}
        assert set(actual) == pending
        pending = set()
        local_words = 0
        for p, row in actual.items():
            assert sorted(p) == list(range(len(p)))
            assert row['width'] >= max(9, len(p))
            positions = [p.index(i) for i in range(k)]
            if row['status'] == 'found':
                state = list(p) + list(range(len(p), row['width']))
                initial_inv = sum(state[i] > state[j] for i in range(len(state))
                                  for j in range(i + 1, len(state)))
                for start, direction in row['word']:
                    assert 0 <= start <= row['width'] - 4 and direction in (-1, 1)
                    old = state[start:start + 4]
                    state[start:start + 4] = (old[1:] + old[:1] if direction == 1
                                             else old[-1:] + old[:-1])
                assert state == list(range(row['width']))
                assert 3 * len(row['word']) - initial_inv <= coefficient * k
                words += 1
                local_words += 1
            else:
                choices = [i for i, value in enumerate(p) if value >= k]
                choices += [len(p), len(p) + 1, len(p) + 2]
                for pos in choices:
                    pending.add(encode(positions + [pos]))
            checked += 1
        stages.append({'k': k, 'cases': len(actual), 'verified_words': local_words,
                       'next_frontier': len(pending)})
    complete = not pending
    assert bool(data['complete']) == complete
    boundary_charges = []
    for row in data['records']:
        if row['status'] != 'found' or len(row['pattern']) != row['k']:
            continue
        last_moved = max((i + 1 for i, value in enumerate(row['pattern']) if i != value), default=0)
        boundary_charges.append(row['k'] - max(2, last_moved))
    result = {'source': path.name, 'coefficient': str(coefficient),
              'complete': complete, 'verified_cases': checked,
              'verified_words': words, 'frontier_size': len(pending),
              'maximum_selected': max(r['k'] for r in data['records']),
              'maximum_width': max(r['width'] for r in data['records']),
              'maximum_helper_charge': max(boundary_charges, default=0),
              'boundary_patterns_checked': len(boundary_charges),
              'stages': stages,
              'scope': ('Complete finite extraction certificate.' if complete else
                        'Incomplete tree: no improved all-n upper bound follows yet.')}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default='adaptive_extraction_c3_2_wide.json')
    args = parser.parse_args()
    folder = Path(__file__).parent
    result = check(folder / args.file)
    (folder / 'adaptive_refined_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
