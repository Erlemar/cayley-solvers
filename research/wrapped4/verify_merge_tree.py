"""Independent, standard-library verification of binary extraction trees."""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path


def check(path):
    data = json.loads(path.read_text())
    coefficient = Fraction(data['coefficient'])
    records = data['records']
    table = {(r['bits'], r['gap']): r for r in records}
    assert len(table) == len(records)
    expected = {(''.join(bits), gap) for bits in itertools.product('AB', repeat=3)
                for gap in range(3)}
    stages = []
    maximum_k = max(len(bits) for bits, _ in table)
    words = 0
    for k in range(3, maximum_k + 1):
        current = {key: row for key, row in table.items() if len(key[0]) == k}
        assert set(current) == expected, f'Coverage error at k={k}'
        expected = set()
        found = 0
        for (bits, gap), row in current.items():
            a = [i for i in range(k) if bits[i] == 'A']
            b = [i for i in range(k) if bits[i] == 'B']
            p = a + list(range(k, k + gap)) + b
            assert p == row['pattern']
            assert row['width'] == max(9, len(p))
            p += list(range(len(p), row['width']))
            inversions = sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
            assert row['inversions'] == inversions
            if row['status'] == 'found':
                for start, direction in row['word']:
                    assert 0 <= start <= len(p) - 4 and direction in (-1, 1)
                    block = p[start:start + 4]
                    p[start:start + 4] = (block[1:] + block[:1] if direction == 1
                                          else block[-1:] + block[:-1])
                assert p == list(range(len(p)))
                assert 3 * len(row['word']) - inversions <= coefficient * k
                found += 1
            else:
                expected.add((bits + 'A', (gap + 2) % 3))
                expected.add((bits + 'B', gap))
        words += found
        stages.append({'k': k, 'cases': len(current), 'verified_words': found,
                       'next_frontier': len(expected)})
    complete = not expected
    assert data['complete'] == complete
    return {'source': path.name, 'coefficient': str(coefficient), 'complete': complete,
            'verified_cases': len(records), 'verified_words': words,
            'maximum_selected': maximum_k, 'maximum_width': max(r['width'] for r in records),
            'frontier_size': len(expected), 'stages': stages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default='merge_extraction_c1_2_wide_trim.json')
    args = parser.parse_args()
    folder = Path(__file__).parent
    result = check(folder / args.file)
    (folder / 'merge_refined_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
