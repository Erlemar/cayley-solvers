"""Combine complete arbitrary-interval and two-list extraction certificates."""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random

from general_sort import apply_to
from linear_error_sort import construct as build_sort, inversions
from refined_merge_sort import CertifiedMerge
from verify_refined_extraction import check


FOLDER = Path(__file__).parent


class CertifiedInterval:
    def __init__(self, filename):
        path = FOLDER / filename
        self.audit = check(path)
        assert self.audit['complete'], 'Incomplete extraction tree'
        data = json.loads(path.read_text())
        self.coefficient = Fraction(data['defect_per_extracted_token'])
        self.width = self.audit['maximum_width']
        self.kmax = self.audit['maximum_selected']
        self.extra = self.audit['maximum_helper_charge']
        self.table = {(r['k'], tuple(r['pattern'])): r for r in data['records']}

    def sort(self, at, move, start, m):
        original = [at(i) for i in range(start, start + m)]
        inv = inversions(original)
        first, left, cost, charged = start, m, 0, 0
        while left > 1:
            active = [at(i) for i in range(first, first + left)]
            if active == sorted(active):
                break
            span = max(left, self.width)
            helpers = [at(i) for i in range(first + left, first + span)]
            before = active + helpers
            target = sorted(active) + helpers
            for k in range(3, self.kmax + 1):
                marked = set(target[:k])
                order = [at(i) for i in range(first, first + span) if at(i) in marked]
                previous = first - 1
                for value in order:
                    pos = next(i for i in range(previous + 1, first + span) if at(i) == value)
                    while pos - previous - 1 >= 3:
                        move(pos - 3, -1)
                        cost += 1
                        pos -= 3
                    previous = pos
                labels = {v: i for i, v in enumerate(target[:k])}
                end = max(i for i in range(first, first + span) if at(i) in marked)
                pattern, filler = [], k
                for pos in range(first, end + 1):
                    value = at(pos)
                    if value in labels:
                        pattern.append(labels[value])
                    else:
                        pattern.append(filler)
                        filler += 1
                row = self.table[k, tuple(pattern)]
                if row['status'] == 'found':
                    assert row['width'] <= span
                    for pos, direction in row['word']:
                        move(first + pos, direction)
                        cost += 1
                    break
            else:
                raise AssertionError('Uncovered extraction')
            after = [at(i) for i in range(first, first + span)]
            assert after == target[:k] + [v for v in before if v not in marked]
            assert after[left:] == helpers
            charged += k
            used = min(k, left)
            first += used
            left -= used
        assert charged <= m + self.extra
        assert [at(i) for i in range(start, start + m)] == sorted(original)
        assert 3 * cost <= inv + self.coefficient * (m + self.extra)
        return cost, inv


def construct(p, interval, merge):
    n = len(p)
    assert n >= max(interval.width, merge.width)
    stage = 0
    def sort(at, move, start, m, baseline):
        nonlocal stage
        stage += 1
        return interval.sort(at, move, start, m) if stage == 1 else merge.sort(at, move, start, m)
    word, row = build_sort(p, interval_sorter=sort)
    c, d = interval.coefficient, merge.coefficient
    numerator = (n * n // 4 + (c + d) * n +
                 c * interval.extra + d * (merge.kmax - 3) + 14)
    assert len(word) <= numerator // 3
    assert apply_to(p, word) == list(range(n))
    row['combined_upper_bound'] = numerator // 3
    return word, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', default='adaptive_extraction_c7_4_pruned.json')
    args = parser.parse_args()
    interval = CertifiedInterval(args.interval)
    merge = CertifiedMerge('merge_extraction_c1_2_wide_trim.json')
    boundary = 0
    # Exhaust every short permutation, with helpers deliberately in a different
    # numerical order, to test that they are treated as formal labels only.
    for size in range(2, 8):
        for p in itertools.permutations(range(size)):
            data = list(p) + list(range(size, size + interval.width))[::-1]
            expected = sorted(p) + data[size:]
            def at(i):
                return data[i]
            def move(i, d):
                block = data[i:i + 4]
                data[i:i + 4] = block[d:] + block[:d]
            interval.sort(at, move, 0, size)
            assert data == expected
            boundary += 1
    rng = random.Random(20260930)
    minimum = max(interval.width, merge.width)
    rows, count = [], 0
    for n in list(range(minimum, 151)) + [250, 500, 1000]:
        candidates = [list(range(n))[::-1], [(i + n // 2) % n for i in range(n)]]
        for _ in range(8):
            p = list(range(n))
            rng.shuffle(p)
            candidates.append(p)
        for p in candidates:
            _, row = construct(p, interval, merge)
            count += 1
        rows.append(row)
    c, d = interval.coefficient, merge.coefficient
    result = {'interval_certificate': interval.audit, 'merge_certificate': merge.audit,
              'helper_boundary_checks': boundary, 'full_word_replays': count,
              'sizes': f'{minimum}..150;250;500;1000',
              'linear_coefficient': str((c + d) / 3),
              'constant': str((c * interval.extra + d * (merge.kmax - 3) + 14) / 3),
              'last_samples': rows[-3:]}
    output = FOLDER / ('combined_sort_checks_c' + str(c).replace('/', '_') + '.json')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
