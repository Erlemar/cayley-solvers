"""Construct full cyclic sorting words from a COMPLETE charged-tail certificate."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import random

from combined_extraction_sort import CertifiedInterval
from general_sort import apply_to
from linear_error_sort import construct as build_sort, inversions
from refined_merge_sort import CertifiedMerge
from verify_partial_extraction import check


FOLDER = Path(__file__).parent


class PartialInterval:
    def __init__(self, filename):
        path = FOLDER / filename
        self.audit = check(path)
        assert self.audit['complete'], 'An incomplete tree cannot prove a universal upper bound'
        data = json.loads(path.read_text())
        self.coefficient = Fraction(data['coefficient'])
        assert 0 <= self.coefficient <= Fraction(7, 4)
        self.width = self.audit['maximum_width']
        self.kmax = self.audit['maximum_selected']
        self.table = {(r['k'], tuple(r['pattern'])): r for r in data['records']}
        self.finish = CertifiedInterval('adaptive_extraction_c7_4_pruned.json')
        self.constant = (Fraction(7, 4) - self.coefficient) * (self.width - 1) + Fraction(21, 2)
        self.batches = 0
        self.changed_tails = 0

    def sort(self, at, move, start, m):
        original = [at(i) for i in range(start, start + m)]
        initial_inv = inversions(original)
        first, left, cost = start, m, 0
        while left >= self.width:
            active = [at(i) for i in range(first, first + left)]
            target = sorted(active)
            if active == target:
                break
            for k in range(3, self.kmax + 1):
                selected = set(target[:k])
                order = [value for value in [at(i) for i in range(first, first + left)] if value in selected]
                previous = first - 1
                for value in order:
                    position = next(i for i in range(previous + 1, first + left) if at(i) == value)
                    while position - previous - 1 >= 3:
                        move(position - 3, -1)
                        cost += 1
                        position -= 3
                    previous = position
                labels = {value: i for i, value in enumerate(target[:k])}
                end = max(i for i in range(first, first + left) if at(i) in selected)
                pattern, filler = [], k
                for i in range(first, end + 1):
                    value = at(i)
                    if value in labels:
                        pattern.append(labels[value])
                    else:
                        pattern.append(filler)
                        filler += 1
                row = self.table[k, tuple(pattern)]
                if row['status'] == 'found':
                    assert row['width'] <= left
                    before = [at(i) for i in range(first, first + row['width'])]
                    for pos, direction in row['word']:
                        move(first + pos, direction)
                        cost += 1
                    after = [at(i) for i in range(first, first + row['width'])]
                    assert after[:k] == target[:k] and sorted(before) == sorted(after)
                    assert 3 * len(row['word']) <= inversions(before) - inversions(after) + self.coefficient * k
                    self.batches += 1
                    self.changed_tails += row['tail_inversions'] > 0
                    first += k
                    left -= k
                    break
            else:
                raise AssertionError('Uncovered partial extraction')
        if left > 1:
            tail_cost, _ = self.finish.sort(at, move, first, left)
            cost += tail_cost
        assert [at(i) for i in range(start, start + m)] == sorted(original)
        assert 3 * cost <= initial_inv + self.coefficient * m + self.constant
        return cost, initial_inv


def construct(p, interval, merge):
    n = len(p)
    assert n >= max(32, interval.width, merge.width)
    stage = 0
    def callback(at, move, start, m, baseline):
        nonlocal stage
        stage += 1
        return interval.sort(at, move, start, m) if stage == 1 else merge.sort(at, move, start, m)
    word, row = build_sort(p, interval_sorter=callback)
    c = interval.coefficient
    constant = (Fraction(7, 4) - c) * (interval.width - 1) + 37
    numerator = row['adjacent_length'] + (c + Fraction(1, 2)) * n + constant
    assert 3 * len(word) <= numerator
    assert apply_to(p, word) == list(range(n))
    row['partial_extraction_upper_bound'] = int((n * n // 4 + (c + Fraction(1, 2)) * n + constant) // 3)
    return word, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('certificate')
    args = parser.parse_args()
    interval = PartialInterval(args.certificate)
    merge = CertifiedMerge('merge_extraction_c1_2_wide_trim.json')
    rng = random.Random(654321)
    minimum = max(32, interval.width, merge.width)
    helpers = 0
    for m in (interval.width, interval.width + 1, 2 * interval.width, 3 * interval.width):
        for _ in range(30):
            active = list(range(m))
            rng.shuffle(active)
            outside = list(range(-max(32, interval.width), 0))[::-1]
            state = active + outside
            expected = sorted(active) + outside
            def at(i):
                return state[i]
            def move(i, d):
                block = state[i:i + 4]
                state[i:i + 4] = block[d:] + block[:d]
            interval.sort(at, move, 0, m)
            assert state == expected
            helpers += 1
    words = moves = 0
    samples = []
    for n in list(range(minimum, 101)) + [150, 300, 1000]:
        candidates = [list(range(n))[::-1], [(i + n // 2) % n for i in range(n)]]
        if n % 2 == 0:
            candidates.append([(i + n // 2 + (1 if i % 2 == 0 else -1)) % n for i in range(n)])
        for _ in range(3):
            p = list(range(n))
            rng.shuffle(p)
            candidates.append(p)
        for p in candidates:
            word, row = construct(p, interval, merge)
            words += 1
            moves += len(word)
        if n >= 150:
            samples.append(row)
    c = interval.coefficient
    constant = (Fraction(7, 4) - c) * (interval.width - 1) + 37
    report = {'certificate': interval.audit, 'complete_sorting_words_replayed': words,
              'full_word_moves_replayed': moves, 'arbitrarily_ordered_helper_checks': helpers,
              'partial_batches_exercised': interval.batches,
              'nonidentity_tail_batches_exercised': interval.changed_tails,
              'minimum_n': minimum, 'largest_n': 1000,
              'diameter_linear_coefficient': str((c + Fraction(1, 2)) / 3),
              'diameter_constant': str(constant / 3), 'last_samples': samples,
              'scope': 'Complete finite certificate plus full constructive implementation checks.'}
    output = FOLDER / (Path(args.certificate).stem + '_sort_checks.json')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
