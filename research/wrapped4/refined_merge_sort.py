"""Apply any complete binary extraction certificate; replay whole sorting words."""
from fractions import Fraction
import json
from pathlib import Path
import random

from adaptive_sort import sort_interval_padded
from general_sort import apply_to
from linear_error_sort import construct as build_sort, inversions
from verify_merge_tree import check


FOLDER = Path(__file__).parent


class CertifiedMerge:
    def __init__(self, filename):
        path = FOLDER / filename
        self.audit = check(path)
        assert self.audit['complete'], 'An incomplete tree cannot prove an all-n bound'
        data = json.loads(path.read_text())
        self.table = {(r['bits'], r['gap']): r for r in data['records']}
        self.coefficient = Fraction(data['coefficient'])
        self.kmax = self.audit['maximum_selected']
        self.width = self.audit['maximum_width']

    def sort(self, at, move, start, m):
        original = [at(i) for i in range(start, start + m)]
        initial_inv = inversions(original)
        first, left, cost, charged = start, m, 0, 0
        while left > 1:
            active = [at(i) for i in range(first, first + left)]
            descents = [i + 1 for i in range(left - 1) if active[i] > active[i + 1]]
            assert len(descents) <= 1
            if not descents:
                break
            split = descents[0]
            a_values = set(active[:split])
            span = max(left, self.width)
            helpers = [at(i) for i in range(first + left, first + span)]
            target = sorted(active) + helpers
            for k in range(3, self.kmax + 1):
                bits = ''.join('A' if value in a_values else 'B' for value in target[:k])
                gap = split - bits.count('A')
                row = self.table[bits, gap % 3]
                if row['status'] == 'found':
                    break
            else:
                raise AssertionError('Uncovered merge')
            selected = set(target[:k])
            untouched = [v for v in active + helpers if v not in selected]
            for offset in range(bits.count('B')):
                pos = first + split + offset
                for _ in range(gap // 3):
                    move(pos - 3, -1)
                    cost += 1
                    pos -= 3
            selected_index = {v: i for i, v in enumerate(target[:k])}
            artificial, filler = [], k
            for pos in range(first, first + row['width']):
                value = at(pos)
                if value in selected_index:
                    artificial.append(selected_index[value])
                else:
                    artificial.append(filler)
                    filler += 1
            assert artificial == row['pattern'] + list(range(len(row['pattern']), row['width']))
            for pos, direction in row['word']:
                move(first + pos, direction)
                cost += 1
            after = [at(i) for i in range(first, first + span)]
            assert after[:k] == target[:k] and after[k:] == untouched
            assert after[left:] == helpers
            consumed = min(k, left)
            charged += k
            first += consumed
            left -= consumed
        assert charged <= m + self.kmax - 2
        assert [at(i) for i in range(start, start + m)] == sorted(original)
        assert 3 * cost <= initial_inv + self.coefficient * (m + self.kmax - 2)
        return cost, initial_inv

    def construct(self, p):
        n = len(p)
        assert n >= max(14, self.width)
        stage = 0
        def interval(at, move, start, m, baseline):
            nonlocal stage
            stage += 1
            if stage == 1:
                return sort_interval_padded(at, move, start, m, baseline, 14)
            assert stage == 2
            return self.sort(at, move, start, m)
        word, row = build_sort(p, interval_sorter=interval)
        bound = (Fraction(n * n // 4 + 2 * n + 24) +
                 self.coefficient * (n + self.kmax - 3)) // 3
        assert len(word) <= bound
        assert apply_to(p, word) == list(range(n))
        row['refined_upper_bound'] = bound
        return word, row


def main():
    sorter = CertifiedMerge('merge_extraction_c1_2_wide_trim.json')
    small = 0
    for size in range(2, 13):
        for mask in range(1 << size):
            data = ([i for i in range(size) if mask >> i & 1]
                    + [i for i in range(size) if not mask >> i & 1]
                    + list(range(size, size + sorter.width)))
            def at(i):
                return data[i]
            def move(i, d):
                block = data[i:i + 4]
                data[i:i + 4] = block[d:] + block[:d]
            sorter.sort(at, move, 0, size)
            assert data == list(range(size + sorter.width))
            small += 1
    rng = random.Random(20260929)
    count = 0
    rows = []
    for n in list(range(max(14, sorter.width), 151)) + [250, 500, 1000]:
        inputs = [list(range(n))[::-1], [(i + n // 2) % n for i in range(n)]]
        for _ in range(8):
            p = list(range(n))
            rng.shuffle(p)
            inputs.append(p)
        for p in inputs:
            _, row = sorter.construct(p)
            count += 1
        rows.append(row)
    result = {'certificate': sorter.audit, 'all_small_merges': small,
              'full_words_verified': count, 'sizes': f'{sorter.width}..150;250;500;1000',
              'last_samples': rows[-3:]}
    (FOLDER / 'refined_merge_sort_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
