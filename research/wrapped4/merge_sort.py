"""Independently verify and apply the c=1 two-list extraction certificate."""
import itertools
import json
from pathlib import Path
import random

from adaptive_sort import sort_interval_padded
from general_sort import apply_to
from linear_error_sort import construct as build_sort, inversions


FOLDER = Path(__file__).parent
DATA = json.loads((FOLDER / 'merge_extraction_c1.json').read_text())
TABLE = {(r['bits'], r['gap']): r for r in DATA['records']}


def independent_certificate_check():
    assert DATA['complete'] and DATA['coefficient'] == '1'
    assert len(TABLE) == len(DATA['records'])
    expected = {(''.join(bits), gap)
                for bits in itertools.product('AB', repeat=3)
                for gap in (0, 1, 2)}
    seen = set()
    summary = []
    for k in range(3, 15):
        actual = {key for key in TABLE if len(key[0]) == k}
        assert expected == actual
        next_expected = set()
        successes = 0
        for bits, gap in actual:
            seen.add((bits, gap))
            row = TABLE[bits, gap]
            # Build the abstract list without calling the discovery code.
            a = []
            b = []
            for label, source in enumerate(bits):
                (a if source == 'A' else b).append(label)
            p = a + list(range(k, k + gap)) + b
            assert p == row['pattern']
            assert row['width'] == max(9, len(p)) <= 16
            p += list(range(len(p), row['width']))
            initial_inv = sum(p[i] > p[j] for i in range(len(p))
                              for j in range(i + 1, len(p)))
            assert initial_inv == row['inversions']
            if row['status'] == 'found':
                for start, direction in row['word']:
                    assert 0 <= start <= len(p) - 4
                    assert direction in (-1, 1)
                    window = p[start:start + 4]
                    p[start:start + 4] = (window[1:] + window[:1] if direction == 1
                                          else window[-1:] + window[:-1])
                assert p == list(range(len(p)))
                assert 3 * len(row['word']) - initial_inv <= k
                successes += 1
            else:
                next_expected.add((bits + 'A', (gap + 2) % 3))
                next_expected.add((bits + 'B', gap))
        summary.append({'k': k, 'cases': len(actual), 'words': successes})
        expected = next_expected
    assert not expected and seen == set(TABLE)
    return {'complete': True, 'cases': len(seen), 'stages': summary,
            'maximum_selected': 14, 'buffer_width': 16}


def sort_merge(at, move, start, m):
    original = [at(i) for i in range(start, start + m)]
    initial_inv = inversions(original)
    first = start
    left = m
    cost = charged = 0
    while left > 1:
        active = [at(i) for i in range(first, first + left)]
        descents = [i + 1 for i in range(left - 1) if active[i] > active[i + 1]]
        assert len(descents) <= 1, 'Input must be two increasing lists'
        if not descents:
            break
        split = descents[0]
        a_values = set(active[:split])
        helpers = [at(i) for i in range(first + left, first + max(left, 16))]
        target = sorted(active) + helpers
        for k in range(3, 15):
            bits = ''.join('A' if value in a_values else 'B' for value in target[:k])
            a = bits.count('A')
            b = k - a
            gap = split - a
            row = TABLE[bits, gap % 3]
            if row['status'] == 'found':
                break
        else:
            raise AssertionError('Certificate did not cover a merge')
        before = active + helpers
        selected = set(target[:k])
        untouched = [v for v in before if v not in selected]
        for offset in range(b):
            pos = first + split + offset
            for _ in range(gap // 3):
                move(pos - 3, -1)
                cost += 1
                pos -= 3
        selected_index = {v: i for i, v in enumerate(target[:k])}
        artificial = []
        filler = k
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
        after = [at(i) for i in range(first, first + max(left, 16))]
        assert after[:k] == target[:k]
        assert after[k:] == untouched
        assert after[left:] == helpers
        consumed = min(k, left)
        charged += k
        first += consumed
        left -= consumed
    assert charged <= m + 12
    assert [at(i) for i in range(start, start + m)] == sorted(original)
    assert 3 * cost <= initial_inv + m + 12
    return cost, initial_inv


def construct(p):
    n = len(p)
    assert n >= 16
    stage = 0
    def interval(at, move, start, m, baseline):
        nonlocal stage
        stage += 1
        if stage == 1:
            return sort_interval_padded(at, move, start, m, baseline, 14)
        assert stage == 2
        return sort_merge(at, move, start, m)
    word, row = build_sort(p, interval_sorter=interval)
    bound = (n * n // 4 + 3 * n + 35) // 3
    assert len(word) <= bound
    assert apply_to(p, word) == list(range(n))
    row['merge_upper_bound'] = bound
    return word, row


def check_all_small_merges():
    count = 0
    for size in range(2, 13):
        for mask in range(1 << size):
            data = ([i for i in range(size) if mask >> i & 1]
                    + [i for i in range(size) if not mask >> i & 1]
                    + list(range(size, size + 16)))
            def at(i):
                return data[i]
            def move(i, d):
                block = data[i:i + 4]
                data[i:i + 4] = block[d:] + block[:d]
            sort_merge(at, move, 0, size)
            assert data == list(range(size + 16))
            count += 1
    return count


def main():
    result = {'certificate': independent_certificate_check()}
    result['all_small_merges'] = check_all_small_merges()
    rng = random.Random(20260925)
    rows = []
    checked = 0
    for n in list(range(16, 151)) + [250, 500, 1000]:
        inputs = [list(range(n))[::-1], [(i + n // 2) % n for i in range(n)]]
        for _ in range(10):
            p = list(range(n))
            rng.shuffle(p)
            inputs.append(p)
        for p in inputs:
            _, row = construct(p)
            checked += 1
        rows.append(row)
    result['full_words_verified'] = checked
    result['sizes'] = '16..150;250;500;1000'
    result['last_samples'] = rows[-3:]
    result['bound_n_ge_16'] = 'floor((floor(n^2/4)+3n+35)/3)'
    (FOLDER / 'merge_sort_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
