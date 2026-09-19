"""Defect-sensitive wrapped-four sorting via crossing deletion from a rotation."""
from collections import deque
from pathlib import Path
import json

from affine_insertion import compile_run
from general_sort import apply_to
from rotation_ascent import balanced, potential
from rotation_bounds import exchange


def adjacent(keys, edge):
    n = len(keys)
    edge %= n
    other = (edge + 1) % n
    shift = n if other == 0 else 0
    keys[edge], keys[other] = keys[other] + shift, keys[edge] - shift


def apply_four(keys, start, direction):
    n = len(keys)
    edges = range(3) if direction == 1 else range(2, -1, -1)
    for offset in edges:
        adjacent(keys, start + offset)


def partial_generator_words():
    """All sixteen punctured generators as exact six-position identities."""
    n = 6
    identity = tuple(range(n))
    queue, words = deque([identity]), {identity: []}
    while queue:
        p = queue.popleft()
        for start in range(n - 3):
            for direction in (-1, 1):
                out = list(p)
                apply_four(out, start, direction)
                other = tuple(out)
                if other not in words:
                    words[other] = words[p] + [(start, direction)]
                    queue.append(other)
    assert len(words) == 720
    result = {}
    for direction in (-1, 1):
        order = list(range(3)) if direction == 1 else list(range(2, -1, -1))
        for mask in range(8):
            p = list(identity)
            for j, edge in enumerate(order):
                if mask & (1 << j):
                    adjacent(p, edge)
            word = words[tuple(p)]
            missing = 3 - mask.bit_count()
            assert len(word) <= 1 + 3 * missing
            assert apply_to(list(identity), word) == p
            result[direction, mask] = word
    return result


PARTIAL = partial_generator_words()


def ascent_and_colors(p):
    n = len(p)
    w = balanced(p)
    ell = potential(w)
    spins = [x - i for i, x in enumerate(w)]
    a = max(max(spins), min(n // 2, min(spins) + n))
    assert 0 <= a < n
    b = n - a
    f = [x + b for x in w]
    marked = [i for i, x in enumerate(f) if i < x < i + n]
    fixed = [i for i in range(n) if i not in marked]
    original = f[:]
    steps = []
    while True:
        inside = [i for i, x in enumerate(f) if i < x < i + n]
        if not inside:
            break
        i = inside[0]
        A = f[i]
        def at(j):
            return f[j % n] + n * (j // n)
        j = next(j for j in range(i + 1, A + 1) if A < at(j) <= i + n)
        B = at(j)
        assert j < B < j + n, 'Both changed residues must be interior'
        increment = 1 + 2 * sum(A < at(k) < B for k in range(i + 1, j))
        steps.append((i, j, A, B, increment))
        f[i], f[j % n] = B, A - n * (j // n)
    T = len(steps)
    d = a * b - ell
    assert sum(row[4] for row in steps) == d and T <= d
    lower_marked = {i for i in marked if f[i] == i}
    e = len(lower_marked)
    assert max(e, len(marked) - e) <= T
    needed = a - e
    assert 0 <= needed <= len(fixed)

    # Choose a block of `needed` consecutive unmarked anchors. Each anchor
    # owns itself and the following marked gap. Average mismatches <= T.
    best = None
    if fixed:
        owner = {}
        for index, start in enumerate(fixed):
            end = fixed[(index + 1) % len(fixed)]
            if end <= start:
                end += n
            for position in range(start, end):
                owner[position % n] = index
        for first in range(len(fixed)):
            selected = {(first + j) % len(fixed) for j in range(needed)}
            wrong = sum((owner[i] in selected) != (i in lower_marked) for i in marked)
            if best is None or wrong < best[0]:
                best = (wrong, selected)
        wrong, selected = best
        assert wrong <= T
        for index, i in enumerate(fixed):
            value = i if index in selected else i + n
            original[i] = f[i] = value
    else:
        wrong = 0
    colors = [f[i] == i for i in range(n)]  # True = negative displacement.
    runs = sum(colors[i] != colors[(i + 1) % n] for i in range(n))
    assert runs <= 2 * T + 2
    assert sum(colors) == a
    initial = [x - b for x in original]
    root = [x - b for x in f]
    assert [x % n for x in initial] == list(p)
    assert sum(initial) == n * (n - 1) // 2 and potential(initial) == ell
    assert potential(root) == a * b
    return initial, root, steps, {'n': n, 'rotation': a, 'adjacent_length': ell,
                                  'rotation_defect': d, 'ascents': T,
                                  'marked': len(marked), 'color_runs': runs,
                                  'miscolored_markers': wrong}


def rotation_word(root, a):
    n, b = len(root), len(root) - a
    if not a or not b:
        assert root == list(range(n))
        return [], {'block_exchanges': 0, 'singleton_pair_exchanges': 0, 'root_charge': 0}
    negative = [root[i] == i - b for i in range(n)]
    cut = next(i for i in range(n) if negative[i] and not negative[(i - 1) % n])
    keys = [root[(i + cut) % n] + n * ((i + cut) // n) - cut for i in range(n)]
    initial = keys[:]
    word = []
    exchanges = singleton_pairs = adjacent_work = 0
    def at(i):
        return keys[i % n] + n * (i // n)
    def append(local):
        for start, direction in local:
            apply_four(keys, start, direction)
            word.append((start % n, direction))
    def swap_blocks(start, left, right):
        nonlocal exchanges, singleton_pairs, adjacent_work
        if not left or not right:
            return
        before = [at(i) for i in range(start, start + left + right)]
        count = len(word)
        if left >= 2 and right >= 2:
            local = exchange(left, right, start)
        elif left == 1:
            local = compile_run(start, 1, right, n)
        else:
            local = compile_run(start + left, -1, left, n)
        append(local)
        after = [at(i) for i in range(start, start + left + right)]
        assert after == before[left:] + before[:left]
        charge = 3 * (len(word) - count) - left * right
        is_pair = left == right == 1
        assert charge <= (14 if is_pair else 10)
        exchanges += 1
        singleton_pairs += is_pair
        adjacent_work += left * right

    # Stable partition into negative and positive displacement classes.
    fixed = 0
    while fixed < n and keys[fixed] < a:
        fixed += 1
    while fixed < a:
        end = fixed
        while keys[end] >= a:
            end += 1
        low_end = end
        while low_end < n and keys[low_end] < a:
            low_end += 1
        swap_blocks(fixed, end - fixed, low_end - end)
        fixed += low_end - end
    assert keys == sorted(initial)
    largest = keys[-1]
    assert largest == n - 1 + a
    pivot_start = len(word)
    append(compile_run(n - 1, 1, a, n))
    assert 3 * (len(word) - pivot_start) <= a + 14
    pivot = a - 1
    assert keys[pivot] == pivot

    # Merge the positive list (minus the pivot) and the negative list.
    start, left, right = a, b - 1, a
    while left and right:
        if at(start) < at(start + left):
            stop = 1
            while stop < left and at(start + stop) < at(start + left):
                stop += 1
            start += stop
            left -= stop
        else:
            take = 1
            while take < right and at(start + left + take) < at(start):
                take += 1
            swap_blocks(start, left, take)
            start += take
            right -= take
    assert keys == list(range(n))
    assert adjacent_work + a == a * b
    m = sum(negative[i] and not negative[(i - 1) % n] for i in range(n))
    assert exchanges <= 2 * m - 1 and singleton_pairs <= 2
    charge = 3 * len(word) - a * b
    assert charge <= 20 * m + 12
    shifted = [((start + cut) % n, direction) for start, direction in word]
    check = root[:]
    for start, direction in shifted:
        apply_four(check, start, direction)
    assert check == list(range(n))
    return shifted, {'block_exchanges': exchanges, 'singleton_pair_exchanges': singleton_pairs,
                     'root_charge': charge, 'negative_runs': m,
                     'root_adjacent_work': adjacent_work + a}


def construct(p, check_each_deletion=False):
    n = len(p)
    assert n >= 6
    initial, upper, steps, row = ascent_and_colors(p)
    word, root_row = rotation_word(upper, row['rotation'])
    letters = []
    for group, (start, direction) in enumerate(word):
        for offset in (range(3) if direction == 1 else range(2, -1, -1)):
            letters.append((group, (start + offset) % n))
    removed = set()
    b = n - row['rotation']
    for i, j, A, B, increment in reversed(steps):
        assert upper[i] == B - b and upper[j % n] + n * (j // n) == A - b
        lo, hi = A - b, B - b
        moving = upper[:]
        found = None
        for index, (_, edge) in enumerate(letters):
            if index in removed:
                continue
            x, y = moving[edge], moving[(edge + 1) % n] + (n if edge == n - 1 else 0)
            same = (x - lo == y - hi and (x - lo) % n == 0)
            opposite = (x - hi == y - lo and (x - hi) % n == 0)
            if same or opposite:
                found = index
                break
            adjacent(moving, edge)
        assert found is not None, 'The reversed pair of lifted strands must cross'
        removed.add(found)
        upper[i], upper[j % n] = A - b, B - b - n * (j // n)
        if check_each_deletion:
            test = upper[:]
            for index, (_, edge) in enumerate(letters):
                if index not in removed:
                    adjacent(test, edge)
            assert test == list(range(n))
    assert upper == initial and len(removed) == len(steps)
    result = []
    for group, (start, direction) in enumerate(word):
        mask = sum(1 << j for j in range(3) if 3 * group + j not in removed)
        result.extend(((start + local) % n, sign) for local, sign in PARTIAL[direction, mask])
    test = initial[:]
    for start, direction in result:
        apply_four(test, start, direction)
    assert test == list(range(n))
    assert apply_to(p, result) == list(range(n))
    assert len(result) <= len(word) + 3 * len(steps)
    assert 3 * len(result) <= row['rotation'] * b + 29 * len(steps) + 32
    deficit = n * n // 4 - row['adjacent_length']
    assert 3 * len(result) <= n * n // 4 + 29 * deficit + 32
    row.update(root_row, word_length=len(result), root_word_length=len(word),
               deleted_crossings=len(removed), adjacent_deficit=deficit)
    return result, row


def write_local_certificate():
    rows = []
    for (direction, mask), word in sorted(PARTIAL.items()):
        rows.append({'direction': direction, 'mask': mask, 'word': word,
                     'missing_adjacent_letters': 3 - mask.bit_count()})
    Path(__file__).with_name('punctured_generator_words.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    import random
    rng = random.Random(20261103)
    write_local_certificate()
    rows = []
    for n in (6, 7, 8, 9, 12, 15, 20, 32, 50, 100):
        for trial in range(8):
            p = list(range(n))
            rng.shuffle(p)
            _, row = construct(p, check_each_deletion=True)
            rows.append(row)
        print(json.dumps({'n': n, 'words_checked': len(rows), 'last': rows[-1]}), flush=True)
    Path(__file__).with_name('ascent_word_deletion_pilot.json').write_text(json.dumps(rows, indent=2))
