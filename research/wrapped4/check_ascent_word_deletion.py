"""Independent replay and small exhaustive checks of the all-size construction.

The mathematical proof is in DEFECT_SENSITIVE_UPPER_BOUND.md. These finite
checks validate its implementation; they are not extrapolated to larger n.
"""
from collections import deque
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import random
import time

from ascent_word_deletion import construct, rotation_word, write_local_certificate


def ordinary_replay(p, word):
    p = list(p)
    n = len(p)
    for start, sign in word:
        assert 0 <= start < n and sign in (-1, 1)
        indices = [(start + j) % n for j in range(4)]
        values = [p[i] for i in indices]
        for j, i in enumerate(indices):
            p[i] = values[(j + sign) % 4]
    return p


def integer_replay(p, word):
    p = list(p)
    n = len(p)
    for start, sign in word:
        places = list(range(start, start + 4))
        values = [p[i % n] + n * (i // n) for i in places]
        for j, i in enumerate(places):
            p[i % n] = values[(j + sign) % 4] - n * (i // n)
    return p


def adjacent_bfs(n):
    identity = tuple(range(n))
    distance, queue = {identity: 0}, deque([identity])
    while queue:
        p = queue.popleft()
        for i in range(n):
            q = list(p)
            q[i], q[(i + 1) % n] = q[(i + 1) % n], q[i]
            q = tuple(q)
            if q not in distance:
                distance[q] = distance[p] + 1
                queue.append(q)
    return distance


def main():
    started = time.monotonic()
    folder = Path(__file__).parent
    write_local_certificate()
    local = json.loads((folder / 'punctured_generator_words.json').read_text())
    assert len(local) == 16
    assert {(r['direction'], r['mask']) for r in local} == set(product((-1, 1), range(8)))
    for row in local:
        target = list(range(6))
        order = list(range(3)) if row['direction'] == 1 else list(range(2, -1, -1))
        for j, i in enumerate(order):
            if row['mask'] & (1 << j):
                target[i], target[i + 1] = target[i + 1], target[i]
        assert all(0 <= i <= 2 and sign in (-1, 1) for i, sign in row['word'])
        assert ordinary_replay(range(6), row['word']) == target
        assert len(row['word']) <= 1 + 3 * (3 - row['mask'].bit_count())
    root_count = root_moves = 0
    for n in range(6, 13):
        for bits in product((0, 1), repeat=n):
            a = sum(bits)
            if not a or a == n:
                continue
            b = n - a
            root = [i - b if bits[i] else i + a for i in range(n)]
            m = sum(bits[i] and not bits[(i - 1) % n] for i in range(n))
            word, row = rotation_word(root, a)
            assert integer_replay(root, word) == list(range(n))
            assert 3 * len(word) <= a * b + 20 * m + 12
            assert row['block_exchanges'] <= 2 * m - 1
            assert row['singleton_pair_exchanges'] <= 2
            root_count += 1
            root_moves += len(word)
    print(json.dumps({'root_lifts_checked': root_count, 'seconds': time.monotonic() - started}), flush=True)
    words = moves = deletions = 0
    for n in range(6, 9):
        exact = adjacent_bfs(n)
        for p, ell in exact.items():
            word, row = construct(p, check_each_deletion=True)
            assert ordinary_replay(p, word) == list(range(n))
            assert row['adjacent_length'] == ell
            assert 3 * len(word) <= n * n // 4 + 29 * (n * n // 4 - ell) + 32
            words += 1
            moves += len(word)
            deletions += row['deleted_crossings']
        print(json.dumps({'exhaustive_through_n': n, 'words': words,
                          'seconds': time.monotonic() - started}), flush=True)
    rng = random.Random(20261104)
    large_count = 0
    samples = []
    for n in (12, 15, 20, 32, 50, 100, 150, 300, 1000):
        trials = 12 if n <= 150 else 3
        for trial in range(trials):
            a = n // 2 + trial % 3 - 1
            p = [(i + a) % n for i in range(n)]
            for _ in range(trial % 7):
                i, j = rng.sample(range(n), 2)
                p[i], p[j] = p[j], p[i]
            word, row = construct(p, check_each_deletion=True)
            assert ordinary_replay(p, word) == list(range(n))
            large_count += 1
            moves += len(word)
            deletions += row['deleted_crossings']
            if n >= 300:
                samples.append(row)
    # Algebraic interpolation is checked over a broad integer grid, independently
    # of the newly constructed word (whose own length need not meet the other bound).
    interpolations = 0
    for n in range(32, 501):
        K = n * n // 4
        for deficit in range(K + 1):
            first = Fraction(9, 4) * n + 37 - deficit
            second = 29 * deficit + 32
            assert min(first, second) <= Fraction(87, 40) * n + Fraction(221, 6)
            interpolations += 1
    report = {'local_punctured_generator_identities': len(local),
              'all_binary_root_lifts_n6_through_n12': root_count,
              'binary_root_word_moves_replayed': root_moves,
              'exhaustive_full_permutations_n6_through_n8': words,
              'additional_near_rotation_words': large_count,
              'full_word_moves_replayed': moves,
              'individual_crossing_deletions_checked': deletions,
              'largest_n': 1000, 'interpolation_integer_pairs_checked': interpolations,
              'global_linear_coefficient': '29/40', 'global_constant_n_ge_32': '221/18',
              'large_samples': samples, 'seconds': time.monotonic() - started,
              'scope': 'Finite identities and implementation checks; the separate analytic proof establishes all n.'}
    (folder / 'ascent_word_deletion_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
