"""Inflate strands in opposite rotation blocks, with explicit cyclic anchoring."""
import argparse
import json
from pathlib import Path


FOLDER = Path(__file__).parent


def target(n, separation):
    p = [(i + n // 2) % n for i in range(n)]
    p[0], p[separation] = p[separation], p[0]
    return p


def strand_data(p, word):
    n = len(p)
    state = list(p)
    displacement = [0] * n
    travel = [0] * n
    crossings = {}
    for start, direction in word:
        assert 0 <= start < n and direction in (-1, 1)
        indices = [(start + j) % n for j in range(4)]
        old = [state[i] for i in indices]
        long = old[0] if direction == 1 else old[3]
        short = old[1:] if direction == 1 else old[:3]
        displacement[long] += 3 * direction
        travel[long] += 3
        for x in short:
            displacement[x] -= direction
            travel[x] += 1
            pair = tuple(sorted((long, x)))
            crossings[pair] = crossings.get(pair, 0) + 1
        for j, i in enumerate(indices):
            state[i] = old[(j + direction) % 4]
    assert state == list(range(n))
    return displacement, travel, crossings


def good_pairs(n, separation, word):
    p = target(n, separation)
    displacement, travel, crossings = strand_data(p, word)
    m = n // 2
    return [(u, v) for u in range(m) for v in range(m + separation + 1, n)
            if travel[u] == travel[v] == m
            and displacement[u] == -displacement[v]
            and abs(displacement[u]) == m
            and crossings.get((u, v), 0) == 1]


def inflate(p, word, selected):
    """Replace each selected token by four ordered clones; replay all moves."""
    n = len(p)
    selected = set(selected)
    weights = [4 if x in selected else 1 for x in range(n)]
    first = []
    total = 0
    for weight in weights:
        first.append(total)
        total += weight
    N = total
    old = list(p)
    initial = [first[x] + j for x in p for j in range(weights[x])]
    state = initial[:]
    positions = [0] * N
    for i, x in enumerate(state):
        positions[x] = i
    result = []

    def move(start, direction):
        start %= N
        indices = [(start + j) % N for j in range(4)]
        entries = [state[i] for i in indices]
        for j, i in enumerate(indices):
            value = entries[(j + direction) % 4]
            state[i] = value
            positions[value] = i
        result.append((start, direction))

    for start, direction in word:
        indices = [(start + j) % n for j in range(4)]
        labels = [old[i] for i in indices]
        expanded_start = positions[first[labels[0]]]
        expected = [first[x] + j for x in labels for j in range(weights[x])]
        assert [state[(expanded_start + j) % N] for j in range(len(expected))] == expected
        if direction == 1:
            a = weights[labels[0]]
            b = sum(weights[x] for x in labels[1:])
            assert b % 3 == 0
            for triple in range(b // 3):
                for j in range(a - 1, -1, -1):
                    move(expanded_start + 3 * triple + j, 1)
        else:
            a = sum(weights[x] for x in labels[:3])
            b = weights[labels[3]]
            assert a % 3 == 0
            for triple in range(a // 3 - 1, -1, -1):
                for j in range(b):
                    move(expanded_start + 3 * triple + j, -1)
        for j, i in enumerate(indices):
            old[i] = labels[(j + direction) % 4]
    assert old == list(range(n))
    origin = positions[0]
    assert state == [(i - origin) % N for i in range(N)]
    return {'initial': initial, 'word': result, 'origin': origin, 'first': first}


def find_bases():
    from near_rotation_search import build_endgame, solve
    source = json.loads((FOLDER / 'twisted_exchange_checks.json').read_text())
    previous = source['cyclic_exception_tests'][:]
    if (FOLDER / 'twist_inflation_small_bases.json').exists():
        previous += json.loads((FOLDER / 'twist_inflation_small_bases.json').read_text())
    rows = []
    endgames = {}
    for n, separation in ((8, 2), (10, 1), (10, 2)):
        p = target(n, separation)
        earlier = next((row for row in previous if row['permutation'] == p), None)
        chosen = None
        for seed in [-1, 0, 7, 13, 31, 3, 11, 23, 37, 41, 5, 17]:
            if seed < 0:
                if earlier is None:
                    continue
                row = earlier
            else:
                if n not in endgames:
                    endgames[n] = build_endgame(n, 4, 2000000)
                row = solve(p, (n * n // 4 - 1) // 3, endgames[n], 4, 500000, seed=seed)
            if row['status'] == 'found':
                pairs = good_pairs(n, separation, row['word'])
                print(json.dumps({'n': n, 'separation': separation, 'seed': seed,
                                  'pair_count': len(pairs), 'selected_pair': pairs[:1],
                                  'nodes': row.get('nodes')}), flush=True)
                if pairs:
                    chosen = {**row, 'separation': separation, 'selected_pair': pairs[0], 'seed': seed}
                    rows.append(chosen)
                    break
        if chosen is None:
            print('NO QUALIFYING WORD', n, separation, flush=True)
    (FOLDER / 'twist_inflation_bases.json').write_text(json.dumps(rows, indent=2))
    return rows


def verify(rows, max_n):
    report = []
    for row in rows:
        n, separation = row['n'], row['separation']
        word = row['word']
        pair = tuple(row['selected_pair'])
        initial_n = n
        checks = 0
        while n <= max_n:
            p = target(n, separation)
            assert len(word) == (n * n // 4 - 1) // 3
            assert pair in good_pairs(n, separation, word)
            checks += 1
            enlarged = inflate(p, word, pair)
            assert enlarged['initial'] == target(n + 6, separation)
            assert enlarged['origin'] == 0
            assert len(enlarged['word']) == len(word) + n + 3
            pair = tuple(enlarged['first'][x] for x in pair)
            word = enlarged['word']
            n += 6
        report.append({'base_n': initial_n, 'separation': separation, 'verified_sizes': checks,
                       'largest_size_replayed': n, 'largest_word_length': len(word)})
    result = {'complete_exceptional_family_coverage': {(r['n'], r['separation']) for r in rows}
              == {(8, 2), (10, 1), (10, 2)}, 'families': report,
              'scope': 'Finite replay checks of the strand-inflation induction.'}
    (FOLDER / 'twist_inflation_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--find-bases', action='store_true')
    parser.add_argument('--max-n', type=int, default=100)
    args = parser.parse_args()
    rows = find_bases() if args.find_bases else json.loads((FOLDER / 'twist_inflation_bases.json').read_text())
    verify(rows, args.max_n)


if __name__ == '__main__':
    main()
