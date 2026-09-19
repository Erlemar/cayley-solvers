"""Standard-library audit, independent of the 4-cycle search implementation."""
from itertools import combinations
import argparse
import json
from pathlib import Path


def adjacent_length(p):
    n = len(p)
    w = list(p)
    while True:
        d = [x - i for i, x in enumerate(w)]
        low, high = min(d), max(d)
        if high - low <= n:
            break
        w[d.index(high)] -= n
        w[d.index(low)] += n
    return sum(abs((w[j] - w[i]) // n) for i in range(n) for j in range(i + 1, n))


def replay(p, word):
    q = list(p)
    n = len(q)
    for start, direction in word:
        assert type(start) is int and 0 <= start < n
        assert direction in (-1, 1)
        a, b, c, d = [(start + offset) % n for offset in range(4)]
        if direction == 1:
            q[a], q[b], q[c], q[d] = q[b], q[c], q[d], q[a]
        else:
            q[a], q[b], q[c], q[d] = q[d], q[a], q[b], q[c]
    assert q == list(range(n))


def candidate_layer(n, deficit):
    """Use the proved ascent theorem, then filter; no core generator imported."""
    target = n * n // 4 - deficit
    all_candidates = set()
    for a in range(n):
        deficit = a * (n - a) - target
        if deficit < 0:
            continue
        frontier = {tuple((i + a) % n for i in range(n))}
        all_candidates.update(frontier)
        for _ in range(deficit):
            children = set()
            for p in frontier:
                for i, j in combinations(range(n), 2):
                    q = list(p)
                    q[i], q[j] = q[j], q[i]
                    children.add(tuple(q))
            frontier = children
            all_candidates.update(frontier)
    return {p for p in all_candidates if adjacent_length(p) == target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deficit', type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    deficit = args.deficit
    stem = 'second_layer_four_n15' if deficit == 2 else f'adjacent_layer_d{deficit}_four_n15'
    path = Path(__file__).with_name(stem + '.json')
    data = json.loads(path.read_text())
    n = data['n']
    target = n * n // 4 - deficit
    lower = target // 3 + target % 3
    assert n == 15
    expected = candidate_layer(n, deficit)
    assert len(expected) == {2: 7282, 3: 126700}[deficit]
    covered = set()
    replay_count = 0
    for row in data['rows']:
        assert row['status'] == 'found'
        p = row['permutation']
        assert adjacent_length(p) == target
        assert len(row['word']) == lower
        inverse = [p.index(i) for i in range(n)]
        inverse_word = [(i, -d) for i, d in reversed(row['word'])]
        images = set()
        for q, word in ((p, row['word']), (inverse, inverse_word)):
            for sign in (-1, 1):
                for shift in range(n):
                    image = tuple((sign * q[(sign * (i - shift)) % n] + shift) % n
                                  for i in range(n))
                    image_word = [((i + shift) % n, d) if sign == 1
                                  else ((shift - i - 3) % n, -d) for i, d in word]
                    replay(image, image_word)
                    replay_count += 1
                    images.add(image)
        assert len(images) == row['orbit_size']
        assert not images & covered
        covered.update(images)
    assert covered == expected
    report = {'n': n, 'auxiliary_length': target, 'states': len(covered),
              'orbits': len(data['rows']), 'words_replayed_including_symmetries': replay_count,
              'proved_exact_four_cycle_length_of_every_state': lower,
              'all_layer_states_covered': True,
              'scope': f'Exact for the full layer ell={target} at n=15, not a diameter proof.'}
    path.with_name(stem + '_independent_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
