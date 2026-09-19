"""Independently replay the all-size noncentral K-3 constructor."""
from collections import Counter
import json
from pathlib import Path
import random

from third_layer_noncentral_construct import construct
from verify_two_defect_certificate import build, replay


def main():
    rng = random.Random(461)
    routes = Counter()
    moves = 0
    rows = json.loads(Path(__file__).with_name('third_layer_noncentral_words.json').read_text())
    models = []
    for row in rows:
        if row['n'] % 2:
            d = row['transposition_separation']
            models.append(((1, 0), (d - 1, row['n'] - d - 1), row['a']))
        else:
            models.append((row['core'], row['gaps'], row['a']))
    # Both signs of the noncentral offset, all four defect-two cores,
    # and cases whose only long gap is in each possible cyclic position.
    for n in tuple(range(14, 55, 2)) + (100, 150):
        for core in ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0)):
            t = len(core)
            for a in (n // 2 - 1, n // 2 + 1):
                for i in range(t):
                    gaps = [0] * t
                    gaps[i] = n - t
                    models.append((core, gaps, a))
                support = [0] + sorted(rng.sample(range(1, n), t - 1))
                gaps = [(support[(i + 1) % t] - support[i]) % n - 1 for i in range(t)]
                models.append((core, gaps, a))
    for n in tuple(range(13, 50, 2)) + (101, 151):
        for a in ((n - 3) // 2, (n + 3) // 2):
            for d in sorted({1, 2, 3, n // 2, n - 3, n - 2, n - 1}):
                models.append(((1, 0), (d - 1, n - d - 1), a))
    for core, gaps, a in models:
        p = build(core, gaps, a)
        word, route = construct(core, gaps, a)
        replay(p, word)
        routes[route] += 1
        moves += len(word)
    report = {'complete_words_replayed': len(models), 'moves_replayed': moves,
              'construction_routes': dict(routes), 'largest_n': 151,
              'scope': 'Implementation checks; full finite coverage is audited separately.'}
    Path(__file__).with_name('third_layer_noncentral_construct_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
