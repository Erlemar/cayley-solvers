"""Replay complete inflated words, including large and very uneven gap sizes."""
import json
from pathlib import Path
import random

from two_defect_construct import construct
from two_defect_roots import root_models
from verify_two_defect_certificate import build, replay


def main():
    checked = moves = branches = 0
    for alpha, gaps, p in root_models():
        word, refined = construct(alpha, gaps)
        replay(p, word)
        checked += 1
        moves += len(word)
        branches += refined
    base_checks = checked
    rng = random.Random(20260920)
    patterns = ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0))
    for n in (24, 25, 26, 27, 28, 29, 30, 35, 50, 75, 100, 150):
        for _ in range(25):
            alpha = rng.choice(patterns)
            t = len(alpha)
            support = [0] + sorted(rng.sample(range(1, n), t - 1))
            gaps = [(support[(i + 1) % t] - support[i]) % n - 1 for i in range(t)]
            a = rng.choice((n // 2, (n + 1) // 2))
            word, refined = construct(alpha, gaps, a)
            replay(build(alpha, gaps, a), word)
            checked += 1
            moves += len(word)
            branches += refined
    # Every support pattern of two added triples at either exceptional root,
    # with both moderate and very large multiplicities.
    for seed_gaps, alpha in (((2, 2, 2, 2), (3, 2, 1, 0)), ((9, 0, 2, 0), (3, 2, 1, 0))):
        active = [i for i, g in enumerate(seed_gaps) if g >= 2]
        for first in active:
            for second in active:
                for multiplier in (1, 2, 10):
                    gaps = list(seed_gaps)
                    gaps[first] += 3 * multiplier
                    gaps[second] += 3 * multiplier
                    n = len(alpha) + sum(gaps)
                    word, refined = construct(alpha, gaps)
                    replay(build(alpha, gaps, n // 2), word)
                    checked += 1
                    moves += len(word)
                    branches += refined
    report = {'complete_words_replayed': checked, 'initial_root_model_checks': base_checks,
              'larger_random_checks': 300, 'exceptional_growth_checks': 60,
              'refinement_steps_exercised': branches, 'four_cycle_moves_replayed': moves,
              'largest_random_n': 150,
              'scope': 'Checks of the explicit all-size constructor; the finite coverage audit is separate.'}
    Path(__file__).with_name('two_defect_construct_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
