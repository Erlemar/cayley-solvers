"""Independent replay of finite and arbitrarily inflated four-cycle-core words."""
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
import random

from four_cycle_core_construct import construct
from verify_four_cycle_core import is_four_cycle, model
from verify_two_defect_certificate import build, replay


def main():
    folder = Path(__file__).parent
    data = json.loads((folder / 'four_cycle_core_certificate.json').read_text())
    patterns = [p for p in permutations(range(4)) if is_four_cycle(p)]
    models = []
    for row in data['nodes']:
        a, alpha, spaces = model(row['permutation'])
        gaps = list(map(len, spaces))
        models.append((alpha, gaps, a))
        if row['children']:
            active = [i for i, g in enumerate(gaps) if g >= 2]
            for i in active:
                for j in active:
                    for factor in (1, 4, 10):
                        grown = gaps[:]
                        grown[i] += 3 * factor
                        grown[j] += 3 * factor
                        models.append((alpha, grown, a + 3 * factor))
    rng = random.Random(92837)
    for n in (24, 25, 26, 27, 28, 29, 30, 31, 32, 40, 75, 100, 151):
        for alpha in patterns:
            for _ in range(5):
                support = [0] + sorted(rng.sample(range(1, n), 3))
                gaps = [(support[(i + 1) % 4] - support[i]) % n - 1 for i in range(4)]
                a = rng.choice((n // 2, (n + 1) // 2))
                models.append((alpha, gaps, a))
    moves = refinements = 0
    excess = Counter()
    for alpha, gaps, a in models:
        word, refined = construct(alpha, gaps, a)
        p = build(alpha, gaps, a)
        replay(p, word)
        ell = len(p) ** 2 // 4 - 3
        lower = ell // 3 + ell % 3
        assert len(word) >= lower and (len(word) - lower) % 2 == 0
        moves += len(word)
        refinements += refined
        excess[len(word) - lower] += 1
    result = {'complete_words_replayed': len(models), 'moves_replayed': moves,
              'refinement_steps_exercised': refinements, 'largest_n': 151,
              'excess_over_lower_bound': dict(excess),
              'scope': 'Implementation audit; complete finite coverage checked separately.'}
    (folder / 'four_cycle_core_construct_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
