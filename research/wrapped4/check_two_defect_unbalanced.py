"""Independent replay and monotone-anchor audit of the unbalanced construction."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random

from two_defect_unbalanced import construct
from two_defect_construct import construct as central_construct
from verify_two_defect_certificate import build, model, replay


PATTERNS = ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0))


def check_anchors(p, a, gaps, word):
    motion, travel, crossings = replay(p, word)
    support = [0]
    for g in gaps[:-1]:
        support.append(support[-1] + g + 1)
    selected = []
    for start, size in zip(support, gaps):
        if size < 3:
            continue
        labels = [p[(start + j) % len(p)] for j in range(1, size + 1)]
        for direction, target in ((1, a), (-1, len(p) - a)):
            good = [x for x in labels if motion[x] == direction * target and travel[x] == target]
            assert good, (len(p), a, gaps, start, direction)
            selected.append((good[0], direction))
    for (x, dx), (y, dy) in combinations(selected, 2):
        assert crossings[tuple(sorted((x, y))) ] == (dx != dy)


def main():
    folder = Path(__file__).parent
    certificate = json.loads((folder / 'two_defect_inflation_complete.json').read_text())
    base_gaps = 0
    for row in certificate['nodes']:
        p = row['permutation']
        a, alpha, spaces = model(p)
        motion, travel, _ = replay(p, row['word'])
        for gap in spaces:
            if len(gap) < 3:
                continue
            assert any(motion[p[j]] == a and travel[p[j]] == a for j in gap)
            assert any(motion[p[j]] == a - len(p) and travel[p[j]] == len(p) - a for j in gap)
            base_gaps += 1
    rng = random.Random(20260921)
    checked = moves = refined = growth = 0
    reductions = Counter()
    # Each ordered block pair in the threshold band, every core, and both
    # clustered and dispersed marked positions. Includes all residue classes.
    block_pairs = [(a, b) for a in range(12, 25) for b in range(12, 25)]
    block_pairs += [(a, b) for a, b in ((12, 60), (60, 12), (12, 121), (121, 12),
                                     (13, 122), (122, 13), (27, 93), (93, 27),
                                     (50, 101), (101, 50))]
    for a, b in block_pairs:
        n = a + b
        for alpha in PATTERNS:
            t = len(alpha)
            samples = [[n - t] + [0] * (t - 1)]
            support = [0] + sorted(rng.sample(range(1, n), t - 1))
            samples.append([(support[(i + 1) % t] - support[i]) % n - 1 for i in range(t)])
            # Maximal retained gaps at the delicate minimum-block threshold.
            if t == 4 and n >= 24:
                g = [5, 5, 5, n - 19]
                samples.append(g)
            for gaps in samples:
                word, meta = construct(alpha, gaps, a)
                p = build(alpha, gaps, a)
                check_anchors(p, a, gaps, word)
                k = a * b - 2
                assert len(word) == k // 3 + k % 3
                checked += 1
                moves += len(word)
                refined += meta['central_refinements']
                growth += meta['one_direction_growth_steps']
                reductions[(abs(a - b) % 3, a >= b)] += 1
    # Final central words also have anchors in every gap >=3 even when their
    # finite base lacks an anchor in a gap of length two.
    central_checks = 0
    for alpha, seed in (((3, 2, 1, 0), (2, 2, 2, 2)),
                        ((3, 2, 1, 0), (9, 0, 2, 0))):
        for i in range(len(seed)):
            if seed[i] < 2:
                continue
            for j in range(len(seed)):
                if seed[j] < 2:
                    continue
                for size in (1, 2, 6):
                    gaps = list(seed)
                    gaps[i] += 3 * size
                    gaps[j] += 3 * size
                    n = len(alpha) + sum(gaps)
                    for a in sorted({n // 2, (n + 1) // 2}):
                        word, _ = central_construct(alpha, gaps, a)
                        check_anchors(build(alpha, gaps, a), a, gaps, word)
                        central_checks += 1
    report = {'certificate_nodes_checked': len(certificate['nodes']),
              'certificate_gaps_at_least_three_checked': base_gaps,
              'unbalanced_words_replayed': checked, 'moves_replayed': moves,
              'central_refinements_exercised': refined,
              'one_direction_growth_steps_exercised': growth,
              'exceptional_central_anchor_checks': central_checks,
              'residue_and_direction_coverage': {str(k): v for k, v in reductions.items()},
              'largest_n': max(a + b for a, b in block_pairs),
              'scope': 'Implementation audit; all-size proof is TWO_DEFECT_UNBALANCED.md.'}
    (folder / 'two_defect_unbalanced_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
