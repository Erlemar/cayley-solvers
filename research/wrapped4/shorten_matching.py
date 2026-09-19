"""Search shorter replacements in a retained sorting word.

Commuting disjoint moves and replacing replay-verified segments preserve
the word's permutation. Failure to find a shortening is inconclusive.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from affine_insertion import balanced_keys
from general_sort import apply_to
from near_rotation_search import potential
from wide_search import build_endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--cap', type=int, default=100000)
    parser.add_argument('--per-round', type=int, default=120)
    args = parser.parse_args()
    folder = Path(__file__).parent
    source = json.loads((folder / 'wide_full_matching_n24_retries.json').read_text())
    original = next(row for row in source if row['status'] == 'found')
    n = original['n']
    p = original['permutation']
    best = original['word']
    endgame = build_endgame(n, 4, 5000000)
    supports = [{(i + j) % n for j in range(4)} for i in range(n)]
    rng = random.Random(20260927)
    records = []
    seen = set()
    out = folder / 'matching_n24_shortening.json'
    for iteration in range(args.rounds):
        word = [tuple(v) for v in best]
        if iteration:
            for _ in range(20 * len(word)):
                pos = rng.randrange(len(word) - 1)
                if not supports[word[pos][0]] & supports[word[pos + 1][0]]:
                    word[pos], word[pos + 1] = word[pos + 1], word[pos]
        assert apply_to(p, word) == list(range(n))
        candidates = []
        for width in range(6, min(31, len(word))):
            for start in range(len(word) - width + 1):
                target = apply_to(list(range(n)), word[start:start + width])
                key = (tuple(target), width - 2)
                if key in seen:
                    continue
                ell = int(potential(np.array(balanced_keys(target), np.int64), n))
                if ell <= 3 * (width - 2):
                    candidates.append((width, -(3 * width - ell), start, target))
        candidates.sort()
        print('ROUND', iteration, 'eligible', len(candidates), flush=True)
        changed = False
        for width, negdefect, start, target in candidates[:args.per_round]:
            seen.add((tuple(target), width - 2))
            row = solve(target, width - 2, endgame, cap=args.cap,
                        seed=(iteration * 7 + start) % (2 * n))
            records.append({'round': iteration, 'start': start, 'old_width': width,
                            'defect': -negdefect, 'result': row})
            if row['status'] == 'found':
                replacement = [(i, -d) for i, d in reversed(row['word'])]
                best = word[:start] + replacement + word[start + width:]
                assert apply_to(p, best) == list(range(n))
                print('SHORTENED', len(word), 'to', len(best), 'span', start, width, flush=True)
                changed = True
            if len(records) % 20 == 0 or changed:
                print('ATTEMPTS', len(records), 'best', len(best), flush=True)
                out.write_text(json.dumps({'n': n, 'permutation': p,
                                           'best_word': best, 'length': len(best),
                                           'records': records}, indent=2))
            if changed:
                break
        if len(best) <= 48:
            break
    out.write_text(json.dumps({'n': n, 'permutation': p,
                               'best_word': best, 'length': len(best),
                               'records': records}, indent=2))
    print('FINAL', len(best), 'attempts', len(records), flush=True)


if __name__ == '__main__':
    main()
