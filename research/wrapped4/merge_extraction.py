"""Discover finite, binary refinement trees for stable merge extraction.

Only complete coverage plus replayed successful words is a certificate.
Unsuccessful searches may be refined regardless of their failure status.
"""
import argparse
from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path

from near_rotation_search import build_endgame, solve
from triple_extraction import inv
from general_sort import apply_to


def pattern(bits, gap):
    k = len(bits)
    return ([i for i, x in enumerate(bits) if x == 'A']
            + list(range(k, k + gap))
            + [i for i, x in enumerate(bits) if x == 'B'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficient', default='1')
    parser.add_argument('--max-k', type=int, default=14)
    parser.add_argument('--cap', type=int, default=20000)
    parser.add_argument('--wide', action='store_true')
    parser.add_argument('--trim', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inverse-first', action='store_true')
    args = parser.parse_args()
    if args.wide:
        from wide_search import build_endgame as wide_endgame, solve as wide_solve
    c = Fraction(args.coefficient)
    cases = {(bits, gap) for bits in map(''.join, itertools.product('AB', repeat=3))
             for gap in range(3)}
    records = []
    cache = {}
    endgames = {}
    suffix = ('_wide' if args.wide else '') + ('_trim' if args.trim else '')
    out = Path(__file__).with_name('merge_extraction_c' + str(c).replace('/', '_') + suffix + '.json')
    complete = False
    start_k = 3
    if args.resume:
        previous = json.loads(out.read_text())
        assert previous['coefficient'] == str(c)
        if previous['complete']:
            print('Certificate is already complete.', flush=True)
            return
        records = previous['records']
        last = max(len(row['bits']) for row in records)
        cases = set()
        for row in records:
            if len(row['bits']) == last and row['status'] != 'found':
                cases.add((row['bits'] + 'A', (row['gap'] - 1) % 3))
                cases.add((row['bits'] + 'B', row['gap']))
            elif row['status'] == 'found':
                full = row['pattern'] + list(range(len(row['pattern']), row['width']))
                cache[tuple(full)] = row['word']
        start_k = last + 1
    for k in range(start_k, args.max_k + 1):
        next_cases = set()
        counts = Counter()
        for bits, gap in sorted(cases):
            prefix = pattern(bits, gap)
            width = max(9, len(prefix))
            p = prefix + list(range(len(prefix), width))
            solver_width = width
            if args.trim:
                solver_width = max(9, max((i + 1 for i, v in enumerate(p) if i != v), default=0))
            solver_input = p[:solver_width]
            inversions = inv(p)
            budget = (inversions + (c * k).__floor__()) // 3
            if (budget - inversions) % 2:
                budget -= 1
            row = {'bits': bits, 'gap': gap, 'pattern': prefix,
                   'width': width, 'solver_width': solver_width,
                   'inversions': inversions, 'budget': budget}
            if solver_width > (40 if args.wide else 16):
                result = {'status': 'too_wide'}
            elif tuple(p) in cache and len(cache[tuple(p)]) <= budget:
                result = {'status': 'found', 'word': cache[tuple(p)]}
            else:
                use_wide = args.wide and solver_width > 16
                if solver_width not in endgames:
                    endgames[solver_width] = (wide_endgame(solver_width, 4, 5000000, False) if use_wide
                                              else build_endgame(solver_width, 4, 2000000, False))
                solver = wide_solve if use_wide else solve
                if args.inverse_first:
                    inverse = [solver_input.index(i) for i in range(solver_width)]
                    result = solver(inverse, budget, endgames[solver_width], 4, args.cap,
                                    seed=7, wrapped=False)
                    if result['status'] == 'found':
                        result['word'] = [(i, -d) for i, d in reversed(result['word'])]
                        assert apply_to(p, result['word']) == list(range(width))
                    elif result['status'] == 'node_limit':
                        result = solver(solver_input, budget, endgames[solver_width], 4, args.cap,
                                        seed=19, wrapped=False)
                else:
                    result = solver(solver_input, budget, endgames[solver_width], 4, args.cap,
                                    seed=19, wrapped=False)
                if result['status'] == 'found':
                    cache[tuple(p)] = result['word']
            counts[result['status']] += 1
            row['status'] = result['status']
            if result['status'] == 'found':
                row['word'] = result['word']
                assert 3 * len(row['word']) - inversions <= c * k
            else:
                next_cases.add((bits + 'A', (gap - 1) % 3))
                next_cases.add((bits + 'B', gap))
            records.append(row)
        print(k, len(cases), dict(counts), 'next', len(next_cases), flush=True)
        complete = not next_cases
        out.write_text(json.dumps({'coefficient': str(c), 'complete': complete,
                                  'records': records}, indent=2))
        if complete or counts['too_wide']:
            break
        cases = next_cases


if __name__ == '__main__':
    main()
