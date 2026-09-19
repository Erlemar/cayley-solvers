"""Find additional extraction identities without mutating a running search's file."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from general_sort import apply_to
from matching_beam import run
from wide_search import build_endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=9)
    parser.add_argument('--cap', type=int, default=100000)
    parser.add_argument('--beam', type=int, default=512)
    parser.add_argument('--source', default='adaptive_extraction_c3_2_wide.json')
    parser.add_argument('--limit', type=int, default=1000000)
    parser.add_argument('--include-exhausted', action='store_true')
    parser.add_argument('--extra-helpers', type=int, default=0)
    parser.add_argument('--radius', type=int, default=4)
    args = parser.parse_args()
    folder = Path(__file__).parent
    data = json.loads((folder / args.source).read_text())
    coefficient = Fraction(data['defect_per_extracted_token'])
    statuses = ('node_limit', 'exhausted') if args.include_exhausted else ('node_limit',)
    cases = [r for r in data['records'] if r['k'] == args.k and r['status'] in statuses]
    if args.radius > 4:
        cases.sort(key=lambda r: r['width'])
    suffix = f'_h{args.extra_helpers}' if args.extra_helpers else ''
    if args.radius != 4:
        suffix += f'_r{args.radius}'
    prefix = '' if coefficient == Fraction(3, 2) else 'c' + str(coefficient).replace('/', '_') + '_'
    output = folder / f'adaptive_supplement_{prefix}k{args.k}{suffix}.json'
    results = json.loads(output.read_text()) if output.exists() else []
    completed = {tuple(r['pattern']) for r in results}
    games = {}
    new_count = 0
    for case in cases:
        if tuple(case['pattern']) in completed:
            continue
        if new_count >= args.limit:
            break
        n = case['width'] + args.extra_helpers
        p = case['pattern'] + list(range(len(case['pattern']), n))
        inv_p = [p.index(i) for i in range(n)]
        reversed_p = [n - 1 - x for x in reversed(p)]
        if n not in games:
            if args.radius > 4:
                games.clear()
            radius = args.radius if n <= 20 else 4
            games[n] = (build_endgame(n, radius, 5000000, False), radius)
        game, radius = games[n]
        word = None
        method = None
        for orientation, state in [('inverse', inv_p), ('reverse', reversed_p)]:
            candidate = solve(state, case['budget'], game, radius, args.cap,
                              seed=13, wrapped=False)
            if candidate['status'] == 'found':
                word = ([(i, -d) for i, d in reversed(candidate['word'])]
                        if orientation == 'inverse' else
                        [(n - 4 - i, -d) for i, d in candidate['word']])
                method = orientation + '_dfs'
                break
        if word is None:
            for orientation, state in [('forward', p), ('inverse', inv_p)]:
                candidate = run(state, case['budget'], args.beam, game,
                                seed=len(results) + 17, wrapped=False, report=False,
                                endgame_radius=radius)
                if candidate['status'] == 'found':
                    word = (candidate['word'] if orientation == 'forward' else
                            [(i, -d) for i, d in reversed(candidate['word'])])
                    method = orientation + '_beam'
                    break
        row = {'k': args.k, 'pattern': case['pattern'], 'width': n,
               'status': 'found' if word is not None else 'inconclusive', 'method': method,
               'coefficient': str(coefficient)}
        if word is not None:
            assert apply_to(p, word) == list(range(n))
            assert all(0 <= i <= n - 4 and d in (-1, 1) for i, d in word)
            inversions = sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n))
            assert 3 * len(word) - inversions <= coefficient * args.k
            row['word'] = word
        results.append(row)
        new_count += 1
        output.write_text(json.dumps(results, indent=2))
        if len(results) % 10 == 0:
            print(args.k, len(results), 'found', sum(r['status'] == 'found' for r in results), flush=True)
    print('FINAL', args.k, len(results), 'found', sum(r['status'] == 'found' for r in results), flush=True)


if __name__ == '__main__':
    main()
