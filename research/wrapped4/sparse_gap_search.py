"""Bounded positive-certificate discovery for a selected finite core support."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from anchored_search import solve as anchored_solve
from rotation_bounds import diameter_lower
from sparse_gap_roots import anchors, assignments
from wide_search import build_endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--support', type=int, default=4)
    parser.add_argument('--cap', type=int, default=10000)
    parser.add_argument('--attempts', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    folder = Path(__file__).parent
    roots = json.loads((folder / 'gap_roots_d3.json').read_text())['rows']
    roots = [r for r in roots if len(r['core']) == args.support]
    if args.limit:
        roots = roots[:args.limit]
    output = folder / f'gap_d3_support{args.support}_words.json'
    old = json.loads(output.read_text()) if output.exists() else []
    lookup = {tuple(r['permutation']): r for r in old}
    rows = []
    n0, endgame = None, None
    started = time.monotonic()
    last_saved = started
    processed = 0
    for index, model in enumerate(roots):
        p, n, a = model['permutation'], model['n'], model['a']
        row = lookup.get(tuple(p), {**model, 'status': 'inconclusive', 'attempts': []})
        rows.append(row)
        if row['status'] == 'anchored':
            continue
        processed += 1
        if n != n0:
            endgame = build_endgame(n, 4, 5000000)
            n0 = n
        ell = n * n // 4 - 3
        budget = diameter_lower(n)
        budget -= (budget - ell) % 2
        row['budget'] = budget
        for number, colors in enumerate(assignments(p, a, args.attempts)):
            trial = anchored_solve(p, budget, endgame, colors, cap=args.cap, seed=number)
            row['attempts'].append({k: v for k, v in trial.items() if k not in ('word', 'permutation')})
            if trial['status'] == 'found':
                info = anchors(p, a, trial['word'])
                assert not info['missing']
                row.update(status='anchored', word=trial['word'], anchor_info=info)
                break
        if row['status'] != 'anchored':
            for seed in (0, 13):
                trial = solve(p, budget, endgame, cap=args.cap, seed=seed)
                row['attempts'].append({k: v for k, v in trial.items() if k not in ('word', 'permutation')})
                if trial['status'] == 'found':
                    info = anchors(p, a, trial['word'])
                    if 'anchor_info' not in row or len(info['missing']) < len(row['anchor_info']['missing']):
                        row.update(status='partial' if info['missing'] else 'anchored',
                                   word=trial['word'], anchor_info=info)
                    if row['status'] == 'anchored':
                        break
        if processed % 25 == 0 or time.monotonic() - last_saved >= 30 or index + 1 == len(roots):
            counts = dict(Counter(r['status'] for r in rows))
            seen = {tuple(r['permutation']) for r in rows}
            output.write_text(json.dumps(rows + [r for r in old if tuple(r['permutation']) not in seen], indent=2))
            print(json.dumps({'done': index + 1, 'total': len(roots), 'n': n,
                              'counts': counts, 'seconds': time.monotonic() - started}), flush=True)
            last_saved = time.monotonic()
    output.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
