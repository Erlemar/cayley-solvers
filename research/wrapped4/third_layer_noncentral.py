"""Finite words for the small noncentral part of auxiliary layer K-3."""
import argparse
from itertools import product
import json
from pathlib import Path

from sparse_gap_roots import canonical
from two_defect_roots import CORES, compositions, from_gaps


def cases():
    keys = set()
    for core in CORES:
        for gaps in compositions(14 - len(core), len(core)):
            keys.add(canonical(6, core, gaps))
        for gaps in product(range(6), repeat=len(core)):
            n = len(core) + sum(gaps)
            if 16 <= n <= 24 and n % 2 == 0:
                keys.add(canonical(n // 2 - 1, core, gaps))
    result = []
    for a, core, gaps in keys:
        p = from_gaps(core, gaps, a)
        result.append({'n': len(p), 'a': a, 'core': core, 'gaps': gaps, 'permutation': p})
    for d in range(1, 7):
        p = [(i + 5) % 13 for i in range(13)]
        p[0], p[d] = p[d], p[0]
        result.append({'n': 13, 'a': 5, 'transposition_separation': d, 'permutation': p})
    return sorted(result, key=lambda r: (r['n'], r['permutation']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', type=int, default=50000)
    parser.add_argument('--seeds', type=int, default=4)
    args = parser.parse_args()
    from wide_search import build_endgame, solve
    from rotation_bounds import diameter_lower
    from verify_two_defect_certificate import replay
    output = Path(__file__).with_name('third_layer_noncentral_words.json')
    saved = json.loads(output.read_text()) if output.exists() else []
    lookup = {tuple(r['permutation']): r for r in saved}
    n0, endgame = None, None
    result = []
    for index, model in enumerate(cases()):
        n, p = model['n'], list(model['permutation'])
        row = lookup.get(tuple(p), {**model, 'status': 'inconclusive', 'attempts': []})
        result.append(row)
        if row['status'] == 'found':
            replay(p, row['word'])
            continue
        if n != n0:
            endgame = build_endgame(n, 4, 5000000)
            n0 = n
        ell = n * n // 4 - 3
        lower = ell // 3 + ell % 3
        upper = diameter_lower(n)
        upper -= (upper - ell) % 2
        for budget in sorted({lower, upper}):
            for seed in range(args.seeds):
                trial = solve(p, budget, endgame, cap=args.cap, seed=seed * 7)
                row['attempts'].append({k: v for k, v in trial.items() if k != 'word'})
                if trial['status'] == 'found':
                    replay(p, trial['word'])
                    row.update(status='found', word=trial['word'], length=len(trial['word']),
                               lower=lower, upper=upper, exact=len(trial['word']) == lower)
                    break
            if row['status'] == 'found':
                break
        output.write_text(json.dumps(result + [r for r in saved if tuple(r['permutation']) not in
                                             {tuple(s['permutation']) for s in result}], indent=2))
        if (index + 1) % 10 == 0 or row['status'] != 'found':
            print(json.dumps({'case': index + 1, 'total': 224, 'n': n,
                              'status': row['status'], 'length': row.get('length')}), flush=True)
    output.write_text(json.dumps(result, indent=2))
    print('FOUND', sum(r['status'] == 'found' for r in result), 'OF', len(result), flush=True)


if __name__ == '__main__':
    main()
