"""Finite candidate bases for inflating the entire auxiliary layer K-2."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from twist_inflation import strand_data


FOLDER = Path(__file__).parent
CORES = ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2))


def compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest


def orbit(p):
    n = len(p)
    inverse = [0] * n
    for i, x in enumerate(p):
        inverse[x] = i
    return {tuple((sign * q[(sign * (i - shift)) % n] + shift) % n for i in range(n))
            for q in (p, inverse) for sign in (-1, 1) for shift in range(n)}


def from_gaps(alpha, gaps, a=None):
    n = len(alpha) + sum(gaps)
    a = n // 2 if a is None else a
    support = [0]
    for g in gaps[:-1]:
        support.append(support[-1] + 1 + g)
    p = [(i + a) % n for i in range(n)]
    for i, x in enumerate(alpha):
        p[support[i]] = (support[x] + a) % n
    return tuple(p)


def root_models():
    for n in range(12, 24):
        for alpha in CORES:
            t = len(alpha)
            for gaps in compositions(n - t, t):
                minimal = t + sum(g if g < 2 else 2 + (g - 2) % 3 for g in gaps)
                if n - 6 >= max(12, minimal):
                    continue
                yield alpha, gaps, from_gaps(alpha, gaps)


@lru_cache(maxsize=1)
def roots():
    groups = {}
    for alpha, gaps, p in root_models():
        canonical = min(orbit(p))
        if canonical not in groups:
            groups[canonical] = {'n': len(p), 'permutation': list(canonical), 'models': 0}
        groups[canonical]['models'] += 1
    return sorted(groups.values(), key=lambda r: (r['n'], r['permutation']))


def marked_structure(p):
    n = len(p)
    for a in {n // 2, (n + 1) // 2}:
        sigma = [(x - a) % n for x in p]
        support = [i for i in range(n) if sigma[i] != i]
        if len(support) in (3, 4):
            break
    else:
        raise AssertionError('Not a central rotation with a small core')
    gaps = [[(support[k] + j) % n for j in range(1, (support[(k + 1) % len(support)] - support[k]) % n)]
            for k in range(len(support))]
    return a, sigma, gaps


def anchors(p, word):
    n = len(p)
    a, sigma, gaps = marked_structure(p)
    b = n - a
    displacement, travel, crossings = strand_data(p, word)
    choices = []
    missing = []
    for index, gap in enumerate(gaps):
        if len(gap) < 2:
            continue
        plus = [i for i in gap if displacement[p[i]] == a and travel[p[i]] == a]
        minus = [i for i in gap if displacement[p[i]] == -b and travel[p[i]] == b]
        if plus and minus:
            assert crossings.get(tuple(sorted((p[plus[0]], p[minus[0]]))), 0) == 1
            choices.append({'gap': index, 'positive_position': plus[0], 'negative_position': minus[0]})
        else:
            missing.append({'gap': index, 'positions': gap, 'positive': len(plus), 'negative': len(minus)})
    return {'a': a, 'anchors': choices, 'missing': missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', type=int, default=0)
    parser.add_argument('--cap', type=int, default=30000)
    parser.add_argument('--seeds', type=int, default=12)
    args = parser.parse_args()
    models = roots()
    stats = {'root_models': sum(r['models'] for r in models), 'distinct_orbits': len(models),
             'orbits_by_size': dict(Counter(r['n'] for r in models))}
    print(json.dumps(stats, indent=2), flush=True)
    (FOLDER / 'two_defect_root_counts.json').write_text(json.dumps(stats, indent=2))
    if not args.pilot:
        return
    from wide_search import build_endgame, solve
    rows = []
    # Select roots across the full size range, not just its easiest beginning.
    selected = [models[(len(models) - 1) * i // max(1, args.pilot - 1)] for i in range(args.pilot)]
    n0, endgame = None, None
    for index, model in enumerate(selected):
        n, p = model['n'], model['permutation']
        if n != n0:
            endgame = build_endgame(n, 4, 5000000)
            n0 = n
        ell = n * n // 4 - 2
        budget = ell // 3 + ell % 3
        best = {'status': 'inconclusive', 'missing': 100}
        attempts = []
        for seed in range(args.seeds):
            inverse = seed % 2 == 1
            q = [p.index(i) for i in range(n)] if inverse else p
            attempt = solve(q, budget, endgame, cap=args.cap, seed=(seed * 7) % (2 * n))
            attempts.append({k: v for k, v in attempt.items() if k not in ('word', 'permutation')})
            if attempt['status'] == 'found':
                word = [(i, -d) for i, d in reversed(attempt['word'])] if inverse else attempt['word']
                info = anchors(p, word)
                if len(info['missing']) < best['missing']:
                    best = {'status': 'found', 'word': word, 'missing': len(info['missing']), 'anchor_info': info}
                if not info['missing']:
                    break
        row = {**model, **best, 'attempts': attempts, 'budget': budget}
        rows.append(row)
        (FOLDER / 'two_defect_inflation_pilot.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps({'case': index + 1, 'n': n, 'missing': best['missing'],
                          'attempts': len(attempts)}), flush=True)
    print('QUALIFIED', sum(r['missing'] == 0 for r in rows), 'OF', len(rows), flush=True)


if __name__ == '__main__':
    main()
