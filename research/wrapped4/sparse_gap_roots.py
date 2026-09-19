"""Geometric symmetry reduction for fixed-deficit gap certificates."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

from adjacent_deficit_layers import cores
from two_defect_roots import compositions, from_gaps
from twist_inflation import strand_data


def cyclic_core(alpha):
    t = len(alpha)
    return min(tuple((alpha[(i + shift) % t] - shift) % t for i in range(t)) for shift in range(t))


def canonical(a, alpha, gaps):
    t = len(alpha)
    n = t + sum(gaps)
    inverse = [0] * t
    for i, x in enumerate(alpha):
        inverse[x] = i
    result = []
    for inv, source in ((False, alpha), (True, inverse)):
        aa = n - a if inv else a
        for reflect in (False, True):
            bb = n - aa if reflect else aa
            perm = [(-source[-i % t]) % t for i in range(t)] if reflect else list(source)
            spaces = [gaps[(-i - 1) % t] for i in range(t)] if reflect else gaps
            for shift in range(t):
                result.append((bb, tuple((perm[(i + shift) % t] - shift) % t for i in range(t)),
                               tuple(spaces[(i + shift) % t] for i in range(t))))
    return min(result)


def gap_models(alpha, minimum=12):
    t = len(alpha)
    for small in product(range(5), repeat=t):
        active = [i for i, g in enumerate(small) if g >= 2]
        m = t + sum(small)
        if not active:
            if m >= minimum:
                yield small
            continue
        for phase in (0, 3):
            n0 = m + phase
            while n0 < minimum:
                n0 += 6
            triples = (n0 - m) // 3
            for counts in compositions(triples, len(active)):
                grown = list(small)
                for i, c in zip(active, counts):
                    grown[i] += 3 * c
                yield tuple(grown)


def generate(deficit):
    patterns = sorted({cyclic_core(alpha) for alpha, e in cores(deficit)})
    records = {}
    raw = 0
    for alpha in patterns:
        for gaps in gap_models(alpha):
            n = len(alpha) + sum(gaps)
            key = canonical(n // 2, alpha, gaps)
            raw += 1
            if key not in records:
                a, core, spaces = key
                records[key] = {'n': n, 'a': a, 'core': core, 'gaps': spaces,
                                'permutation': from_gaps(core, spaces, a), 'models': 0}
            records[key]['models'] += 1
    rows = sorted(records.values(), key=lambda r: (r['n'], r['a'], r['core'], r['gaps']))
    stats = {'deficit': deficit, 'cyclic_core_types': len(patterns), 'root_models': raw,
             'root_orbits': len(rows), 'orbits_by_size': dict(Counter(r['n'] for r in rows))}
    return rows, stats


def structure(p, a):
    n = len(p)
    support = [i for i, x in enumerate(p) if x != (i + a) % n]
    return [[(support[k] + j) % n for j in range(1, (support[(k + 1) % len(support)] - support[k]) % n)]
            for k in range(len(support))]


def anchors(p, a, word):
    n = len(p)
    gaps = structure(p, a)
    motion, travel, crossing = strand_data(p, word)
    found = []
    missing = []
    for i, gap in enumerate(gaps):
        if len(gap) < 2:
            continue
        plus = [pos for pos in gap if motion[p[pos]] == a and travel[p[pos]] == a]
        minus = [pos for pos in gap if motion[p[pos]] == -(n - a) and travel[p[pos]] == n - a]
        if plus and minus:
            found.append({'gap': i, 'positive_position': plus[0], 'negative_position': minus[0]})
        else:
            missing.append(i)
    return {'anchors': found, 'missing': missing}


def assignments(p, a, count=8):
    active = [g for g in structure(p, a) if len(g) >= 2]
    masks = list(range(1 << len(active)))
    # Alternate the orientations instead of trying only the first adjacent masks.
    masks.sort(key=lambda m: (abs(2 * m.bit_count() - len(active)), m))
    for bits in masks[:count]:
        colors = [0] * len(p)
        for i, gap in enumerate(active):
            first, second = gap[0], gap[-1]
            if bits & (1 << i):
                first, second = second, first
            colors[p[first]] = 1
            colors[p[second]] = -1
        yield colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deficit', type=int, default=3)
    parser.add_argument('--pilot', type=int, default=0)
    parser.add_argument('--cap', type=int, default=30000)
    parser.add_argument('--attempts', type=int, default=8)
    args = parser.parse_args()
    folder = Path(__file__).parent
    rows, stats = generate(args.deficit)
    print(json.dumps(stats, indent=2), flush=True)
    (folder / f'gap_roots_d{args.deficit}.json').write_text(json.dumps({'stats': stats, 'rows': rows}, indent=2))
    if not args.pilot:
        return
    from anchored_search import solve
    from rotation_bounds import diameter_lower
    from wide_search import build_endgame
    selected = [rows[(len(rows) - 1) * i // max(1, args.pilot - 1)] for i in range(args.pilot)]
    result = []
    previous_n, endgame = None, None
    for index, model in enumerate(selected):
        n, a, p = model['n'], model['a'], list(model['permutation'])
        if n != previous_n:
            endgame = build_endgame(n, 4, 5000000)
            previous_n = n
        ell = a * (n - a) - args.deficit
        budget = diameter_lower(n)
        budget -= (budget - ell) % 2
        record = {**model, 'budget': budget, 'status': 'inconclusive', 'attempts': []}
        for number, colors in enumerate(assignments(p, a, args.attempts)):
            attempt = solve(p, budget, endgame, colors, cap=args.cap, seed=number)
            record['attempts'].append({k: v for k, v in attempt.items() if k != 'word'})
            if attempt['status'] == 'found':
                info = anchors(p, a, attempt['word'])
                assert not info['missing']
                record.update(status='found', word=attempt['word'], anchor_info=info)
                break
        result.append(record)
        (folder / f'gap_pilot_d{args.deficit}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'case': index + 1, 'n': n, 'status': record['status'],
                          'attempts': len(record['attempts'])}), flush=True)
    print('FOUND', sum(r['status'] == 'found' for r in result), 'OF', len(result), flush=True)


if __name__ == '__main__':
    main()
