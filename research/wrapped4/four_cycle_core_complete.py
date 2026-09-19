"""Assemble/refine a finite H(n) certificate for central rotation times a 4-cycle."""
import argparse
from collections import Counter
import json
from pathlib import Path

from anchored_search import solve
from rotation_bounds import diameter_lower
from sparse_gap_roots import anchors, assignments
from two_defect_construct import transport
from two_defect_roots import from_gaps, orbit
from wide_search import build_endgame


def structure(p):
    n = len(p)
    for a in sorted({n // 2, (n + 1) // 2}):
        sigma = [(x - a) % n for x in p]
        support = [i for i, x in enumerate(sigma) if x != i]
        if len(support) == 4:
            ranks = {x: i for i, x in enumerate(support)}
            core = [ranks[sigma[x]] for x in support]
            assert core[core[0]] != 0 and core[core[core[core[0]]]] == 0
            gaps = [(support[(i + 1) % 4] - support[i]) % n - 1 for i in range(4)]
            return a, core, gaps
    raise AssertionError('Not a central four-cycle core')


def children(p, word):
    a, core, gaps = structure(p)
    missing = set(anchors(p, a, word)['missing'])
    active = [i for i, g in enumerate(gaps) if g >= 2]
    result = set()
    for index, first in enumerate(active):
        for second in active[index:]:
            if first not in missing and second not in missing:
                continue
            grown = gaps[:]
            grown[first] += 3
            grown[second] += 3
            result.add(min(orbit(from_gaps(core, grown, a + 3))))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', type=int, default=100000)
    parser.add_argument('--max-n', type=int, default=29)
    args = parser.parse_args()
    folder = Path(__file__).parent
    source = json.loads((folder / 'gap_d3_support4_words.json').read_text())
    output = folder / 'four_cycle_core_certificate.json'
    previous = json.loads(output.read_text()) if output.exists() else {'nodes': []}
    table = {tuple(r['permutation']): r for r in previous['nodes']}
    roots = []
    for row in source:
        p = row['permutation']
        key = min(orbit(p))
        roots.append(key)
        assert 'word' in row
        if key not in table:
            table[key] = {'n': len(key), 'permutation': list(key),
                          'word': transport(p, row['word'], key)}
    pending = roots[:]
    visited = set()
    n0, endgame = None, None
    while pending:
        p = pending.pop()
        if p in visited:
            continue
        visited.add(p)
        row = table.setdefault(p, {'n': len(p), 'permutation': list(p)})
        a, _, _ = structure(p)
        if 'word' not in row:
            n = len(p)
            if n > args.max_n:
                continue
            if n != n0:
                endgame = build_endgame(n, 4, 5000000)
                n0 = n
            ell = n * n // 4 - 3
            budget = diameter_lower(n)
            budget -= (budget - ell) % 2
            attempts = []
            for number, colors in enumerate(assignments(p, a, 16)):
                trial = solve(list(p), budget, endgame, colors, cap=args.cap, seed=number)
                attempts.append({k: v for k, v in trial.items() if k not in ('word', 'permutation')})
                if trial['status'] == 'found':
                    row['word'] = trial['word']
                    break
            row['attempts'] = attempts
            print(json.dumps({'child_n': n, 'found': 'word' in row, 'attempts': len(attempts)}), flush=True)
        if 'word' in row:
            info = anchors(p, a, row['word'])
            child_keys = children(p, row['word'])
            row.update(a=a, anchor_info=info, children=[list(q) for q in child_keys])
            pending.extend(child_keys)
    nodes = [table[p] for p in visited]
    complete = all('word' in r and all(tuple(q) in visited for q in r['children']) for r in nodes)
    data = {'complete': complete, 'roots': [list(p) for p in roots],
            'nodes': sorted(nodes, key=lambda r: (r['n'], r['permutation'])),
            'scope': 'Positive H(n) word identities and gap-growth coverage for central four-cycle cores.'}
    output.write_text(json.dumps(data, indent=2))
    print(json.dumps({'complete': complete, 'roots': len(roots), 'nodes': len(nodes),
                      'branches': sum(bool(r.get('children')) for r in nodes),
                      'by_size': dict(Counter(r['n'] for r in nodes))}), flush=True)


if __name__ == '__main__':
    main()
