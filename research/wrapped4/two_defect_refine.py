"""Close unresolved anchored bases by complete +6 gap refinements."""
import argparse
import json
from pathlib import Path

from anchored_defect_pilot import assignments
from anchored_search import solve as anchored_solve
from two_defect_roots import anchors, from_gaps, marked_structure, orbit
from wide_search import build_endgame, solve


def children(p):
    a, sigma, gaps = marked_structure(p)
    support = [i for i, x in enumerate(sigma) if x != i]
    ranks = {x: i for i, x in enumerate(support)}
    alpha = tuple(ranks[sigma[i]] for i in support)
    lengths = [len(gap) for gap in gaps]
    active = [i for i, g in enumerate(lengths) if g >= 2]
    result = set()
    for first_index, i in enumerate(active):
        for j in active[first_index:]:
            grown = lengths[:]
            grown[i] += 3
            grown[j] += 3
            result.add(min(orbit(from_gaps(alpha, grown, a + 3))))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', type=int, default=200000)
    parser.add_argument('--max-n', type=int, default=29)
    args = parser.parse_args()
    folder = Path(__file__).parent
    data = json.loads((folder / 'two_defect_inflation_certificate.json').read_text())
    table = {tuple(row['permutation']): row for row in data['rows']}
    root_keys = list(table)
    output = folder / 'two_defect_inflation_refined.json'
    endgame_n, endgame = None, None
    def save():
        result = {'roots': [list(p) for p in root_keys],
                  'rows': sorted(table.values(), key=lambda r: (r['n'], r['permutation'])),
                  'complete': all(row['status'] in ('found', 'branch') for row in table.values())}
        output.write_text(json.dumps(result, indent=2))
        return result
    processed = set()
    while True:
        pending = sorted((p for p, row in table.items() if row['status'] not in ('found', 'branch') and p not in processed),
                         key=lambda p: (len(p), p))
        if not pending:
            break
        p = pending[0]
        processed.add(p)
        n = len(p)
        if n > args.max_n:
            continue
        row = table[p]
        if n != endgame_n:
            endgame = build_endgame(n, 4, 5000000)
            endgame_n = n
        ell = n * n // 4 - 2
        budget = ell // 3 + ell % 3
        if not row.get('attempts'):
            row['attempts'] = []
            for mode in range(3):
                for index, colors in enumerate(assignments(list(p), mode)):
                    attempt = anchored_solve(list(p), budget, endgame, colors, cap=args.cap, seed=index + mode)
                    row['attempts'].append({k: v for k, v in attempt.items() if k != 'word'})
                    if attempt['status'] == 'found':
                        row.update(status='found', word=attempt['word'], anchor_info=anchors(p, attempt['word']))
                        assert not row['anchor_info']['missing']
                        break
                if row['status'] == 'found':
                    break
        if row['status'] != 'found':
            point = solve(list(p), budget, endgame, cap=1000000, seed=7)
            assert point['status'] == 'found', (n, p, point['status'])
            child_keys = children(p)
            row.update(status='branch', word=point['word'], children=[list(q) for q in child_keys])
            for q in child_keys:
                if q not in table:
                    table[q] = {'n': len(q), 'permutation': list(q), 'status': 'inconclusive', 'budget':
                                (len(q) * len(q) // 4 - 2) // 3 + (len(q) * len(q) // 4 - 2) % 3}
        save()
        print(json.dumps({'n': n, 'status': row['status'], 'children': len(row.get('children', [])),
                          'total_nodes': len(table), 'unresolved': sum(r['status'] not in ('found', 'branch') for r in table.values())}), flush=True)
    result = save()
    print(json.dumps({'complete': result['complete'], 'nodes': len(table),
                      'anchored': sum(r['status'] == 'found' for r in table.values()),
                      'branches': sum(r['status'] == 'branch' for r in table.values()),
                      'max_n': max(map(len, table))}), flush=True)


if __name__ == '__main__':
    main()
