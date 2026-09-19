"""Assemble the complete positive certificate, using partial anchor coverage."""
import json
from pathlib import Path

from two_defect_roots import anchors, from_gaps, marked_structure, orbit


def needed_children(p, word):
    a, sigma, gaps = marked_structure(p)
    support = [i for i, x in enumerate(sigma) if x != i]
    ranks = {x: i for i, x in enumerate(support)}
    alpha = tuple(ranks[sigma[i]] for i in support)
    lengths = [len(gap) for gap in gaps]
    active = [i for i, g in enumerate(lengths) if g >= 2]
    missing = {row['gap'] for row in anchors(p, word)['missing']}
    result = set()
    for first_index, i in enumerate(active):
        for j in active[first_index:]:
            if i not in missing and j not in missing:
                continue
            grown = lengths[:]
            grown[i] += 3
            grown[j] += 3
            result.add(min(orbit(from_gaps(alpha, grown, a + 3))))
    return sorted(result)


def main():
    folder = Path(__file__).parent
    source = json.loads((folder / 'two_defect_inflation_refined.json').read_text())
    table = {tuple(row['permutation']): row for row in source['rows']}
    partial = json.loads((folder / 'two_defect_mask_covers.json').read_text())
    for data in partial:
        if data['n'] == 15:
            good = next(r for r in data['covers'] if r['status'] == 'found' and r['covered'] == [0])
            table[tuple(data['permutation'])]['word'] = good['word']
    roots = [tuple(p) for p in source['roots']]
    pending = roots[:]
    checked = {}
    while pending:
        p = pending.pop()
        if p in checked:
            continue
        row = table[p]
        assert 'word' in row, (len(p), p)
        info = anchors(p, row['word'])
        child_keys = needed_children(p, row['word'])
        for q in child_keys:
            assert q in table and len(q) == len(p) + 6
        checked[p] = {'n': len(p), 'permutation': list(p), 'word': row['word'],
                      'anchor_info': info, 'children': [list(q) for q in child_keys]}
        pending.extend(child_keys)
    result = {'roots': [list(p) for p in roots],
              'nodes': sorted(checked.values(), key=lambda r: (r['n'], r['permutation'])),
              'complete': True,
              'scope': 'Positive word, anchor, and refinement certificate; search failures are not used.'}
    (folder / 'two_defect_inflation_complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'roots': len(roots), 'nodes': len(checked),
                      'leaves': sum(not r['children'] for r in checked.values()),
                      'branches': sum(bool(r['children']) for r in checked.values()),
                      'max_n': max(map(len, checked))}), flush=True)


if __name__ == '__main__':
    main()
