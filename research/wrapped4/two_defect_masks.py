"""Cover exact growth subsets at the two exceptional anchored roots."""
import json
from pathlib import Path

from anchored_defect_pilot import assignments
from anchored_search import solve
from two_defect_roots import anchors, marked_structure
from wide_search import build_endgame


def main():
    folder = Path(__file__).parent
    source = json.loads((folder / 'two_defect_inflation_certificate.json').read_text())
    result = []
    for row in source['rows']:
        if row['status'] == 'found':
            continue
        p = row['permutation']
        n = len(p)
        _, _, gaps = marked_structure(p)
        active = [i for i, gap in enumerate(gaps) if len(gap) >= 2]
        endgame = build_endgame(n, 4, 5000000)
        covers = []
        subsets = sorted((tuple(i for bit, i in enumerate(active) if mask & (1 << bit))
                          for mask in range(1, 1 << len(active))), key=lambda s: (-len(s), s))
        for required in subsets:
            inherited = next((r for r in covers if r['status'] == 'found' and set(required) <= set(r['covered'])), None)
            if inherited is not None:
                continue
            record = {'required': list(required), 'status': 'inconclusive', 'attempts': []}
            if list(required) != active:
                for mode in range(3):
                    for index, colors in enumerate(assignments(p, mode, required)):
                        found = solve(p, row['budget'], endgame, colors, cap=200000, seed=index + mode)
                        record['attempts'].append({k: v for k, v in found.items() if k != 'word'})
                        if found['status'] == 'found':
                            info = anchors(p, found['word'])
                            covered = [c['gap'] for c in info['anchors']]
                            assert set(required) <= set(covered)
                            record.update(status='found', word=found['word'], covered=covered)
                            break
                    if record['status'] == 'found':
                        break
            covers.append(record)
            print(json.dumps({'n': n, 'required': required, 'status': record['status'],
                              'covered': record.get('covered'), 'attempts': len(record['attempts'])}), flush=True)
        result.append({'n': n, 'permutation': p, 'budget': row['budget'], 'covers': covers})
        (folder / 'two_defect_mask_covers.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
