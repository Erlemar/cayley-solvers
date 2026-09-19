"""Try monotone-anchor assignments on the unresolved pilot roots."""
import argparse
import json
from pathlib import Path

from anchored_search import solve
from two_defect_roots import anchors, marked_structure
from wide_search import build_endgame


def assignments(p, mode=0, required=None):
    _, _, gaps = marked_structure(p)
    active = [gap for i, gap in enumerate(gaps) if len(gap) >= 2 and (required is None or i in required)]
    for bits in range(1 << len(active)):
        colors = [0] * len(p)
        for index, gap in enumerate(active):
            first, second = (gap[0], gap[-1]) if mode == 0 else (gap[0], gap[1]) if mode == 1 else (gap[-2], gap[-1])
            if bits & (1 << index):
                first, second = second, first
            colors[p[first]] = 1
            colors[p[second]] = -1
        yield colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', type=int, default=100000)
    args = parser.parse_args()
    folder = Path(__file__).parent
    rows = json.loads((folder / 'two_defect_inflation_pilot.json').read_text())
    out = folder / 'two_defect_anchored_pilot.json'
    results = []
    n0, endgame = None, None
    for row in rows:
        if row['missing'] == 0:
            continue
        p = row['permutation']
        n = len(p)
        if n != n0:
            endgame = build_endgame(n, 4, 5000000)
            n0 = n
        record = {'n': n, 'permutation': p, 'budget': row['budget'], 'status': 'inconclusive', 'attempts': []}
        for mode in range(3):
            for index, colors in enumerate(assignments(p, mode)):
                found = solve(p, row['budget'], endgame, colors, cap=args.cap, seed=index)
                record['attempts'].append({k: v for k, v in found.items() if k != 'word'})
                if found['status'] == 'found':
                    info = anchors(p, found['word'])
                    assert not info['missing']
                    record.update(status='found', word=found['word'], anchor_info=info)
                    break
            if record['status'] == 'found':
                break
        results.append(record)
        out.write_text(json.dumps(results, indent=2))
        print(json.dumps({'n': n, 'status': record['status'], 'attempts': len(record['attempts']),
                          'nodes': sum(r['nodes'] for r in record['attempts'])}), flush=True)


if __name__ == '__main__':
    main()
