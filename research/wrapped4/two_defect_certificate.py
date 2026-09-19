"""Find complete finite anchored bases for the all-size deficit-two theorem."""
import argparse
import json
from pathlib import Path

from anchored_defect_pilot import assignments
from anchored_search import solve
from two_defect_roots import anchors, roots
from wide_search import build_endgame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', type=int, default=100000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    folder = Path(__file__).parent
    output = folder / 'two_defect_inflation_certificate.json'
    cached = {}
    for name in ('two_defect_inflation_pilot.json', 'two_defect_anchored_pilot.json'):
        for row in json.loads((folder / name).read_text()):
            if 'word' in row and not anchors(row['permutation'], row['word'])['missing']:
                cached[tuple(row['permutation'])] = row
    if args.resume and output.exists():
        for row in json.loads(output.read_text())['rows']:
            if row['status'] == 'found':
                cached[tuple(row['permutation'])] = row
    rows = []
    n0, endgame = None, None
    models = roots()
    for index, model in enumerate(models):
        if args.limit and index >= args.limit:
            break
        p = model['permutation']
        n = len(p)
        ell = n * n // 4 - 2
        budget = ell // 3 + ell % 3
        record = {**model, 'budget': budget, 'status': 'inconclusive', 'attempts': []}
        if tuple(p) in cached:
            record.update(status='found', word=cached[tuple(p)]['word'], reused=True)
        else:
            if n != n0:
                endgame = build_endgame(n, 4, 5000000)
                n0 = n
            for mode in range(3):
                for anchor_index, colors in enumerate(assignments(p, mode)):
                    found = solve(p, budget, endgame, colors, cap=args.cap, seed=anchor_index + mode)
                    record['attempts'].append({k: v for k, v in found.items() if k != 'word'})
                    if found['status'] == 'found':
                        record.update(status='found', word=found['word'])
                        break
                if record['status'] == 'found':
                    break
        if record['status'] == 'found':
            assert len(record['word']) == budget
            record['anchor_info'] = anchors(p, record['word'])
            assert not record['anchor_info']['missing']
        rows.append(record)
        if (index + 1) % 10 == 0 or index + 1 == len(models) or record['status'] != 'found':
            data = {'expected_orbits': len(models), 'rows': rows,
                    'complete': len(rows) == len(models) and all(r['status'] == 'found' for r in rows)}
            output.write_text(json.dumps(data, indent=2))
        if (index + 1) % 25 == 0 or record['status'] != 'found':
            print(json.dumps({'case': index + 1, 'total': len(models), 'n': n,
                              'status': record['status'], 'attempts': len(record['attempts']),
                              'nodes': sum(r['nodes'] for r in record['attempts'])}), flush=True)
    data = {'expected_orbits': len(models), 'rows': rows,
            'complete': len(rows) == len(models) and all(r['status'] == 'found' for r in rows)}
    output.write_text(json.dumps(data, indent=2))
    print(json.dumps({'complete': data['complete'], 'cases': len(rows),
                      'found': sum(r['status'] == 'found' for r in rows)}), flush=True)


if __name__ == '__main__':
    main()
