"""Certify 4-cycle upper bounds on the entire auxiliary layer K-2."""
import argparse
from collections import Counter
import json
from pathlib import Path

from check_rotation_defect import predicted_second_layer, second_layer_count
from general_sort import apply_to
from near_rotation_search import build_endgame, solve
from rotation_bounds import diameter_lower, parity_rounded_third
from adjacent_deficit_layers import layer


def orbit(p):
    n = len(p)
    inverse = [0] * n
    for i, x in enumerate(p):
        inverse[x] = i
    return {tuple((direction * q[(direction * (i - shift)) % n] + shift) % n
                  for i in range(n))
            for q in (p, inverse) for direction in (-1, 1) for shift in range(n)}


def representatives(states):
    remaining = set(states)
    result = []
    for p in sorted(states):
        if p not in remaining:
            continue
        images = orbit(p)
        assert images <= states
        assert images <= remaining
        result.append((p, len(images)))
        remaining.difference_update(images)
    return result


def verify(data):
    n = data['n']
    deficit = data.get('deficit', 2)
    expected = layer(n, deficit)
    if deficit == 2:
        assert expected == predicted_second_layer(n)
        assert len(expected) == second_layer_count(n)
    covered = set()
    counts = Counter()
    for row in data['rows']:
        p = tuple(row['permutation'])
        images = orbit(p)
        assert len(images) == row['orbit_size']
        assert images <= expected and not images & covered
        covered.update(images)
        if row['status'] == 'found':
            word = row['word']
            assert apply_to(list(p), word) == list(range(n))
            assert len(word) == row['length'] <= data['diameter_candidate']
            assert len(word) >= data['lower_bound']
            counts[f'upper_{len(word)}'] += len(images)
        else:
            counts['unresolved'] += len(images)
    return {'n': n, 'layer': f'cyclic_adjacent_length=floor(n^2/4)-{deficit}',
            'expected_states': len(expected), 'covered_states': len(covered),
            'orbit_representatives': len(data['rows']), 'counts': dict(counts),
            'complete_upper_bound': len(covered) == len(expected) and not counts['unresolved'],
            'scope': 'Only this auxiliary layer; not a Cayley diameter computation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=15)
    parser.add_argument('--cap', type=int, default=50000)
    parser.add_argument('--deficit', type=int, default=2)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--retry-unresolved', action='store_true')
    args = parser.parse_args()
    n = args.n
    assert 6 <= n <= 16
    deficit = args.deficit
    stem = f'second_layer_four_n{n}' if deficit == 2 else f'adjacent_layer_d{deficit}_four_n{n}'
    output = Path(__file__).with_name(stem + '.json')
    if args.verify or args.retry_unresolved:
        data = json.loads(output.read_text())
        if args.retry_unresolved:
            endgame = build_endgame(n, 4, 2000000)
            for index, row in enumerate(data['rows']):
                if row['status'] == 'found':
                    continue
                p = row['permutation']
                for inverse, seed in ((False, 19), (True, 7), (False, 31)):
                    q = [p.index(i) for i in range(n)] if inverse else p
                    attempt = solve(q, row['budget'], endgame, 4, args.cap, seed=seed)
                    row['attempts'].append({**{k: v for k, v in attempt.items() if k not in ('word', 'permutation')},
                                            'inverse_target': inverse, 'seed': seed})
                    if attempt['status'] == 'found':
                        if inverse:
                            attempt['word'] = [(i, -d) for i, d in reversed(attempt['word'])]
                        row.update({k: v for k, v in attempt.items() if k != 'permutation'})
                        assert apply_to(p, row['word']) == list(range(n))
                        break
                output.write_text(json.dumps(data, indent=2))
                print(json.dumps({'case': index + 1, 'status': row['status'],
                                  'length': row.get('length'), 'orbit_size': row['orbit_size']}), flush=True)
    else:
        states = layer(n, deficit)
        reps = representatives(states)
        ell = n * n // 4 - deficit
        lower = parity_rounded_third(ell)
        candidate = diameter_lower(n)
        upper_budget = candidate - ((candidate - ell) % 2)
        data = {'n': n, 'deficit': deficit, 'adjacent_length': ell, 'lower_bound': lower,
                'diameter_candidate': candidate, 'rows': []}
        print(f'n={n}: {len(states)} states, {len(reps)} orbits, budgets {lower}..{upper_budget}', flush=True)
        endgame = build_endgame(n, 4, 2000000)
        print(f'endgame: {len(endgame)} states', flush=True)
        for index, (p, size) in enumerate(reps):
            row = solve(list(p), lower, endgame, 4, args.cap, seed=7)
            attempts = [{k: v for k, v in row.items() if k not in ('word', 'permutation')}]
            if row['status'] != 'found' and upper_budget > lower:
                row = solve(list(p), upper_budget, endgame, 4, args.cap, seed=19)
                attempts.append({k: v for k, v in row.items() if k not in ('word', 'permutation')})
            row['attempts'] = attempts
            row['orbit_size'] = size
            data['rows'].append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(reps):
                output.write_text(json.dumps(data, indent=2))
            if (index + 1) % 100 == 0 or row['status'] != 'found':
                print(json.dumps({'case': index + 1, 'of': len(reps), 'status': row['status'],
                              'length': row.get('length'), 'orbit_size': size,
                              'nodes': sum(r['nodes'] for r in attempts)}), flush=True)
    report = verify(data)
    output.with_name(stem + '_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
