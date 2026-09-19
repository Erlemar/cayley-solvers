"""Retry only node-limited leaves; successful words are directly replayed."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from general_sort import apply_to
from near_rotation_search import build_endgame, solve
from wide_search import build_endgame as wide_endgame, solve as wide_solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--cap', type=int, default=100000)
    args = parser.parse_args()
    folder = Path(__file__).parent
    path = folder / 'adaptive_extraction_c3_2_wide.json'
    data = json.loads(path.read_text())
    k = data['stages'][-1]['k']
    backup = path.with_name(path.stem + f'_before_inverse_k{k}.json')
    if not backup.exists():
        backup.write_text(path.read_text())
    cases = [row for row in data['records'] if row['k'] == k and row['status'] == 'node_limit'][:args.limit]
    endgames = {}
    count = found = exhausted = 0
    coefficient = Fraction(data['defect_per_extracted_token'])
    def save():
        tally = Counter(r['status'] for r in data['records'] if r['k'] == k)
        for status in ('found', 'exhausted', 'node_limit', 'too_wide'):
            data['stages'][-1][status] = tally[status]
        data['complete'] = tally['found'] == data['stages'][-1]['patterns']
        path.write_text(json.dumps(data, indent=2))
    for row in cases:
        n = row['width']
        full = row['pattern'] + list(range(len(row['pattern']), n))
        if n not in endgames:
            builder = wide_endgame if n > 16 else build_endgame
            endgames[n] = builder(n, 4, 5000000, False)
        solver = wide_solve if n > 16 else solve
        inverse = [full.index(i) for i in range(n)]
        result = solver(inverse, row['budget'], endgames[n], 4, args.cap, seed=7, wrapped=False)
        if result['status'] == 'found':
            word = [(i, -d) for i, d in reversed(result['word'])]
            assert apply_to(full, word) == list(range(n))
            assert all(0 <= i <= n - 4 for i, _ in word)
            inv = sum(full[i] > full[j] for i in range(n) for j in range(i + 1, n))
            assert 3 * len(word) - inv <= coefficient * k
            row.update(status='found', word=word, length=len(word), permutation=full,
                       search_orientation='inverse')
            found += 1
        elif result['status'] == 'exhausted':
            row['inverse_retry_status'] = 'exhausted'
            exhausted += 1
        count += 1
        if count % 50 == 0:
            save()
            print(count, 'found', found, 'inverse exhausted', exhausted, flush=True)
    save()
    print('FINAL', count, 'found', found, 'inverse exhausted', exhausted, flush=True)


if __name__ == '__main__':
    main()
