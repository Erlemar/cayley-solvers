"""Compose stored identities on invariant intervals, with restored helpers."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from general_sort import apply_to


def minimal_core(p):
    moved = [i for i, value in enumerate(p) if i != value]
    if not moved:
        return (), 0
    first, last = min(moved), max(moved) + 1
    return tuple(value - first for value in p[first:last]), first


def build_library(folder, files):
    library = {}
    def insert(key, word):
        if not key:
            return
        left = max(0, -min(i for i, d in word))
        right = max(0, max(i + 4 for i, d in word) - len(key))
        variants = library.setdefault(key, [])
        if any(len(w) <= len(word) and l <= left and r <= right for w, l, r in variants):
            return
        variants[:] = [(w, l, r) for w, l, r in variants
                       if not (len(word) <= len(w) and left <= l and right <= r)]
        variants.append((word, left, right))
    for file in files:
        data = json.loads((folder / file).read_text())
        for row in data['records']:
            if row['status'] != 'found' or not row['word']:
                continue
            key, first = minimal_core(row['pattern'])
            word = [(i - first, d) for i, d in row['word']]
            inverse = tuple(key.index(i) for i in range(len(key)))
            inverse_word = [(i, -d) for i, d in reversed(word)]
            for q, w in ((key, word), (inverse, inverse_word)):
                insert(q, w)
                reflected = tuple(len(q) - 1 - value for value in reversed(q))
                insert(reflected, [(len(q) - 4 - i, -d) for i, d in w])
    return library


def compose(p, library, available=40):
    n = len(p)
    cuts = [0]
    highest = -1
    for i, value in enumerate(p):
        highest = max(highest, value)
        if highest == i:
            cuts.append(i + 1)
    best = {0: []}
    for end in cuts[1:]:
        for start in cuts:
            if start >= end:
                break
            if start not in best:
                continue
            key, offset = minimal_core([v - start for v in p[start:end]])
            if not key:
                candidate = best[start]
                if end not in best or len(candidate) < len(best[end]):
                    best[end] = candidate
                continue
            begin = start + offset
            for word, left, right in library.get(key, []):
                if left > begin or begin + len(key) + right > available:
                    continue
                candidate = best[start] + [(begin + i, d) for i, d in word]
                if end not in best or len(candidate) < len(best[end]):
                    best[end] = candidate
    return best.get(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='adaptive_extraction_c7_4_pruned.json')
    parser.add_argument('--k', type=int)
    args = parser.parse_args()
    folder = Path(__file__).parent
    data = json.loads((folder / args.source).read_text())
    coefficient = Fraction(data['defect_per_extracted_token'])
    k = args.k or max(r['k'] for r in data['records'])
    library = build_library(folder, ['adaptive_extraction_c2.json',
                                    'adaptive_extraction_c3_2_pruned.json', args.source])
    cases = [r for r in data['records'] if r['k'] == k and r['status'] != 'found']
    rows = []
    for case in cases:
        p = case['pattern']
        word = compose(p, library)
        inversions = sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
        success = word is not None and 3 * len(word) - inversions <= coefficient * k
        row = {'k': k, 'pattern': p, 'coefficient': str(coefficient),
               'status': 'found' if success else 'inconclusive', 'method': 'interval_composition'}
        if success:
            width = max(9, len(p), max((i + 4 for i, d in word), default=0))
            assert all(i >= 0 and d in (-1, 1) for i, d in word)
            full = p + list(range(len(p), width))
            assert apply_to(full, word) == list(range(width))
            row.update(word=word, width=width)
        else:
            row['best_composed_length'] = None if word is None else len(word)
            row['budget'] = case['budget']
        rows.append(row)
    slug = str(coefficient).replace('/', '_')
    output = folder / f'adaptive_supplement_c{slug}_k{k}_composed.json'
    output.write_text(json.dumps(rows, indent=2))
    print('Library cores', len(library), 'cases', len(rows),
          'found', sum(r['status'] == 'found' for r in rows), flush=True)


if __name__ == '__main__':
    main()
