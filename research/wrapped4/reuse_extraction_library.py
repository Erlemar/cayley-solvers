"""Apply the complete 7/4 identity library to the unfinished 3/2 tree."""
from fractions import Fraction
import json
from pathlib import Path

from compose_extraction_words import build_library, compose
from general_sort import apply_to


def main():
    folder = Path(__file__).parent
    source = 'adaptive_extraction_c3_2_pruned.json'
    data = json.loads((folder / source).read_text())
    coefficient = Fraction(data['defect_per_extracted_token'])
    library = build_library(folder, ['adaptive_extraction_c2.json', source,
                                    'adaptive_extraction_c7_4_pruned.json'])
    records = []
    tried = 0
    by_k = {}
    for row in data['records']:
        if row['status'] == 'found':
            continue
        p = row['pattern']
        k = row['k']
        tried += 1
        word = compose(p, library)
        inversions = sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
        if word is None or 3 * len(word) - inversions > coefficient * k:
            continue
        width = max(9, len(p), max((i + 4 for i, d in word), default=0))
        assert all(0 <= i <= width - 4 and d in (-1, 1) for i, d in word)
        assert apply_to(p + list(range(len(p), width)), word) == list(range(width))
        records.append({'k': k, 'pattern': p, 'width': width, 'word': word,
                        'status': 'found', 'coefficient': str(coefficient),
                        'method': 'reuse_and_interval_composition'})
        by_k[k] = by_k.get(k, 0) + 1
    output = folder / 'adaptive_supplement_c3_2_reused.json'
    output.write_text(json.dumps(records, indent=2))
    print(json.dumps({'library_cores': len(library), 'tried': tried,
                      'new_identities': len(records), 'by_k': by_k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
