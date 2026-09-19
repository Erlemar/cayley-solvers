"""Merge new verified identities and remove branches beneath solved ancestors."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from adaptive_extraction import initial_patterns, refinements
from verify_refined_extraction import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='adaptive_extraction_c3_2_wide.json')
    parser.add_argument('--output', default='adaptive_extraction_c3_2_pruned.json')
    args = parser.parse_args()
    folder = Path(__file__).parent
    source = folder / args.source
    data = json.loads(source.read_text())
    coefficient = Fraction(data['defect_per_extracted_token'])
    table = {(r['k'], tuple(r['pattern'])): r for r in data['records']}
    additions = 0
    for file in sorted(folder.glob('adaptive_supplement_*.json')):
        for extra in json.loads(file.read_text()):
            if extra['status'] != 'found':
                continue
            key = extra['k'], tuple(extra['pattern'])
            if key not in table:
                continue
            pattern = extra['pattern']
            inversions = sum(pattern[i] > pattern[j] for i in range(len(pattern))
                             for j in range(i + 1, len(pattern)))
            if 3 * len(extra['word']) - inversions > coefficient * extra['k']:
                continue
            old = table[key]
            if old['status'] == 'found' and len(old['word']) <= len(extra['word']):
                continue
            full = extra['pattern'] + list(range(len(extra['pattern']), extra['width']))
            table[key] = {**old, **extra, 'permutation': full,
                          'length': len(extra['word']), 'supplement_source': file.name}
            additions += 1
    pending = initial_patterns()
    records = []
    stages = []
    for k in range(3, max(key[0] for key in table) + 1):
        current = [table[k, p] for p in sorted(pending)]
        counts = Counter(r['status'] for r in current)
        stages.append({'k': k, 'patterns': len(current),
                       **{s: counts[s] for s in ('found', 'exhausted', 'node_limit', 'too_wide')}})
        records.extend(current)
        pending = set()
        for row in current:
            if row['status'] != 'found':
                pending.update(refinements(row['pattern'], k))
        if not pending:
            break
    result = {'defect_per_extracted_token': data['defect_per_extracted_token'],
              'stages': stages, 'complete': not pending,
              'maximum_width': max(r['width'] for r in records), 'records': records,
              'pruned_from': source.name, 'additional_identities': additions}
    output = folder / args.output
    output.write_text(json.dumps(result, indent=2))
    audit = check(output)
    output.with_name(output.stem + '_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != 'stages'}, indent=2), flush=True)
    print('Old cases', len(data['records']), 'new', len(records), 'additional identities', additions, flush=True)


if __name__ == '__main__':
    main()
