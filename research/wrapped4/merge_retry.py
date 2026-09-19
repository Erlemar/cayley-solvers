"""Retry inconclusive merge searches without changing their original records."""
import json
from collections import Counter
from pathlib import Path

from near_rotation_search import build_endgame, solve


def main():
    folder = Path(__file__).parent
    original = json.loads((folder / 'merge_n16_defect14.json').read_text())
    endgame = build_endgame(16, 4, 2000000, False)
    results = []
    for index, row in enumerate(original['rows']):
        if row['status'] != 'node_limit':
            continue
        attempts = []
        for seed in (19, 3, 11):
            attempt = solve(row['permutation'], row['budget'], endgame, 4,
                            1000000, seed=seed, wrapped=False)
            attempt['seed'] = seed
            attempts.append(attempt)
            if attempt['status'] != 'node_limit':
                break
        result = {'original_case': index, 'status': attempts[-1]['status'],
                  'attempts': attempts}
        results.append(result)
        print(index, result['status'], [r['nodes'] for r in attempts], flush=True)
        (folder / 'merge_n16_defect14_retries.json').write_text(
            json.dumps(results, indent=2))
    print('SUMMARY', dict(Counter(r['status'] for r in results)), flush=True)


if __name__ == '__main__':
    main()
