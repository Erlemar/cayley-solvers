"""Standard-library-only independent replay of matching search certificates."""
import json
from pathlib import Path


def main():
    folder = Path(__file__).parent
    best = {}
    checked = 0
    files = sorted({path for pattern in ('matching_beam_*.json',
                                        'matching_bidirectional_*.json',
                                        'matching_prefix_*.json')
                    for path in folder.glob(pattern)})
    for path in files:
        row = json.loads(path.read_text())
        if row.get('status') != 'found':
            continue
        n = row['n']
        assert n >= 8 and n % 4 == 0
        expected = [(i + n // 2 + (1 if i % 2 == 0 else -1)) % n for i in range(n)]
        assert row['permutation'] == expected
        state = expected[:]
        for i, direction in row['word']:
            assert 0 <= i < n and direction in (-1, 1)
            positions = [(i + j) % n for j in range(4)]
            old = [state[pos] for pos in positions]
            for j, pos in enumerate(positions):
                state[pos] = old[(j + direction) % 4]
        assert state == list(range(n))
        assert len(row['word']) == row['length'] <= row['budget']
        m = n // 2
        ell = m * (m - 1)
        lower = 2 * ((ell + 7) // 6)
        assert lower <= row['length']
        checked += 1
        if n not in best or row['length'] < best[n]['upper_bound']:
            best[n] = {'n': n, 'permutation': expected, 'adjacent_length': ell,
                       'lower_bound': lower, 'upper_bound': row['length'],
                       'exact': lower == row['length'], 'word': row['word'],
                       'source': path.name}
    assert best[24]['exact'] and best[24]['upper_bound'] == 46
    result = {'replayed_successful_words': checked,
              'lower_bound_proof': 'ALTERNATING_MATCHING.md',
              'rows': [best[n] for n in sorted(best)]}
    (folder / 'alternating_matching_certificates.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'replayed_successful_words': checked,
                      'bounds': [{k: v for k, v in row.items() if k not in ('word', 'permutation')}
                                 for row in result['rows']]}, indent=2))


if __name__ == '__main__':
    main()
