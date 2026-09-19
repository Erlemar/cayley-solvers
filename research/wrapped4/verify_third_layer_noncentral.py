"""Independent finite coverage and replay for the noncentral K-3 families."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from verify_two_defect_certificate import orbit, replay


def main():
    folder = Path(__file__).parent
    rows = json.loads((folder / 'third_layer_noncentral_words.json').read_text())
    expected = set()
    raw = Counter()
    for n in range(14, 25, 2):
        for size in (3, 4):
            for tail in combinations(range(1, n), size - 1):
                support = (0,) + tail
                gaps = [(support[(i + 1) % size] - support[i]) % n - 1 for i in range(size)]
                if n != 14 and max(gaps) > 5:
                    continue
                patterns = ((1, 2, 0), (2, 0, 1)) if size == 3 else ((1, 0, 3, 2), (3, 2, 1, 0))
                for core in patterns:
                    p = [(i + n // 2 - 1) % n for i in range(n)]
                    for i, x in enumerate(core):
                        p[support[i]] = (support[x] + n // 2 - 1) % n
                    expected.add(min(orbit(p)))
                    raw[n] += 1
    for separation in range(1, 7):
        p = [(i + 5) % 13 for i in range(13)]
        p[0], p[separation] = p[separation], p[0]
        expected.add(min(orbit(p)))
        raw[13] += 1
    actual = set()
    lengths = Counter()
    for row in rows:
        p, word = row['permutation'], row['word']
        n = len(p)
        replay(p, word)
        ell = n * n // 4 - 3
        assert len(word) == ell // 3 + ell % 3
        actual.add(min(orbit(p)))
        lengths[n, len(word)] += 1
    assert actual == expected and len(actual) == len(rows)
    report = {'complete': True, 'independent_models_by_size': dict(sorted(raw.items())),
              'independent_orbits': len(expected), 'words_replayed': len(rows),
              'moves_replayed': sum(len(r['word']) for r in rows),
              'all_words_attain_adjacent_and_parity_lower_bound': True,
              'lengths_by_size': {str(k): v for k, v in sorted(lengths.items())},
              'scope': 'All finite cases in THIRD_LAYER_NONCENTRAL.md; no search failures are used.'}
    (folder / 'third_layer_noncentral_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
