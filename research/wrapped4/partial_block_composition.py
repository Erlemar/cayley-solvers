"""Compose verified stable identities on invariant consecutive blocks."""
import argparse
import json
from pathlib import Path


def replay(p, word):
    out = list(p)
    for i, d in word:
        assert 0 <= i <= len(out) - 4 and d in (-1, 1)
        block = out[i:i + 4]
        out[i:i + 4] = block[d:] + block[:d]
    return out


def core(p):
    changes = [i for i, x in enumerate(p) if i != x]
    if not changes:
        return (), 0
    first, end = changes[0], changes[-1] + 1
    return tuple(x - first for x in p[first:end]), first


class StableBlocks:
    def __init__(self, filenames):
        self.library = {}
        folder = Path(__file__).parent
        for filename in filenames:
            data = json.loads((folder / filename).read_text())
            for row in data['records']:
                if row['status'] != 'found':
                    continue
                n, p, word = row['width'], list(row['pattern']), row['word']
                p += list(range(len(p), n))
                if replay(p, word) != list(range(n)):
                    continue
                inverse = [p.index(i) for i in range(n)]
                backward = [(i, -d) for i, d in reversed(word)]
                for state, path in ((p, word), (inverse, backward)):
                    self.remember(state, path)
                    mirror = [n - 1 - state[n - 1 - i] for i in range(n)]
                    reflected = [(n - 4 - i, -d) for i, d in path]
                    self.remember(mirror, reflected)

    def remember(self, p, word):
        key, offset = core(p)
        if not key:
            return
        relative = tuple((i - offset, d) for i, d in word)
        left = min(i for i, d in relative)
        right = max(i + 4 for i, d in relative)
        candidates = self.library.setdefault(key, [])
        if any(len(w) <= len(relative) and lo >= left and hi <= right
               for lo, hi, w in candidates):
            return
        candidates[:] = [(lo, hi, w) for lo, hi, w in candidates
                         if not (len(relative) <= len(w) and left >= lo and right <= hi)]
        candidates.append((left, right, relative))

    def solve(self, pattern, k, allowance, maximum_width=40):
        p = list(pattern)
        n = len(p)
        if n > maximum_width:
            return None
        cuts, high = [0], -1
        for i, x in enumerate(p):
            high = max(high, x)
            if high == i:
                cuts.append(i + 1)
        dp = {0: (0, n, [])}
        for end in cuts[1:]:
            best = None
            for start in cuts:
                if start >= end:
                    break
                if start not in dp:
                    continue
                length0, width0, word0 = dp[start]
                local = [x - start for x in p[start:end]]
                key, offset = core(local)
                if not key:
                    options = [(0, 0, ())]
                else:
                    options = self.library.get(key, ())
                shift = start + offset
                for left, right, relative in options:
                    if shift + left < 0 or shift + right > maximum_width:
                        continue
                    candidate = (length0 + len(relative), max(width0, shift + right),
                                 word0 + [(i + shift, d) for i, d in relative])
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
            if best is not None:
                dp[end] = best
        if n not in dp:
            return None
        length, width, word = dp[n]
        inv = sum(x > y for i, x in enumerate(p) for y in p[i + 1:])
        if 3 * length - inv > allowance:
            return None
        full = p + list(range(n, width))
        after = replay(full, word)
        assert after == list(range(width))
        return {'status': 'found', 'width': width, 'word': word, 'length': length,
                'endpoint': after, 'tail_inversions': 0, 'charge': 3 * length - inv,
                'discovery': 'composition_of_verified_stable_block_identities'}


def main():
    from fractions import Fraction
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--known', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(__file__).parent
    data = json.loads((folder / args.source).read_text())
    coefficient = Fraction(data['coefficient'])
    blocks = StableBlocks(args.known)
    records = []
    attempted = 0
    for row in data['records']:
        if row['status'] == 'found':
            continue
        attempted += 1
        result = blocks.solve(row['pattern'], row['k'], (coefficient * row['k']).__floor__())
        if result:
            records.append({**result, 'k': row['k'], 'pattern': row['pattern']})
    report = {'coefficient': str(coefficient), 'records': records,
              'attempted': attempted, 'library_keys': len(blocks.library),
              'scope': 'Replayed positive identities only, not a complete extraction tree.'}
    path = folder / args.output
    assert not path.exists(), 'Preserve prior positive outputs'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({'attempted': attempted, 'found': len(records),
                      'library_keys': len(blocks.library)}), flush=True)


if __name__ == '__main__':
    main()
