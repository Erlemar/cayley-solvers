"""A bounded, spread-out pilot on unresolved partial-extraction cases."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

from partial_extraction_beam import solve
from partial_extraction_search import endgame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--width', type=int, default=1000)
    parser.add_argument('--min-k', type=int, default=13)
    args = parser.parse_args()
    folder = Path(__file__).parent
    data = json.loads((folder / args.source).read_text())
    coefficient = Fraction(data['coefficient'])
    candidates = [r for r in data['records'] if r['status'] == 'node_limit' and r['k'] >= args.min_k]
    count = min(args.count, len(candidates))
    selected = [candidates[(len(candidates) - 1) * i // max(1, count - 1)] for i in range(count)]
    output = folder / ('partial_beam_c' + str(coefficient).replace('/', '_') + '.json')
    positives, attempts = [], []
    start = time.monotonic()
    for index, row in enumerate(selected):
        k, width, p = row['k'], row['width'], row['pattern']
        full = p + list(range(len(p), width))
        for seed in (0, 1):
            trial = solve(full, k, (coefficient * k).__floor__(), args.width, seed)
            if trial['status'] == 'found':
                positives.append({**trial, 'k': k, 'pattern': p, 'width': width})
                break
        attempts.append({'k': k, 'pattern': p, 'status': trial['status'], 'seconds': trial['seconds']})
        output.write_text(json.dumps({'coefficient': str(coefficient), 'records': positives,
                                     'attempts': attempts, 'scope': 'Positive identities from a bounded pilot, not full coverage.'}, indent=2))
        print(json.dumps({'case': index + 1, 'total': count, 'found': len(positives),
                          'status': trial['status'], 'seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
