"""Positive-word pilot with extra unmarked positions; no coverage claim."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

from partial_extraction_beam import solve as beam
from partial_extraction_search import solve, endgame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--min-k', type=int, default=8)
    parser.add_argument('--max-k', type=int, default=40)
    parser.add_argument('--pad', type=int, default=3)
    parser.add_argument('--beam', type=int, default=1000)
    parser.add_argument('--heuristic', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(__file__).parent
    data = json.loads((folder / args.source).read_text())
    coefficient = Fraction(data['coefficient'])
    candidates = [r for r in data['records'] if r['status'] != 'found'
                  and args.min_k <= r['k'] <= args.max_k and r['width'] + args.pad <= 40]
    count = min(args.count, len(candidates))
    selected = [candidates[(len(candidates) - 1) * i // max(1, count - 1)] for i in range(count)]
    output = folder / args.output
    assert not output.exists(), 'Use a new output name to preserve prior positive words'
    records, attempts = [], []
    started = time.monotonic()
    previous_k = None
    for index, row in enumerate(selected):
        k, p, width = row['k'], row['pattern'], row['width'] + args.pad
        if k != previous_k:
            endgame.cache_clear()
            previous_k = k
        full = p + list(range(len(p), width))
        allowance = (coefficient * k).__floor__()
        trial = solve(full, k, allowance, 10000)
        if trial['status'] != 'found':
            for seed in (0, 1):
                trial = beam(full, k, allowance, args.beam, seed, args.heuristic)
                if trial['status'] == 'found':
                    break
        if trial['status'] == 'found':
            records.append({**trial, 'k': k, 'pattern': p, 'width': width})
        attempts.append({'k': k, 'pattern': p, 'width': width, 'status': trial['status']})
        output.write_text(json.dumps({'coefficient': str(coefficient), 'records': records,
                                     'attempts': attempts, 'scope': 'Positive identities only; incomplete coverage.'}, indent=2))
        print(json.dumps({'case': index + 1, 'of': count, 'found': len(records), 'k': k,
                          'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
