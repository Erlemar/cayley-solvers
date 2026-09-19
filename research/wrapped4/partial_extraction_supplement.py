"""Retry bounded node-limit cases with the quotient endgame, preserving positives."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

from partial_extraction_search import endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--cap', type=int, default=100000)
    args = parser.parse_args()
    folder = Path(__file__).parent
    data = json.loads((folder / args.source).read_text())
    coefficient = Fraction(data['coefficient'])
    output = folder / ('partial_supplement_c' + str(coefficient).replace('/', '_') + '.json')
    old = json.loads(output.read_text()) if output.exists() else {'records': [], 'attempts': []}
    successes = {(r['k'], tuple(r['pattern'])): r for r in old['records']}
    attempted = {(r['k'], tuple(r['pattern'])) for r in old['attempts'] if r['cap'] >= args.cap}
    rows = [r for r in data['records'] if r['status'] == 'node_limit'
            and (r['k'], tuple(r['pattern'])) not in successes
            and (r['k'], tuple(r['pattern'])) not in attempted][:args.count]
    start = time.monotonic()
    k0 = None
    def save():
        output.write_text(json.dumps({'coefficient': str(coefficient), 'records': list(successes.values()),
                                     'attempts': old['attempts'],
                                     'scope': 'Additional positive identities only; no claim of full tree coverage.'}, indent=2))
    for index, row in enumerate(rows):
        k, width, pattern = row['k'], row['width'], row['pattern']
        if k != k0:
            endgame.cache_clear()
            k0 = k
        p = pattern + list(range(len(pattern), width))
        for seed, priority in ((7, 1), (13, 0)):
            trial = solve(p, k, (coefficient * k).__floor__(), args.cap, seed, priority)
            if trial['status'] == 'found':
                successes[k, tuple(pattern)] = {**trial, 'k': k, 'width': width, 'pattern': pattern}
                break
        old['attempts'].append({'k': k, 'pattern': pattern, 'status': trial['status'],
                                'cap': args.cap, 'seconds': trial['seconds']})
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            save()
            print(json.dumps({'done': index + 1, 'total': len(rows), 'k': k,
                              'positive_identities': len(successes), 'seconds': time.monotonic() - start}), flush=True)
    save()


if __name__ == '__main__':
    main()
