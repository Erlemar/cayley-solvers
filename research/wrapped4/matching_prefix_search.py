"""Test a fixed four-move preparation followed only by three-unit descents."""
import argparse
import json
from pathlib import Path

from general_sort import apply_to
from matching_beam import run
from wide_search import build_endgame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=36)
    parser.add_argument('--width', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    n = args.n
    assert n % 12 == 0 and n >= 12
    p = [(i + n // 2 + (1 if i % 2 == 0 else -1)) % n for i in range(n)]
    prefix = [(3, -1), (5, -1), (7, -1), (0, 1)]
    prepared = apply_to(p, prefix)
    budget = (n * n // 4 - n // 2) // 3 + 2
    endgame = build_endgame(n, 4, 5000000)
    result = run(prepared, budget - len(prefix), args.width, endgame, args.seed)
    if result['status'] == 'found':
        result['word'] = prefix + result['word']
        result['length'] = len(result['word'])
        assert len(result['word']) == budget
        assert apply_to(p, result['word']) == list(range(n))
    result.update(n=n, permutation=p, budget=budget, prefix=prefix,
                  width=args.width, seed=args.seed,
                  scope='Finite constructive tests, not an all-n induction.')
    name = f'matching_prefix_n{n}_w{args.width}_s{args.seed}.json'
    Path(__file__).with_name(name).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('word', 'trace', 'permutation')}), flush=True)


if __name__ == '__main__':
    main()
