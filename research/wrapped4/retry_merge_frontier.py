"""Replace inconclusive terminal-frontier cases by independently replayed words."""
import argparse
import json
from pathlib import Path

from general_sort import apply_to
from matching_beam import run
from wide_search import build_endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--extra-orientations', action='store_true')
    args = parser.parse_args()
    folder = Path(__file__).parent
    path = folder / 'merge_extraction_c1_2_wide_trim.json'
    backup = folder / 'merge_extraction_c1_2_wide_trim_before_beam.json'
    if not backup.exists():
        backup.write_text(path.read_text())
    data = json.loads(path.read_text())
    k = max(len(row['bits']) for row in data['records'])
    cases = [(i, row) for i, row in enumerate(data['records'])
             if len(row['bits']) == k and row['status'] == 'node_limit'][:args.limit]
    endgames = {}
    log_path = folder / 'merge_frontier_retries.json'
    attempts = json.loads(log_path.read_text()) if log_path.exists() else []
    old_count = len(attempts)
    for index, row in cases:
        full = row['pattern'] + list(range(len(row['pattern']), row['width']))
        n = row['solver_width']
        p = full[:n]
        assert full[n:] == list(range(n, len(full)))
        if n not in endgames:
            endgames[n] = build_endgame(n, 4, 5000000, False)
        inverse = [p.index(i) for i in range(n)]
        inv_result = solve(inverse, row['budget'], endgames[n], 4, 100000,
                           seed=7, wrapped=False)
        if inv_result['status'] == 'found':
            word = [(i, -d) for i, d in reversed(inv_result['word'])]
            result = {'status': 'found', 'word': word, 'method': 'inverse_dfs'}
        else:
            result = run(p, row['budget'], args.width, endgames[n],
                         seed=index, wrapped=False, report=False)
            result['method'] = 'beam'
        if result['status'] != 'found' and args.extra_orientations:
            reversed_p = [n - 1 - value for value in reversed(p)]
            retry = solve(reversed_p, row['budget'], endgames[n], 4, 300000,
                          seed=13, wrapped=False)
            if retry['status'] == 'found':
                result = {'status': 'found', 'word': [(n - 4 - i, -d) for i, d in retry['word']],
                          'method': 'reversed_dfs'}
            else:
                retry = run(inverse, row['budget'], args.width, endgames[n],
                            seed=index + 7, wrapped=False, report=False)
                if retry['status'] == 'found':
                    result = {'status': 'found', 'word': [(i, -d) for i, d in reversed(retry['word'])],
                              'method': 'inverse_beam'}
        if result['status'] == 'found':
            assert apply_to(full, result['word']) == list(range(len(full)))
            assert all(0 <= i <= n - 4 for i, _ in result['word'])
            assert 3 * len(result['word']) - row['inversions'] <= k / 2
            row['status'] = 'found'
            row['word'] = result['word']
            row['discovery_method'] = result['method']
        attempts.append({'record_index': index, 'bits': row['bits'], 'gap': row['gap'],
                         'status': result['status'], 'method': result['method']})
        data['complete'] = all(r['status'] == 'found' for r in data['records'] if len(r['bits']) == k)
        if len(attempts) % 20 == 0:
            path.write_text(json.dumps(data, indent=2))
            log_path.write_text(json.dumps(attempts, indent=2))
            print('ATTEMPTS', len(attempts), 'found', sum(a['status'] == 'found' for a in attempts), flush=True)
    path.write_text(json.dumps(data, indent=2))
    log_path.write_text(json.dumps(attempts, indent=2))
    print('SUMMARY', sum(a['status'] == 'found' for a in attempts[old_count:]), 'of', len(attempts) - old_count, flush=True)


if __name__ == '__main__':
    main()
