"""Finite extraction trees permitting a charged permutation of the unselected tail."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

from adaptive_extraction import initial_patterns, refinements
from partial_extraction_search import endgame, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficient', type=Fraction, default=Fraction(3, 2))
    parser.add_argument('--max-k', type=int, default=10)
    parser.add_argument('--cap', type=int, default=30000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse-known', nargs='*', default=[])
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--reuse-only', action='store_true')
    parser.add_argument('--stable-blocks', action='store_true')
    parser.add_argument('--beam-heuristic', type=int, default=0)
    parser.add_argument('--output-file')
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--max-frontier', type=int, default=20000)
    args = parser.parse_args()
    output = Path(__file__).with_name('partial_extraction_c' + str(args.coefficient).replace('/', '_') + '.json')
    if args.output_file:
        output = Path(__file__).with_name(args.output_file)
    def core(p):
        changed = [i for i, x in enumerate(p) if i != x]
        if not changed:
            return (), 0
        first, end = changed[0], changed[-1] + 1
        return tuple(x - first for x in p[first:end]), first
    known = {}
    def remember(pattern, word, width, filename):
        key, offset = core(pattern)
        known.setdefault(key, []).append((word, width, offset, filename))
    for filename in args.reuse_known:
        source = json.loads(Path(__file__).with_name(filename).read_text())
        for row in source['records']:
            if row['status'] != 'found':
                continue
            remember(row['pattern'], row['word'], row['width'], filename)
    blocks = None
    if args.stable_blocks:
        from partial_block_composition import StableBlocks
        blocks = StableBlocks(args.reuse_known)
    def reuse(pattern, k, allowance):
        key, offset = core(pattern)
        for original_word, width, old_offset, filename in known.get(key, []):
            shift = offset - old_offset
            if any(i + shift < 0 for i, _ in original_word):
                continue
            word = [(i + shift, d) for i, d in original_word]
            span = max(width + shift, len(pattern))
            before = list(pattern) + list(range(len(pattern), span))
            after = before[:]
            for i, d in word:
                assert 0 <= i <= span - 4 and d in (-1, 1)
                block = after[i:i + 4]
                after[i:i + 4] = block[d:] + block[:d]
            if after[:k] != list(range(k)):
                continue
            inv = sum(before[i] > before[j] for i in range(span) for j in range(i + 1, span))
            tail = sum(after[i] > after[j] for i in range(k, span) for j in range(i + 1, span))
            charge = 3 * len(word) + tail - inv
            if charge <= allowance:
                return {'status': 'found', 'word': word, 'length': len(word), 'width': span,
                        'endpoint': after, 'tail_inversions': tail, 'charge': charge,
                        'discovery': 'replayed_saved_identity', 'source': filename}
        return None
    pending = initial_patterns()
    rows, stages = [], []
    start_k = 3
    if args.resume:
        old = json.loads(output.read_text())
        assert old['coefficient'] == str(args.coefficient)
        if old['complete']:
            print('Already complete', flush=True)
            return
        rows, stages = old['records'], old['stages']
        previous = max(r['k'] for r in rows)
        pending = set()
        for row in rows:
            if row['k'] == previous and row['status'] != 'found':
                pending.update(refinements(row['pattern'], previous))
        start_k = previous + 1
    start = time.monotonic()
    for k in range(start_k, args.max_k + 1):
        if len(pending) > args.max_frontier:
            print(json.dumps({'stopped_before_k': k, 'frontier': len(pending),
                              'reason': 'configured_frontier_limit'}), flush=True)
            break
        endgame.cache_clear()
        unresolved = set()
        count = Counter()
        ordered = sorted(pending)
        time_limit = False
        for index, p in enumerate(ordered):
            width = max(9, len(p) + args.padding)
            full = list(p) + list(range(len(p), width))
            allowance = (args.coefficient * k).__floor__()
            result = reuse(p, k, allowance)
            if result is None and blocks is not None:
                result = blocks.solve(p, k, allowance)
            if result is None:
                if args.reuse_only:
                    result = {'status': 'not_searched'}
                else:
                    result = solve(full, k, allowance, args.cap) if width <= 40 else {'status': 'too_wide'}
            else:
                width = result['width']
            if result['status'] == 'node_limit':
                retry = solve(full, k, allowance, args.cap, seed=7, priority=1)
                if retry['status'] == 'found':
                    result = retry
            if result['status'] == 'node_limit' and args.beam:
                from partial_extraction_beam import solve as beam_solve
                for seed in (0, 1):
                    retry = beam_solve(full, k, allowance, args.beam, seed, args.beam_heuristic)
                    if retry['status'] == 'found':
                        result = retry
                        break
            row = {**result, 'k': k, 'pattern': p, 'width': width}
            rows.append(row)
            count[result['status']] += 1
            if result['status'] == 'found':
                remember(p, row['word'], width, str(output.name))
            if result['status'] != 'found':
                unresolved.add(p)
            if args.max_seconds and time.monotonic() - start >= args.max_seconds:
                for other in ordered[index + 1:]:
                    rows.append({'status': 'not_searched', 'k': k, 'pattern': other,
                                 'width': max(9, len(other) + args.padding)})
                    unresolved.add(other)
                    count['not_searched'] += 1
                time_limit = True
                break
        stage = {'k': k, 'patterns': len(pending), 'counts': dict(count), 'seconds': time.monotonic() - start}
        stages.append(stage)
        complete = not unresolved
        output.write_text(json.dumps({'coefficient': str(args.coefficient), 'complete': complete,
                                     'stages': stages, 'records': rows}, indent=2))
        print(json.dumps(stage), flush=True)
        if complete or count['too_wide'] or time_limit:
            break
        pending = set()
        for p in unresolved:
            pending.update(refinements(p, k))


if __name__ == '__main__':
    main()
