"""Heuristic breadth-limited search, with replayable words and no negative claims."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from affine_insertion import balanced_keys
from general_sort import apply_to
from near_rotation_search import advance_lift, potential
from wide_search import STATE, apply_packed, build_endgame, pack


@njit(cache=True)
def mobility(w, n, wrapped=True):
    value = 0
    for i in range(n if wrapped else n - 3):
        a = w[i]
        b = w[(i + 1) % n] + (n if i + 1 >= n else 0)
        c = w[(i + 2) % n] + (n if i + 2 >= n else 0)
        d = w[(i + 3) % n] + (n if i + 3 >= n else 0)
        value += int(a > b and a > c and a > d)
        value += int(d < a and d < b and d < c)
    return value


@njit(cache=True)
def expand(states, lifts, lengths, lasts, n, remaining, endgame, seed, wrapped=True, endgame_radius=4):
    capacity = len(states) * 2 * n
    out_states = np.empty((capacity, 4), np.uint64)
    out_lifts = np.empty((capacity, n), np.int64)
    out_lengths = np.empty(capacity, np.int64)
    parents = np.empty(capacity, np.int64)
    actions = np.empty(capacity, np.int64)
    scores = np.empty(capacity, np.int64)
    seen = Dict.empty(key_type=STATE, value_type=types.int64)
    count = 0
    found = -1
    for parent in range(len(states)):
        state = (states[parent, 0], states[parent, 1], states[parent, 2], states[parent, 3])
        for action in range(2 * (n if wrapped else n - 3)):
            if action == (lasts[parent] ^ 1):
                continue
            new_lift, new_length = advance_lift(lifts[parent], lengths[parent], action, n, wrapped)
            if new_length > 3 * remaining:
                continue
            next_state = apply_packed(state, action, n)
            if next_state in seen:
                continue
            seen[next_state] = count
            out_states[count] = next_state
            out_lifts[count] = new_lift
            out_lengths[count] = new_length
            parents[count] = parent
            actions[count] = action
            h = (next_state[0] ^ next_state[1] ^ next_state[2] ^ next_state[3]) + np.uint64(seed + 1)
            h ^= h >> np.uint64(30)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(27)
            h *= np.uint64(0x94D049BB133111EB)
            h ^= h >> np.uint64(31)
            scores[count] = new_length * 10000 - mobility(new_lift, n, wrapped) * 100 + int(h % np.uint64(500))
            if new_length <= 3 * endgame_radius and next_state in endgame and endgame[next_state] <= remaining:
                found = count
                count += 1
                return (out_states[:count], out_lifts[:count], out_lengths[:count],
                        parents[:count], actions[:count], scores[:count], found)
            count += 1
    return (out_states[:count], out_lifts[:count], out_lengths[:count],
            parents[:count], actions[:count], scores[:count], found)


def run(p, budget, width, endgame, seed, wrapped=True, report=True, endgame_radius=4):
    n = len(p)
    states = np.array([pack(p)], np.uint64)
    lifts = np.array([balanced_keys(p) if wrapped else p], np.int64)
    lengths = np.array([int(potential(lifts[0], n))], np.int64)
    lasts = np.array([-1], np.int64)
    history = []
    trace = []
    begin = time.monotonic()
    for depth in range(1, budget + 1):
        new_states, new_lifts, new_lengths, parents, actions, scores, found = expand(
            states, lifts, lengths, lasts, n, budget - depth, endgame, seed + 1337 * depth, wrapped, endgame_radius)
        if found >= 0:
            path = [int(actions[found])]
            index = int(parents[found])
            for old_parents, old_actions in reversed(history):
                path.append(int(old_actions[index]))
                index = int(old_parents[index])
            path.reverse()
            state = tuple(np.uint64(x) for x in new_states[found])
            while int(endgame[state]):
                distance = int(endgame[state])
                for action in range(2 * (n if wrapped else n - 3)):
                    next_state = tuple(np.uint64(x) for x in apply_packed(state, action, n))
                    if next_state in endgame and int(endgame[next_state]) == distance - 1:
                        path.append(action)
                        state = next_state
                        break
                else:
                    raise AssertionError('Missing endgame step')
            word = [(a // 2, 1 if a % 2 == 0 else -1) for a in path]
            assert len(word) <= budget
            assert apply_to(p, word) == list(range(n))
            assert wrapped or all(0 <= i <= n - 4 for i, _ in word)
            return {'status': 'found', 'word': word, 'length': len(word),
                    'seconds': time.monotonic() - begin, 'trace': trace}
        if not len(scores):
            return {'status': 'beam_empty', 'seconds': time.monotonic() - begin, 'trace': trace}
        keep = (np.argpartition(scores, width - 1)[:width] if len(scores) > width
                else np.arange(len(scores)))
        states = new_states[keep].copy()
        lifts = new_lifts[keep].copy()
        lengths = new_lengths[keep].copy()
        lasts = actions[keep].copy()
        history.append((parents[keep].copy(), lasts))
        row = {'depth': depth, 'candidates': len(scores), 'kept': len(keep),
               'min_adjacent_length': int(min(lengths)), 'max_adjacent_length': int(max(lengths))}
        trace.append(row)
        if report and depth % 5 == 0:
            print(json.dumps(row), flush=True)
    return {'status': 'budget_reached', 'seconds': time.monotonic() - begin, 'trace': trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--budget', type=int, default=48)
    parser.add_argument('--width', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    n = args.n
    p = [(i + n // 2 + (1 if i % 2 == 0 else -1)) % n for i in range(n)]
    endgame = build_endgame(n, 4, 5000000)
    result = run(p, args.budget, args.width, endgame, args.seed)
    result.update(n=n, permutation=p, budget=args.budget, width=args.width, seed=args.seed,
                  scope='Heuristic search: only replayed successful words certify bounds.')
    out = Path(__file__).with_name(f'matching_beam_n{n}_b{args.budget}_w{args.width}_s{args.seed}.json')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('word', 'trace', 'permutation')}), flush=True)


if __name__ == '__main__':
    main()
