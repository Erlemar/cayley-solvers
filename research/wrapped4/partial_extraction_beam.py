"""Breadth-limited positive-word search for the charged partial-extraction goal."""
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from partial_extraction_search import endgame, local_inversions, project
from partial_prefix_pdb import score as pdb_score, tables as pdb_tables
from wide_search import STATE, apply_packed, pack


@njit(cache=True)
def expand(states, inversions, marked_inversions, lasts, n, k, depth, remaining,
           initial_inv, allowance, finish, seed, heuristic, groups, distances):
    capacity = len(states) * 2 * (n - 3)
    output = np.empty((capacity, 4), np.uint64)
    out_inv = np.empty(capacity, np.int64)
    out_marked = np.empty(capacity, np.int64)
    parents = np.empty(capacity, np.int64)
    actions = np.empty(capacity, np.int64)
    scores = np.empty(capacity, np.int64)
    seen = Dict.empty(key_type=STATE, value_type=types.int64)
    count, found = 0, -1
    for parent in range(len(states)):
        state = (states[parent, 0], states[parent, 1], states[parent, 2], states[parent, 3])
        for action in range(2 * (n - 3)):
            if action == (lasts[parent] ^ 1):
                continue
            values = np.empty(4, np.int64)
            for j in range(4):
                position = action // 2 + j
                values[j] = int((state[position // 10] >> np.uint64(6 * (position % 10))) & np.uint64(63))
            old_i, old_m = local_inversions(values, k)
            changed = np.empty(4, np.int64)
            for j in range(4):
                changed[j] = values[(j + (1 if action % 2 == 0 else 3)) % 4]
            new_i, new_m = local_inversions(changed, k)
            inv = inversions[parent] + new_i - old_i
            marked = marked_inversions[parent] + new_m - old_m
            if 3 * depth + inv - initial_inv > allowance or marked > 3 * remaining:
                continue
            other = apply_packed(state, action, n)
            if other in seen:
                continue
            if remaining <= 4 and marked:
                goal_key = project(other, k, n)
                if goal_key not in finish or finish[goal_key] > remaining:
                    continue
            seen[other] = count
            output[count] = other
            out_inv[count], out_marked[count] = inv, marked
            parents[count], actions[count] = parent, action
            h = (other[0] ^ other[1] ^ other[2] ^ other[3]) + np.uint64(seed + 1337 * depth + 1)
            h ^= h >> np.uint64(30)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(27)
            h *= np.uint64(0x94D049BB133111EB)
            h ^= h >> np.uint64(31)
            if heuristic:
                rank = pdb_score(other, n, k, groups, distances) * (2 if seed % 2 else 4) + inv
            else:
                rank = marked * (1 if seed % 2 else 4) + inv
            scores[count] = rank * 1024 + int(h % np.uint64(1024))
            if marked == 0:
                found = count
                count += 1
                return (output[:count], out_inv[:count], out_marked[:count],
                        parents[:count], actions[:count], scores[:count], found)
            count += 1
    return (output[:count], out_inv[:count], out_marked[:count],
            parents[:count], actions[:count], scores[:count], found)


def solve(p, k, allowance, width=1000, seed=0, heuristic=0):
    n = len(p)
    initial_inv = sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n))
    initial_marked = sum(p[i] > p[j] and min(p[i], p[j]) < k for i in range(n) for j in range(i + 1, n))
    assert initial_marked > 0
    budget = (initial_inv + allowance) // 3
    states = np.array([pack(p)], np.uint64)
    invs, marked = np.array([initial_inv]), np.array([initial_marked])
    lasts = np.array([-1])
    history = []
    start = time.monotonic()
    finish = endgame(n, k)
    groups, distances = pdb_tables(n, k) if heuristic else (np.empty((0, 3), np.int64), np.empty((0, 0), np.int16))
    total = 0
    for depth in range(1, budget + 1):
        grown, gi, gm, parents, actions, scores, found = expand(
            states, invs, marked, lasts, n, k, depth, budget - depth,
            initial_inv, allowance, finish, seed, heuristic, groups, distances)
        total += len(scores)
        if found >= 0:
            path = [int(actions[found])]
            index = int(parents[found])
            for prev_parents, prev_actions in reversed(history):
                path.append(int(prev_actions[index]))
                index = int(prev_parents[index])
            word = [(a // 2, 1 if a % 2 == 0 else -1) for a in reversed(path)]
            after = list(p)
            for i, d in word:
                block = after[i:i + 4]
                after[i:i + 4] = block[d:] + block[:d]
            assert after[:k] == list(range(k))
            tail = sum(after[i] > after[j] for i in range(k, n) for j in range(i + 1, n))
            charge = 3 * len(word) + tail - initial_inv
            assert charge <= allowance
            return {'status': 'found', 'word': word, 'endpoint': after, 'length': len(word),
                    'tail_inversions': tail, 'charge': charge, 'seconds': time.monotonic() - start,
                    'expanded_candidates': total, 'beam_width': width, 'seed': seed, 'heuristic': heuristic}
        if not len(scores):
            break
        keep = np.argpartition(scores, width - 1)[:width] if len(scores) > width else np.arange(len(scores))
        states, invs, marked = grown[keep].copy(), gi[keep].copy(), gm[keep].copy()
        lasts = actions[keep].copy()
        history.append((parents[keep].copy(), lasts))
    return {'status': 'inconclusive_beam', 'seconds': time.monotonic() - start,
            'expanded_candidates': total, 'beam_width': width, 'seed': seed, 'heuristic': heuristic}
