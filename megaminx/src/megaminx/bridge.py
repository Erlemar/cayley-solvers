"""Residual-state composition for neural bridge compression.

Given a valid path A that takes state S_i to state S_j, the *residual state*
X is the state that, when solved by the standard solver (path B with
apply_path(X, B) = solved_state), produces a bridge B such that
apply_path(S_i, B) = S_j.

Derivation under the apply convention ``new[k] = old[gen[k]]``:

    A is a sequence of moves with composed permutation
        composed(A)[k] = inv_S_i[S_j[k]]
    so that S_i[composed(A)] = S_j.

    To solve a residual state X means: find path B with
        apply_path(X, B) = solved.
    Under the convention, apply_path(X, B) = X[composed(B)], so we need
        composed(B) = inv(X).

    To use B as a bridge from S_i to S_j we need composed(B) = composed(A),
    i.e., inv(X) = composed(A), i.e., X = inv(composed(A)).

    inv(composed(A))[k] = position of k in composed(A)
                       = inv_S_j[S_i[k]]
    so X = inv_S_j[S_i] (numpy-style indexing).

Test guarantee: the *original* window path A is itself a valid solve of X
(apply_path(X, A) == solved), so the residual is well-formed by construction.
A shorter bridge B' (len(B') < len(A)) found by the solver gives a verified
improvement when spliced back in: orig[:i] + B' + orig[j:].
"""

from __future__ import annotations

import numpy as np


def make_residual(s_i, s_j):
    """X[k] = inv_S_j[S_i[k]]. Returns a length-N numpy int array.

    Solving X (finding path B with apply_path(X, B) == solved) yields a
    bridge B such that apply_path(S_i, B) == S_j.
    """
    s_i = np.asarray(s_i, dtype=np.int64)
    s_j = np.asarray(s_j, dtype=np.int64)
    n = s_i.shape[0]
    assert s_j.shape[0] == n, "states must have the same length"
    inv_s_j = np.empty(n, dtype=np.int64)
    inv_s_j[s_j] = np.arange(n, dtype=np.int64)
    return inv_s_j[s_i]


def compute_prefix_states(initial_state, path, puzzle):
    """Return a list of length len(path)+1 of state tuples along the path.

    states[0] == initial_state, states[k] = apply_path(initial_state, path[:k]).
    """
    states = [tuple(initial_state)]
    cur = tuple(initial_state)
    for move in path:
        cur = puzzle.apply_move(cur, move)
        states.append(cur)
    return states


def verify_bridge_replacement(initial_state, candidate_path, solved_state, puzzle):
    """Confirm apply_path(initial_state, candidate_path) == solved_state."""
    end = puzzle.apply_path(initial_state, candidate_path)
    return tuple(end) == tuple(solved_state)
