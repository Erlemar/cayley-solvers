"""Training data generation via non-backtracking random walks.

For each walk, we start at the solved state and take k_max steps. At each step i we pick
a random generator (excluding the inverse of the previous action so we don't trivially
undo our progress) and emit the pair (state_after_i_steps, i).

The model is trained to predict i from the state — the "diffusion distance" heuristic.
Because random walks revisit nearby states, i is an upper bound on the true distance,
which is fine (the NN learns a smoothed version that still guides search correctly).

GPU implementation: we run N walks in parallel. State is a (N, 72) int64 tensor and we
advance all walks one step at a time by fancy-indexing against a (n_gen, 72) generator
tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from cayley.puzzle import PictureCube


@dataclass
class GeneratorTable:
    """Dense representation of all generators as a single (n_gen, state_size) array plus
    an int array mapping each generator index to the index of its inverse."""

    perms: np.ndarray           # (n_gen, state_size) int64
    names: tuple[str, ...]      # length n_gen
    inverse_idx: np.ndarray     # (n_gen,) int64, inverse_idx[i] = j means names[j] == inverse(names[i])

    @classmethod
    def from_puzzle(cls, puzzle: PictureCube) -> "GeneratorTable":
        names = puzzle.move_names
        n = len(names)
        state_size = len(puzzle.solved_state)
        perms = np.zeros((n, state_size), dtype=np.int64)
        for i, name in enumerate(names):
            perms[i] = np.array(puzzle.generators[name], dtype=np.int64)
        name_to_idx = {name: i for i, name in enumerate(names)}
        inverse_idx = np.zeros(n, dtype=np.int64)
        for i, name in enumerate(names):
            inv_name = puzzle.inverse_name(name)
            inverse_idx[i] = name_to_idx[inv_name]
        return cls(perms=perms, names=names, inverse_idx=inverse_idx)


def generate_walks_numpy(
    puzzle: PictureCube,
    n_walks: int,
    k_max: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data via non-backtracking random walks on the CPU.

    Returns:
        states: (n_walks * k_max, state_size) int64 — the state after step i of each walk
        depths: (n_walks * k_max,) int64 — the walk step (1..k_max) that produced each state
    """
    rng = np.random.default_rng(seed)
    gens = GeneratorTable.from_puzzle(puzzle)
    n_gen = gens.perms.shape[0]
    state_size = gens.perms.shape[1]
    solved = np.array(puzzle.solved_state, dtype=np.int64)

    states = np.broadcast_to(solved, (n_walks, state_size)).copy()  # (n_walks, state_size)
    prev_action = np.full(n_walks, -1, dtype=np.int64)

    out_states = np.empty((n_walks * k_max, state_size), dtype=np.int64)
    out_depths = np.empty(n_walks * k_max, dtype=np.int64)

    for k in range(1, k_max + 1):
        # Sample action per walk, resample where it equals the inverse of the previous.
        action = rng.integers(0, n_gen, size=n_walks)
        # Non-backtracking: when prev_action != -1 AND action == inverse(prev_action), resample.
        # Do a couple of passes; collisions are at most ~5% per pass so this converges fast.
        for _ in range(4):
            bad = (prev_action >= 0) & (action == gens.inverse_idx[np.where(prev_action >= 0, prev_action, 0)])
            if not bad.any():
                break
            action[bad] = rng.integers(0, n_gen, size=int(bad.sum()))

        # Apply: new[i, j] = state[i, perms[action[i], j]]
        gen_rows = gens.perms[action]  # (n_walks, state_size)
        states = np.take_along_axis(states, gen_rows, axis=1)
        prev_action = action

        off = (k - 1) * n_walks
        out_states[off : off + n_walks] = states
        out_depths[off : off + n_walks] = k

    return out_states, out_depths


def generate_walks_torch(
    puzzle: PictureCube,
    n_walks: int,
    k_max: int,
    seed: int = 0,
    device: str = "cuda",
    n_back: int = 1,
    max_resamples: int = 6,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """GPU version of generate_walks_numpy.

    `n_back`: ban the inverse of any of the last `n_back` actions. The Jan-2025 chat
    reports a large improvement (Kaggle 10,281) using `n_back=40`. Note: the 18 generators
    form 9 inverse pairs, so `n_back=40` sometimes forces repeat-heavy sequences — that's
    OK, it just means training sees more interesting subsequences. If no valid action is
    left (all 9 inverse pairs would be banned), we bail out of the ban check and sample
    freely.

    Returns tensors on the requested device:
        states: (n_walks * k_max, state_size) int64
        depths: (n_walks * k_max,) int64
    """
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(device)              # (n_gen, state_size)
    inv = torch.from_numpy(gens.inverse_idx).to(device)          # (n_gen,)
    n_gen = perms.shape[0]
    state_size = perms.shape[1]

    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
    states = solved.unsqueeze(0).expand(n_walks, state_size).clone()

    # History of last `n_back` action indices per walk; -1 means "empty slot".
    history = torch.full((n_walks, n_back), -1, dtype=torch.int64, device=device)

    out_states = torch.empty((n_walks * k_max, state_size), dtype=torch.int64, device=device)
    out_depths = torch.empty(n_walks * k_max, dtype=torch.int64, device=device)

    for k in range(1, k_max + 1):
        action = torch.randint(0, n_gen, (n_walks,), generator=g, device=device)
        if n_back > 0:
            # Compute banned set per walk: inverses of any action in the history.
            # history: (n_walks, n_back), inv: (n_gen,). Banned inverses: inv[history]
            # with -1 slots mapping to a sentinel. Use torch.where to mask.
            hist_safe = torch.where(history >= 0, history, torch.zeros_like(history))
            banned = torch.where(history >= 0, inv[hist_safe], torch.full_like(hist_safe, -1))
            # banned: (n_walks, n_back). Resample while any walk's action is banned.
            for _ in range(max_resamples):
                bad = (action.unsqueeze(1) == banned).any(dim=1)
                if not bool(bad.any()):
                    break
                n_bad = int(bad.sum().item())
                new_actions = torch.randint(0, n_gen, (n_bad,), generator=g, device=device)
                action = action.clone()
                action[bad] = new_actions
            # else: after max_resamples, accept remaining bad actions (happens when all gens banned)

        gen_rows = perms[action]                                 # (n_walks, state_size)
        states = torch.gather(states, 1, gen_rows)
        # Shift history left and append current action at the end.
        if n_back > 0:
            history = torch.cat([history[:, 1:], action.unsqueeze(1)], dim=1)

        off = (k - 1) * n_walks
        out_states[off : off + n_walks] = states
        out_depths[off : off + n_walks] = k

    return out_states, out_depths


def sample_from_bfs_table(
    bfs_table,  # cayley.bfs_table.BfsTable
    n_samples: int,
    device: str,
    seed: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Sample `n_samples` (state, distance) pairs uniformly from a BFS table.

    The table maps permutation tuples to shortest-word tuples; distance = len(word).
    These are EXACT labels (unlike random-walk depths which are upper bounds), so mixing
    them into training provides unbiased signal near the goal. Identity (depth 0) is
    naturally included and beneficial — it's the only "solved" state the model sees.
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    all_perms = list(bfs_table.table.keys())
    n = len(all_perms)
    idx = rng.integers(0, n, size=n_samples)
    state_size = len(all_perms[0])

    states_np = np.empty((n_samples, state_size), dtype=np.int64)
    depths_np = np.empty(n_samples, dtype=np.int64)
    for i, j in enumerate(idx):
        perm = all_perms[j]
        word = bfs_table.table[perm]
        states_np[i] = perm
        depths_np[i] = len(word)

    return (
        torch.from_numpy(states_np).to(device),
        torch.from_numpy(depths_np).to(device),
    )


def load_kociemba_walks(
    path: str,
    device: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Load a precomputed set of (state, distance) pairs derived from Kociemba solutions.

    Each entry is a state reached while replaying a Kociemba solution backwards from the
    scrambled state towards solved; distance is the number of remaining moves along that
    path to reach solved. Upper bound on true distance, but drawn from a realistic
    test-like state distribution (unlike random walks from solved).

    Build with `scripts/build_kociemba_walks.py`.
    """
    import pickle

    import torch

    with open(path, "rb") as f:
        data = pickle.load(f)
    states = torch.from_numpy(data["states"]).to(device)
    depths = torch.from_numpy(data["depths"]).to(device)
    return states, depths
