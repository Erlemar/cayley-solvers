"""MITM-augmented beam search.

Wraps `KhoruzhiiSolver` and terminates the beam as soon as ANY beam state lands in
the precomputed BFS-d6 shell (19.4M states, depth ≤ 6 from solved). Final path is
beam-prefix + reversed-inverse(bfs_table_path_from_identity_to_state).

Effect: search depth (D - d_shell) instead of D. On hard puzzles this is the
difference between "solver burns 60 steps and times out" and "solver hits the
shell at step 30 and we splice the known-optimal tail".

Quality: zero loss (BFS path is exact). Strict improvement or no-op vs single-puzzle
beam search.

Loads `BfsBytesTable` from production (`megaminx/src/megaminx/bfs_bytes.py`).
The .pkl was pickled with that class, so we need the production module on sys.path
to unpickle.
"""
from __future__ import annotations

import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from beam_search import (
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    Profile,
    _model_predict,
    _state_hash,
    _sync,
)


# `bfs_bytes_d*.pkl` was pickled with class `megaminx.bfs_bytes.BfsBytesTable`. To
# unpickle on another machine without the production code, register an alias so
# pickle can find a structurally compatible class. We also try the real production
# import first if it's available — that keeps the original class identity when
# beam_lab is run inside the main repo.
@dataclass
class BfsBytesTable:
    """bytes-keyed BFS table. `table[state_bytes] = path_bytes`. Structurally identical
    to `megaminx.bfs_bytes.BfsBytesTable` so the same .pkl unpickles into either."""

    table: dict[bytes, bytes]
    max_depth: int
    puzzle_name: str
    move_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "BfsBytesTable":
        with open(path, "rb") as f:
            return pickle.load(f)


def _register_bfs_bytes_alias() -> None:
    """Make `megaminx.bfs_bytes.BfsBytesTable` resolve to OUR `BfsBytesTable` so the
    .pkl can be unpickled on machines without the production code on PYTHONPATH.

    If the production module is already importable, do nothing — we want to use the
    real class in that case (so any helper methods defined there still work)."""
    try:
        import megaminx.bfs_bytes  # noqa: F401
        return  # production import works; nothing to do
    except ImportError:
        pass

    import types
    # Construct fake module hierarchy: megaminx and megaminx.bfs_bytes
    if "megaminx" not in sys.modules:
        sys.modules["megaminx"] = types.ModuleType("megaminx")
    fake = types.ModuleType("megaminx.bfs_bytes")
    fake.BfsBytesTable = BfsBytesTable
    sys.modules["megaminx.bfs_bytes"] = fake


_register_bfs_bytes_alias()


def load_bfs_table(path: str | Path) -> BfsBytesTable:
    """Load a BFS bytes-keyed table. Works whether or not the production
    `megaminx.bfs_bytes` is on PYTHONPATH."""
    t0 = time.time()
    with open(path, "rb") as f:
        table = pickle.load(f)
    n = len(table.table)
    md = getattr(table, "max_depth", "?")
    print(f"[MITM] loaded {path}: {n:,} states, max_depth={md}, "
          f"{(time.time() - t0):.1f}s")
    return table


class MitmKhoruzhiiSolver(KhoruzhiiSolver):
    """Beam search that terminates on entry to a precomputed BFS-d shell.

    Construction precomputes the hash of every state in `mitm_table` once on the
    GPU. At each beam step, `torch.isin(beam_hashes, mitm_hashes)` flags any match.
    On match, the matched state's BFS path is spliced onto the end of the beam path.
    """

    def __init__(
        self,
        puzzle,
        model,
        mitm_table: BfsBytesTable,
        device: str = "cuda",
        internal_batch_size: int = 16384,
        random_seed: int = 0,
        state_dtype: torch.dtype = torch.int8,
        pad_to_batch_size: bool = False,
    ):
        super().__init__(
            puzzle=puzzle, model=model, device=device,
            internal_batch_size=internal_batch_size,
            random_seed=random_seed, state_dtype=state_dtype,
            pad_to_batch_size=pad_to_batch_size,
        )
        self.mitm_table = mitm_table

        # Map each generator name to the index of its inverse, so we can invert a
        # BFS path (which goes identity → state) to get state → identity.
        inv_names = [puzzle.inverse_name(n) for n in puzzle.move_names]
        self.inv_idx_np = np.array(
            [puzzle.move_names.index(n) for n in inv_names], dtype=np.int64
        )

        # Hash all 19M shell states once on the GPU. ~152 MB at int64.
        N = len(mitm_table.table)
        print(f"[MITM] hashing {N:,} target states on {device}…", flush=True)
        t0 = time.time()
        all_states = np.empty((N, self.state_size), dtype=np.int8)
        # Keep keys parallel to all_states so we can recover the BFS path on match.
        self._all_keys: list[bytes] = list(mitm_table.table.keys())
        for i, k in enumerate(self._all_keys):
            all_states[i] = np.frombuffer(k, dtype=np.int8)
        all_hashes = torch.empty(N, dtype=torch.int64, device=device)
        bs = 65536
        for i in range(0, N, bs):
            chunk = torch.from_numpy(all_states[i : i + bs]).to(device=device, dtype=torch.int64)
            all_hashes[i : i + bs] = torch.sum(self.hash_vec * chunk, dim=1)
        # Sorted form for fast isin().
        sorted_h, _ = torch.sort(all_hashes)
        self._mitm_hashes_sorted = sorted_h
        self._mitm_hashes = all_hashes  # unsorted; index matches self._all_keys
        print(f"[MITM] shell hashing done in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------- helpers

    def _mitm_find_match(self, states: torch.Tensor, hashed: torch.Tensor):
        """Return (beam_idx, key_bytes) of the first beam state in the MITM set,
        or None if no match. Verifies hash collisions by comparing actual state bytes."""
        in_set = torch.isin(hashed, self._mitm_hashes_sorted, assume_unique=False)
        hit_positions = torch.nonzero(in_set, as_tuple=True)[0]
        if hit_positions.numel() == 0:
            return None
        hit_states_cpu = states[hit_positions].to(torch.int8).cpu().numpy()
        for i in range(hit_states_cpu.shape[0]):
            key = hit_states_cpu[i].tobytes()
            if key in self.mitm_table.table:
                return int(hit_positions[i].item()), key
        return None

    def _invert_bfs_path(self, path_bytes: bytes) -> list[int]:
        """`path_bytes`: BFS word from identity to a state, as raw uint8 gen indices.
        Return reversed list with each gi → its inverse — applies to that state to
        reach identity (= solved)."""
        return [int(self.inv_idx_np[gi]) for gi in reversed(list(path_bytes))]

    # ----------------------------------------------------------------- solve

    def solve(self, initial_state, cfg: KhoruzhiiSearchConfig):
        """Solve one puzzle. Returns (found, path_len, names, profile).

        Termination conditions:
            (a) any beam state == solved (single-puzzle baseline)
            (b) any beam state in the MITM shell  ← new path
        """
        prof = Profile()
        _sync()
        t_start = time.time()
        B = cfg.beam_width
        num_steps = cfg.num_steps

        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)
        if torch.equal(state, self.V0):
            prof.total_s = time.time() - t_start
            prof.found = True
            return True, 0, [], prof

        # Immediate MITM check on the initial state.
        init_h = _state_hash(state.unsqueeze(0), self.hash_vec)
        init_match = self._mitm_find_match(state.unsqueeze(0), init_h)
        if init_match is not None:
            _, key = init_match
            path_bytes = self.mitm_table.table[key]
            names = [self.move_names[gi] for gi in self._invert_bfs_path(path_bytes)]
            prof.total_s = time.time() - t_start
            prof.found = True
            prof.path_len = len(names)
            return True, len(names), names, prof

        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        final_states = None
        final_j = None
        mitm_beam_pos = None
        mitm_tail_names: list[str] = []

        for attempt in range(cfg.num_attempts):
            states = state.unsqueeze(0).clone()
            # Initial parent hash for incremental hashing.
            states_hashed = _state_hash(states, self.hash_vec, self.internal_batch_size)
            tree_move = torch.full((num_steps, B), -1, dtype=torch.int8, device=self.device)
            tree_idx = torch.full((num_steps, B), -1, dtype=torch.int32, device=self.device)
            states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
            reached = None

            for j in range(num_steps):
                states, _values, moves, parents, chosen_hashes = self._do_greedy_step(
                    states, states_hashed, states_bad_hashed, B, prof
                )
                if states.numel() == 0:
                    break

                # Update parent hash for next iteration (incremental hash needs it).
                states_hashed = chosen_hashes

                _sync(); t0 = time.time()
                states_hash_log.append(chosen_hashes)
                _sync(); prof.hash_s += time.time() - t0

                leaves = states.size(0)
                tree_move[j, :leaves] = moves.to(torch.int8)
                tree_idx[j, :leaves] = parents.to(torch.int32)
                prof.per_step_states.append(int(leaves))
                prof.n_steps = j + 1

                # (a) solved-state termination (fast path, identical to single)
                if (states == self.V0).all(dim=1).any():
                    reached = j
                    break

                # (b) MITM termination
                match = self._mitm_find_match(states, chosen_hashes)
                if match is not None:
                    beam_pos, key = match
                    path_bytes = self.mitm_table.table[key]
                    mitm_tail_names = [
                        self.move_names[gi] for gi in self._invert_bfs_path(path_bytes)
                    ]
                    mitm_beam_pos = beam_pos
                    reached = j
                    break

                if j > 3 and self._check_stagnation(states_hash_log):
                    new_bad = torch.cat(list(states_hash_log))
                    states_bad_hashed = torch.unique(torch.cat([states_bad_hashed, new_bad]))
                    break

            if reached is not None:
                final_states = states
                final_j = reached
                break

        if final_states is None or final_j is None:
            prof.total_s = time.time() - t_start
            prof.found = False
            return False, 0, [], prof

        # Reconstruct beam path back to either solved or the MITM-match position.
        tree_idx_cpu = tree_idx[: final_j + 1].flip((0,)).cpu().numpy()
        tree_move_cpu = tree_move[: final_j + 1].flip((0,)).cpu().numpy()
        if mitm_beam_pos is not None:
            start_pos = mitm_beam_pos
        else:
            start_pos = int(torch.nonzero(
                (final_states == self.V0).all(dim=1), as_tuple=True
            )[0][0])
        path = [int(tree_idx_cpu[0, start_pos])]
        for k in range(1, final_j + 1):
            path.append(int(tree_idx_cpu[k, path[-1]]))
        moves_seq = [
            int(tree_move_cpu[k, path[k - 1]] if k > 0 else tree_move_cpu[k, start_pos])
            for k in range(final_j + 1)
        ]
        moves_seq.reverse()
        names = [self.move_names[m] for m in moves_seq] + mitm_tail_names

        prof.total_s = time.time() - t_start
        prof.found = True
        prof.path_len = len(names)
        return True, len(names), names, prof
