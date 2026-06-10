"""MITM-augmented beam search for Megaminx.

Wraps `KhoruzhiiSolver` and terminates the beam as soon as ANY beam state lands in the
pre-computed BFS target set (bfs_bytes_d*.pkl). Instead of searching all the way to the
solved state, we reach the d-shell around solved and append the known-optimal tail.

Effect: for a path of true distance D with the BFS shell at depth d, beam search needs
only (D - d) steps instead of D. On hard puzzles where beam drifts after ~80 steps, this
could turn "unsolved" into "solved with d6-optimal tail".

Implementation strategy:
  - Precompute hashes of all BFS target states once, on GPU
  - Each step: `torch.isin(beam_hashes, mitm_hashes)` detects any match
  - On match: transfer matched state to CPU, lookup in dict, invert BFS path, splice
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from cayley.khoruzhii_search import (
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    _state_hash,
    _model_predict,
    _q_predict,
)

from megaminx.bfs_bytes import BfsBytesTable


class MitmKhoruzhiiSolver(KhoruzhiiSolver):
    """Beam search that terminates on entry to a BFS target-set shell."""

    def __init__(
        self,
        puzzle,
        model,
        mitm_table: BfsBytesTable,
        device: str = "cuda",
        internal_batch_size: int = 2**14,
        random_seed: int = 0,
        state_dtype: torch.dtype = torch.int8,
        use_q_function: bool = False,
        pdb_lookup=None,
    ):
        super().__init__(
            puzzle=puzzle, model=model, device=device,
            internal_batch_size=internal_batch_size,
            random_seed=random_seed, state_dtype=state_dtype,
            use_q_function=use_q_function, pdb_lookup=pdb_lookup,
        )
        self.mitm_table = mitm_table

        # Precompute inv_idx: for each generator index gi, the index of its inverse.
        inv_names = [puzzle.inverse_name(n) for n in puzzle.move_names]
        self.inv_idx_np = np.array(
            [puzzle.move_names.index(n) for n in inv_names], dtype=np.int64
        )

        # Precompute hashes of the whole BFS target set on GPU (for torch.isin).
        # 19M states x 8 bytes = ~152 MB fits trivially in 16 GB.
        print(f"[MITM] hashing {len(mitm_table.table):,} target states...", flush=True)
        N = len(mitm_table.table)
        all_states = np.empty((N, self.state_size), dtype=np.int8)
        self._all_keys: list[bytes] = [None] * N
        for i, k in enumerate(mitm_table.table.keys()):
            all_states[i] = np.frombuffer(k, dtype=np.int8)
            self._all_keys[i] = k
        # Compute hashes in batches on GPU.
        bs = 2**16
        all_hashes = torch.empty(N, dtype=torch.int64, device=device)
        for i in range(0, N, bs):
            chunk = torch.from_numpy(all_states[i : i + bs]).to(device=device, dtype=torch.int64)
            all_hashes[i : i + bs] = torch.sum(self.hash_vec * chunk, dim=1)
        # sort for fast isin; keep a permutation to recover original index.
        sorted_h, sort_perm = torch.sort(all_hashes)
        self._mitm_hashes_sorted = sorted_h
        self._mitm_sort_perm = sort_perm
        # Also keep the original hash-to-key map for per-state lookup.
        self._mitm_hashes = all_hashes  # unsorted; index matches self._all_keys

    def _mitm_find_match(self, states: torch.Tensor, hashed: torch.Tensor):
        """Return (beam_idx, state_key_bytes) for the first beam state in the MITM set,
        or None if no match. `hashed` is the (B,) hash vector for the current beam."""
        # Check collisions against the MITM hash set via isin.
        in_set = torch.isin(hashed, self._mitm_hashes_sorted, assume_unique=False)
        hit_positions = torch.nonzero(in_set, as_tuple=True)[0]
        if hit_positions.numel() == 0:
            return None
        # Hash collisions are possible; verify each candidate by actual state bytes.
        # Pull hit states to CPU, compare bytes against table keys.
        hit_states_cpu = states[hit_positions].to(torch.int8).cpu().numpy()
        for i in range(hit_states_cpu.shape[0]):
            key = hit_states_cpu[i].tobytes()
            if key in self.mitm_table.table:
                return int(hit_positions[i].item()), key
        return None

    def _invert_bfs_path(self, path_bytes: bytes) -> list[int]:
        """path_bytes is the word from identity to state (uint8 gen indices).
        Return the inverse: reversed list, each gi replaced by its inverse gi.
        Applying the inverse to state reaches identity (= solved)."""
        gis = list(path_bytes)
        return [int(self.inv_idx_np[gi]) for gi in reversed(gis)]

    def solve(self, initial_state, cfg: KhoruzhiiSearchConfig):
        """Try to solve one puzzle. Returns (found, path_length, path_names).

        Termination: solved state reached OR some beam state found in MITM set.
        On MITM match, path = beam_path_to_match + inverse(bfs_path_from_identity_to_match).
        """
        B = cfg.beam_width
        num_steps = cfg.num_steps
        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)

        # Immediate check: initial state in MITM set?
        init_h = _state_hash(state.unsqueeze(0), self.hash_vec)
        init_match = self._mitm_find_match(state.unsqueeze(0), init_h)
        if init_match is not None:
            _, key = init_match
            path_bytes = self.mitm_table.table[key]
            names = [self.move_names[gi] for gi in self._invert_bfs_path(path_bytes)]
            return True, len(names), names

        if torch.equal(state, self.V0):
            return True, 0, []

        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        final_states = None
        final_j = None
        mitm_beam_pos = None
        mitm_tail_names: list[str] = []

        for attempt in range(cfg.num_attempts):
            states = state.unsqueeze(0).clone()
            tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
            states_hash_log: deque[torch.Tensor] = deque(maxlen=4)

            reached = None
            for j in range(num_steps):
                if cfg.beam_decay != 1.0:
                    step_beam = max(cfg.min_beam_width, int(round(B * (cfg.beam_decay ** j))))
                else:
                    step_beam = B
                states, y_pred, moves, idx = self._do_greedy_step(states, states_bad_hashed, step_beam)
                if states.numel() == 0:
                    break
                hashed_now = _state_hash(states, self.hash_vec)
                states_hash_log.append(hashed_now)
                leaves = states.size(0)
                tree_move[j, :leaves] = moves.cpu()
                tree_idx[j, :leaves] = idx.cpu()

                # Check solved-state first (fast path).
                if (states == self.V0).all(dim=1).any():
                    reached = j
                    break

                # MITM check.
                match = self._mitm_find_match(states, hashed_now)
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
            return False, 0, []

        # Reconstruct beam path back to the matching position in final beam.
        tree_idx = tree_idx[: final_j + 1].flip((0,))
        tree_move = tree_move[: final_j + 1].flip((0,))

        if mitm_beam_pos is not None:
            start_pos = mitm_beam_pos
        else:
            start_pos = torch.nonzero(
                (final_states == self.V0).all(dim=1), as_tuple=True
            )[0].item()

        path = [tree_idx[0, start_pos].item()]
        for k in range(1, final_j + 1):
            path.append(tree_idx[k, path[-1]].item())

        moves_seq_cpu = [
            (tree_move[k, path[k - 1]] if k > 0 else tree_move[k, start_pos]).item()
            for k in range(final_j + 1)
        ]
        moves_seq_cpu.reverse()  # built from end to start
        names = [self.move_names[m] for m in moves_seq_cpu] + mitm_tail_names
        return True, len(names), names
