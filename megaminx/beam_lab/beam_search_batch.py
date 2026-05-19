"""Multi-puzzle parallel beam search.

Solves K puzzles in lockstep, batching the model forward call across all of them.
The model forward is the bottleneck (~50-70% of single-puzzle wall) and shape-stable,
so packing K puzzles' beams into one big call gives near-linear speedup until VRAM
caps.

Key design choices:
- Each puzzle keeps its own beam tensor and tree (independent search).
- Phase 1 (neighbor gen + hash + dedup) is per-puzzle but done back-to-back.
- Phase 2 (model forward) is ONE call on concatenated unique candidates from all
  alive puzzles.
- Phase 3 (top-K + bookkeeping) splits values back per-puzzle.
- Solved/dead puzzles drop out of the alive set; remaining alive ones continue.

When K=1 this should produce identical results to the single-puzzle solver, modulo
the (deterministic) reordering of operations. Useful as a sanity check.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from beam_search import (
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    Profile,
    _model_predict,
    _state_hash,
    _sync,
)


@dataclass
class BatchProfile:
    """Aggregate profile for a batch run + per-puzzle profiles."""
    total_s: float = 0.0
    n_steps: int = 0
    K: int = 0
    n_solved: int = 0
    # Cumulative across all steps:
    phase1_s: float = 0.0   # neighbor gen + hash + dedup, summed across puzzles
    model_s: float = 0.0    # the ONE batched model call per step (not per puzzle)
    phase3_s: float = 0.0   # top-k + bookkeeping
    # One Profile entry per puzzle (recorded from each puzzle's perspective):
    per_puzzle: list[Profile] = field(default_factory=list)


class BatchedKhoruzhiiSolver(KhoruzhiiSolver):
    """K-puzzle parallel beam search. Inherits helpers from KhoruzhiiSolver."""

    # ----------------------------------------------------------------- per-puzzle helpers

    def _phase1_dedup_candidates(self, states, bad_hashed):
        """For one puzzle's beam: generate 24 children, hash, dedup against blacklist.

        Returns (unique_states, unique_parent_idx, unique_move_idx, unique_hashed) — all
        candidate states ready for model scoring. Or None if nothing alive.

        Uses implicit (parent_idx, move_idx) decoding from idx1: parent = idx1 // n_gen,
        move = idx1 % n_gen. Avoids allocating two (n * n_gen,) int64 tensors per step.
        """
        # Generate (n * n_gen, S) neighbors and hash them.
        neighbors_flat = self._get_neighbors(states).flatten(end_dim=1)
        hashed = _state_hash(neighbors_flat, self.hash_vec, self.internal_batch_size)
        # Dedup against this puzzle's blacklist.
        idx1 = self._unique_hashed_idx(hashed, bad_hashed)
        if idx1.numel() == 0:
            return None
        # Implicit parent/move from flat index.
        parents_unique = idx1 // self.n_gen
        moves_unique = idx1 % self.n_gen
        return neighbors_flat[idx1], parents_unique, moves_unique, hashed[idx1]

    def _phase3_pick_topk(self, candidates, idx0, moves, values, B):
        """Top-B by ascending predicted value. Returns (next_states, values, moves, parent_idx).

        Uses torch.topk (O(N log B)) instead of argsort (O(N log N)).
        """
        if values.numel() <= B:
            chosen = torch.arange(values.numel(), device=self.device)
        else:
            _, chosen = torch.topk(values, B, largest=False, sorted=False)
        return candidates[chosen], values[chosen], moves[chosen], idx0[chosen]

    # ----------------------------------------------------------------- entrypoint

    def solve_batch(self, initial_states_list, cfg: KhoruzhiiSearchConfig):
        """Solve K puzzles in lockstep with one batched model call per beam step.

        Returns: list of (found, path_len, names, profile) per puzzle, in the same order
        as the input. profile.total_s for each puzzle is the wall from solve_batch entry
        to the puzzle's resolution (or to batch end if it never resolves) — i.e. shared
        wall time across the batch.
        """
        _sync()
        t_batch_start = time.time()
        bp = BatchProfile(K=len(initial_states_list))
        bp.per_puzzle = [Profile() for _ in initial_states_list]

        K = len(initial_states_list)
        B = cfg.beam_width
        num_steps = cfg.num_steps

        # Per-puzzle state. None means "this puzzle is done" (solved or dead).
        states_per: list[torch.Tensor | None] = []
        for s in initial_states_list:
            states_per.append(
                torch.tensor(list(s), dtype=self.state_dtype, device=self.device).unsqueeze(0)
            )

        bad_per: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.int64, device=self.device) for _ in range(K)
        ]
        # Tree backpointers on GPU — int8 for moves, int32 for parent indices.
        # Avoids per-step .cpu() syncs that stall the next step's kernel launches.
        tree_move_per: list[torch.Tensor] = [
            torch.full((num_steps, B), -1, dtype=torch.int8, device=self.device) for _ in range(K)
        ]
        tree_idx_per: list[torch.Tensor] = [
            torch.full((num_steps, B), -1, dtype=torch.int32, device=self.device) for _ in range(K)
        ]
        hash_log_per: list[deque] = [deque(maxlen=4) for _ in range(K)]
        final_states_per: list[torch.Tensor | None] = [None] * K
        final_j_per: list[int | None] = [None] * K
        per_resolved_at: list[float | None] = [None] * K  # wall time when each puzzle finished

        # Immediate solved-state check (before any beam expansion).
        for k, s in enumerate(states_per):
            if torch.equal(s.squeeze(0), self.V0):
                final_states_per[k] = s
                final_j_per[k] = -1  # 0-move solution
                per_resolved_at[k] = time.time() - t_batch_start
                states_per[k] = None
                bp.per_puzzle[k].found = True
                bp.per_puzzle[k].path_len = 0

        # Main loop: lockstep, K puzzles in parallel.
        for j in range(num_steps):
            alive = [k for k in range(K) if states_per[k] is not None]
            if not alive:
                break
            bp.n_steps = j + 1

            # ---- Phase 1: per-puzzle neighbor gen + hash + dedup ----
            _sync()
            t0 = time.time()
            phase1_results: list[tuple | None] = [None] * K
            for k in alive:
                phase1_results[k] = self._phase1_dedup_candidates(states_per[k], bad_per[k])
            _sync()
            bp.phase1_s += time.time() - t0

            # Drop puzzles whose dedup left zero candidates (dead end).
            still_alive = [k for k in alive if phase1_results[k] is not None]
            for k in alive:
                if phase1_results[k] is None:
                    states_per[k] = None
                    per_resolved_at[k] = time.time() - t_batch_start
                    bp.per_puzzle[k].found = False

            if not still_alive:
                break

            # Stack candidates from all alive puzzles for ONE batched model call.
            cands_list = [phase1_results[k][0] for k in still_alive]
            sizes = [c.size(0) for c in cands_list]
            big_cands = torch.cat(cands_list, dim=0)

            # ---- Phase 2: ONE model forward across all alive puzzles ----
            _sync()
            t0 = time.time()
            big_values = _model_predict(self.model, big_cands, self.internal_batch_size)
            _sync()
            bp.model_s += time.time() - t0
            # Charge model time back to each puzzle proportional to its share.
            total_n = sum(sizes)
            for ki, k in enumerate(still_alive):
                bp.per_puzzle[k].model_s += (sizes[ki] / total_n) * (time.time() - t0)

            # ---- Phase 3: per-puzzle top-K + bookkeeping ----
            _sync()
            t0 = time.time()
            offset = 0
            for ki, k in enumerate(still_alive):
                n_k = sizes[ki]
                puz_cands = big_cands[offset : offset + n_k]
                puz_idx0 = phase1_results[k][1]
                puz_moves = phase1_results[k][2]
                puz_hashed = phase1_results[k][3]
                puz_values = big_values[offset : offset + n_k]
                offset += n_k

                # Top-B
                next_states, _, sel_moves, sel_idx0 = self._phase3_pick_topk(
                    puz_cands, puz_idx0, puz_moves, puz_values, B
                )
                # Save tree for path reconstruction (kept on GPU; copy to CPU only at the end).
                leaves = next_states.size(0)
                tree_move_per[k][j, :leaves] = sel_moves.to(torch.int8)
                tree_idx_per[k][j, :leaves] = sel_idx0.to(torch.int32)
                bp.per_puzzle[k].n_steps = j + 1
                bp.per_puzzle[k].per_step_states.append(int(leaves))

                # Hash log for stagnation detection
                hash_log_per[k].append(_state_hash(next_states, self.hash_vec))

                # Solved check
                if (next_states == self.V0).all(dim=1).any():
                    final_states_per[k] = next_states
                    final_j_per[k] = j
                    per_resolved_at[k] = time.time() - t_batch_start
                    states_per[k] = None
                    bp.n_solved += 1
                    continue

                # Stagnation
                if j > 3 and self._stagnation(hash_log_per[k]):
                    new_bad = torch.cat(list(hash_log_per[k]))
                    bad_per[k] = torch.unique(torch.cat([bad_per[k], new_bad]))
                    states_per[k] = None  # this puzzle gives up
                    per_resolved_at[k] = time.time() - t_batch_start
                    bp.per_puzzle[k].found = False
                    continue

                # Not done — carry beam forward
                states_per[k] = next_states

            _sync()
            bp.phase3_s += time.time() - t0

        bp.total_s = time.time() - t_batch_start

        # ---- Reconstruct paths for solved puzzles ----
        results: list[tuple] = []
        for k in range(K):
            prof = bp.per_puzzle[k]
            prof.total_s = per_resolved_at[k] or bp.total_s

            if final_states_per[k] is None:
                results.append((False, 0, [], prof))
                continue

            if final_j_per[k] == -1:
                # Initial state was solved.
                prof.found = True
                prof.path_len = 0
                results.append((True, 0, [], prof))
                continue

            # Trace path backward through the per-puzzle tree (single CPU copy at end).
            j = final_j_per[k]
            ti_cpu = tree_idx_per[k][: j + 1].flip((0,)).cpu().numpy()
            tm_cpu = tree_move_per[k][: j + 1].flip((0,)).cpu().numpy()
            v0_pos = int(torch.nonzero(
                (final_states_per[k] == self.V0).all(dim=1), as_tuple=True
            )[0][0])
            path = [int(ti_cpu[0, v0_pos])]
            for kk in range(1, j + 1):
                path.append(int(ti_cpu[kk, path[-1]]))
            moves_seq = [
                int(tm_cpu[kk, path[kk - 1]] if kk > 0 else tm_cpu[kk, v0_pos])
                for kk in range(j + 1)
            ]
            moves_seq.reverse()
            names = [self.move_names[m] for m in moves_seq]
            prof.found = True
            prof.path_len = len(names)
            results.append((True, len(names), names, prof))

        return results, bp

    # Quick helper alias matching parent API
    def _stagnation(self, log):
        return self._check_stagnation(log)
