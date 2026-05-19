"""Beam-stack search — discrepancy rescue for residual hard puzzles.

Standard beam keeps the top-B candidates per layer. Beam-stack also stores the
"second slice" (B+1)..(2B) at *checkpoint* layers. When main beam fails, restart
from a saved runner-up at the deepest checkpoint layer.

Targeted at pid-492 type puzzles where m05 at beam 131k is one wrong choice
away from a solve. Easy puzzles take exactly the same wall time as plain beam
(retries don't trigger).

References:
- Beam-stack search (Furcy & Koenig, ICAPS 2005)
- Limited Discrepancy Beam Search (Furcy 2006)

Implementation strategy:
- Two-pass topk per step: top 2B by value (sorted). First B = main beam, next B
  = "runners". The 2B forward already scored; just split.
- Main pass keeps a SINGLE (num_steps, B) tree. At every `checkpoint_every`
  layers, snapshot the runner local indices + the bad-hash blacklist active at
  the time. The main tree stays live so we can walk it back to layer 0 from
  any runner-up parent at retry time.
- On failure, pop deepest checkpoint, restart beam with `runner_states` as the
  initial beam (NOT a single state — so we get diverse exploration from up
  to B alternative paths).
- On retry success, splice: prefix [initial → runner_state via main tree] +
  retry path. Verify with puzzle.apply_path.
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
class StackProfile:
    total_s: float = 0.0
    n_attempts: int = 0
    found: bool = False
    path_len: int = 0
    success_attempt: int = -1   # 0 = main beam, ≥1 = retry from runner-up
    n_steps_main: int = 0
    n_steps_retries: list[int] = field(default_factory=list)


@dataclass
class _Checkpoint:
    """Snapshot of state at a search layer for retry."""
    layer: int                   # the search layer at which we stored
    runner_states: torch.Tensor  # (R, S) — runner-up states at this layer
    runner_parents: torch.Tensor # (R,) parent index in main beam at layer-1
    runner_moves: torch.Tensor   # (R,) move that produced each runner
    bad_hashed: torch.Tensor     # blacklist active at the time


class BeamStackSolver(KhoruzhiiSolver):
    """Beam search with runner-up checkpointing + retry."""

    # ---------------------------------------------------------------- helpers

    def _do_2b_step(self, states, bad_hashed, B):
        """Greedy step that returns BOTH top-B (main) and (B+1)..(2B) (runners).

        Returns 8-tuple:
            (main_states, main_values, main_moves, main_parents, main_hashes,
             runner_states, runner_moves, runner_parents)
        or None if no candidates.
        """
        bs = self.internal_batch_size

        if hasattr(self, "_get_neighbors_hashed") and getattr(self, "changed_pos", None) is not None:
            states_hashed = _state_hash(states, self.hash_vec, bs)
            neighbors_hashed = self._get_neighbors_hashed(states, states_hashed)
            neighbors_flat = None
        else:
            neighbors_flat = self._get_neighbors(states).flatten(end_dim=1)
            neighbors_hashed = _state_hash(neighbors_flat, self.hash_vec, bs)

        idx1 = self._unique_hashed_idx(neighbors_hashed, bad_hashed)
        if idx1.numel() == 0:
            return None

        parent_of_idx1 = idx1 // self.n_gen
        move_of_idx1 = idx1 % self.n_gen

        if neighbors_flat is None:
            candidate_states = self._apply_move(states[parent_of_idx1], move_of_idx1)
        else:
            candidate_states = neighbors_flat[idx1]

        value = _model_predict(self.model, candidate_states, bs)

        # Top-2B (sorted ascending). Split into main + runner.
        target_k = min(2 * B, value.numel())
        if value.numel() <= target_k:
            order = torch.argsort(value)
        else:
            _, order = torch.topk(value, target_k, largest=False, sorted=True)

        main_local = order[:B]
        runner_local = order[B : 2 * B] if order.numel() > B else order[:0]

        return (
            candidate_states[main_local], value[main_local],
            move_of_idx1[main_local], parent_of_idx1[main_local],
            neighbors_hashed[idx1[main_local]],
            candidate_states[runner_local],
            move_of_idx1[runner_local], parent_of_idx1[runner_local],
        )

    def _walk_main_tree(self, main_tree_idx, main_tree_move, end_layer, leaf_idx):
        """Trace path from initial state to a state at `end_layer` whose index
        in that layer's beam is `leaf_idx`. Returns list of move indices length
        end_layer+1."""
        # main_tree_idx[k, j] = parent index in layer k-1 for state j of layer k.
        # main_tree_move[k, j] = move that produced state j of layer k from its parent.
        ti = main_tree_idx[: end_layer + 1].cpu().numpy()
        tm = main_tree_move[: end_layer + 1].cpu().numpy()
        # Walk backward from end_layer.
        idx = int(leaf_idx)
        moves_rev = []
        for k in range(end_layer, -1, -1):
            moves_rev.append(int(tm[k, idx]))
            idx = int(ti[k, idx])
        # moves_rev is now from end_layer→0; reverse to get 0→end_layer order.
        moves_rev.reverse()
        return moves_rev

    # ---------------------------------------------------------------- main entry

    def solve_with_backtrack(
        self,
        initial_state,
        cfg: KhoruzhiiSearchConfig,
        checkpoint_every: int = 10,
        max_retries: int = 3,
    ):
        prof = StackProfile()
        _sync()
        t_start = time.time()

        B = cfg.beam_width
        num_steps = cfg.num_steps
        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)
        if torch.equal(state, self.V0):
            prof.total_s = time.time() - t_start
            prof.found = True
            return True, 0, [], prof

        # ------ main pass with checkpointing ------
        prof.n_attempts = 1
        states = state.unsqueeze(0).clone()
        main_tree_idx = torch.full((num_steps, B), -1, dtype=torch.int32, device=self.device)
        main_tree_move = torch.full((num_steps, B), -1, dtype=torch.int8, device=self.device)
        states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        checkpoints: list[_Checkpoint] = []

        reached = None
        last_j = 0
        for j in range(num_steps):
            last_j = j
            step = self._do_2b_step(states, states_bad_hashed, B)
            if step is None:
                break
            (next_states, _, chosen_moves, chosen_parents, chosen_hashes,
             runner_states, runner_moves, runner_parents) = step

            states_hash_log.append(chosen_hashes)
            leaves = next_states.size(0)
            main_tree_move[j, :leaves] = chosen_moves.to(torch.int8)
            main_tree_idx[j, :leaves] = chosen_parents.to(torch.int32)

            if (next_states == self.V0).all(dim=1).any():
                states = next_states
                reached = j
                break

            if j > 3 and self._check_stagnation(states_hash_log):
                if runner_states.size(0) > 0:
                    checkpoints.append(_Checkpoint(
                        layer=j,
                        runner_states=runner_states.clone(),
                        runner_parents=runner_parents.clone(),
                        runner_moves=runner_moves.clone(),
                        bad_hashed=states_bad_hashed.clone(),
                    ))
                new_bad = torch.cat(list(states_hash_log))
                states_bad_hashed = torch.unique(torch.cat([states_bad_hashed, new_bad]))
                break

            if (j + 1) % checkpoint_every == 0 and runner_states.size(0) > 0:
                checkpoints.append(_Checkpoint(
                    layer=j,
                    runner_states=runner_states.clone(),
                    runner_parents=runner_parents.clone(),
                    runner_moves=runner_moves.clone(),
                    bad_hashed=states_bad_hashed.clone(),
                ))

            states = next_states

        # ------ main pass found? ------
        if reached is not None:
            prof.found = True
            prof.success_attempt = 0
            prof.n_steps_main = reached + 1
            names = self._reconstruct(main_tree_idx, main_tree_move, reached, states)
            prof.path_len = len(names)
            prof.total_s = time.time() - t_start
            return True, prof.path_len, names, prof

        prof.n_steps_main = last_j

        # ------ retry phase: deepest checkpoint first ------
        for retry_idx in range(min(max_retries, len(checkpoints))):
            ck = checkpoints[-(retry_idx + 1)]
            prof.n_attempts += 1
            retry = self._beam_retry(
                initial_states=ck.runner_states,
                cfg=cfg,
                states_bad_hashed=ck.bad_hashed,
                max_steps=max(num_steps - ck.layer, 30),
            )
            prof.n_steps_retries.append(retry["last_step"])
            if not retry["found"]:
                continue

            # Reconstruct full path.
            # 1) Prefix: initial → runner_state (path-of-length L+1)
            #    runner is derived from main beam parent at layer ck.layer-1, then
            #    one more move (runner_moves[winner_idx]).
            winner_in_runner = retry["winner_in_initial_beam"]
            parent_local = int(ck.runner_parents[winner_in_runner].item())
            runner_move = int(ck.runner_moves[winner_in_runner].item())
            if ck.layer >= 1:
                prefix_moves = self._walk_main_tree(
                    main_tree_idx, main_tree_move,
                    end_layer=ck.layer - 1, leaf_idx=parent_local,
                )
            else:
                prefix_moves = []
            prefix_moves.append(runner_move)
            # 2) Retry path from runner to solved.
            full_moves = prefix_moves + retry["move_seq_from_runner"]
            full_names = [self.move_names[m] for m in full_moves]

            prof.found = True
            prof.success_attempt = retry_idx + 1
            prof.path_len = len(full_names)
            prof.total_s = time.time() - t_start
            return True, prof.path_len, full_names, prof

        prof.total_s = time.time() - t_start
        prof.found = False
        return False, 0, [], prof

    # ---------------------------------------------------------------- retry sub-search

    def _beam_retry(self, initial_states, cfg, states_bad_hashed, max_steps):
        """Run a fresh beam search starting from a SET of initial states (the
        runners-up). Returns dict with {found, move_seq_from_runner,
        winner_in_initial_beam, last_step}."""
        B = cfg.beam_width
        states = initial_states.clone()
        # We need to remember which row of `initial_states` each beam cell descends from
        # so the caller can splice the prefix correctly. Index 0..R-1 at layer 0.
        R = initial_states.size(0)
        retry_tree_idx = torch.full((max_steps, B), -1, dtype=torch.int32, device=self.device)
        retry_tree_move = torch.full((max_steps, B), -1, dtype=torch.int8, device=self.device)
        # Track ancestor in initial_states for each beam state. Layer 0: identity.
        ancestor = torch.arange(R, dtype=torch.int32, device=self.device)
        # If states is empty, bail early.
        if R == 0:
            return dict(found=False, move_seq_from_runner=[], winner_in_initial_beam=-1, last_step=0)

        states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
        last_j = 0
        for j in range(max_steps):
            last_j = j
            step = self._do_2b_step(states, states_bad_hashed, B)
            if step is None:
                break
            (next_states, _, chosen_moves, chosen_parents, chosen_hashes, *_runners) = step
            states_hash_log.append(chosen_hashes)
            leaves = next_states.size(0)
            retry_tree_move[j, :leaves] = chosen_moves.to(torch.int8)
            retry_tree_idx[j, :leaves] = chosen_parents.to(torch.int32)
            # Update ancestor: each new state inherits its parent's ancestor.
            new_ancestor = ancestor[chosen_parents]
            ancestor = new_ancestor

            if (next_states == self.V0).all(dim=1).any():
                # Pick a solver row and splice
                v0_pos = int(torch.nonzero((next_states == self.V0).all(dim=1), as_tuple=True)[0][0])
                ti_cpu = retry_tree_idx[: j + 1].flip((0,)).cpu().numpy()
                tm_cpu = retry_tree_move[: j + 1].flip((0,)).cpu().numpy()
                # Walk backward
                idx = v0_pos
                moves_rev = []
                for k in range(j + 1):
                    moves_rev.append(int(tm_cpu[k, idx]))
                    idx = int(ti_cpu[k, idx])
                moves_rev.reverse()
                return dict(
                    found=True,
                    move_seq_from_runner=moves_rev,
                    winner_in_initial_beam=int(ancestor[v0_pos].item()),
                    last_step=j,
                )

            if j > 3 and self._check_stagnation(states_hash_log):
                new_bad = torch.cat(list(states_hash_log))
                states_bad_hashed = torch.unique(torch.cat([states_bad_hashed, new_bad]))
                break

            states = next_states

        return dict(found=False, move_seq_from_runner=[], winner_in_initial_beam=-1, last_step=last_j)

    def _reconstruct(self, tree_idx, tree_move, final_j, final_states):
        ti_cpu = tree_idx[: final_j + 1].flip((0,)).cpu().numpy()
        tm_cpu = tree_move[: final_j + 1].flip((0,)).cpu().numpy()
        v0_pos = int(torch.nonzero((final_states == self.V0).all(dim=1), as_tuple=True)[0][0])
        path = [int(ti_cpu[0, v0_pos])]
        for k in range(1, final_j + 1):
            path.append(int(ti_cpu[k, path[-1]]))
        moves_seq = [
            int(tm_cpu[k, path[k - 1]] if k > 0 else tm_cpu[k, v0_pos])
            for k in range(final_j + 1)
        ]
        moves_seq.reverse()
        return [self.move_names[m] for m in moves_seq]
