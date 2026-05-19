"""beam_search_v0 — pre-optimization baseline.

This is the version of beam_search.py *before* the 6 Tier-1 micro-optimizations
landed (argsort instead of topk, no_grad instead of inference_mode, CPU tree
backpointers with per-step .cpu() syncs, double model chunking, no hash reuse,
explicit idx0/moves allocations).

Used for direct A/B benchmarking against beam_search.py to confirm the opts
deliver the claimed speedup.

Same Profile object so the harness can compare per-stage timings.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from beam_search import KhoruzhiiSearchConfig, Profile, _sync


def _state_hash_v0(states, hash_vec, batch_size=16384):
    out = torch.empty(states.size(0), dtype=torch.int64, device=states.device)
    for i in range(0, states.size(0), batch_size):
        chunk = states[i : i + batch_size].to(torch.int64)
        out[i : i + batch_size] = torch.sum(hash_vec * chunk, dim=1)
    return out


def _model_predict_v0(model, states, batch_size=16384):
    """Original: torch.no_grad() (not inference_mode). Note: model.inference_chunk_size
    is NOT disabled, so the model also chunks internally → double chunking."""
    model.eval()
    out = torch.empty(states.size(0), dtype=torch.float16, device=states.device)
    with torch.no_grad():
        for i in range(0, states.size(0), batch_size):
            out[i : i + batch_size] = model(states[i : i + batch_size]).flatten().to(torch.float16)
    return out


class KhoruzhiiSolverV0:
    """Pre-optimization beam search. Identical to KhoruzhiiSolver before 2026-04-26 edits."""

    def __init__(
        self,
        puzzle,
        model,
        device: str = "cuda",
        internal_batch_size: int = 16384,
        random_seed: int = 0,
        state_dtype: torch.dtype = torch.int8,
    ):
        self.puzzle = puzzle
        self.model = model.to(device).eval()
        self.device = device
        self.internal_batch_size = internal_batch_size
        self.state_dtype = state_dtype

        n_gen = len(puzzle.move_names)
        self.n_gen = n_gen
        self.state_size = len(puzzle.solved_state)
        self.all_moves = torch.zeros((n_gen, self.state_size), dtype=torch.int64, device=device)
        for i, name in enumerate(puzzle.move_names):
            self.all_moves[i] = torch.tensor(puzzle.generators[name], dtype=torch.int64)
        self.move_names = puzzle.move_names
        self.V0 = torch.tensor(puzzle.solved_state, dtype=state_dtype, device=device)

        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)
        self.hash_vec = torch.randint(
            0, int(1e15), (self.state_size,), dtype=torch.int64, device=device, generator=gen
        )

    def _get_neighbors(self, states):
        B = states.size(0)
        bs = self.internal_batch_size
        out = torch.empty((B, self.n_gen, self.state_size), dtype=states.dtype, device=self.device)
        for i in range(0, B, bs):
            chunk = states[i : i + bs]
            out[i : i + bs] = torch.gather(
                chunk.unsqueeze(1).expand(chunk.size(0), self.n_gen, self.state_size),
                2,
                self.all_moves.unsqueeze(0).expand(chunk.size(0), self.n_gen, self.state_size),
            )
        return out

    def _apply_move(self, states, moves):
        bs = self.internal_batch_size
        out = torch.empty(states.size(0), self.state_size, dtype=states.dtype, device=self.device)
        for i in range(0, states.size(0), bs):
            s = states[i : i + bs]
            m = moves[i : i + bs]
            out[i : i + bs] = torch.gather(s, 1, self.all_moves[m])
        return out

    @staticmethod
    def _unique_hashed_idx(hashed, states_bad_hashed):
        idx1 = torch.arange(hashed.size(0), dtype=torch.int64, device=hashed.device)
        if states_bad_hashed.numel() > 0:
            mask1 = ~torch.isin(hashed, states_bad_hashed)
            hashed = hashed[mask1]
            idx1 = idx1[mask1]
        sorted_h, idx2 = torch.sort(hashed)
        if sorted_h.numel() == 0:
            return idx1
        mask2 = torch.cat([torch.tensor([True], device=hashed.device), sorted_h[1:] - sorted_h[:-1] > 0])
        return idx1[idx2[mask2]]

    def _do_greedy_step(self, states, states_bad_hashed, B, prof: Profile):
        n = states.size(0)
        bs = self.internal_batch_size
        # ORIGINAL: explicit idx0 / moves arrays of size (n * n_gen,) allocated every step
        idx0 = torch.arange(n, device=self.device).repeat_interleave(self.n_gen)
        moves = torch.arange(self.n_gen, device=self.device).repeat(n)

        _sync(); t0 = time.time()
        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, n, bs):
            chunk_states = states[i : i + bs]
            chunk_neighbors = self._get_neighbors(chunk_states).flatten(end_dim=1)
            neighbors_hashed[i * self.n_gen : (i + chunk_states.size(0)) * self.n_gen] = _state_hash_v0(
                chunk_neighbors, self.hash_vec, bs
            )
        _sync(); prof.neighbor_s += time.time() - t0

        _sync(); t0 = time.time()
        idx1 = self._unique_hashed_idx(neighbors_hashed, states_bad_hashed)
        _sync(); prof.dedup_s += time.time() - t0

        if idx1.numel() == 0:
            empty_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_v = torch.empty(0, dtype=torch.float16, device=self.device)
            empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty_s, empty_v, empty_i, empty_i

        _sync(); t0 = time.time()
        candidate_states = self._apply_move(states[idx0[idx1]], moves[idx1])
        _sync(); prof.apply_s += time.time() - t0

        _sync(); t0 = time.time()
        value = _model_predict_v0(self.model, candidate_states, bs)
        _sync(); prof.model_s += time.time() - t0

        # ORIGINAL: full argsort then slice [:B]
        _sync(); t0 = time.time()
        idx2 = torch.argsort(value)[:B]
        chosen_idx1 = idx1[idx2]
        next_states = candidate_states[idx2]
        _sync(); prof.topk_s += time.time() - t0

        return next_states, value[idx2], moves[chosen_idx1], idx0[chosen_idx1]

    def _check_stagnation(self, log):
        if len(log) < 4:
            return False
        recent = torch.cat(list(log)[2:])
        earlier = torch.cat(list(log)[:2])
        return bool(torch.isin(recent, earlier).all().item())

    def solve(self, initial_state, cfg: KhoruzhiiSearchConfig):
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

        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        final_states = None
        final_j = None

        for attempt in range(cfg.num_attempts):
            states = state.unsqueeze(0).clone()
            # ORIGINAL: tree on CPU (forces .cpu() sync each step)
            tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
            states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
            reached = None

            for j in range(num_steps):
                states, _, moves, idx = self._do_greedy_step(states, states_bad_hashed, B, prof)
                if states.numel() == 0:
                    break
                # ORIGINAL: re-hash chosen states for stagnation log (ignores neighbors_hashed reuse)
                _sync(); t0 = time.time()
                states_hash_log.append(_state_hash_v0(states, self.hash_vec))
                _sync(); prof.hash_s += time.time() - t0

                leaves = states.size(0)
                # ORIGINAL: .cpu() forces CUDA sync on every step
                tree_move[j, :leaves] = moves.cpu()
                tree_idx[j, :leaves] = idx.cpu()
                prof.per_step_states.append(int(leaves))
                prof.n_steps = j + 1

                if (states == self.V0).all(dim=1).any():
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

        tree_idx = tree_idx[: final_j + 1].flip((0,))
        tree_move = tree_move[: final_j + 1].flip((0,))
        v0_pos = torch.nonzero((final_states == self.V0).all(dim=1), as_tuple=True)[0].item()
        path = [tree_idx[0, v0_pos].item()]
        for k in range(1, final_j + 1):
            path.append(tree_idx[k, path[-1]].item())
        moves_seq = [
            (tree_move[k, path[k - 1]] if k > 0 else tree_move[k, v0_pos]).item()
            for k in range(final_j + 1)
        ]
        moves_seq.reverse()
        names = [self.move_names[m] for m in moves_seq]

        prof.total_s = time.time() - t_start
        prof.found = True
        prof.path_len = len(names)
        return True, len(names), names, prof
