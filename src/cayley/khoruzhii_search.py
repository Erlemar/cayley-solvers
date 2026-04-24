"""Port of khoruzhii/cayleypy-cube beam search to our codebase.

Original: https://github.com/khoruzhii/cayleypy-cube/blob/main/pilgrim/searcher.py

Why this vs cayleypy's built-in BeamSearchAlgorithm:
 - Self-contained: ~150 lines, directly manages beam tensors on GPU.
 - Memory-efficient: uses fp16 for model values, stores search tree as two int64
   arrays of shape (num_steps, beam_width). At beam 2^18 on 50 steps: ~420 MB.
   cayleypy's library has internal state buffers that inflate peak memory ~30%.
 - Supports stagnation detection + retry with blacklisting.

This lets us run beam widths 2^16 - 2^18 on 16GB VRAM, which cayleypy OOMs on.

Adapted to our ResMLPDistance model (embedding or one-hot, int64 state input).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from cayley.model import ResMLPDistance
from cayley.puzzle import PictureCube


@dataclass
class KhoruzhiiSearchConfig:
    beam_width: int = 2**14
    num_steps: int = 60
    num_attempts: int = 1
    internal_batch_size: int = 2**14
    # Distance-adaptive beam: if set, beam at step k = max(min_beam, round(beam_width * decay**k)).
    # N5 from NEW_IDEAS_SYNTHESIS: wide beam early (uncertain), narrow beam late (accurate).
    # decay < 1.0 shrinks per step; default 1.0 = constant beam (original behaviour).
    beam_decay: float = 1.0
    min_beam_width: int = 2**14


def _state_hash(states: torch.Tensor, hash_vec: torch.Tensor, batch_size: int = 2**14) -> torch.Tensor:
    """Linear hash: sum(hash_vec * state). Simple, fast, decent collision rate."""
    out = torch.empty(states.size(0), dtype=torch.int64, device=states.device)
    for i in range(0, states.size(0), batch_size):
        chunk = states[i : i + batch_size].to(torch.int64)
        out[i : i + batch_size] = torch.sum(hash_vec * chunk, dim=1)
    return out


def _model_predict(
    model: ResMLPDistance, states: torch.Tensor, batch_size: int = 2**14
) -> torch.Tensor:
    """Run model.forward on states in chunks, return fp16 1D tensor of values."""
    model.eval()
    out = torch.empty(states.size(0), dtype=torch.float16, device=states.device)
    with torch.no_grad():
        for i in range(0, states.size(0), batch_size):
            chunk = states[i : i + batch_size]
            out[i : i + batch_size] = model(chunk).flatten().to(torch.float16)
    return out


def _q_predict(
    model: ResMLPDistance, states: torch.Tensor, batch_size: int, n_gen: int
) -> torch.Tensor:
    """Run model.forward on states in chunks; each forward returns (chunk, n_gen)
    neighbor scores. Returns fp16 2D tensor of shape (N, n_gen)."""
    model.eval()
    out = torch.empty((states.size(0), n_gen), dtype=torch.float16, device=states.device)
    with torch.no_grad():
        for i in range(0, states.size(0), batch_size):
            chunk = states[i : i + batch_size]
            pred = model(chunk)
            if pred.dim() == 1:
                # Shouldn't happen with output_dim=n_gen, but be safe.
                pred = pred.view(-1, n_gen)
            out[i : i + batch_size] = pred.to(torch.float16)
    return out


class KhoruzhiiSolver:
    """Drop-in replacement for our Solver using the khoruzhii-style beam search."""

    def __init__(
        self,
        puzzle: PictureCube,
        model: ResMLPDistance,
        device: str = "cuda",
        internal_batch_size: int = 2**14,
        random_seed: int = 0,
        state_dtype: torch.dtype = torch.int8,
        use_q_function: bool = False,
        pdb_lookup=None,
    ):
        self.puzzle = puzzle
        self.model = model.to(device).eval()
        self.device = device
        self.internal_batch_size = internal_batch_size
        # State buffer dtype. Default int8 — sticker values are 0..71, fit in signed 8-bit
        # (-128..127). Shrinks the beam state tensor 8× vs int64 → larger beams at same VRAM.
        # Generator permutations (all_moves) stay int64 because they're gather indices.
        # Hashing casts state to int64 internally to avoid overflow.
        if state_dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"state_dtype must be an integer type; got {state_dtype!r}")
        self.state_dtype = state_dtype
        self.use_q_function = use_q_function
        if use_q_function:
            base = getattr(model, "_orig_mod", model)
            expected = len(puzzle.move_names)
            got = getattr(base, "output_dim", 1)
            if got != expected:
                raise ValueError(
                    f"use_q_function=True requires model.output_dim={expected} (n_gen), got {got}"
                )
        # Optional admissible re-scorer: pdb_lookup(states) → (B,) float lower bound.
        # Final value = max(neural_value, pdb_value). Safe (still ≤ true distance).
        self.pdb_lookup = pdb_lookup

        # Generator permutations as a (n_gen, state_size) int64 tensor — gather indices.
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

    def _get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """(B, S) → (B*n_gen, S), with neighbors[i*n_gen+g] = apply(states[i], g)."""
        B = states.size(0)
        bs = self.internal_batch_size
        out = torch.empty((B, self.n_gen, self.state_size), dtype=states.dtype, device=self.device)
        for i in range(0, B, bs):
            chunk = states[i : i + bs]
            # expand: (chunk, 1, S) × (1, n_gen, S) → gather along S
            out[i : i + bs] = torch.gather(
                chunk.unsqueeze(1).expand(chunk.size(0), self.n_gen, self.state_size),
                2,
                self.all_moves.unsqueeze(0).expand(chunk.size(0), self.n_gen, self.state_size),
            )
        return out

    def _apply_move(self, states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """For each (state_i, move_i) pair, return apply(state_i, move_i)."""
        bs = self.internal_batch_size
        out = torch.empty(states.size(0), self.state_size, dtype=states.dtype, device=self.device)
        for i in range(0, states.size(0), bs):
            s = states[i : i + bs]
            m = moves[i : i + bs]
            out[i : i + bs] = torch.gather(s, 1, self.all_moves[m])
        return out

    @staticmethod
    def _unique_hashed_idx(
        hashed: torch.Tensor, states_bad_hashed: torch.Tensor
    ) -> torch.Tensor:
        """Return indices of unique, non-blacklisted hashes."""
        idx1 = torch.arange(hashed.size(0), dtype=torch.int64, device=hashed.device)
        if states_bad_hashed.numel() > 0:
            mask1 = ~torch.isin(hashed, states_bad_hashed)
            hashed = hashed[mask1]
            idx1 = idx1[mask1]
        sorted_h, idx2 = torch.sort(hashed)
        if sorted_h.numel() == 0:
            return idx1
        mask2 = torch.cat(
            [torch.tensor([True], device=hashed.device), sorted_h[1:] - sorted_h[:-1] > 0]
        )
        return idx1[idx2[mask2]]

    def _do_greedy_step(
        self, states: torch.Tensor, states_bad_hashed: torch.Tensor, B: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of beam search. Returns (next_states, values, moves, parent_idx)."""
        n = states.size(0)
        bs = self.internal_batch_size

        # For each state and each generator, remember the (parent_idx, gen_idx).
        # idx0 = parent index repeated n_gen times. moves = 0..n_gen-1 repeated for each parent.
        idx0 = torch.arange(n, device=self.device).repeat_interleave(self.n_gen)
        moves = torch.arange(self.n_gen, device=self.device).repeat(n)

        # Compute + hash neighbors in batches.
        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, n, bs):
            chunk_states = states[i : i + bs]
            chunk_neighbors = self._get_neighbors(chunk_states).flatten(end_dim=1)
            neighbors_hashed[i * self.n_gen : (i + chunk_states.size(0)) * self.n_gen] = _state_hash(
                chunk_neighbors, self.hash_vec, bs
            )

        # Keep unique, non-blacklisted.
        idx1 = self._unique_hashed_idx(neighbors_hashed, states_bad_hashed)
        if idx1.numel() == 0:
            empty_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_v = torch.empty(0, dtype=torch.float16, device=self.device)
            empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty_s, empty_v, empty_i, empty_i

        if self.use_q_function:
            # One forward per parent yields (n, n_gen) predicted neighbor values. Flattened
            # as [p0*n_gen + m, p1*n_gen + m, ...] which matches idx0*n_gen+moves ordering.
            q_all = _q_predict(self.model, states, bs, self.n_gen)  # (n, n_gen) fp16
            q_flat = q_all.view(-1)
            value = q_flat[idx1]
            # Top-B by ascending value, then materialize only those states.
            idx2 = torch.argsort(value)[:B]
            chosen_idx1 = idx1[idx2]
            next_states = self._apply_move(states[idx0[chosen_idx1]], moves[chosen_idx1])
            return next_states, value[idx2], moves[chosen_idx1], idx0[chosen_idx1]
        # V-model path: evaluate value on each unique candidate.
        candidate_states = self._apply_move(states[idx0[idx1]], moves[idx1])
        value = _model_predict(self.model, candidate_states, bs)
        if self.pdb_lookup is not None:
            pdb_val = self.pdb_lookup(candidate_states).to(torch.float16)
            value = torch.maximum(value, pdb_val)

        # Top-B by ascending value.
        idx2 = torch.argsort(value)[:B]
        chosen_idx1 = idx1[idx2]
        next_states = candidate_states[idx2]
        return next_states, value[idx2], moves[chosen_idx1], idx0[chosen_idx1]

    def _check_stagnation(self, states_hash_log: deque) -> bool:
        """Return True if the last two layers' hashes are all contained in the first two layers'."""
        if len(states_hash_log) < 4:
            return False
        recent = torch.cat(list(states_hash_log)[2:])
        earlier = torch.cat(list(states_hash_log)[:2])
        return bool(torch.isin(recent, earlier).all().item())

    def solve(
        self, initial_state, cfg: KhoruzhiiSearchConfig
    ) -> tuple[bool, int, list[str]]:
        """Try to solve one puzzle. Returns (found, path_length, path_names)."""
        B = cfg.beam_width
        num_steps = cfg.num_steps
        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)
        if torch.equal(state, self.V0):
            return True, 0, []

        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        final_states = None
        final_j = None

        for attempt in range(cfg.num_attempts):
            states = state.unsqueeze(0).clone()
            tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
            states_hash_log: deque[torch.Tensor] = deque(maxlen=4)

            reached = None
            for j in range(num_steps):
                # Adaptive beam (N5): shrink beam_width geometrically with step index.
                # At step j: B_j = max(min_beam, round(B * decay**j)).
                if cfg.beam_decay != 1.0:
                    step_beam = max(cfg.min_beam_width, int(round(B * (cfg.beam_decay ** j))))
                else:
                    step_beam = B
                states, y_pred, moves, idx = self._do_greedy_step(states, states_bad_hashed, step_beam)
                if states.numel() == 0:
                    break
                states_hash_log.append(_state_hash(states, self.hash_vec))
                leaves = states.size(0)
                tree_move[j, :leaves] = moves.cpu()
                tree_idx[j, :leaves] = idx.cpu()

                if (states == self.V0).all(dim=1).any():
                    reached = j
                    break
                if j > 3 and self._check_stagnation(states_hash_log):
                    # Blacklist the recent hashes so the next attempt can try different paths.
                    new_bad = torch.cat(list(states_hash_log))
                    states_bad_hashed = torch.unique(torch.cat([states_bad_hashed, new_bad]))
                    break

            if reached is not None:
                final_states = states
                final_j = reached
                break

        if final_states is None or final_j is None:
            return False, 0, []

        # Reconstruct path. `tree_idx` and `tree_move` at row j reference the state at row j-1.
        tree_idx = tree_idx[: final_j + 1].flip((0,))
        tree_move = tree_move[: final_j + 1].flip((0,))

        v0_pos = torch.nonzero((final_states == self.V0).all(dim=1), as_tuple=True)[0].item()
        path = [tree_idx[0, v0_pos].item()]
        for k in range(1, final_j + 1):
            path.append(tree_idx[k, path[-1]].item())

        moves_seq_cpu = [
            (tree_move[k, path[k - 1]] if k > 0 else tree_move[k, v0_pos]).item()
            for k in range(final_j + 1)
        ]
        moves_seq_cpu.reverse()  # we built it from end to start
        names = [self.move_names[m] for m in moves_seq_cpu]
        return True, len(names), names
