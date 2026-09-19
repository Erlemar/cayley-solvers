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
from torch import nn

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
    # Cross-layer history dedup. When > 0, candidates whose hash appeared in any
    # of the last `history_depth` layers are excluded, so the beam cannot spend
    # width re-selecting states it just came from or oscillate in a small
    # recurrent set. 0 = original behaviour (dedup within the current layer
    # only). CayleyPy's library beam runs history_depth=10 by default and the
    # public tetraminx notebooks inherit that.
    history_depth: int = 0


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


def _qv_predict(model, states: torch.Tensor, batch_size: int, n_gen: int):
    """Q (N, n_gen) AND V (N,) from a SINGLE trunk pass per chunk.

    Dual-head models compute the trunk once and read two linear heads off it, so
    V(parent) costs one extra matmul on top of the Q the beam already needs -- it is
    free in forward-pass terms. Requires `model.return_value = True`.
    """
    model.eval()
    q = torch.empty((states.size(0), n_gen), dtype=torch.float16, device=states.device)
    v = torch.empty(states.size(0), dtype=torch.float16, device=states.device)
    with torch.no_grad():
        for i in range(0, states.size(0), batch_size):
            out = model(states[i : i + batch_size])
            qq, vv = out if isinstance(out, tuple) else (out, None)
            q[i : i + batch_size] = qq.to(torch.float16)
            if vv is not None:
                v[i : i + batch_size] = vv.to(torch.float16)
    return q, v


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
        q_progressive_topk: bool = True,
        q_topk_min: int = 16384,
        qv_rerank_model: nn.Module | None = None,
        qv_alpha: float = 2.0,
        qv_band_lo: float = 0.0,
        qv_consistency_lambda: float = 0.0,
        pdb_lookup=None,
        pdb_combine_mode: str = "replace_in_set",
        macros: list[tuple[list[int] | tuple[int, ...], list[str]]] | None = None,
        cutoff_model: nn.Module | None = None,
        cutoff_lambda: float = 0.0,
        cutoff_pool_mult: float = 1.0,
        cutoff_normalize: bool = True,
        cutoff_batch_size: int | None = None,
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
        # Progressive top-k candidate selection (Q path only). With a Q head every
        # (parent, action) pair already has a score before any child is built, so
        # there is no reason to materialise and hash all n_actions*B children: take
        # the best `q_topk_min`/2B by Q, hash only those, and double k only if the
        # unique count comes up short. Measured on tetraminx at B=65536 this takes a
        # step from 36.5 ms to 24.9 ms (V path is 465 ms). Set False for the old
        # exhaustive path.
        self.q_progressive_topk = bool(q_progressive_topk)
        self.q_topk_min = int(q_topk_min)
        # Optional scalar-V reranker over the Q shortlist (see _do_greedy_step_q_topk).
        self.qv_rerank_model = (qv_rerank_model.to(device).eval()
                                if qv_rerank_model is not None else None)
        self.qv_alpha = float(qv_alpha)
        if self.qv_rerank_model is not None and self.qv_alpha < 1.0:
            raise ValueError("qv_alpha must be >= 1")
        # MARGINAL-BAND RERANK. Reranking all alpha*B candidates wastes the expensive
        # model on the top ~90 pct, which enter the beam whatever V says; only the band
        # straddling the cutoff is genuinely uncertain. band_lo=0.9 auto-accepts the top
        # 0.9*B by Q and reranks only ranks [0.9B, alpha*B), cutting V forwards ~5x at
        # alpha=1.3. 0 = rerank everything (original behaviour).
        self.qv_band_lo = float(qv_band_lo)
        if not 0.0 <= self.qv_band_lo < 1.0:
            raise ValueError("qv_band_lo must be in [0, 1)")
        # V(parent) CONSISTENCY. Free: the dual head yields V(s) from the same trunk
        # pass as Q(s,.). The model asserts Q(s,a) ~= V(s) - 1 for a distance-reducing
        # move, so |Q - (V-1)| measures self-disagreement. Penalising it uses the value
        # head as an inference-time regulariser rather than as a scorer -- no extra
        # forwards at all.
        self.qv_consistency_lambda = float(qv_consistency_lambda)
        # Optional admissible re-scorer: pdb_lookup(states) → (B,) float lower bound.
        # combine_mode controls how it merges with neural V:
        #   "replace_in_set": PDB returns sentinel (>=1e5) for out-of-set; replaces V for in-set.
        #   "max": value = max(V_neural, PDB) — for full-coverage PDBs that never sentinel.
        if pdb_combine_mode not in ("replace_in_set", "max"):
            raise ValueError(f"pdb_combine_mode must be 'replace_in_set' or 'max', got {pdb_combine_mode!r}")
        self.pdb_lookup = pdb_lookup
        self.pdb_combine_mode = pdb_combine_mode
        self.cutoff_model = cutoff_model.to(device).eval() if cutoff_model is not None else None
        self.cutoff_lambda = float(cutoff_lambda)
        self.cutoff_pool_mult = float(cutoff_pool_mult)
        self.cutoff_normalize = bool(cutoff_normalize)
        self.cutoff_batch_size = int(cutoff_batch_size or internal_batch_size)
        if self.cutoff_model is not None and self.cutoff_pool_mult < 1.0:
            raise ValueError("cutoff_pool_mult must be >= 1.0")

        # Generator permutations as a (n_gen, state_size) int64 tensor — gather indices.
        n_gen = len(puzzle.move_names)
        self.n_gen = n_gen
        self.state_size = len(puzzle.solved_state)
        # Validate Q-head output_dim now that we know n_gen + n_macros. The Q-head can be
        # either a vanilla shortlister (output_dim = n_gen) when no macros are passed, or
        # a Macro-Q shortlister (output_dim = n_gen + n_macros) when macros are passed.
        # We finish the check after counting macros below.
        n_macros_arg = len(macros) if macros else 0
        self.all_moves = torch.zeros((n_gen, self.state_size), dtype=torch.int64, device=device)
        for i, name in enumerate(puzzle.move_names):
            self.all_moves[i] = torch.tensor(puzzle.generators[name], dtype=torch.int64)
        self.move_names = puzzle.move_names
        self.V0 = torch.tensor(puzzle.solved_state, dtype=state_dtype, device=device)

        # Macro extension: each macro is (perm, word) where perm is the macro's net
        # permutation (length state_size) and word is the list of generator names it
        # expands to. Each macro becomes an additional action — at each beam step,
        # the parent state is extended by (24 + n_macros) children. Path reconstruction
        # expands macro actions to their words (so output is in real moves, not action
        # steps). Macros are a no-op when use_q_function=True (Q-head has fixed
        # output_dim=n_gen and doesn't know about macros).
        #
        # Cost-aware scoring: each action has a real-move cost (1 for a gen, len(word)
        # for a macro). When the beam compares candidates at a single step, candidates
        # reached via a macro of cost L would otherwise look "free" (V dropped, but L
        # real moves were spent). We add (cost - 1) to the predicted V so a macro must
        # deliver more V-drop than its extra cost to be selected. This is an
        # admissible-style cost adjustment for the greedy beam.
        if macros:
            for perm, word in macros:
                assert len(perm) == self.state_size, "macro perm size mismatch"
                for nm in word:
                    assert nm in puzzle.generators, f"macro word has unknown name {nm}"
            self.n_macros = len(macros)
            macro_perms = torch.tensor(
                [list(p) for p, _ in macros], dtype=torch.int64, device=device
            )
            self.all_moves = torch.cat([self.all_moves, macro_perms], dim=0)
            self.macro_words = [list(w) for _, w in macros]
            # action_cost[i] = real-move cost of action i. gens cost 1, macros cost len(word).
            costs = [1] * self.n_gen + [len(w) for _, w in macros]
            self.action_cost = torch.tensor(costs, dtype=torch.float16, device=device)
        else:
            self.n_macros = 0
            self.macro_words = []
            self.action_cost = None
        self.n_actions = self.n_gen + self.n_macros

        # Q-head validation — see initial setup; deferred until n_actions is known.
        if use_q_function:
            base = getattr(model, "_orig_mod", model)
            expected = self.n_actions  # n_gen for vanilla shortlister, n_gen+n_macros for macro-Q
            got = getattr(base, "output_dim", 1)
            if got != expected:
                raise ValueError(
                    f"use_q_function=True requires model.output_dim={expected} "
                    f"({self.n_gen} primitives + {self.n_macros} macros), got {got}"
                )

        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)
        self.hash_vec = torch.randint(
            0, int(1e15), (self.state_size,), dtype=torch.int64, device=device, generator=gen
        )

    def _get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """(B, S) → (B*n_actions, S). Action 0..n_gen-1 is a generator; action
        n_gen..n_actions-1 is a macro applied as a single permutation."""
        B = states.size(0)
        bs = self.internal_batch_size
        out = torch.empty((B, self.n_actions, self.state_size), dtype=states.dtype, device=self.device)
        for i in range(0, B, bs):
            chunk = states[i : i + bs]
            out[i : i + bs] = torch.gather(
                chunk.unsqueeze(1).expand(chunk.size(0), self.n_actions, self.state_size),
                2,
                self.all_moves.unsqueeze(0).expand(chunk.size(0), self.n_actions, self.state_size),
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

    def _hash_flat_candidates(self, states: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        """Hash the children named by flat indices (parent * n_actions + action)."""
        bs = self.internal_batch_size
        out = torch.empty(flat_idx.numel(), dtype=torch.int64, device=self.device)
        for i in range(0, flat_idx.numel(), bs):
            f = flat_idx[i : i + bs]
            parent = torch.div(f, self.n_actions, rounding_mode="floor")
            move = torch.remainder(f, self.n_actions)
            child = torch.gather(states.index_select(0, parent), 1, self.all_moves[move])
            out[i : i + bs] = _state_hash(child, self.hash_vec, bs)
        return out

    @staticmethod
    def _first_unique_positions(hashed: torch.Tensor) -> torch.Tensor:
        """Positions of the first occurrence of each hash, in input order.

        Callers pass a score-sorted array, so "first" means "best scoring".
        """
        pos = torch.arange(hashed.numel(), dtype=torch.int64, device=hashed.device)
        unique, inverse = torch.unique(hashed, return_inverse=True)
        first = torch.full((unique.numel(),), hashed.numel(), dtype=torch.int64, device=hashed.device)
        first.scatter_reduce_(0, inverse, pos, reduce="amin", include_self=True)
        return pos[first.index_select(0, inverse) == pos]

    def _empty_step(self, states: torch.Tensor):
        empty_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
        empty_v = torch.empty(0, dtype=torch.float16, device=self.device)
        empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
        return empty_s, empty_v, empty_i, empty_i

    def _do_greedy_step_q_topk(
        self, states: torch.Tensor, states_bad_hashed: torch.Tensor, B: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q-path beam step that only builds and hashes the top candidates.

        Equivalent to the exhaustive path up to tie-breaking: k doubles until B
        distinct, non-blacklisted children are in hand or every candidate has been
        considered.
        """
        bs = self.internal_batch_size
        if self.qv_consistency_lambda != 0.0 and getattr(self.model, "has_value_head", False):
            base = getattr(self.model, "_orig_mod", self.model)
            base.return_value = True
            q_all, v_parent = _qv_predict(self.model, states, bs, self.n_actions)
            base.return_value = False
        else:
            q_all = _q_predict(self.model, states, bs, self.n_actions)  # (n, n_actions) fp16
            v_parent = None
        q_flat = q_all.reshape(-1)
        score_flat = q_flat.float()
        if v_parent is not None:
            # The model asserts Q(s,a) ~= V(s) - 1 for a distance-reducing move; charge
            # children whose Q disagrees with the parent's own distance estimate.
            expect = (v_parent.float() - 1.0).unsqueeze(1).expand_as(q_all).reshape(-1)
            score_flat = score_flat + self.qv_consistency_lambda * (score_flat - expect).abs()
        if self.action_cost is not None:
            # Macros cost more than one real move, so charge them before ranking.
            penalty = (self.action_cost.float() - 1.0).view(1, -1).expand_as(q_all)
            score_flat = score_flat + penalty.reshape(-1)

        # Q-shortlist -> V-rerank: take alpha*B candidates by Q, then let a scalar
        # V model choose the final B among them. This is the only mode in which a
        # dual-head model's VALUE head does work at inference -- otherwise the beam
        # reads the Q head and the value head acts only through trunk interference.
        # It is also the arrangement that won on megaminx (AZ v4's V head was the
        # scorer), so it is the closest analogue worth testing here.
        select_B = B if self.qv_rerank_model is None else max(B, int(round(B * self.qv_alpha)))

        total = score_flat.numel()
        k = min(total, max(2 * select_B, self.q_topk_min))
        while True:
            _, cand = torch.topk(score_flat, k=k, largest=False, sorted=True)
            hashed = self._hash_flat_candidates(states, cand)
            if states_bad_hashed.numel() > 0:
                keep = ~torch.isin(hashed, states_bad_hashed)
                cand, hashed = cand[keep], hashed[keep]
            if cand.numel() > 0:
                unique_pos = self._first_unique_positions(hashed)
                if unique_pos.numel() >= select_B or k == total:
                    cand = cand.index_select(0, unique_pos[:select_B])
                    break
            if k == total:
                return self._empty_step(states)
            k = min(total, k * 2)

        parent = torch.div(cand, self.n_actions, rounding_mode="floor")
        move = torch.remainder(cand, self.n_actions)
        next_states = self._apply_move(states.index_select(0, parent), move)
        if self.qv_rerank_model is not None and next_states.size(0) > B:
            # `cand` is Q-ascending (topk sorted, first-unique preserves order), so the
            # first n_head entries are the confident ones.
            n_head = int(self.qv_band_lo * B)
            if n_head <= 0:
                v = _model_predict(self.qv_rerank_model, next_states, self.internal_batch_size)
                keep = torch.argsort(v.float())[:B]
            else:
                band = next_states[n_head:]
                v_band = _model_predict(self.qv_rerank_model, band, self.internal_batch_size)
                take = torch.argsort(v_band.float())[: B - n_head] + n_head
                keep = torch.cat((torch.arange(n_head, device=next_states.device), take))
            return (next_states.index_select(0, keep), q_flat.index_select(0, cand).index_select(0, keep),
                    move.index_select(0, keep), parent.index_select(0, keep))
        return next_states, q_flat.index_select(0, cand), move, parent

    def _do_greedy_step(
        self, states: torch.Tensor, states_bad_hashed: torch.Tensor, B: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of beam search. Returns (next_states, values, moves, parent_idx)."""
        n = states.size(0)
        bs = self.internal_batch_size

        if self.use_q_function and self.q_progressive_topk:
            return self._do_greedy_step_q_topk(states, states_bad_hashed, B)

        # For each state and each action (generator or macro), remember (parent_idx, action_idx).
        idx0 = torch.arange(n, device=self.device).repeat_interleave(self.n_actions)
        moves = torch.arange(self.n_actions, device=self.device).repeat(n)

        # Compute + hash neighbors in batches.
        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, n, bs):
            chunk_states = states[i : i + bs]
            chunk_neighbors = self._get_neighbors(chunk_states).flatten(end_dim=1)
            neighbors_hashed[i * self.n_actions : (i + chunk_states.size(0)) * self.n_actions] = _state_hash(
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
            # One forward per parent yields (n, n_actions) predicted neighbor values.
            # Flattened as [p0*n_actions + a, p1*n_actions + a, ...] matches the
            # idx0*n_actions+moves ordering set up above. n_actions = n_gen for vanilla
            # m23, or n_gen + n_macros for the m24 Macro-Q shortlister.
            q_all = _q_predict(self.model, states, bs, self.n_actions)  # (n, n_actions) fp16
            q_flat = q_all.view(-1)
            value = q_flat[idx1]
            # Cost-aware adjustment: macros cost > 1 real move, so a macro candidate must
            # deliver more V-drop than its extra cost to win. Same rule the V-path uses.
            if self.action_cost is not None:
                chosen_actions = moves[idx1]
                cost_penalty = self.action_cost[chosen_actions] - 1.0  # 0 for gens, >0 for macros
                score = value + cost_penalty
            else:
                score = value
            idx2 = torch.argsort(score)[:B]
            chosen_idx1 = idx1[idx2]
            next_states = self._apply_move(states[idx0[chosen_idx1]], moves[chosen_idx1])
            return next_states, value[idx2], moves[chosen_idx1], idx0[chosen_idx1]
        # V-model path: evaluate value on each unique candidate.
        candidate_states = self._apply_move(states[idx0[idx1]], moves[idx1])
        value = _model_predict(self.model, candidate_states, bs)
        if self.pdb_lookup is not None:
            pdb_val = self.pdb_lookup(candidate_states).to(torch.float16)
            if self.pdb_combine_mode == "replace_in_set":
                # PDB returns sentinel (>=1e5) for out-of-set. Replace V on in-set.
                in_pdb_set = pdb_val < 1e5
                value = torch.where(in_pdb_set, pdb_val, value)
            else:  # "max"
                # Full-coverage PDB. value = max(V, PDB) — admissible floor.
                value = torch.maximum(value, pdb_val)

        # Cost-aware adjustment: penalize candidates reached via expensive actions
        # (i.e., macros costing > 1 real move) so a macro must deliver V-drop greater
        # than its extra cost. Score = V + (action_cost - 1).
        if self.action_cost is not None:
            chosen_actions = moves[idx1]
            cost_penalty = self.action_cost[chosen_actions] - 1.0  # 0 for gens, >0 for macros
            score = value + cost_penalty
        else:
            score = value

        # Optional beam-retention head: keep the global V as the coarse scorer,
        # then rerank only an overgenerated near-cutoff pool. This is deliberately
        # local; the auxiliary head cannot pull arbitrary bad states into the beam.
        if (
            self.cutoff_model is not None
            and self.cutoff_lambda != 0.0
            and self.cutoff_pool_mult > 1.0
            and score.numel() > B
        ):
            pool_n = min(score.numel(), max(B, int(round(B * self.cutoff_pool_mult))))
            pool_idx = torch.argsort(score)[:pool_n]
            cutoff = _model_predict(
                self.cutoff_model, candidate_states[pool_idx], self.cutoff_batch_size
            ).float()
            if self.cutoff_normalize:
                cutoff = (cutoff - cutoff.mean()) / cutoff.std(unbiased=False).clamp_min(1.0e-3)
            final_score = score[pool_idx].float() + self.cutoff_lambda * cutoff
            idx2 = pool_idx[torch.argsort(final_score)[:B]]
        else:
            # Top-B by ascending score.
            idx2 = torch.argsort(score)[:B]
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
        self, initial_state, cfg: KhoruzhiiSearchConfig,
        goal_check_fn=None, stats: dict | None = None,
    ) -> tuple[bool, int, list[str]]:
        """Try to solve one puzzle. Returns (found, path_length, path_names).

        goal_check_fn: optional callable (states tensor) -> bool tensor of shape (B,).
            Default is `lambda s: (s == self.V0).all(dim=1)` (full-puzzle solve).
            Use a custom predicate (e.g., F2L-correct check) for staged solving.
        stats: optional dict. When given, per-step ancestor-diversity is appended to
            `stats["steps"]` -- how many distinct depth-`stats["ancestor_depth"]`
            openings (default 2) the surviving beam still descends from. Purely
            diagnostic; leaving it None is a no-op on the search itself.
        """
        if goal_check_fn is None:
            goal_check_fn = lambda s: (s == self.V0).all(dim=1)
        B = cfg.beam_width
        num_steps = cfg.num_steps
        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)
        if goal_check_fn(state.unsqueeze(0))[0].item():
            return True, 0, []

        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        final_states = None
        final_j = None

        for attempt in range(cfg.num_attempts):
            states = state.unsqueeze(0).clone()
            tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
            states_hash_log: deque[torch.Tensor] = deque(
                maxlen=max(4, cfg.history_depth))

            reached = None
            anc = None
            k_anc = int(stats.get("ancestor_depth", 2)) if stats is not None else 0
            for j in range(num_steps):
                # Adaptive beam (N5): shrink beam_width geometrically with step index.
                # At step j: B_j = max(min_beam, round(B * decay**j)).
                if cfg.beam_decay != 1.0:
                    step_beam = max(cfg.min_beam_width, int(round(B * (cfg.beam_decay ** j))))
                else:
                    step_beam = B
                excluded = states_bad_hashed
                if cfg.history_depth > 0 and states_hash_log:
                    recent = list(states_hash_log)[-cfg.history_depth:]
                    excluded = torch.cat([states_bad_hashed, *recent])
                states, y_pred, moves, idx = self._do_greedy_step(states, excluded, step_beam)
                if states.numel() == 0:
                    break
                if stats is not None:
                    # idx[i] is the index of child i's parent in the PREVIOUS beam, so
                    # carrying the tag through index_select tracks each survivor back to
                    # the depth-k_anc state it descends from.
                    if j == k_anc - 1:
                        anc = torch.arange(states.size(0), device=idx.device)
                        n_anc = states.size(0)
                        # keep the opening states so the caller can identify WHICH
                        # opening a known-good path uses, not just how many survive
                        stats["ancestor_states"] = states.detach().cpu()
                    elif j >= k_anc and anc is not None:
                        anc = anc.index_select(0, idx)
                    if anc is not None:
                        full = torch.bincount(anc, minlength=n_anc)
                        counts = full[full > 0].double()
                        tot = counts.sum()
                        p = counts / tot
                        stats.setdefault("steps", []).append(dict(
                            attempt=attempt, step=j, depth=j + 1, beam=int(tot.item()),
                            openings=int(counts.numel()),
                            top1=float((counts.max() / tot).item()),
                            effective=float(torch.exp(-(p * p.log()).sum()).item()),
                            counts=full.cpu().tolist(),
                        ))
                states_hash_log.append(_state_hash(states, self.hash_vec))
                leaves = states.size(0)
                tree_move[j, :leaves] = moves.cpu()
                tree_idx[j, :leaves] = idx.cpu()

                if goal_check_fn(states).any():
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

        v0_pos = torch.nonzero(goal_check_fn(final_states), as_tuple=True)[0][0].item()
        path = [tree_idx[0, v0_pos].item()]
        for k in range(1, final_j + 1):
            path.append(tree_idx[k, path[-1]].item())

        moves_seq_cpu = [
            (tree_move[k, path[k - 1]] if k > 0 else tree_move[k, v0_pos]).item()
            for k in range(final_j + 1)
        ]
        moves_seq_cpu.reverse()  # we built it from end to start
        names: list[str] = []
        for m in moves_seq_cpu:
            if m < self.n_gen:
                names.append(self.move_names[m])
            else:
                names.extend(self.macro_words[m - self.n_gen])
        return True, len(names), names
