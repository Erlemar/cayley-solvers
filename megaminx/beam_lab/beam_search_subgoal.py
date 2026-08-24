"""Hierarchical subgoal search v0 -- the "macro-jump beam" (big-swings Dir 3).

Faithful, minimal instantiation of kSubS/AdaSubS for a permutation puzzle, with
the literature's three hardest components (state-generator + reachability
verifier + conditional low-level policy) ELIMINATED by generating moves directly:
a subgoal = the state reached after k policy-driven atomic steps, so every
subgoal is reachable by construction.

Mechanism (the only thing that differs from a normal beam): the teacher value V
is queried to SELECT the beam only every k-th step. On the k-1 intermediate
steps the beam is pruned by the POLICY alone (cumulative -log pi along the
within-macro path), with NO V evaluation. This is exactly the "leap over a
faulty value function by taking longer steps" robustness mechanism (Zawalski et
al. 2406.03361): over a depth-L solve V makes L/k noisy decisions instead of L.

Why the AZ v4 POLICY (not the m23 Q-shortlister) is the proposer: m23 is
distilled from the teacher V, so it inherits V's mid-depth optimism -- using it
between V-steps would NOT escape V's errors. The AZ v4 policy head is an
action-likelihood trained toward improved play (~50% top-1, genuinely diverse),
largely independent of V's distance errors -- the right "leap over faulty V"
signal.

k=1 makes every step a V-step (policy-shortlist -> V-select): that is the
BASELINE arm. Arm B is the same solver at k>1. The ONLY difference between arms
is the V-gating period, so the A/B isolates the subgoal mechanism cleanly.
"""
from __future__ import annotations

import time
from collections import deque

import torch

from beam_search import (
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    Profile,
    _model_predict,
    _state_hash,
    _sync,
)


class SubgoalSolver(KhoruzhiiSolver):
    """Beam search with V-selection gated to every k-th step (policy between)."""

    def __init__(
        self,
        puzzle,
        teacher_v,         # V-head (output_dim=1) -- selects on V-steps
        policy,            # policy head (output_dim=n_gen) -- proposes + selects between
        device: str = "cuda",
        internal_batch_size: int = 16384,
        random_seed: int = 0,
        state_dtype: torch.dtype = torch.int8,
        alpha: float = 8.0,        # V-step shortlist size = alpha*B (by cumulative policy)
        policy_temp: float = 1.0,  # softmax temperature on policy logits
        pad_to_batch_size: bool = False,
    ):
        super().__init__(
            puzzle, teacher_v, device=device,
            internal_batch_size=internal_batch_size,
            random_seed=random_seed, state_dtype=state_dtype,
            pad_to_batch_size=pad_to_batch_size,
        )
        assert self.n_macros == 0, "subgoal v0 uses primitive generators only"
        self.policy = policy.to(device).eval()
        self.alpha = float(alpha)
        self.policy_temp = float(policy_temp)
        base = getattr(policy, "_orig_mod", policy)
        pdim = getattr(base, "output_dim", 1)
        if pdim != self.n_gen:
            raise ValueError(f"policy output_dim={pdim} != n_gen={self.n_gen}")
        self._sg_cum = None  # within-macro cumulative -log pi, aligned to the live beam

    @torch.inference_mode()
    def _policy_logpi(self, parents: torch.Tensor) -> torch.Tensor:
        """log pi(.|parent) for each parent. Returns (n_parents, n_gen) float32."""
        bs = self.internal_batch_size
        out = torch.empty((parents.size(0), self.n_gen), dtype=torch.float32, device=self.device)
        for i in range(0, parents.size(0), bs):
            chunk = parents[i:i + bs]
            sz = chunk.size(0)
            if self.pad_to_batch_size and sz < bs:
                pad = chunk[-1:].expand(bs - sz, -1)
                logits = self.policy(torch.cat([chunk, pad], 0).long()).float()[:sz]
            else:
                logits = self.policy(chunk.long()).float()
            out[i:i + sz] = torch.log_softmax(logits / self.policy_temp, dim=-1)
        return out

    def _do_subgoal_step(self, states, states_hashed, states_bad, B, use_V, prof):
        """One beam step. use_V: select by teacher V (and reset macro); else select
        by cumulative policy cost with NO V eval."""
        bs = self.internal_batch_size

        # policy on parents -> cumulative within-macro -log pi for every child
        _sync(); t0 = time.time()
        log_pi = self._policy_logpi(states)                       # (n, n_gen)
        _sync(); prof.model_s += time.time() - t0

        _sync(); t0 = time.time()
        neighbors_hashed = self._get_neighbors_hashed(states, states_hashed)
        _sync(); prof.neighbor_s += time.time() - t0
        _sync(); t0 = time.time()
        idx1 = self._unique_hashed_idx(neighbors_hashed, states_bad)
        _sync(); prof.dedup_s += time.time() - t0
        if idx1.numel() == 0:
            e_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            e_v = torch.empty(0, dtype=torch.float16, device=self.device)
            e_i = torch.empty(0, dtype=torch.int64, device=self.device)
            return e_s, e_v, e_i, e_i, e_v

        parent_of = idx1 // self.n_actions
        move_of = idx1 % self.n_actions
        neg_log_pi = (-log_pi[parent_of, move_of])                # (n_unique,) >= 0
        cand_cum = self._sg_cum[parent_of] + neg_log_pi           # within-macro cumulative

        if use_V:
            # shortlist top-alpha*B by cumulative policy, then V-rerank to top-B
            _sync(); t0 = time.time()
            target_k = min(int(self.alpha * B), cand_cum.numel())
            if cand_cum.numel() <= target_k:
                sl = torch.arange(cand_cum.numel(), device=self.device)
            else:
                _, sl = torch.topk(cand_cum, target_k, largest=False, sorted=False)
            sl_idx1 = idx1[sl]; sl_parent = parent_of[sl]; sl_move = move_of[sl]
            _sync(); prof.topk_s += time.time() - t0
            _sync(); t0 = time.time()
            sl_states = self._apply_move(states[sl_parent], sl_move)
            _sync(); prof.apply_s += time.time() - t0
            _sync(); t0 = time.time()
            v = _model_predict(self.model, sl_states, bs, pad_to_batch_size=self.pad_to_batch_size)
            _sync(); prof.model_s += time.time() - t0
            self._n_v_evals += int(sl_states.size(0))
            self._n_v_steps += 1
            _sync(); t0 = time.time()
            if v.numel() <= B:
                chosen = torch.arange(v.numel(), device=self.device)
            else:
                _, chosen = torch.topk(v, B, largest=False, sorted=False)
            _sync(); prof.topk_s += time.time() - t0
            next_states = sl_states[chosen]; next_values = v[chosen]
            chosen_moves = sl_move[chosen]; chosen_parents = sl_parent[chosen]
            chosen_hashes = neighbors_hashed[sl_idx1[chosen]]
            # macro boundary: just landed on a V-selected subgoal -> reset macro cost
            self._sg_cum = torch.zeros(next_states.size(0), dtype=torch.float32, device=self.device)
        else:
            # policy-step: top-B by cumulative policy, NO V eval
            _sync(); t0 = time.time()
            if cand_cum.numel() <= B:
                chosen = torch.arange(cand_cum.numel(), device=self.device)
            else:
                _, chosen = torch.topk(cand_cum, B, largest=False, sorted=False)
            ch_idx1 = idx1[chosen]; chosen_parents = parent_of[chosen]; chosen_moves = move_of[chosen]
            _sync(); prof.topk_s += time.time() - t0
            _sync(); t0 = time.time()
            next_states = self._apply_move(states[chosen_parents], chosen_moves)
            _sync(); prof.apply_s += time.time() - t0
            next_values = cand_cum[chosen]
            chosen_hashes = neighbors_hashed[ch_idx1]
            self._sg_cum = cand_cum[chosen]      # carry within-macro cumulative
            self._n_policy_steps += 1
        return next_states, next_values, chosen_moves, chosen_parents, chosen_hashes

    def solve(self, initial_state, cfg: KhoruzhiiSearchConfig, k: int = 1):
        """Solve one puzzle. V-steps occur when (j+1) % k == 0; k=1 -> every step.
        Returns (found, path_len, names, prof); prof carries n_v_evals / n_v_steps /
        n_policy_steps for matched-budget accounting."""
        prof = Profile()
        self._n_v_evals = 0; self._n_v_steps = 0; self._n_policy_steps = 0
        _sync(); t_start = time.time()
        B = cfg.beam_width
        num_steps = cfg.num_steps

        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)
        if torch.equal(state, self.V0):
            prof.total_s = time.time() - t_start; prof.found = True
            prof.n_v_evals = 0; prof.n_v_steps = 0; prof.n_policy_steps = 0
            return True, 0, [], prof

        states = state.unsqueeze(0).clone()
        states_hashed = _state_hash(states, self.hash_vec, self.internal_batch_size)
        states_bad = torch.empty(0, dtype=torch.int64, device=self.device)
        self._sg_cum = torch.zeros(1, dtype=torch.float32, device=self.device)
        tree_move = torch.full((num_steps, B), -1, dtype=torch.int8, device=self.device)
        tree_idx = torch.full((num_steps, B), -1, dtype=torch.int32, device=self.device)
        states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
        reached = None

        for j in range(num_steps):
            use_V = ((j + 1) % k == 0)
            states, _vals, moves, parents, chosen_hashes = self._do_subgoal_step(
                states, states_hashed, states_bad, B, use_V, prof)
            if states.numel() == 0:
                break
            states_hashed = chosen_hashes
            states_hash_log.append(chosen_hashes)
            leaves = states.size(0)
            tree_move[j, :leaves] = moves.to(torch.int8)
            tree_idx[j, :leaves] = parents.to(torch.int32)
            prof.per_step_states.append(int(leaves))
            prof.n_steps = j + 1
            if (states == self.V0).all(dim=1).any():
                reached = j
                break
            # no stagnation early-break: keep both A/B arms identical except k

        prof.n_v_evals = self._n_v_evals
        prof.n_v_steps = self._n_v_steps
        prof.n_policy_steps = self._n_policy_steps
        if reached is None:
            prof.total_s = time.time() - t_start; prof.found = False
            return False, 0, [], prof

        tree_idx_cpu = tree_idx[:reached + 1].flip((0,)).cpu().numpy()
        tree_move_cpu = tree_move[:reached + 1].flip((0,)).cpu().numpy()
        v0_pos = int(torch.nonzero((states == self.V0).all(dim=1), as_tuple=True)[0][0])
        path = [int(tree_idx_cpu[0, v0_pos])]
        for kk in range(1, reached + 1):
            path.append(int(tree_idx_cpu[kk, path[-1]]))
        moves_seq = [int(tree_move_cpu[kk, path[kk - 1]] if kk > 0 else tree_move_cpu[kk, v0_pos])
                     for kk in range(reached + 1)]
        moves_seq.reverse()
        names = [self.move_names[m] for m in moves_seq]
        prof.total_s = time.time() - t_start; prof.found = True; prof.path_len = len(names)
        return True, len(names), names, prof
