"""Q-shortlister beam search.

Two-stage candidate scoring:
  1. STUDENT (24-output Q-head, m23): one forward per parent → 24 child scores per parent.
     Pick top-α·B candidates across all parents.
  2. TEACHER (V-head, m05): score only the α·B shortlisted candidates → pick final top-B.

Quality preserved if student's top-α·B contains teacher's top-B (validation gate at
≥99% recall). Recall measured offline with `09_eval_q_recall.py`.

Speedup: teacher's per-step forwards drop from ~17·B (unique children) to α·B.
At α=2 with 100% recall, that's an 8.5× teacher reduction. Adding the cheap student
pass (B forwards) gives net ~5-6× total model wall reduction.

Usage:
    --mode qshort --student models/m23_q_shortlister/epoch_0499.pt --alpha 2
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import torch

from beam_search import (
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    Profile,
    _model_predict,
    _state_hash,
    _sync,
)


class QShortlisterSolver(KhoruzhiiSolver):
    """Beam search with student-shortlister + teacher-reranker."""

    def __init__(
        self,
        puzzle,
        teacher,        # V-head model (output_dim=1) — m05
        student,        # Q-head model (output_dim=n_actions) — m23 (n_gen) or m24 (n_gen + n_macros)
        device: str = "cuda",
        internal_batch_size: int = 16384,
        random_seed: int = 0,
        state_dtype: torch.dtype = torch.int8,
        alpha: float = 2.0,
        pad_to_batch_size: bool = False,
        macros: list[tuple[list[int] | tuple[int, ...], list[str]]] | None = None,
        policy_model=None,
        lambda_policy: float = 0.0,
        phs_cumulative: bool = False,
        cutoff_model=None,
        cutoff_lambda: float = 0.0,
        cutoff_pool_mult: float = 1.0,
        cutoff_normalize: bool = True,
    ):
        # Initialize parent with teacher as the "model" — most helpers use self.model.
        super().__init__(
            puzzle, teacher, device=device,
            internal_batch_size=internal_batch_size,
            random_seed=random_seed, state_dtype=state_dtype,
            pad_to_batch_size=pad_to_batch_size,
            macros=macros,
            policy_model=policy_model,
            lambda_policy=lambda_policy,
            phs_cumulative=phs_cumulative,
        )
        self.teacher = teacher
        self.student = student.to(device).eval()
        self.alpha = alpha
        self.cutoff_model = cutoff_model.to(device).eval() if cutoff_model is not None else None
        self.cutoff_lambda = float(cutoff_lambda)
        self.cutoff_pool_mult = float(cutoff_pool_mult)
        self.cutoff_normalize = bool(cutoff_normalize)
        if self.cutoff_pool_mult < 1.0:
            raise ValueError(f"cutoff_pool_mult must be >= 1.0, got {self.cutoff_pool_mult}")
        # Validate student's output_dim matches n_actions (n_gen for vanilla m23,
        # n_gen + n_macros for the m24 Macro-Q shortlister).
        student_base = getattr(student, "_orig_mod", student)
        student_dim = getattr(student_base, "output_dim", 1)
        if student_dim != self.n_actions:
            raise ValueError(
                f"student.output_dim={student_dim} != n_actions={self.n_actions} "
                f"({self.n_gen} primitives + {self.n_macros} macros); this isn't "
                f"a Q-head matching the configured action set."
            )

    @torch.inference_mode()
    def _student_score(self, parents: torch.Tensor) -> torch.Tensor:
        """Score parents with the Q-head. Returns (n_parents, n_actions) float16.

        One forward per parent → all child scores in one go (n_gen primitives +
        n_macros macros when macros are configured).
        """
        bs = self.internal_batch_size
        out = torch.empty((parents.size(0), self.n_actions), dtype=torch.float16, device=self.device)
        for i in range(0, parents.size(0), bs):
            chunk = parents[i : i + bs]
            sz = chunk.size(0)
            if self.pad_to_batch_size and sz < bs:
                pad = chunk[-1:].expand(bs - sz, -1)
                padded = torch.cat([chunk, pad], dim=0)
                vals = self.student(padded).to(torch.float16)
                out[i : i + sz] = vals[:sz]
            else:
                out[i : i + sz] = self.student(chunk).to(torch.float16)
        return out

    def _do_qshort_step(self, states, states_hashed, states_bad_hashed, B, prof: Profile):
        """One beam step using student-shortlist + teacher-rerank.

        Returns (next_states, next_values, chosen_moves, chosen_parents, chosen_hashes).
        """
        n = states.size(0)
        bs = self.internal_batch_size

        # --- (1) Student: one forward per parent → (n, n_actions) ---
        _sync(); t0 = time.time()
        student_q = self._student_score(states)  # (n, n_actions) fp16
        # Each (parent_idx, action_idx) pair has a student score in this 2D tensor.
        # Flatten to (n * n_actions,) parallel to neighbors_hashed.
        student_q_flat = student_q.flatten().float()
        _sync(); model_student_s = time.time() - t0

        # --- (2) Hash all n_actions*n children (incremental for primitives, full for macros) ---
        _sync(); t0 = time.time()
        if hasattr(self, "_get_neighbors_hashed") and getattr(self, "changed_pos", None) is not None:
            neighbors_hashed = self._get_neighbors_hashed(states, states_hashed)
        else:
            neighbors_flat = self._get_neighbors(states).flatten(end_dim=1)
            neighbors_hashed = _state_hash(neighbors_flat, self.hash_vec, bs)
        _sync(); prof.neighbor_s += time.time() - t0

        # --- (3) Dedup ---
        _sync(); t0 = time.time()
        idx1 = self._unique_hashed_idx(neighbors_hashed, states_bad_hashed)
        _sync(); prof.dedup_s += time.time() - t0

        if idx1.numel() == 0:
            empty_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_v = torch.empty(0, dtype=torch.float16, device=self.device)
            empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty_s, empty_v, empty_i, empty_i, empty_v

        parent_of_idx1 = idx1 // self.n_actions
        move_of_idx1 = idx1 % self.n_actions

        # --- (4) Student shortlist: top αB by student Q (cost-adjusted for macros,
        # policy-adjusted for Idea 4 if enabled) ---
        # The shortlist gates which children get teacher V evaluation. Apply the
        # same cost-aware penalty (cost - 1) the teacher will use to score them so
        # macros don't over-saturate the shortlist with cheap V hits.
        # Policy term (Idea 4): if policy_model is set, also include +lambda *
        # (-log pi) so unlikely actions are filtered early. Same penalty applied
        # at teacher final pick below for consistency.
        _sync(); t0 = time.time()
        student_values = student_q_flat[idx1]  # (n_unique,) student scores
        if self.action_cost is not None:
            student_score = student_values + (self.action_cost[move_of_idx1].float() - 1.0)
        else:
            student_score = student_values
        # Compute log pi(a|parent) once per parent if policy is configured. Re-used
        # at teacher rerank step. log_pi_per_cand has the same shape as idx1.
        policy_cost = None  # per-candidate additive cost, aligned to idx1 (float32)
        if self.policy_model is not None and self.lambda_policy > 0:
            with torch.inference_mode():
                pol_out = self.policy_model(states.long())  # (n, n_gen)
            log_pi = torch.log_softmax(pol_out.float(), dim=-1)  # (n, n_gen)
            neg_log_pi = (-log_pi[parent_of_idx1, move_of_idx1]).float()  # (numel(idx1),) >= 0
            if self.phs_cumulative:
                # Cumulative path policy-cost: parent's running cost + this step's -log pi.
                policy_cost = self._phs_cum[parent_of_idx1] + neg_log_pi
            else:
                policy_cost = neg_log_pi  # local (memoryless) penalty
            student_score = student_score + self.lambda_policy * policy_cost.to(student_score.dtype)
        target_k = min(int(self.alpha * B), student_score.numel())
        if student_score.numel() <= target_k:
            shortlist_local = torch.arange(student_score.numel(), device=self.device)
        else:
            _, shortlist_local = torch.topk(
                student_score, target_k, largest=False, sorted=False
            )
        # The shortlist references positions within idx1 (the unique-after-dedup index)
        shortlist_idx1 = idx1[shortlist_local]
        shortlist_parents = parent_of_idx1[shortlist_local]
        shortlist_moves = move_of_idx1[shortlist_local]
        _sync(); prof.topk_s += (time.time() - t0) * 0.5  # half topk-time = student selection

        # --- (5) Materialize shortlist states ---
        _sync(); t0 = time.time()
        shortlist_states = self._apply_move(states[shortlist_parents], shortlist_moves)
        _sync(); prof.apply_s += time.time() - t0

        # --- (6) Teacher score: full V on shortlist only ---
        _sync(); t0 = time.time()
        teacher_value = _model_predict(
            self.teacher, shortlist_states, bs, pad_to_batch_size=self.pad_to_batch_size
        )
        _sync(); prof.model_s += (time.time() - t0) + model_student_s

        # --- (7) Teacher final pick: top B by teacher V (cost-adjusted for macros,
        # policy-adjusted for Idea 4 if enabled) ---
        _sync(); t0 = time.time()
        if self.action_cost is not None:
            cost_penalty = self.action_cost[shortlist_moves].float() - 1.0
            teacher_score = teacher_value.float() + cost_penalty
        else:
            teacher_score = teacher_value.float() if policy_cost is not None else teacher_value
        # Policy adjustment: reuse the per-candidate policy cost from the shortlist
        # stage (cumulative or local), indexed to the shortlist subset.
        if policy_cost is not None:
            teacher_score = teacher_score + self.lambda_policy * policy_cost[shortlist_local].to(teacher_score.dtype)
        if (
            self.cutoff_model is not None
            and self.cutoff_lambda != 0.0
            and self.cutoff_pool_mult > 1.0
            and teacher_score.numel() > B
        ):
            pool_n = min(teacher_score.numel(), max(B, int(round(B * self.cutoff_pool_mult))))
            if pool_n >= teacher_score.numel():
                pool_idx = torch.arange(teacher_score.numel(), device=self.device)
            else:
                _, pool_idx = torch.topk(teacher_score, pool_n, largest=False, sorted=False)
            cutoff_value = _model_predict(
                self.cutoff_model,
                shortlist_states[pool_idx],
                bs,
                pad_to_batch_size=self.pad_to_batch_size,
            ).float()
            if self.cutoff_normalize and cutoff_value.numel() > 1:
                cutoff_std = cutoff_value.std(unbiased=False).clamp_min(1e-3)
                cutoff_value = (cutoff_value - cutoff_value.mean()) / cutoff_std
            final_score = teacher_score[pool_idx].float() + self.cutoff_lambda * cutoff_value
            if final_score.numel() <= B:
                chosen_in_pool = torch.arange(final_score.numel(), device=self.device)
            else:
                _, chosen_in_pool = torch.topk(final_score, B, largest=False, sorted=False)
            chosen_local = pool_idx[chosen_in_pool]
        elif teacher_score.numel() <= B:
            chosen_local = torch.arange(teacher_score.numel(), device=self.device)
        else:
            _, chosen_local = torch.topk(teacher_score, B, largest=False, sorted=False)
        next_states = shortlist_states[chosen_local]
        next_values = teacher_value[chosen_local]
        chosen_moves = shortlist_moves[chosen_local]
        chosen_parents = shortlist_parents[chosen_local]
        # Advance the cumulative path policy-cost to the surviving beam (PHS mode).
        if policy_cost is not None and self.phs_cumulative:
            self._phs_cum = policy_cost[shortlist_local][chosen_local]
        chosen_hashes = neighbors_hashed[shortlist_idx1[chosen_local]]
        _sync(); prof.topk_s += (time.time() - t0) * 0.5  # other half of topk-time

        return next_states, next_values, chosen_moves, chosen_parents, chosen_hashes

    def solve(self, initial_state, cfg: KhoruzhiiSearchConfig):
        """Same protocol as KhoruzhiiSolver.solve, but uses _do_qshort_step internally."""
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
            states_hashed = _state_hash(states, self.hash_vec, self.internal_batch_size)
            # Root beam carries zero accumulated policy-cost (PHS cumulative mode).
            if self.phs_cumulative:
                self._phs_cum = torch.zeros(1, dtype=torch.float32, device=self.device)
            tree_move = torch.full((num_steps, B), -1, dtype=torch.int8, device=self.device)
            tree_idx = torch.full((num_steps, B), -1, dtype=torch.int32, device=self.device)
            states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
            reached = None

            for j in range(num_steps):
                states, _values, moves, parents, chosen_hashes = self._do_qshort_step(
                    states, states_hashed, states_bad_hashed, B, prof
                )
                if states.numel() == 0:
                    break

                states_hashed = chosen_hashes
                _sync(); t0 = time.time()
                states_hash_log.append(chosen_hashes)
                _sync(); prof.hash_s += time.time() - t0

                leaves = states.size(0)
                tree_move[j, :leaves] = moves.to(torch.int8)
                tree_idx[j, :leaves] = parents.to(torch.int32)
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

        tree_idx_cpu = tree_idx[: final_j + 1].flip((0,)).cpu().numpy()
        tree_move_cpu = tree_move[: final_j + 1].flip((0,)).cpu().numpy()
        v0_pos = int(torch.nonzero((final_states == self.V0).all(dim=1), as_tuple=True)[0][0])
        path = [int(tree_idx_cpu[0, v0_pos])]
        for k in range(1, final_j + 1):
            path.append(int(tree_idx_cpu[k, path[-1]]))
        moves_seq = [
            int(tree_move_cpu[k, path[k - 1]] if k > 0 else tree_move_cpu[k, v0_pos])
            for k in range(final_j + 1)
        ]
        moves_seq.reverse()
        # Expand macro action indices to their generator words. Primitive actions
        # (m < n_gen) emit a single generator name; macro actions (m >= n_gen)
        # emit macro_words[m - n_gen] in order.
        names: list[str] = []
        for m in moves_seq:
            if m < self.n_gen:
                names.append(self.move_names[m])
            else:
                names.extend(self.macro_words[m - self.n_gen])

        prof.total_s = time.time() - t_start
        prof.found = True
        prof.path_len = len(names)
        return True, len(names), names, prof
