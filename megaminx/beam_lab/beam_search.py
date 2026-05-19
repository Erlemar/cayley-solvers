"""KhoruzhiiSolver — beam search with profiling hooks for optimization work.

Self-contained copy of the production solver, with one addition: each `solve()`
call now records per-stage timings into a `Profile` object you can read back.

Profile-collected fields per call:
    - total_s          # wall clock from solve() entry to return
    - n_steps          # number of beam-search layers actually expanded
    - hash_s           # cumulative time spent hashing states
    - dedup_s          # cumulative time in unique-hash filtering
    - neighbor_s       # cumulative time in `_get_neighbors`
    - apply_s          # cumulative time in `_apply_move`
    - model_s          # cumulative time in model forward
    - topk_s           # cumulative time in argsort/top-k

These are wall-clock timings on a backend synchronization boundary so they are
real seconds, not async-launch overhead.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch


@dataclass
class KhoruzhiiSearchConfig:
    beam_width: int = 131072
    num_steps: int = 120
    num_attempts: int = 1
    internal_batch_size: int = 16384
    use_incremental_hash: bool = True
    reuse_full_neighbors: bool = False
    # Geometric narrowing: step_beam = max(min_beam_width, int(B * beam_decay**j)).
    # 1.0 = constant beam (default); 0.97 = beam shrinks to ~3% of original at step 120.
    beam_decay: float = 1.0
    min_beam_width: int = 0
    # Stochastic sampling temperature. 0.0 = greedy top-B (default).
    # T > 0: sample B without replacement from softmax(-value/T) via Gumbel-top-k.
    # Higher T = more exploration (closer to uniform), lower T → greedy.
    temperature: float = 0.0
    # Async tuning. Defer the (CPU-syncing) solved-position check to every K steps:
    # 1 = check every step (default, current behavior); larger = trades a few extra
    # forward passes for fewer CPU syncs. Stagnation check still runs every step.
    solved_check_every: int = 1


@dataclass
class Profile:
    total_s: float = 0.0
    n_steps: int = 0
    found: bool = False
    path_len: int = 0
    hash_s: float = 0.0
    dedup_s: float = 0.0
    neighbor_s: float = 0.0
    apply_s: float = 0.0
    model_s: float = 0.0
    topk_s: float = 0.0
    per_step_states: list[int] = field(default_factory=list)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def _noop_sync():
    """No-op replacement for _sync used when profile=False."""


def _state_hash(states, hash_vec, batch_size=16384):
    out = torch.empty(states.size(0), dtype=torch.int64, device=states.device)
    for i in range(0, states.size(0), batch_size):
        chunk = states[i : i + batch_size].to(torch.int64)
        out[i : i + batch_size] = torch.sum(hash_vec * chunk, dim=1)
    return out


def _model_predict(model, states, batch_size=16384, pad_to_batch_size: bool = False):
    """Run model on states in chunks. Uses inference_mode (stricter than no_grad).

    NOTE: callers should set `model.inference_chunk_size = None` to avoid double-chunking
    (the model also chunks internally at `inference_chunk_size`; see `setup_model_for_inference`).

    `pad_to_batch_size`: when True, every forward call has shape
    `(batch_size, state_size)` — short final chunks are padded with copies of the
    last state, then the padded slots are dropped after model forward. This is
    REQUIRED for torch.compile to avoid recompile loops on variable shapes.
    Set this together with `setup_model_for_compile()`.
    """
    model.eval()
    n = states.size(0)
    out = torch.empty(n, dtype=torch.float16, device=states.device)
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            chunk = states[i : i + batch_size]
            sz = chunk.size(0)
            if pad_to_batch_size and sz < batch_size:
                pad = chunk[-1:].expand(batch_size - sz, -1)
                padded = torch.cat([chunk, pad], dim=0)
                # Cast to int64 — TRT engines built with dtype=int64 require it; eager
                # model's `x.long()` makes int64 input a no-op, so this is safe both ways.
                vals = model(padded.long()).flatten().to(torch.float16)
                out[i : i + sz] = vals[:sz]
            else:
                out[i : i + sz] = model(chunk.long()).flatten().to(torch.float16)
    return out


def setup_model_for_compile(model, batch_size: int = 16384, mode: str = "reduce-overhead"):
    """Wrap the model with torch.compile, pre-warm at the fixed batch_size.

    Call AFTER `setup_model_for_inference`. Returns the compiled model.

    The pre-warm forward triggers compilation; subsequent calls with the same
    shape reuse the compiled graph. If you call with different shapes (e.g.
    forgot to enable `pad_to_batch_size` in `_model_predict`), torch will
    recompile — which is the 5.8× slowdown the project has observed historically.
    """
    import torch as _t
    compiled = _t.compile(model, mode=mode, dynamic=False, fullgraph=False)
    # Pre-warm: trigger compile with a synthetic batch matching our search shape.
    # State is (batch, state_size); use zeros and run twice (first call compiles).
    state_size = getattr(model, "state_size", 120)
    device = next(model.parameters()).device
    dummy = _t.zeros((batch_size, state_size), dtype=_t.int64, device=device)
    with _t.inference_mode():
        compiled(dummy)
        compiled(dummy)  # second call should hit the compiled cache
    return compiled


class CudaGraphedModel(torch.nn.Module):
    """Wrap a model so fixed-batch forwards replay a captured CUDA graph.

    Direct torch.cuda.CUDAGraph capture (no torch.compile / dynamo). The
    hypothesis is that bypassing dynamo gives a small additional win over
    `setup_model_for_compile(mode="reduce-overhead")`, which uses CUDAGraphs
    via dynamo and incurs some Python-level dispatch overhead per call.

    Constraints:
      - Must run on CUDA (no CPU/MPS support — that's the whole point).
      - Captures one graph at exactly `batch_size`. Smaller batches are padded
        with the last row (matching `_model_predict`'s pad_to_batch_size logic),
        larger batches fall back to the underlying model.
      - Inputs must be int64 with shape `(B, state_size)` on the same device.
      - Output is `static_output.clone()` so the caller can store/index without
        the graph clobbering the buffer on the next replay.
    """

    def __init__(self, model, batch_size: int = 16384, state_size: int | None = None):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CudaGraphedModel requires CUDA")
        self.model = model
        self.batch_size = batch_size
        self.state_size = state_size if state_size is not None else getattr(model, "state_size", 120)
        device = next(model.parameters()).device
        # Pre-allocate the static input buffer the captured graph reads from.
        self.static_input = torch.zeros(
            (self.batch_size, self.state_size), dtype=torch.int64, device=device
        )
        # Warm up on a side stream — required by the CUDAGraph API to ensure
        # all kernels are JIT'd and any one-shot allocations have happened.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.inference_mode():
                for _ in range(3):
                    _ = model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # Capture the forward pass.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.inference_mode():
                self.static_output = model(self.static_input)

    def forward(self, x):
        n = x.size(0)
        if n == self.batch_size:
            self.static_input.copy_(x)
            self.graph.replay()
            return self.static_output.clone()
        if n < self.batch_size:
            self.static_input[:n].copy_(x)
            # Pad with last row so unused outputs are well-defined (we slice them off).
            self.static_input[n:].copy_(x[-1:].expand(self.batch_size - n, -1))
            self.graph.replay()
            return self.static_output[:n].clone()
        # Larger than capture size: fall back to eager. Shouldn't happen if
        # _model_predict is chunking at internal_batch_size.
        return self.model(x)


def setup_model_for_cuda_graphs(model, batch_size: int = 16384):
    """Wrap the model in a CUDA graph capture. Returns the wrapped module.

    Call AFTER `setup_model_for_inference`. Pair with `pad_to_batch_size=True`
    so callers always submit full-batch tensors (the wrapper pads short batches
    too, but skipping the wrapper's pad path is one less copy).
    """
    return CudaGraphedModel(model, batch_size=batch_size)


def setup_model_for_inference(model) -> None:
    """Disable the model's internal chunking so the outer batch flows through in one call.

    Without this, calling `_model_predict` with internal_batch_size=16384 would have the
    model split each 16384-batch into two 8192-batches internally and concat them — pure
    waste. After this call, the outer chunking is the only level.
    """
    if hasattr(model, "inference_chunk_size"):
        model.inference_chunk_size = None
    base = getattr(model, "_orig_mod", model)
    if hasattr(base, "inference_chunk_size"):
        base.inference_chunk_size = None


class KhoruzhiiSolver:
    """Drop-in beam search solver. Use `solve()` to attempt one puzzle.

    The `internal_batch_size` controls per-chunk model + hash batch size — ~16k
    is a good default for 16 GB cards. Drop it if you OOM at large beam.
    """

    def __init__(
        self,
        puzzle,
        model,
        device: str = "cuda",
        internal_batch_size: int = 16384,
        random_seed: int = 0,
        state_dtype: torch.dtype = torch.int8,
        pad_to_batch_size: bool = False,
        profile: bool = True,
        macros: list[tuple[list[int] | tuple[int, ...], list[str]]] | None = None,
        policy_model=None,
        lambda_policy: float = 0.0,
    ):
        self.puzzle = puzzle
        self.model = model.to(device).eval()
        self.device = device
        self.internal_batch_size = internal_batch_size
        self.state_dtype = state_dtype
        # When True, model forward calls are padded to exactly `internal_batch_size`
        # — required for torch.compile to avoid recompile thrash on variable shapes.
        self.pad_to_batch_size = pad_to_batch_size
        # When False, the per-phase _sync_p() barriers in _do_greedy_step/solve become
        # no-ops. CPU can race ahead and queue many GPU kernels without stalling on
        # `cuda.synchronize()` calls. Per-phase timings (model_s, neighbor_s, etc.)
        # become 0/inaccurate — production runs with profile=False trade observability
        # for any CPU-stall reduction.
        self.profile = profile

        n_gen = len(puzzle.move_names)
        self.n_gen = n_gen
        self.state_size = len(puzzle.solved_state)
        # Generator perms first; macros (if any) appended after. Each macro is
        # (perm, word_list): perm is the macro's net 120-perm, word_list is the
        # generator names it expands to during path reconstruction. Macros add to
        # the action set as length-|word| atoms; cost-aware scoring (action_cost
        # below) penalizes macro candidates so a macro must deliver V-drop greater
        # than its extra cost (cost - 1) to win against primitives.
        gen_perms = torch.zeros((n_gen, self.state_size), dtype=torch.int64, device=device)
        for i, name in enumerate(puzzle.move_names):
            gen_perms[i] = torch.tensor(puzzle.generators[name], dtype=torch.int64)
        if macros:
            for perm, word in macros:
                assert len(perm) == self.state_size, "macro perm size mismatch"
                for nm in word:
                    assert nm in puzzle.generators, f"macro word has unknown name {nm}"
            self.n_macros = len(macros)
            macro_perms = torch.tensor(
                [list(p) for p, _ in macros], dtype=torch.int64, device=device
            )
            self.all_moves = torch.cat([gen_perms, macro_perms], dim=0)
            self.macro_words = [list(w) for _, w in macros]
            costs = [1] * n_gen + [len(w) for _, w in macros]
            self.action_cost = torch.tensor(costs, dtype=torch.float16, device=device)
        else:
            self.n_macros = 0
            self.all_moves = gen_perms
            self.macro_words = []
            self.action_cost = None
        self.n_actions = n_gen + self.n_macros
        self.move_names = puzzle.move_names
        self.V0 = torch.tensor(puzzle.solved_state, dtype=state_dtype, device=device)

        # Policy-guided scoring (Idea 4): score(child) = V(child) - lambda * log pi(a|parent).
        # Applied as an ADDITIVE penalty at top-B selection time. Lower score = better,
        # so -log pi (which is non-negative, large when pi is small) acts as a penalty
        # for unlikely actions. lambda=0 reduces to pure V scoring (current behavior).
        # Policy model output_dim must match n_gen; only valid for primitive actions.
        # In v0 we forbid policy + macros together (the policy was trained on n_gen
        # actions and doesn't know macros); pick one mechanism per solve.
        self.lambda_policy = float(lambda_policy)
        self.policy_model = None
        if policy_model is not None and lambda_policy > 0:
            if self.n_macros > 0:
                raise ValueError(
                    "policy_model + macros are not compatible in v0. The policy head "
                    "outputs n_gen logits; macros expand the action set beyond that. "
                    "Pick one mechanism per solve."
                )
            base = getattr(policy_model, "_orig_mod", policy_model)
            pdim = getattr(base, "output_dim", 1)
            if pdim != self.n_gen:
                raise ValueError(
                    f"policy_model output_dim={pdim} != n_gen={self.n_gen}; "
                    f"policy must output one logit per primitive generator."
                )
            self.policy_model = policy_model.to(device).eval()
            for p in self.policy_model.parameters():
                p.requires_grad = False

        # Bind the conditional sync used by _do_greedy_step/solve (profile gated).
        self._sync_p = _sync if profile else _noop_sync

        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)
        self.hash_vec = torch.randint(
            0, int(1e15), (self.state_size,), dtype=torch.int64, device=device, generator=gen
        )
        self._init_incremental_hash_tables()
        self.V0_hash = _state_hash(self.V0.unsqueeze(0), self.hash_vec, self.internal_batch_size)[0]

    def _init_incremental_hash_tables(self):
        # Incremental hash is only valid for primitive generators (small fan-out
        # of changed positions). Macros are full 120-element perms; their hash is
        # computed via full _state_hash on the materialized neighbor in
        # _get_neighbors_hashed_with_macros below.
        pos = torch.arange(self.state_size, dtype=torch.int64, device=self.device).expand(
            self.n_gen, self.state_size
        )
        gen_perms = self.all_moves[: self.n_gen]
        changed = gen_perms != pos
        max_changed = int(changed.sum(dim=1).max().item())
        self.changed_pos = torch.zeros((self.n_gen, max_changed), dtype=torch.int64, device=self.device)
        self.changed_src = torch.zeros((self.n_gen, max_changed), dtype=torch.int64, device=self.device)
        for move_idx in range(self.n_gen):
            move_pos = torch.nonzero(changed[move_idx], as_tuple=True)[0]
            n_changed = move_pos.numel()
            self.changed_pos[move_idx, :n_changed] = move_pos
            self.changed_src[move_idx, :n_changed] = gen_perms[move_idx, move_pos]
        self.changed_hash_vec = self.hash_vec[self.changed_pos]

    # ------------------------------------------------------------------ helpers

    def _get_neighbors(self, states):
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

    def _apply_move(self, states, moves):
        bs = self.internal_batch_size
        out = torch.empty(states.size(0), self.state_size, dtype=states.dtype, device=self.device)
        for i in range(0, states.size(0), bs):
            s = states[i : i + bs]
            m = moves[i : i + bs]
            out[i : i + bs] = torch.gather(s, 1, self.all_moves[m])
        return out

    def _get_neighbors_hashed(self, states, states_hashed):
        bs = self.internal_batch_size
        # Incremental hash for primitives.
        prim_out = torch.empty(states.size(0) * self.n_gen, dtype=torch.int64, device=self.device)
        for i in range(0, states.size(0), bs):
            chunk = states[i : i + bs]
            base_hash = states_hashed[i : i + chunk.size(0)]
            cur = chunk[:, self.changed_pos].to(torch.int64)
            src = chunk[:, self.changed_src].to(torch.int64)
            delta = ((src - cur) * self.changed_hash_vec).sum(dim=2)
            prim_out[i * self.n_gen : (i + chunk.size(0)) * self.n_gen] = (
                base_hash[:, None] + delta
            ).reshape(-1)
        if self.n_macros == 0:
            return prim_out
        # Macros: compute full neighbor and hash. For each parent, this gives
        # n_macros child hashes via materialized states. Cost is O(n*n_macros*S)
        # but n_macros is typically small (45) so still much cheaper than the
        # primitive incremental shortcut wins back.
        n = states.size(0)
        macro_out = torch.empty(n * self.n_macros, dtype=torch.int64, device=self.device)
        macro_perms = self.all_moves[self.n_gen :]  # (n_macros, S)
        for i in range(0, n, bs):
            chunk = states[i : i + bs]
            chunk_n = chunk.size(0)
            children = torch.gather(
                chunk.unsqueeze(1).expand(chunk_n, self.n_macros, self.state_size),
                2,
                macro_perms.unsqueeze(0).expand(chunk_n, self.n_macros, self.state_size),
            ).flatten(end_dim=1)
            macro_out[i * self.n_macros : (i + chunk_n) * self.n_macros] = _state_hash(
                children, self.hash_vec, bs
            )
        # Interleave to match the flat (n * n_actions) layout: for each parent,
        # n_gen primitive hashes followed by n_macros macro hashes.
        prim_view = prim_out.view(n, self.n_gen)
        macro_view = macro_out.view(n, self.n_macros)
        return torch.cat([prim_view, macro_view], dim=1).reshape(-1)

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

    # ----------------------------------------------------------------- step

    def _do_greedy_step(
        self,
        states,
        states_hashed,
        states_bad_hashed,
        B,
        prof: Profile,
        use_incremental_hash=True,
        reuse_full_neighbors=False,
        temperature: float = 0.0,
    ):
        """One beam-expansion step.

        Returns (next_states, values, move_indices, parent_indices, chosen_hashes).
        The trailing `chosen_hashes` lets callers reuse them for the stagnation log
        instead of rehashing.

        `temperature > 0` switches the top-B selection from greedy (lowest values)
        to sampling-without-replacement via the Gumbel-top-k trick:
        scores = -value/T + Gumbel(0,1); take top-B by scores. T → 0 collapses
        to greedy; larger T mixes in random exploration.
        """
        n = states.size(0)
        bs = self.internal_batch_size

        # --- neighbors + hash
        self._sync_p(); t0 = time.time()
        neighbors_flat = None
        if use_incremental_hash:
            neighbors_hashed = self._get_neighbors_hashed(states, states_hashed)
        elif reuse_full_neighbors:
            neighbors_flat = self._get_neighbors(states).flatten(end_dim=1)
            neighbors_hashed = _state_hash(neighbors_flat, self.hash_vec, bs)
        else:
            neighbors_hashed = torch.empty(n * self.n_actions, dtype=torch.int64, device=self.device)
            for i in range(0, n, bs):
                chunk_states = states[i : i + bs]
                chunk_neighbors = self._get_neighbors(chunk_states).flatten(end_dim=1)
                neighbors_hashed[i * self.n_actions : (i + chunk_states.size(0)) * self.n_actions] = _state_hash(
                    chunk_neighbors, self.hash_vec, bs
                )
        self._sync_p(); prof.neighbor_s += time.time() - t0

        # --- dedup
        self._sync_p(); t0 = time.time()
        idx1 = self._unique_hashed_idx(neighbors_hashed, states_bad_hashed)
        self._sync_p(); prof.dedup_s += time.time() - t0

        if idx1.numel() == 0:
            empty_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_v = torch.empty(0, dtype=torch.float16, device=self.device)
            empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty_s, empty_v, empty_i, empty_i, empty_v

        # Implicit (parent_idx, action_idx) decoding from the flat neighbor index.
        # idx1 references positions in a (n * n_actions) flat layout where row r has
        # parent = r // n_actions and action = r % n_actions. Actions 0..n_gen-1 are
        # primitive generators; n_gen..n_actions-1 are macros (when present).
        parent_of_idx1 = idx1 // self.n_actions
        move_of_idx1 = idx1 % self.n_actions

        # --- materialize candidate states + score with model
        self._sync_p(); t0 = time.time()
        if neighbors_flat is None:
            candidate_states = self._apply_move(states[parent_of_idx1], move_of_idx1)
        else:
            candidate_states = neighbors_flat[idx1]
        self._sync_p(); prof.apply_s += time.time() - t0

        self._sync_p(); t0 = time.time()
        value = _model_predict(
            self.model, candidate_states, bs, pad_to_batch_size=self.pad_to_batch_size
        )
        self._sync_p(); prof.model_s += time.time() - t0

        # --- top-B (torch.topk replaces full argsort: O(N log B) vs O(N log N))
        # Cost-aware adjustment for macros: a macro costing L real moves must
        # deliver V-drop > L-1 to win against a primitive. Apply (cost - 1) penalty.
        if self.action_cost is not None:
            cost_penalty = self.action_cost[move_of_idx1] - 1.0  # 0 for gens, >0 for macros
            score = value + cost_penalty
        else:
            score = value
        # Policy-guided adjustment (Idea 4): subtract lambda * log pi(a|parent),
        # which is +lambda * (-log pi). One policy forward per beam step on the
        # current parents (n forwards total), then index by (parent, action) for
        # each candidate. Macros are forbidden in policy mode (v0), so action
        # indices are guaranteed to be < n_gen.
        if self.policy_model is not None and self.lambda_policy > 0:
            self._sync_p(); t_pol = time.time()
            with torch.inference_mode():
                pol_out = self.policy_model(states.long())  # (n, n_gen) logits
            log_pi = torch.log_softmax(pol_out.float(), dim=-1)  # (n, n_gen)
            log_pi_per_cand = log_pi[parent_of_idx1, move_of_idx1].to(score.dtype)
            score = score + self.lambda_policy * (-log_pi_per_cand)
            self._sync_p(); prof.model_s += time.time() - t_pol
        self._sync_p(); t0 = time.time()
        # `largest=False` because we want lowest predicted distances. `sorted=False` —
        # we don't care about internal ordering of the chosen B for downstream ops.
        if score.numel() <= B:
            chosen_local = torch.arange(score.numel(), device=self.device)
        elif temperature > 0:
            # Gumbel-top-k: sample B without replacement from softmax(-score/T).
            # Cast to float32 for stable log/exp; clamp uniform sample away from 0.
            sc32 = score.to(torch.float32)
            u = torch.rand_like(sc32).clamp_(min=1e-20)
            gumbel = -torch.log(-torch.log(u))
            gscores = gumbel - sc32 / float(temperature)
            _, chosen_local = torch.topk(gscores, B, largest=True, sorted=False)
        else:
            _, chosen_local = torch.topk(score, B, largest=False, sorted=False)
        chosen_idx1 = idx1[chosen_local]
        next_states = candidate_states[chosen_local]
        next_values = value[chosen_local]
        chosen_moves = move_of_idx1[chosen_local]
        chosen_parents = parent_of_idx1[chosen_local]
        # Reuse already-computed neighbor hashes for the stagnation log (idea #6).
        chosen_hashes = neighbors_hashed[chosen_idx1]
        self._sync_p(); prof.topk_s += time.time() - t0

        return next_states, next_values, chosen_moves, chosen_parents, chosen_hashes

    def _find_solved_pos(self, states, states_hashed):
        hit_positions = torch.nonzero(states_hashed == self.V0_hash, as_tuple=True)[0]
        if hit_positions.numel() == 0:
            return None
        exact = (states[hit_positions] == self.V0).all(dim=1)
        if not exact.any():
            return None
        return int(hit_positions[torch.nonzero(exact, as_tuple=True)[0][0]].item())

    # ---------------------------------------------------------- solve loop

    def _check_stagnation(self, states_hash_log):
        if len(states_hash_log) < 4:
            return False
        recent = torch.cat(list(states_hash_log)[2:])
        earlier = torch.cat(list(states_hash_log)[:2])
        return bool(torch.isin(recent, earlier).all().item())

    def solve(self, initial_state, cfg: KhoruzhiiSearchConfig):
        prof = Profile()
        self._sync_p(); t_start = time.time()
        B = cfg.beam_width
        num_steps = cfg.num_steps
        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)
        if torch.equal(state, self.V0):
            prof.total_s = time.time() - t_start
            prof.found = True
            return True, 0, [], prof

        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        initial_hashed = _state_hash(state.unsqueeze(0), self.hash_vec, self.internal_batch_size)
        final_states = None
        final_j = None
        final_pos = None

        for attempt in range(cfg.num_attempts):
            states = state.unsqueeze(0).clone()
            states_hashed = initial_hashed.clone()
            # Tree backpointers on GPU — int8 for moves (24 << 127), int32 for parent
            # indices (B can be > 2^15). Avoids one CUDA sync per step from .cpu() copies.
            tree_move = torch.full((num_steps, B), -1, dtype=torch.int8, device=self.device)
            tree_idx = torch.full((num_steps, B), -1, dtype=torch.int32, device=self.device)
            states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
            reached = None

            for j in range(num_steps):
                if cfg.beam_decay != 1.0:
                    step_beam = max(cfg.min_beam_width, int(round(B * (cfg.beam_decay ** j))))
                else:
                    step_beam = B
                states, _values, moves, parents, chosen_hashes = self._do_greedy_step(
                    states,
                    states_hashed,
                    states_bad_hashed,
                    step_beam,
                    prof,
                    cfg.use_incremental_hash,
                    cfg.reuse_full_neighbors,
                    temperature=cfg.temperature,
                )
                if states.numel() == 0:
                    break
                # Stagnation log uses the already-computed chosen hashes (idea #6).
                self._sync_p(); t0 = time.time()
                states_hash_log.append(chosen_hashes)
                self._sync_p(); prof.hash_s += time.time() - t0

                leaves = states.size(0)
                tree_move[j, :leaves] = moves.to(torch.int8)
                tree_idx[j, :leaves] = parents.to(torch.int32)
                states_hashed = chosen_hashes
                prof.per_step_states.append(int(leaves))
                prof.n_steps = j + 1

                # Deferred solved-check: skip the (CPU-syncing) check on intermediate steps.
                # Always check on the last step. solved_check_every=1 = current behavior.
                check_due = (cfg.solved_check_every <= 1) or (
                    (j + 1) % cfg.solved_check_every == 0
                ) or (j + 1 == num_steps)
                if check_due:
                    solved_pos = self._find_solved_pos(states, chosen_hashes)
                    if solved_pos is not None:
                        reached = j
                        final_pos = solved_pos
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

        # Path reconstruction: copy the relevant trace off GPU just once, at the end.
        tree_idx_cpu = tree_idx[: final_j + 1].flip((0,)).cpu().numpy()
        tree_move_cpu = tree_move[: final_j + 1].flip((0,)).cpu().numpy()
        v0_pos = final_pos
        path = [int(tree_idx_cpu[0, v0_pos])]
        for k in range(1, final_j + 1):
            path.append(int(tree_idx_cpu[k, path[-1]]))
        moves_seq = [
            int(tree_move_cpu[k, path[k - 1]] if k > 0 else tree_move_cpu[k, v0_pos])
            for k in range(final_j + 1)
        ]
        moves_seq.reverse()
        # Expand macro action indices to their generator words. Primitive actions
        # (m < n_gen) emit a single generator name; macro actions (m >= n_gen) emit
        # macro_words[m - n_gen] in order.
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
