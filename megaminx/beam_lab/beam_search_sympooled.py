"""Sym-pooled beam search — research direction 3C (own_research_directions_2026-06-12.md).

Production `--sym-ensemble K` runs K INDEPENDENT beam searches of width B, one
per rotated copy of the scramble, and keeps the shortest translated path.
Width is statically partitioned: every rotation gets exactly B slots no matter
how its subtree is doing.

`SymPooledSolver.solve_pooled` runs ONE beam of total width K*B seeded with
all K rotated copies at once, reallocating width across rotation frames every
step. Equal-wall vs sequential sym-K: both score ~K*B*L candidates over an
L-step solve.

Allocation design (v1, after a measured v0 failure):
  v0 ranked ALL candidates from all roots in one global top-(K*B) by raw V.
  That assumes V is comparable across rotation frames — it is NOT: the model
  is deliberately non-invariant across rotations (that is what makes the
  sym-ensemble diverse), so per-frame bias plus near-solved V noise made the
  pool defect from the leading root late in the run (occupancy flipped
  [3072,1024] -> [1024,3072] near the finish on the smoke pid), starving the
  leader below its sequential width exactly when it mattered. v0 solved
  nothing where seq solved easily.

  v1 never compares V across frames:
    level 1 (allocation): per-root widths w_r = floor + swing, swing split by
      softmax(progress_r / tau), where progress_r = EMA of (minV_r(0) -
      minV_r(t)) — each root's V-descent since ITS OWN start. Baseline
      subtraction cancels per-frame bias; EMA smooths noise; a root nearing
      solved has maximal progress, so the leader GAINS width at the finish by
      construction.
    level 2 (selection): within each root, plain top-w_r by V — exactly the
      sequential ranking, just with a dynamic width.
  `alloc="global"` keeps the v0 behavior for comparison.

Frame semantics live entirely in the CALLER: this class receives a list of
root states (already rotated / inverted copies of one scramble) and returns
the raw path plus the index of the root the solution descends from; the
caller translates moves back to the original frame. Duplicate roots (e.g.
superflip, fixed by every rotation) are merged up front and reported under
the first occurrence's id. Dedup is global across roots — identical states
have identical futures and the walkback translates through whichever root's
chain survived.

v0 scope: plain V scoring only — no macros, no policy/PHS (use the production
sequential path for those).
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
)


@dataclass
class PooledProfile(Profile):
    # Per-step count of beam slots owned by each root — the scientific readout:
    # how the pool reallocates width over depth.
    per_step_root_counts: list[list[int]] = field(default_factory=list)
    # Per-step per-root min candidate V (float) — progress trace for analysis.
    per_step_root_minv: list[list[float]] = field(default_factory=list)
    # Index (into the CALLER's root list) of the root the solution came from.
    root_id: int = -1
    n_roots_effective: int = 0


def _alloc_widths(
    B_total: int,
    n_roots: int,
    n_cand_per_root: list[int],
    progress: list[float],
    floor_frac: float,
    tau: float,
) -> list[int]:
    """Split B_total slots across roots: floor share + progress-softmax swing.

    Dead roots (no candidates) get 0. Widths are capped by each root's
    available candidate count, with deficit redistributed to roots that still
    have headroom (waterfall, repeated to a fixpoint). Returns integer widths
    summing to min(B_total, total candidates).
    """
    import math

    alive = [r for r in range(n_roots) if n_cand_per_root[r] > 0]
    if not alive:
        return [0] * n_roots
    total_cand = sum(n_cand_per_root)
    budget = min(B_total, total_cand)

    floor = int(floor_frac * B_total / n_roots) if len(alive) > 1 else 0
    swing = budget - floor * len(alive)
    if swing < 0:  # floor over-subscribed (tiny budget): pure proportional floor
        floor = budget // len(alive)
        swing = budget - floor * len(alive)

    mx = max(progress[r] for r in alive)
    expw = {r: math.exp((progress[r] - mx) / max(tau, 1e-6)) for r in alive}
    z = sum(expw.values())
    target = {r: floor + swing * expw[r] / z for r in alive}

    # Integerize then waterfall-redistribute over candidate-count caps.
    w = {r: min(int(target[r]), n_cand_per_root[r]) for r in alive}
    for _ in range(n_roots + 2):
        deficit = budget - sum(w.values())
        if deficit <= 0:
            break
        headroom = [r for r in alive if w[r] < n_cand_per_root[r]]
        if not headroom:
            break
        # Give to highest-progress roots first, one pass.
        headroom.sort(key=lambda r: -progress[r])
        for r in headroom:
            add = min(deficit, n_cand_per_root[r] - w[r])
            take = max(1, add if len(headroom) == 1 else deficit // len(headroom))
            take = min(take, add)
            w[r] += take
            deficit -= take
            if deficit <= 0:
                break
    return [w.get(r, 0) for r in range(n_roots)]


class SymPooledSolver(KhoruzhiiSolver):
    """Multi-root pooled beam. See module docstring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_macros > 0:
            raise ValueError("SymPooledSolver v0 does not support macros")
        if self.policy_model is not None or self.phs_cumulative:
            raise ValueError("SymPooledSolver v0 does not support policy/PHS scoring")

    # ----------------------------------------------------------------- step

    def _do_pooled_step(
        self,
        states,
        states_hashed,
        root_of,
        n_roots,
        states_bad_hashed,
        widths: list[int] | None,
        B_total: int,
        prof: Profile,
        use_incremental_hash=True,
        temperature: float = 0.0,
    ):
        """One pooled beam-expansion step.

        `widths`: per-root slot budget (level-2 selection: within-root top-w_r
        by V; V is never compared across roots). `widths=None` falls back to
        v0 global top-B_total (alloc="global" mode / single root).

        Returns (next_states, values, moves, parents, chosen_hashes, next_root,
        minv_per_root, n_cand_per_root).
        """
        n = states.size(0)
        bs = self.internal_batch_size

        # --- neighbors + hash (same paths as the parent step)
        self._sync_p(); t0 = time.time()
        if use_incremental_hash:
            neighbors_hashed = self._get_neighbors_hashed(states, states_hashed)
        else:
            neighbors_hashed = torch.empty(n * self.n_actions, dtype=torch.int64, device=self.device)
            for i in range(0, n, bs):
                chunk_states = states[i : i + bs]
                chunk_neighbors = self._get_neighbors(chunk_states).flatten(end_dim=1)
                neighbors_hashed[i * self.n_actions : (i + chunk_states.size(0)) * self.n_actions] = _state_hash(
                    chunk_neighbors, self.hash_vec, bs
                )
        self._sync_p(); prof.neighbor_s += time.time() - t0

        # --- dedup (global across roots)
        self._sync_p(); t0 = time.time()
        idx1 = self._unique_hashed_idx(neighbors_hashed, states_bad_hashed)
        self._sync_p(); prof.dedup_s += time.time() - t0

        empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
        if idx1.numel() == 0:
            empty_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_v = torch.empty(0, dtype=torch.float16, device=self.device)
            return empty_s, empty_v, empty_i, empty_i, empty_v, empty_i, [], []

        parent_of_idx1 = idx1 // self.n_actions
        move_of_idx1 = idx1 % self.n_actions

        # --- materialize + score
        self._sync_p(); t0 = time.time()
        candidate_states = self._apply_move(states[parent_of_idx1], move_of_idx1)
        self._sync_p(); prof.apply_s += time.time() - t0

        self._sync_p(); t0 = time.time()
        value = _model_predict(
            self.model, candidate_states, bs, pad_to_batch_size=self.pad_to_batch_size
        )
        self._sync_p(); prof.model_s += time.time() - t0

        cand_root = root_of[parent_of_idx1]

        # --- selection
        self._sync_p(); t0 = time.time()
        n_cand = value.numel()
        sel_key = value.to(torch.float32)
        if temperature > 0:
            u = torch.rand_like(sel_key).clamp_(min=1e-20)
            gumbel = -torch.log(-torch.log(u))
            sel_key = sel_key / float(temperature) - gumbel

        minv_per_root: list[float] = []
        n_cand_per_root: list[int] = []
        if widths is None or n_roots == 1:
            # v0 / single-root: global top-B_total. Bit-compatible with the
            # parent's greedy step when n_roots == 1.
            if n_cand <= B_total:
                chosen_local = torch.arange(n_cand, device=self.device)
            else:
                _, chosen_local = torch.topk(sel_key, B_total, largest=False, sorted=False)
            if n_roots == 1:
                minv_per_root = [float(value.min().item())]
                n_cand_per_root = [n_cand]
        else:
            picks = []
            for r in range(n_roots):
                ridx = torch.nonzero(cand_root == r, as_tuple=True)[0]
                n_r = int(ridx.numel())
                n_cand_per_root.append(n_r)
                if n_r == 0:
                    minv_per_root.append(float("inf"))
                    continue
                minv_per_root.append(float(value[ridx].min().item()))
                w_r = min(widths[r], n_r)
                if w_r <= 0:
                    continue
                if w_r >= n_r:
                    picks.append(ridx)
                else:
                    _, loc = torch.topk(sel_key[ridx], w_r, largest=False, sorted=False)
                    picks.append(ridx[loc])
            chosen_local = torch.cat(picks) if picks else empty_i

        chosen_idx1 = idx1[chosen_local]
        next_states = candidate_states[chosen_local]
        next_values = value[chosen_local]
        chosen_moves = move_of_idx1[chosen_local]
        chosen_parents = parent_of_idx1[chosen_local]
        chosen_hashes = neighbors_hashed[chosen_idx1]
        next_root = cand_root[chosen_local]
        self._sync_p(); prof.topk_s += time.time() - t0

        return (next_states, next_values, chosen_moves, chosen_parents,
                chosen_hashes, next_root, minv_per_root, n_cand_per_root)

    # ---------------------------------------------------------- solve loop

    def solve_pooled(
        self,
        root_states,
        cfg: KhoruzhiiSearchConfig,
        floor_frac: float = 0.5,
        alloc: str = "progress",
        tau: float = 1.0,
        ema: float = 0.7,
    ):
        """Pooled beam search from multiple roots.

        `root_states`: list of state sequences (rotated and/or inverted copies
        of one scramble). `cfg.beam_width` is the TOTAL pooled width (use K*B
        to equal-wall a sequential sym-K at width B).

        `alloc`: "progress" (v1 default — per-root widths from the
        bias-cancelled progress softmax; see module docstring) or "global"
        (v0 — raw global top-B across frames; kept for comparison).
        `floor_frac`: guaranteed per-root share of the equal split.
        `tau`: softmax temperature on progress, in V units (moves).
        `ema`: smoothing of the progress signal, 0 = no memory.

        Returns (found, path_len, move_names, root_id, prof). `move_names`
        solves `root_states[root_id]`; the caller translates to the original
        frame. `root_id` indexes the ORIGINAL root list even when duplicate
        roots were merged.
        """
        prof = PooledProfile()
        self._sync_p(); t_start = time.time()
        B_total = cfg.beam_width
        num_steps = cfg.num_steps

        # Merge duplicate roots (superflip-like states are fixed by many rotations).
        seen: dict[tuple, int] = {}
        kept_states: list[tuple] = []
        kept_ids: list[int] = []
        for i, st in enumerate(root_states):
            key = tuple(int(x) for x in st)
            if key in seen:
                continue
            seen[key] = i
            kept_states.append(key)
            kept_ids.append(i)
        n_roots = len(kept_states)
        prof.n_roots_effective = n_roots

        solved_key = tuple(int(x) for x in self.puzzle.solved_state)
        for st, rid in zip(kept_states, kept_ids):
            if st == solved_key:
                prof.total_s = time.time() - t_start
                prof.found = True
                prof.root_id = rid
                return True, 0, [], rid, prof

        roots_t = torch.tensor(kept_states, dtype=self.state_dtype, device=self.device)
        roots_hashed = _state_hash(roots_t, self.hash_vec, self.internal_batch_size)

        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=self.device)
        final_j = None
        final_pos = None
        tree_move = None
        tree_idx = None
        found_any = False

        for attempt in range(cfg.num_attempts):
            states = roots_t.clone()
            states_hashed = roots_hashed.clone()
            root_of = torch.arange(n_roots, dtype=torch.int64, device=self.device)
            tree_move = torch.full((num_steps, B_total), -1, dtype=torch.int8, device=self.device)
            tree_idx = torch.full((num_steps, B_total), -1, dtype=torch.int32, device=self.device)
            states_hash_log: deque[torch.Tensor] = deque(maxlen=4)
            occupancy_gpu: list[torch.Tensor] = []
            reached = None

            # Progress-allocation state.
            base_minv: list[float] | None = None  # per-root V baseline (step 0)
            prog_ema: list[float] = [0.0] * n_roots
            dead: list[bool] = [False] * n_roots

            for j in range(num_steps):
                if cfg.beam_decay != 1.0:
                    step_beam = max(cfg.min_beam_width, int(round(B_total * (cfg.beam_decay ** j))))
                else:
                    step_beam = B_total

                widths: list[int] | None = None
                if alloc == "progress" and n_roots > 1:
                    if base_minv is None:
                        # First expansion: equal split (no progress info yet).
                        widths = [step_beam // n_roots] * n_roots
                    else:
                        progress = [
                            (-1e9 if dead[r] else prog_ema[r]) for r in range(n_roots)
                        ]
                        n_avail = [0 if dead[r] else 10**9 for r in range(n_roots)]
                        widths = _alloc_widths(
                            step_beam, n_roots, n_avail, progress, floor_frac, tau,
                        )

                (states, _values, moves, parents, chosen_hashes, root_of,
                 minv_per_root, n_cand_per_root) = self._do_pooled_step(
                    states,
                    states_hashed,
                    root_of,
                    n_roots,
                    states_bad_hashed,
                    widths,
                    step_beam,
                    prof,
                    cfg.use_incremental_hash,
                    temperature=cfg.temperature,
                )
                if states.numel() == 0:
                    break

                # Update progress signal from this step's candidate min-V.
                if alloc == "progress" and n_roots > 1 and minv_per_root:
                    if base_minv is None:
                        base_minv = list(minv_per_root)
                    for r in range(n_roots):
                        if n_cand_per_root[r] == 0:
                            dead[r] = True
                            continue
                        if minv_per_root[r] == float("inf"):
                            continue
                        p_now = base_minv[r] - minv_per_root[r]
                        prog_ema[r] = ema * prog_ema[r] + (1.0 - ema) * p_now
                    if self.profile:
                        prof.per_step_root_minv.append(
                            [round(v, 3) if v != float("inf") else None for v in minv_per_root]
                        )

                self._sync_p(); t0 = time.time()
                states_hash_log.append(chosen_hashes)
                self._sync_p(); prof.hash_s += time.time() - t0

                leaves = states.size(0)
                tree_move[j, :leaves] = moves.to(torch.int8)
                tree_idx[j, :leaves] = parents.to(torch.int32)
                states_hashed = chosen_hashes
                prof.per_step_states.append(int(leaves))
                prof.n_steps = j + 1
                if self.profile:
                    occupancy_gpu.append(torch.bincount(root_of, minlength=n_roots))

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

            if self.profile and occupancy_gpu:
                prof.per_step_root_counts = [
                    [int(x) for x in row] for row in torch.stack(occupancy_gpu).cpu().tolist()
                ]
            if reached is not None:
                found_any = True
                final_j = reached
                break

        if not found_any or final_j is None:
            prof.total_s = time.time() - t_start
            prof.found = False
            return False, 0, [], -1, prof

        # Path reconstruction — identical to the parent walkback. After the
        # flip, row k corresponds to step final_j - k, so the last appended
        # parent index lands in step 0's frontier = the root list.
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
        names = [self.move_names[m] for m in moves_seq]

        root_local = path[-1]
        root_id = kept_ids[root_local]

        prof.total_s = time.time() - t_start
        prof.found = True
        prof.path_len = len(names)
        prof.root_id = root_id
        return True, len(names), names, root_id, prof
