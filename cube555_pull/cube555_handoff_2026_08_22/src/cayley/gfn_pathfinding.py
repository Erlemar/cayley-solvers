"""GFN-pathfinding: PyTorch port of "Learning Shortest Paths with Generative Flow
Networks" (Morozov, Maksimov, Tiapkin, Samsonov 2026, arXiv:2603.01786).

Construction (paper section 3.2):
  - GFlowNet source s0 = the SOLVED state; forward policy P_F scrambles away from
    solved; backward policy P_B is the solver.
  - Every state is terminal (stop edge to sink), reward R(s) = 1 uniformly, so the
    partition function Z = |group| is KNOWN exactly (no learned logZ).
  - Theorem 3.4: minimizing total flow makes P_B traverse only geodesics. Practical
    loss = trajectory balance over all prefixes + flow regularization lambda*F(s),
    with F(s) = 1/P_F(stop|s) (no separate flow net).
  - Inference: beam search on cumulative log P_B (policy beam, one forward pass
    scores all children), greedy = W=1.

Conventions (match cayley/megaminx repo):
  - state: length-S int vector; apply(state, gen)[i] = state[gen[i]].
  - env.gens: (A, S) index-map permutations, closed under inverse via env.inv_idx
    (gens[inv_idx[a]] composed after gens[a] = identity).
  - Backward-head logit j at state x  <=>  the solving move "apply gens[j] to x".
    (Derivation: the parent of s_{t+1} reached by fwd action a is
    apply(s_{t+1}, gens[inv_idx[a]]), so indexing the bwd head at inv_idx[a] makes
    bwd index j coincide with gen index j.)

Numeric notes (megaminx-scale):
  - lnZ ~ 156.6 -> near-root flows F ~ Z/A ~ 4e66 overflow float32 exp. reg_mode
    "flow" computes the regularizer in float64 (paper-faithful; pick lambda so that
    lambda*F(d1) lands ~1e10-1e12, the paper's rubik3 operating point). reg_mode
    "logflow" penalizes logsumexp(log F) instead (bounded gradients, same argmin
    direction, different geometry).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cayley.model import ResBlock

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class GFNEnv:
    """Cayley-graph GFN environment (paper construction, section 3.2)."""

    gens: torch.Tensor  # (A, S) long, index-map perms
    inv_idx: torch.Tensor  # (A,) long, gens[inv_idx[a]] = inverse of gens[a]
    solved: torch.Tensor  # (S,) long (colored or permutation state)
    num_classes: int
    true_log_z: float
    s0_preimages: torch.Tensor  # (A, S) long; fwd action a from this state returns to solved

    @property
    def n_actions(self) -> int:
        return int(self.gens.shape[0])

    @property
    def state_size(self) -> int:
        return int(self.gens.shape[1])

    def to(self, device: torch.device | str) -> "GFNEnv":
        return GFNEnv(
            gens=self.gens.to(device),
            inv_idx=self.inv_idx.to(device),
            solved=self.solved.to(device),
            num_classes=self.num_classes,
            true_log_z=self.true_log_z,
            s0_preimages=self.s0_preimages.to(device),
        )


def apply_gens(states: torch.Tensor, gen_rows: torch.Tensor) -> torch.Tensor:
    """apply(state, gen) row-wise. states (B, S), gen_rows (B, S) long -> (B, S)."""
    return torch.gather(states, 1, gen_rows)


def build_env(
    gens_list: list[list[int]],
    inv_idx_list: list[int],
    solved_list: list[int],
    num_classes: int,
    true_log_z: float,
) -> GFNEnv:
    gens = torch.tensor(gens_list, dtype=torch.long)
    inv_idx = torch.tensor(inv_idx_list, dtype=torch.long)
    solved = torch.tensor(solved_list, dtype=torch.long)
    a, s = gens.shape

    # Verify inverse pairing on the permutation level (arange is injective).
    arange = torch.arange(s, dtype=torch.long)
    for j in range(a):
        fwd = apply_gens(arange.unsqueeze(0), gens[j].unsqueeze(0))
        back = apply_gens(fwd, gens[inv_idx[j]].unsqueeze(0))
        if not torch.equal(back.squeeze(0), arange):
            raise ValueError(f"inv_idx wrong for generator {j}")

    # s0_preimages[a] = apply(solved, gens[inv_idx[a]]): the unique state from which
    # forward action a transitions INTO the solved state (masked during P_F, since
    # the construction removes forward edges into s0).
    pre = apply_gens(solved.unsqueeze(0).expand(a, s), gens[inv_idx])
    # sanity: applying fwd action a to its preimage must give solved
    check = apply_gens(pre, gens)
    if not torch.equal(check, solved.unsqueeze(0).expand(a, s)):
        raise ValueError("s0_preimages construction failed self-check")

    return GFNEnv(
        gens=gens,
        inv_idx=inv_idx,
        solved=solved,
        num_classes=num_classes,
        true_log_z=true_log_z,
        s0_preimages=pre,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GFNPolicyNet(nn.Module):
    """ResMLP trunk with a single output layer split into backward-policy logits (A)
    and forward-policy logits (A + 1, last = stop). Mirrors the paper's architecture
    shape; the trunk reuses this repo's proven ResBlock (two-linear residual)."""

    def __init__(
        self,
        state_size: int,
        num_classes: int,
        n_actions: int,
        hidden: int = 1024,
        num_res_blocks: int = 3,
        encoding: str = "embedding",
        embed_dim: int = 16,
    ):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.n_actions = n_actions
        self.encoding = encoding
        if encoding == "onehot":
            in_dim = state_size * num_classes
            self.embedding = None
        elif encoding == "embedding":
            in_dim = state_size * embed_dim
            self.embedding = nn.Embedding(num_classes, embed_dim)
        else:
            raise ValueError(f"unknown encoding {encoding!r}")
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(num_res_blocks)])
        self.out = nn.Linear(hidden, n_actions + n_actions + 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding == "onehot":
            oh = F.one_hot(x.long(), num_classes=self.num_classes)
            return oh.to(self.inp.weight.dtype).flatten(start_dim=-2)
        return self.embedding(x.long()).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.inp(self.encode(x)))
        for blk in self.blocks:
            h = blk(h)
        out = self.out(h)
        bwd_logits = out[..., : self.n_actions]
        fwd_logits = out[..., self.n_actions :]
        return bwd_logits, fwd_logits

    @torch.no_grad()
    def forward_chunked(
        self, x: torch.Tensor, chunk: int = 16384
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs, fs = [], []
        for i in range(0, x.shape[0], chunk):
            b, f = self.forward(x[i : i + chunk])
            bs.append(b)
            fs.append(f)
        return torch.cat(bs, 0), torch.cat(fs, 0)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Forward-policy masking
# ---------------------------------------------------------------------------


def s0_entry_mask(states: torch.Tensor, env: GFNEnv) -> torch.Tensor:
    """(N, A) bool: True where fwd action a from states[n] would enter the solved
    state (must be masked; the construction removes forward edges into s0)."""
    # states (N, S) vs preimages (A, S) -> (N, A)
    return (states.unsqueeze(1) == env.s0_preimages.unsqueeze(0)).all(dim=-1)


def masked_fwd_log_softmax(fwd_logits: torch.Tensor, mask_a: torch.Tensor) -> torch.Tensor:
    """log_softmax over A+1 fwd logits with the (N, A) action mask applied.
    Stop column (last) is never masked."""
    full_mask = torch.cat(
        [mask_a, torch.zeros(mask_a.shape[0], 1, dtype=torch.bool, device=mask_a.device)],
        dim=1,
    )
    return torch.log_softmax(fwd_logits.masked_fill(full_mask, NEG_INF), dim=-1)


# ---------------------------------------------------------------------------
# On-policy trajectory sampling (paper Algorithm 1, step 3)
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_forward_trajectories(
    model: GFNPolicyNet,
    env: GFNEnv,
    batch_size: int,
    nmax: int,
    eps_explore: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (states (T+1, B, S), actions (T, B)) on-policy from P_F starting at
    solved, stop action masked (fixed-length partial trajectories)."""
    device = env.gens.device
    s = env.solved.unsqueeze(0).expand(batch_size, env.state_size).contiguous()
    states = [s]
    actions = []
    for _ in range(nmax):
        _, fwd_logits = model(s)
        logits = fwd_logits[:, : env.n_actions].clone()  # drop stop column: masked during sampling
        if eps_explore > 0.0:
            explore = torch.rand(batch_size, device=device) < eps_explore
            logits[explore] = 0.0
        logits = logits.masked_fill(s0_entry_mask(s, env), NEG_INF)
        a = torch.distributions.Categorical(logits=logits).sample()
        s = apply_gens(s, env.gens[a])
        states.append(s)
        actions.append(a)
    return torch.stack(states, 0), torch.stack(actions, 0)


# ---------------------------------------------------------------------------
# Loss: prefix trajectory balance + flow regularization (paper eq. 8)
# ---------------------------------------------------------------------------


def regularized_tb_loss(
    model: GFNPolicyNet,
    env: GFNEnv,
    states: torch.Tensor,  # (T+1, B, S)
    actions: torch.Tensor,  # (T, B)
    reg_coef: float,
    reg_mode: str = "flow",
    log_z: torch.Tensor | None = None,  # learnable override; None = env.true_log_z fixed
) -> tuple[torch.Tensor, dict]:
    tp1, b, s = states.shape
    t = tp1 - 1
    flat = states.reshape(tp1 * b, s)

    bwd_logits, fwd_logits = model(flat)
    log_pb = torch.log_softmax(bwd_logits, dim=-1).reshape(tp1, b, -1)
    log_pf = masked_fwd_log_softmax(fwd_logits, s0_entry_mask(flat, env)).reshape(tp1, b, -1)
    log_flows = -log_pf[..., -1]  # (T+1, B): log F(s) = -log P_F(stop|s)

    # taken forward action log-probs at s_t; taken backward at s_{t+1}, index inv_idx[a]
    log_pf_taken = torch.gather(log_pf[:-1], 2, actions.unsqueeze(-1)).squeeze(-1)  # (T, B)
    log_pb_taken = torch.gather(
        log_pb[1:], 2, env.inv_idx[actions].unsqueeze(-1)
    ).squeeze(-1)  # (T, B)

    zeros = torch.zeros(1, b, device=flat.device, dtype=log_pf_taken.dtype)
    pre_f = torch.cat([zeros, torch.cumsum(log_pf_taken, dim=0)], dim=0)  # (T+1, B)
    pre_b = torch.cat([zeros, torch.cumsum(log_pb_taken, dim=0)], dim=0)

    lz = env.true_log_z if log_z is None else log_z
    residual = lz + pre_f - pre_b - log_flows
    tb = (residual**2).mean()

    if reg_coef > 0.0:
        if reg_mode == "flow":
            # paper-faithful: lambda * sum_i F(s_i), i >= 1. float64: megaminx flows
            # near the root are ~exp(150), inf in float32.
            traj_flow = torch.exp(torch.logsumexp(log_flows[1:].to(torch.float64), dim=0))
            reg = reg_coef * traj_flow.mean()
        elif reg_mode == "logflow":
            reg = reg_coef * torch.logsumexp(log_flows[1:], dim=0).mean()
        else:
            raise ValueError(f"unknown reg_mode {reg_mode!r}")
    else:
        reg = torch.zeros((), device=flat.device, dtype=torch.float64)

    loss = tb + reg
    metrics = {
        "tb": float(tb.detach()),
        "reg": float(reg.detach()),
        "log_flow_last": float(log_flows[-1].detach().mean()),
        "log_flow_d1": float(log_flows[1].detach().mean()),
        "residual_rms": float(residual.detach().pow(2).mean().sqrt()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Inference: rollouts and policy beam on cumulative log P_B
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_solve(
    model: GFNPolicyNet,
    env: GFNEnv,
    starts: torch.Tensor,  # (B, S)
    max_steps: int,
    greedy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy (or sampled) P_B rollout. Returns (solved (B,) bool, length (B,) long);
    length is meaningful only where solved."""
    s = starts.clone()
    b = s.shape[0]
    device = s.device
    done = (s == env.solved.unsqueeze(0)).all(dim=-1)
    length = torch.zeros(b, dtype=torch.long, device=device)
    for _ in range(max_steps):
        if bool(done.all()):
            break
        bwd_logits, _ = model(s)
        if greedy:
            a = bwd_logits.argmax(dim=-1)
        else:
            a = torch.distributions.Categorical(logits=bwd_logits).sample()
        nxt = apply_gens(s, env.gens[a])
        s = torch.where(done.unsqueeze(-1), s, nxt)
        length = length + (~done).long()
        done = done | (s == env.solved.unsqueeze(0)).all(dim=-1)
    return done, length


def _hash_states(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """(N, S) -> (N,) int64 hash (wraparound multiply-add; collision prob ~ N^2/2^64)."""
    return (states.long() * weights.unsqueeze(0)).sum(dim=-1)


@torch.no_grad()
def policy_beam_solve(
    model: GFNPolicyNet,
    env: GFNEnv,
    start: torch.Tensor,  # (S,)
    width: int,
    max_steps: int,
    chunk: int = 16384,
    hash_seed: int = 0,
) -> tuple[bool, int, list[int]]:
    """Beam search on cumulative log P_B (paper section 3.4) with duplicate-state
    dropping. Returns (solved, length, path of gen indices). First solve found is
    shortest in beam steps."""
    device = env.gens.device
    a_n, s_n = env.n_actions, env.state_size
    g = torch.Generator(device="cpu").manual_seed(hash_seed)
    weights = torch.randint(
        -(2**62), 2**62, (s_n,), dtype=torch.long, generator=g
    ).to(device)

    if torch.equal(start.to(device), env.solved):
        return True, 0, []

    # uint8 beam states: (W, A, S) children tensor stays ~8x smaller than int64
    solved_u8 = env.solved.to(torch.uint8)
    beam = start.to(device).to(torch.uint8).unsqueeze(0)  # (W, S)
    scores = torch.zeros(1, device=device)
    parents_hist: list[torch.Tensor] = []
    actions_hist: list[torch.Tensor] = []

    for step in range(max_steps):
        w = beam.shape[0]
        bwd_logits, _ = model.forward_chunked(beam, chunk=chunk)
        log_pb = torch.log_softmax(bwd_logits.float(), dim=-1)  # (W, A)
        cand_scores = (scores.unsqueeze(1) + log_pb).reshape(-1)  # (W*A,)
        # children: apply gen a to every beam state
        children = torch.gather(
            beam.unsqueeze(1).expand(w, a_n, s_n),
            2,
            env.gens.unsqueeze(0).expand(w, a_n, s_n),
        ).reshape(w * a_n, s_n)
        cand_parent = torch.arange(w, device=device).repeat_interleave(a_n)
        cand_action = torch.arange(a_n, device=device).repeat(w)

        solved_mask = (children == solved_u8.unsqueeze(0)).all(dim=-1)
        if bool(solved_mask.any()):
            # earliest step -> shortest; pick the best-scored solving candidate
            idx = torch.nonzero(solved_mask).squeeze(-1)
            best = idx[cand_scores[idx].argmax()]
            path = [int(cand_action[best])]
            p = int(cand_parent[best])
            for parents, acts in zip(reversed(parents_hist), reversed(actions_hist)):
                path.append(int(acts[p]))
                p = int(parents[p])
            path.reverse()
            return True, step + 1, path

        # dedup by state hash, keep best score per state
        h = _hash_states(children, weights)
        order = cand_scores.argsort(descending=True)
        h_ord = h[order]
        uniq, inv = torch.unique(h_ord, return_inverse=True)
        first_pos = torch.full(
            (uniq.shape[0],), h_ord.shape[0], dtype=torch.long, device=device
        ).scatter_reduce(0, inv, torch.arange(h_ord.shape[0], device=device), reduce="amin")
        keep = order[first_pos]

        if keep.shape[0] > width:
            top = cand_scores[keep].topk(width).indices
            keep = keep[top]

        parents_hist.append(cand_parent[keep].cpu())
        actions_hist.append(cand_action[keep].cpu())
        beam = children[keep]
        scores = cand_scores[keep]

    return False, -1, []


# ---------------------------------------------------------------------------
# Eval-state generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def random_walk_states(
    env: GFNEnv,
    depths: list[int],
    n_per_depth: int,
    seed: int = 0,
    non_backtracking: bool = True,
) -> dict[int, torch.Tensor]:
    """Random-walk states from solved at the requested depths (depth = walk length,
    an upper bound on true distance). Returns {depth: (n, S) long} on env device."""
    device = env.gens.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: dict[int, torch.Tensor] = {}
    max_d = max(depths)
    want = set(depths)
    s = env.solved.unsqueeze(0).expand(n_per_depth, env.state_size).contiguous()
    prev_a = torch.full((n_per_depth,), -1, dtype=torch.long)
    for d in range(1, max_d + 1):
        a = torch.randint(0, env.n_actions, (n_per_depth,), generator=g)
        if non_backtracking:
            for _ in range(8):  # resample immediate backtracks
                bad = a == env.inv_idx.cpu()[prev_a.clamp(min=0)]
                bad &= prev_a >= 0
                if not bool(bad.any()):
                    break
                a[bad] = torch.randint(0, env.n_actions, (int(bad.sum()),), generator=g)
        a_dev = a.to(device)
        s = apply_gens(s, env.gens[a_dev])
        prev_a = a
        if d in want:
            out[d] = s.clone()
    return out


def bfs_exact_states(
    env: GFNEnv, max_depth: int, sample_per_depth: int, seed: int = 0
) -> dict[int, torch.Tensor]:
    """Exact-distance states via BFS from solved (CPU, numpy dedup). Returns
    {depth: (n, S) long} sampled per sphere. Practical to max_depth ~4 on megaminx."""
    import numpy as np

    rng = np.random.default_rng(seed)
    gens = env.gens.cpu().numpy()
    solved = env.solved.cpu().numpy().astype(np.uint8)
    seen = {solved.tobytes()}
    frontier = solved[None, :]
    out: dict[int, torch.Tensor] = {}
    for d in range(1, max_depth + 1):
        children = frontier[:, gens].reshape(-1, frontier.shape[1])  # (F*A, S)
        # dedup within layer
        children = np.unique(children, axis=0)
        keep = np.fromiter(
            (c.tobytes() not in seen for c in children), dtype=bool, count=len(children)
        )
        frontier = children[keep]
        for c in frontier:
            seen.add(c.tobytes())
        take = min(sample_per_depth, len(frontier))
        idx = rng.choice(len(frontier), size=take, replace=False)
        out[d] = torch.tensor(frontier[idx].astype(np.int64), device=env.gens.device)
    return out


def bfs_geodesic_dataset(
    env: GFNEnv, max_depth: int, cap_per_depth: int | None = None, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """BFS from solved, returning (states, bwd_moves) for warm-starting the backward
    (solver) head by imitation of EXACT shortest moves.

    For a state s' first discovered from parent p via forward move a
    (s' = apply(p, gens[a])), the shortest backward move (toward solved) is
    inv_idx[a]: apply(s', gens[inv_idx[a]]) = p, which is one step closer. That
    gen index is the geodesic label for P_B(s'). (One valid geodesic move per
    state suffices to teach direction; there may be others.)

    Returns states (N, S) long and bwd_moves (N,) long on env device. Excludes the
    solved state itself. Practical to depth ~6 on megaminx, deeper on subgroups.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    gens = env.gens.cpu().numpy()
    inv_idx = env.inv_idx.cpu().numpy()
    solved = env.solved.cpu().numpy().astype(np.uint8)
    a_n, s_n = gens.shape
    seen = {solved.tobytes()}
    frontier = solved[None, :]
    all_states, all_moves = [], []
    for d in range(1, max_depth + 1):
        # expand every frontier state by every generator; child gets bwd label inv[a]
        children = frontier[:, gens].reshape(-1, s_n)  # (F*A, S)
        move_of_child = np.tile(inv_idx, frontier.shape[0])  # (F*A,)
        # first occurrence of each distinct child (any geodesic parent is fine)
        _, uniq_pos = np.unique(children, axis=0, return_index=True)
        uniq_pos = np.sort(uniq_pos)
        cand, cand_moves = children[uniq_pos], move_of_child[uniq_pos]
        keep = np.fromiter((c.tobytes() not in seen for c in cand), bool, len(cand))
        new_states, new_moves = cand[keep], cand_moves[keep]  # full new sphere
        for c in new_states:
            seen.add(c.tobytes())
        frontier = new_states  # BFS continues on the FULL sphere, uncapped
        # dataset may cap per depth to bound memory / balance depths
        if cap_per_depth is not None and len(new_states) > cap_per_depth:
            sel = rng.choice(len(new_states), size=cap_per_depth, replace=False)
            all_states.append(new_states[sel].astype(np.int64))
            all_moves.append(new_moves[sel].astype(np.int64))
        else:
            all_states.append(new_states.astype(np.int64))
            all_moves.append(new_moves.astype(np.int64))
    states = torch.tensor(np.concatenate(all_states), device=env.gens.device)
    moves = torch.tensor(np.concatenate(all_moves), device=env.gens.device)
    return states, moves
