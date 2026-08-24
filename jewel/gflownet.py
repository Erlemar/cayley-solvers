"""Paper-faithful GFlowNet construction for Christopher's Jewel.

The source is the solved state.  The forward policy scrambles away from it and
has an additional stop action; the backward policy is the solving policy.  All
actions exposed by this module use Jewel's internal 0..11 ordering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ball import ExactBall
from .official import OfficialPuzzle
from .puzzle import ACTION_NAMES, GROUP_ORDER, INVERSE_ACTION, JewelState, apply_path


NEG_INF = float("-inf")


@dataclass(slots=True)
class GFNEnv:
    generators: torch.Tensor
    inverse_actions: torch.Tensor
    solved: torch.Tensor
    true_log_z: float
    source_preimages: torch.Tensor

    @property
    def n_actions(self) -> int:
        return int(self.generators.shape[0])

    @property
    def state_size(self) -> int:
        return int(self.generators.shape[1])

    def to(self, device: str | torch.device) -> "GFNEnv":
        return GFNEnv(
            self.generators.to(device),
            self.inverse_actions.to(device),
            self.solved.to(device),
            self.true_log_z,
            self.source_preimages.to(device),
        )


def apply_generators(states: torch.Tensor, generator_rows: torch.Tensor) -> torch.Tensor:
    return torch.gather(states, 1, generator_rows)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.norm1(self.linear1(inputs)))
        hidden = self.norm2(self.linear2(hidden))
        return F.relu(inputs + hidden)


class GFNPolicyNet(nn.Module):
    """Shared trunk with a 12-way backward and 13-way forward head."""

    def __init__(
        self,
        *,
        hidden: int,
        num_res_blocks: int,
        encoding: str,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.encoding = encoding
        if encoding == "onehot":
            input_size = 48 * 48
            self.embedding = None
        elif encoding == "embedding":
            input_size = 48 * embed_dim
            self.embedding = nn.Embedding(48, embed_dim)
        else:
            raise ValueError(f"unknown encoding: {encoding}")
        self.input = nn.Linear(input_size, hidden)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden) for _ in range(num_res_blocks)])
        self.output = nn.Linear(hidden, 25)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        if self.encoding == "onehot":
            return F.one_hot(states.long(), num_classes=48).to(self.input.weight.dtype).flatten(1)
        assert self.embedding is not None
        return self.embedding(states.long()).flatten(1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.input(self.encode(states)))
        for block in self.blocks:
            hidden = block(hidden)
        output = self.output(hidden)
        return output[:, :12], output[:, 12:]


def source_entry_mask(states: torch.Tensor, env: GFNEnv) -> torch.Tensor:
    return (states[:, None, :] == env.source_preimages[None, :, :]).all(dim=-1)


def masked_forward_log_softmax(forward_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_mask = torch.cat(
        [mask, torch.zeros((len(mask), 1), dtype=torch.bool, device=mask.device)], dim=1
    )
    return torch.log_softmax(forward_logits.masked_fill(full_mask, NEG_INF), dim=-1)


@torch.no_grad()
def sample_forward_trajectories(
    model: GFNPolicyNet,
    env: GFNEnv,
    batch_size: int,
    nmax: int,
    eps_explore: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = env.solved[None].expand(batch_size, -1).contiguous()
    states = [state]
    actions: list[torch.Tensor] = []
    for _ in range(nmax):
        _, forward_logits = model(state)
        logits = forward_logits[:, :12].clone()
        if eps_explore:
            explore = torch.rand(batch_size, device=state.device) < eps_explore
            logits[explore] = 0.0
        logits = logits.masked_fill(source_entry_mask(state, env), NEG_INF)
        action = torch.distributions.Categorical(logits=logits).sample()
        state = apply_generators(state, env.generators[action])
        states.append(state)
        actions.append(action)
    return torch.stack(states), torch.stack(actions)


def regularized_tb_loss(
    model: GFNPolicyNet,
    env: GFNEnv,
    states: torch.Tensor,
    actions: torch.Tensor,
    reg_coef: float,
    reg_mode: str = "flow",
) -> tuple[torch.Tensor, dict]:
    trajectory_plus_one, batch, state_size = states.shape
    flat = states.reshape(trajectory_plus_one * batch, state_size)
    backward_logits, forward_logits = model(flat)
    log_pb = torch.log_softmax(backward_logits.float(), dim=-1).reshape(
        trajectory_plus_one, batch, 12
    )
    log_pf = masked_forward_log_softmax(
        forward_logits.float(), source_entry_mask(flat, env)
    ).reshape(trajectory_plus_one, batch, 13)
    log_flows = -log_pf[..., -1]

    log_pf_taken = torch.gather(log_pf[:-1], 2, actions[..., None]).squeeze(-1)
    backward_actions = env.inverse_actions[actions]
    log_pb_taken = torch.gather(log_pb[1:], 2, backward_actions[..., None]).squeeze(-1)
    initial = torch.zeros((1, batch), device=flat.device, dtype=log_pf_taken.dtype)
    forward_prefix = torch.cat([initial, torch.cumsum(log_pf_taken, dim=0)])
    backward_prefix = torch.cat([initial, torch.cumsum(log_pb_taken, dim=0)])
    residual = env.true_log_z + forward_prefix - backward_prefix - log_flows
    tb = residual.square().mean()

    if reg_coef <= 0.0:
        regularizer = torch.zeros((), device=flat.device, dtype=torch.float64)
    elif reg_mode == "flow":
        trajectory_flow = torch.exp(torch.logsumexp(log_flows[1:].double(), dim=0))
        regularizer = reg_coef * trajectory_flow.mean()
    elif reg_mode == "logflow":
        regularizer = reg_coef * torch.logsumexp(log_flows[1:], dim=0).mean()
    else:
        raise ValueError(f"unknown regularizer mode: {reg_mode}")
    loss = tb + regularizer
    return loss, {
        "tb": float(tb.detach()),
        "reg": float(regularizer.detach()),
        "log_flow_last": float(log_flows[-1].detach().mean()),
        "log_flow_d1": float(log_flows[1].detach().mean()),
        "residual_rms": float(residual.detach().square().mean().sqrt()),
    }


@torch.no_grad()
def rollout_solve(
    model: GFNPolicyNet,
    env: GFNEnv,
    starts: torch.Tensor,
    max_steps: int,
    greedy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = starts.clone()
    done = (states == env.solved[None]).all(dim=-1)
    lengths = torch.zeros(len(states), dtype=torch.long, device=states.device)
    for _ in range(max_steps):
        if bool(done.all()):
            break
        backward_logits, _ = model(states)
        action = (
            backward_logits.argmax(dim=-1)
            if greedy
            else torch.distributions.Categorical(logits=backward_logits).sample()
        )
        next_states = apply_generators(states, env.generators[action])
        states = torch.where(done[:, None], states, next_states)
        lengths += (~done).long()
        done |= (states == env.solved[None]).all(dim=-1)
    return done, lengths


@torch.no_grad()
def rollout_paths(
    model: GFNPolicyNet,
    env: GFNEnv,
    starts: torch.Tensor,
    max_steps: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Batched greedy P_B rollout retaining one path per input state."""
    states = starts.clone()
    done = (states == env.solved[None]).all(dim=-1)
    paths: list[list[int]] = [[] for _ in range(len(states))]
    for _ in range(max_steps):
        if bool(done.all()):
            break
        backward_logits, _ = model(states)
        actions = backward_logits.argmax(dim=-1)
        active = ~done
        for index in torch.nonzero(active).squeeze(-1).cpu().tolist():
            paths[index].append(int(actions[index]))
        next_states = apply_generators(states, env.generators[actions])
        states = torch.where(done[:, None], states, next_states)
        done |= (states == env.solved[None]).all(dim=-1)
    return done, paths


@torch.no_grad()
def random_walk_states(
    env: GFNEnv,
    depths: Sequence[int],
    n_per_depth: int,
    seed: int,
    non_backtracking: bool = True,
) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    states = env.solved[None].expand(n_per_depth, -1).contiguous()
    previous = torch.full((n_per_depth,), -1, dtype=torch.long)
    output: dict[int, torch.Tensor] = {}
    requested = set(depths)
    for depth in range(1, max(depths) + 1):
        actions = torch.randint(0, 12, (n_per_depth,), generator=generator)
        if non_backtracking:
            inverse_cpu = env.inverse_actions.cpu()
            while True:
                bad = previous >= 0
                bad &= actions == inverse_cpu[previous.clamp(min=0)]
                if not bool(bad.any()):
                    break
                actions[bad] = torch.randint(0, 12, (int(bad.sum()),), generator=generator)
        actions_device = actions.to(states.device)
        states = apply_generators(states, env.generators[actions_device])
        previous = actions
        if depth in requested:
            output[depth] = states.clone()
    return output


@dataclass(slots=True)
class JewelGFNConfig:
    hidden: int = 512
    num_res_blocks: int = 3
    encoding: str = "onehot"
    embed_dim: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


def build_jewel_env(official: OfficialPuzzle, device: str | torch.device = "cpu") -> GFNEnv:
    """Build the reversed-graph environment in internal action order."""
    generators: list[list[int]] = []
    for internal_action in range(12):
        official_action = int(official.internal_to_official_action[internal_action])
        generators.append(official.generators[ACTION_NAMES[official_action]].astype(int).tolist())
    generator_tensor = torch.tensor(generators, dtype=torch.long)
    inverse_tensor = torch.tensor(INVERSE_ACTION.astype(int), dtype=torch.long)
    solved = torch.tensor(official.central_state.astype(int), dtype=torch.long)
    preimages = apply_generators(
        solved[None].expand(12, -1), generator_tensor[inverse_tensor]
    )
    returned = apply_generators(preimages, generator_tensor)
    if not torch.equal(returned, solved[None].expand_as(returned)):
        raise ValueError("forward/source action alignment failed")
    return GFNEnv(
        generator_tensor,
        inverse_tensor,
        solved,
        log(GROUP_ORDER),
        preimages,
    ).to(device)


def build_gflownet(config: JewelGFNConfig = JewelGFNConfig()) -> GFNPolicyNet:
    return GFNPolicyNet(
        hidden=config.hidden,
        num_res_blocks=config.num_res_blocks,
        encoding=config.encoding,
        embed_dim=config.embed_dim,
    )


def load_gflownet(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[GFNPolicyNet, JewelGFNConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = JewelGFNConfig(**checkpoint["config"])
    model = build_gflownet(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


class GFNBackwardPolicy:
    """Cached ``JewelState -> P_B(action|state)`` adapter for tree search."""

    def __init__(
        self,
        model: GFNPolicyNet,
        official: OfficialPuzzle,
        *,
        device: str | torch.device | None = None,
        cache: bool = True,
    ) -> None:
        self.model = model
        self.official = official
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.cache_enabled = cache
        self.cache: dict[int, np.ndarray] = {}

    @torch.no_grad()
    def __call__(self, state: JewelState) -> np.ndarray:
        rank = state.rank()
        if self.cache_enabled and rank in self.cache:
            return self.cache[rank]
        stickers = torch.from_numpy(self.official.from_structured(state).astype(np.int64))[None].to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            backward_logits, _ = self.model(stickers)
        probabilities = backward_logits.float().softmax(dim=-1)[0].cpu().numpy()
        if self.cache_enabled:
            self.cache[rank] = probabilities
        return probabilities

    @torch.no_grad()
    def batch(self, states: Sequence[JewelState]) -> np.ndarray:
        if not states:
            return np.empty((0, 12), dtype=np.float32)
        stickers = np.stack([self.official.from_structured(state) for state in states]).astype(np.int64)
        tensor = torch.from_numpy(stickers).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            backward_logits, _ = self.model(tensor)
        probabilities = backward_logits.float().softmax(dim=-1).cpu().numpy()
        if self.cache_enabled:
            for state, probs in zip(states, probabilities):
                self.cache[state.rank()] = probs
        return probabilities


@torch.no_grad()
def greedy_solve(
    start: JewelState,
    policy: GFNBackwardPolicy,
    *,
    ball: ExactBall | None = None,
    max_steps: int = 64,
) -> list[int] | None:
    state = start
    path: list[int] = []
    seen = {state.rank()}
    for _ in range(max_steps):
        if state.rank() == 0:
            return path
        if ball is not None:
            suffix = ball.exact_path(state)
            if suffix is not None:
                result = path + suffix
                assert apply_path(start, result).rank() == 0
                return result
        action = int(np.argmax(policy(state)))
        from .puzzle import apply_action

        state = apply_action(state, action)
        path.append(action)
        rank = state.rank()
        if rank in seen:
            return None
        seen.add(rank)
    return None


__all__ = [
    "GFNBackwardPolicy",
    "JewelGFNConfig",
    "build_gflownet",
    "build_jewel_env",
    "greedy_solve",
    "load_gflownet",
    "apply_generators",
    "random_walk_states",
    "regularized_tb_loss",
    "rollout_solve",
    "rollout_paths",
    "sample_forward_trajectories",
    "source_entry_mask",
]
