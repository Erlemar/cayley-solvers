"""Goal-conditioned local control and search for exact 6x6x6 trajectories.

The cube transition function is known exactly.  The learned component therefore
predicts useful moves and a short-horizon cost for the *relative* permutation
between a current state and an exact waypoint.  If ``current == goal`` that
relative state is the identity, and primitive generators act on it with the
same pullback permutations as on an ordinary cube state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ORBIT_COUNT = 9
ORBIT_SIZE = 24
STATE_SIZE = ORBIT_COUNT * ORBIT_SIZE
TERMINAL_ACTION = 255


@dataclass(frozen=True)
class BridgeTrajectoryDataset:
    """Compact exact trajectories used to sample waypoint windows online."""

    states: np.ndarray
    next_actions: np.ndarray
    remaining_lengths: np.ndarray
    path_offsets: np.ndarray
    path_pids: np.ndarray
    path_splits: np.ndarray
    orbit_positions: np.ndarray
    generator_permutations: np.ndarray
    inverse_actions: np.ndarray
    move_names: tuple[str, ...]
    source_digest: str

    @property
    def path_count(self) -> int:
        return len(self.path_pids)

    @property
    def state_count(self) -> int:
        return len(self.states)

    def validate(self) -> None:
        if self.states.ndim != 2 or self.states.shape[1] != STATE_SIZE:
            raise ValueError(f"states must have shape (N, {STATE_SIZE})")
        if self.states.dtype != np.uint16:
            raise ValueError("states must use uint16")
        for name, array in (
            ("next_actions", self.next_actions),
            ("remaining_lengths", self.remaining_lengths),
        ):
            if array.shape != (self.state_count,):
                raise ValueError(f"{name} must contain one value per state")
        if self.path_offsets.shape != (self.path_count + 1,):
            raise ValueError("path_offsets must delimit every stored trajectory")
        if self.path_splits.shape != (self.path_count,):
            raise ValueError("path_splits must contain one split per path")
        if self.path_offsets[0] != 0 or self.path_offsets[-1] != self.state_count:
            raise ValueError("path_offsets do not span the state table")
        if np.any(np.diff(self.path_offsets) < 1):
            raise ValueError("every path must contain at least its terminal state")
        if self.orbit_positions.shape != (ORBIT_COUNT, ORBIT_SIZE):
            raise ValueError("orbit_positions has the wrong shape")
        if set(self.orbit_positions.reshape(-1).tolist()) != set(range(STATE_SIZE)):
            raise ValueError("orbit_positions must partition all positions")
        if self.generator_permutations.shape != (len(self.move_names), STATE_SIZE):
            raise ValueError("generator_permutations has the wrong shape")
        if self.inverse_actions.shape != (len(self.move_names),):
            raise ValueError("inverse_actions has the wrong shape")
        if not set(np.unique(self.path_splits)).issubset({0, 1, 2}):
            raise ValueError("path_splits may only contain train=0, validation=1, test=2")
        terminal = self.remaining_lengths == 0
        if not np.array_equal(self.next_actions == TERMINAL_ACTION, terminal):
            raise ValueError("terminal action markers disagree with remaining lengths")
        if np.any(self.next_actions[~terminal] >= len(self.move_names)):
            raise ValueError("a nonterminal action is outside the move vocabulary")
        for path_index in range(self.path_count):
            start = int(self.path_offsets[path_index])
            stop = int(self.path_offsets[path_index + 1])
            expected = np.arange(stop - start - 1, -1, -1, dtype=np.uint16)
            if not np.array_equal(self.remaining_lengths[start:stop], expected):
                raise ValueError(f"remaining lengths are inconsistent for path {path_index}")

    def indices_for_split(self, split: int, *, nonterminal: bool = True) -> np.ndarray:
        if split not in (0, 1, 2):
            raise ValueError("split must be 0, 1, or 2")
        pieces: list[np.ndarray] = []
        for path_index, path_split in enumerate(self.path_splits):
            if int(path_split) != split:
                continue
            start = int(self.path_offsets[path_index])
            stop = int(self.path_offsets[path_index + 1])
            if nonterminal:
                stop -= 1
            if stop > start:
                pieces.append(np.arange(start, stop, dtype=np.int64))
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def path_index_for_pid(self, pid: int) -> int:
        matches = np.flatnonzero(self.path_pids == pid)
        if len(matches) != 1:
            raise KeyError(f"expected exactly one trajectory for pid {pid}, found {len(matches)}")
        return int(matches[0])

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                states=self.states,
                next_actions=self.next_actions,
                remaining_lengths=self.remaining_lengths,
                path_offsets=self.path_offsets,
                path_pids=self.path_pids,
                path_splits=self.path_splits,
                orbit_positions=self.orbit_positions,
                generator_permutations=self.generator_permutations,
                inverse_actions=self.inverse_actions,
                move_names=np.asarray(self.move_names),
                source_digest=np.asarray(self.source_digest),
            )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "BridgeTrajectoryDataset":
        with np.load(path, allow_pickle=False) as archive:
            dataset = cls(
                states=archive["states"],
                next_actions=archive["next_actions"],
                remaining_lengths=archive["remaining_lengths"],
                path_offsets=archive["path_offsets"],
                path_pids=archive["path_pids"],
                path_splits=archive["path_splits"],
                orbit_positions=archive["orbit_positions"],
                generator_permutations=archive["generator_permutations"],
                inverse_actions=archive["inverse_actions"],
                move_names=tuple(str(value) for value in archive["move_names"].tolist()),
                source_digest=str(archive["source_digest"].item()),
            )
        dataset.validate()
        return dataset


def digest_files(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        source = Path(path)
        digest.update(source.name.encode("utf-8"))
        with open(source, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def global_to_local_lookup(orbit_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(orbit_positions)
    if positions.shape != (ORBIT_COUNT, ORBIT_SIZE):
        raise ValueError("orbit_positions has the wrong shape")
    lookup = np.empty(STATE_SIZE, dtype=np.uint8)
    for orbit in positions:
        lookup[orbit] = np.arange(ORBIT_SIZE, dtype=np.uint8)
    return lookup


def relative_global_state(current: Sequence[int], goal: Sequence[int]) -> np.ndarray:
    """Return current stickers expressed as their desired positions in ``goal``."""

    current_array = np.asarray(current, dtype=np.int64)
    goal_array = np.asarray(goal, dtype=np.int64)
    if current_array.shape != (STATE_SIZE,) or goal_array.shape != (STATE_SIZE,):
        raise ValueError(f"current and goal must each contain {STATE_SIZE} stickers")
    goal_inverse = np.empty(STATE_SIZE, dtype=np.uint16)
    goal_inverse[goal_array] = np.arange(STATE_SIZE, dtype=np.uint16)
    return goal_inverse[current_array]


def relative_orbit_states_numpy(
    current: np.ndarray,
    goal: np.ndarray,
    orbit_positions: np.ndarray,
) -> np.ndarray:
    """Vectorized absolute-state pairs to lossless 9x24 local permutations."""

    current_array = np.asarray(current)
    goal_array = np.asarray(goal)
    if current_array.ndim == 1:
        current_array = current_array[None, :]
    if goal_array.ndim == 1:
        goal_array = goal_array[None, :]
    if current_array.shape != goal_array.shape or current_array.shape[1] != STATE_SIZE:
        raise ValueError("current and goal batches must have matching shape (B, 216)")
    batch_size = len(current_array)
    goal_inverse = np.empty((batch_size, STATE_SIZE), dtype=np.uint16)
    goal_inverse[
        np.arange(batch_size)[:, None],
        goal_array.astype(np.int64, copy=False),
    ] = np.arange(STATE_SIZE, dtype=np.uint16)[None, :]
    relative = np.take_along_axis(
        goal_inverse,
        current_array.astype(np.int64, copy=False),
        axis=1,
    )
    positions = np.asarray(orbit_positions, dtype=np.int64)
    lookup = global_to_local_lookup(positions)
    return lookup[relative[:, positions]]


def relative_orbit_states_torch(
    current: torch.Tensor,
    goal: torch.Tensor,
    orbit_positions: torch.Tensor,
    global_to_local: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of :func:`relative_orbit_states_numpy`."""

    if current.ndim != 2 or current.shape[1] != STATE_SIZE or current.shape != goal.shape:
        raise ValueError("current and goal must have matching shape (B, 216)")
    batch_size = current.shape[0]
    positions = torch.arange(STATE_SIZE, device=current.device).expand(batch_size, -1)
    goal_inverse = torch.empty_like(positions)
    goal_inverse.scatter_(1, goal.long(), positions)
    relative = goal_inverse.gather(1, current.long())
    orbit_values = relative[:, orbit_positions.long()]
    return global_to_local.long()[orbit_values]


@dataclass(frozen=True)
class OrbitBridgeConfig:
    action_count: int = 36
    orbit_dim: int = 128
    transformer_layers: int = 3
    attention_heads: int = 8
    hidden_dim: int = 512
    residual_blocks: int = 3
    dropout: float = 0.0
    maximum_horizon: int = 32

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class BridgeResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.linear_in(self.norm(inputs)))
        return inputs + self.dropout(self.linear_out(hidden))


class OrbitBridgePolicyValueNet(nn.Module):
    """Cross-orbit policy/value model for short exact waypoint problems."""

    def __init__(self, config: OrbitBridgeConfig) -> None:
        super().__init__()
        if config.action_count <= 0 or config.maximum_horizon <= 0:
            raise ValueError("action_count and maximum_horizon must be positive")
        if config.orbit_dim % config.attention_heads:
            raise ValueError("orbit_dim must be divisible by attention_heads")
        self.config = config
        self.orbit_stem = nn.Sequential(
            nn.Linear(ORBIT_SIZE * ORBIT_SIZE, config.orbit_dim),
            nn.LayerNorm(config.orbit_dim),
            nn.SiLU(),
        )
        self.orbit_embedding = nn.Parameter(torch.empty(ORBIT_COUNT, config.orbit_dim))
        nn.init.normal_(self.orbit_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.orbit_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.orbit_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_orbit = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.fusion = nn.Sequential(
            nn.Linear(ORBIT_COUNT * config.orbit_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            BridgeResidualBlock(config.hidden_dim, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.policy_head = nn.Linear(config.hidden_dim, config.action_count)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def encode_hidden(self, relative_orbits: torch.Tensor) -> torch.Tensor:
        if relative_orbits.ndim != 3 or relative_orbits.shape[1:] != (
            ORBIT_COUNT,
            ORBIT_SIZE,
        ):
            raise ValueError("relative_orbits must have shape (B, 9, 24)")
        encoded = F.one_hot(relative_orbits.long(), num_classes=ORBIT_SIZE)
        encoded = encoded.reshape(-1, ORBIT_COUNT, ORBIT_SIZE * ORBIT_SIZE)
        encoded = encoded.to(self.orbit_stem[0].weight.dtype)
        tokens = self.orbit_stem(encoded) + self.orbit_embedding[None]
        tokens = self.cross_orbit(tokens)
        hidden = self.fusion(tokens.flatten(start_dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, relative_orbits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(relative_orbits)
        logits = self.policy_head(hidden)
        value = F.softplus(self.value_head(hidden).squeeze(-1))
        return logits, value


def bridge_policy_value_loss(
    outputs: tuple[torch.Tensor, torch.Tensor],
    actions: torch.Tensor,
    horizons: torch.Tensor,
    *,
    maximum_horizon: int,
    value_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, values = outputs
    policy = F.cross_entropy(logits.float(), actions.long())
    normalized_targets = horizons.float() / maximum_horizon
    normalized_values = values.float() / maximum_horizon
    value = F.smooth_l1_loss(normalized_values, normalized_targets)
    total = policy + value_weight * value
    return total, {
        "loss": total.detach(),
        "policy_loss": policy.detach(),
        "value_loss": value.detach(),
    }


@dataclass(frozen=True)
class BridgeSearchResult:
    solved: bool
    actions: tuple[int, ...]
    expanded_states: int
    generated_states: int
    final_beam_size: int
    depth: int


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.inference_mode()
def bridge_beam_search(
    current: Sequence[int],
    goal: Sequence[int],
    model: nn.Module,
    orbit_positions: np.ndarray,
    generator_permutations: np.ndarray,
    inverse_actions: np.ndarray,
    *,
    beam_width: int = 4096,
    branch_width: int = 8,
    maximum_steps: int = 32,
    value_weight: float = 1.0,
    policy_weight: float = 0.15,
    model_batch_size: int = 8192,
) -> BridgeSearchResult:
    """Search from ``current`` to an exact waypoint using known cube dynamics."""

    if beam_width <= 0 or branch_width <= 0 or maximum_steps < 0:
        raise ValueError("invalid bridge search dimensions")
    action_count = len(generator_permutations)
    if branch_width > action_count:
        raise ValueError("branch_width exceeds the action count")
    relative = relative_global_state(current, goal)
    identity = np.arange(STATE_SIZE, dtype=np.uint16)
    if np.array_equal(relative, identity):
        return BridgeSearchResult(True, (), 0, 0, 1, 0)

    device = _model_device(model)
    states = torch.from_numpy(relative.astype(np.int16))[None].to(device)
    generators = torch.as_tensor(generator_permutations, dtype=torch.long, device=device)
    inverse = torch.as_tensor(inverse_actions, dtype=torch.long, device=device)
    orbit_tensor = torch.as_tensor(orbit_positions, dtype=torch.long, device=device)
    lookup_tensor = torch.as_tensor(
        global_to_local_lookup(np.asarray(orbit_positions)),
        dtype=torch.long,
        device=device,
    )
    goal_tensor = torch.arange(STATE_SIZE, dtype=states.dtype, device=device)
    hash_generator = torch.Generator(device="cpu").manual_seed(666)
    hash_weights_1 = torch.randint(
        -(2**62),
        2**62,
        (STATE_SIZE,),
        generator=hash_generator,
        dtype=torch.int64,
    ).to(device)
    hash_weights_2 = torch.randint(
        -(2**62),
        2**62,
        (STATE_SIZE,),
        generator=hash_generator,
        dtype=torch.int64,
    ).to(device)

    last_actions = torch.full((1,), -1, dtype=torch.long, device=device)
    cumulative_nll = torch.zeros(1, dtype=torch.float32, device=device)
    history: list[tuple[np.ndarray, np.ndarray]] = []
    expanded = 0
    generated = 0

    def predict(batch_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        for start in range(0, len(batch_states), model_batch_size):
            selected = batch_states[start : start + model_batch_size]
            local = lookup_tensor[selected[:, orbit_tensor].long()]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, values = model(local)
            logits_parts.append(logits.float())
            value_parts.append(values.float())
        return torch.cat(logits_parts), torch.cat(value_parts)

    def reconstruct(parent_index: int, action: int) -> tuple[int, ...]:
        reverse_path = [action]
        cursor = parent_index
        for parents, actions in reversed(history):
            reverse_path.append(int(actions[cursor]))
            cursor = int(parents[cursor])
        return tuple(reversed(reverse_path))

    for depth in range(1, maximum_steps + 1):
        logits, _ = predict(states)
        expanded += len(states)
        if torch.any(last_actions >= 0):
            rows = torch.nonzero(last_actions >= 0, as_tuple=False).flatten()
            logits[rows, inverse[last_actions[rows]]] = -torch.inf
        log_probs = F.log_softmax(logits, dim=1)
        chosen_log_probs, chosen_actions = log_probs.topk(branch_width, dim=1)
        parent_count = len(states)
        parent_states = states[:, None, :].expand(-1, branch_width, -1)
        gather_indices = generators[chosen_actions]
        children = parent_states.gather(2, gather_indices).reshape(-1, STATE_SIZE)
        flat_actions = chosen_actions.reshape(-1)
        flat_parents = torch.arange(parent_count, device=device).repeat_interleave(branch_width)
        child_nll = (
            cumulative_nll[:, None] - chosen_log_probs
        ).reshape(-1)
        generated += len(children)

        solved_mask = children.eq(goal_tensor).all(dim=1)
        if torch.any(solved_mask):
            solved_indices = torch.nonzero(solved_mask, as_tuple=False).flatten()
            winner = solved_indices[child_nll[solved_indices].argmin()]
            parent = int(flat_parents[winner].item())
            action = int(flat_actions[winner].item())
            path = reconstruct(parent, action)
            return BridgeSearchResult(
                True,
                path,
                expanded,
                generated,
                len(states),
                depth,
            )

        _, child_values = predict(children)
        scores = depth + value_weight * child_values + policy_weight * child_nll
        hashes_1 = (children.long() * hash_weights_1).sum(dim=1)
        hashes_2 = (children.long() * hash_weights_2).sum(dim=1)
        order = torch.argsort(scores)
        ordered_1 = hashes_1[order].cpu().tolist()
        ordered_2 = hashes_2[order].cpu().tolist()
        keep_order_positions: list[int] = []
        seen: set[tuple[int, int]] = set()
        for order_position, key in enumerate(zip(ordered_1, ordered_2, strict=True)):
            if key in seen:
                continue
            seen.add(key)
            keep_order_positions.append(order_position)
            if len(keep_order_positions) >= beam_width:
                break
        selected = order[
            torch.as_tensor(keep_order_positions, dtype=torch.long, device=device)
        ]
        states = children[selected]
        cumulative_nll = child_nll[selected]
        last_actions = flat_actions[selected]
        history.append(
            (
                flat_parents[selected].cpu().numpy().astype(np.int32, copy=False),
                flat_actions[selected].cpu().numpy().astype(np.uint8, copy=False),
            )
        )

    return BridgeSearchResult(
        False,
        (),
        expanded,
        generated,
        len(states),
        maximum_steps,
    )


def checkpoint_manifest(
    config: OrbitBridgeConfig,
    dataset: BridgeTrajectoryDataset,
    report: dict[str, object],
) -> dict[str, object]:
    return {
        "kind": "cube666_goal_conditioned_bridge_v1",
        "model_config": config.as_dict(),
        "move_names": list(dataset.move_names),
        "orbit_positions": dataset.orbit_positions.tolist(),
        "source_digest": dataset.source_digest,
        "report": report,
    }


def manifest_digest(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
