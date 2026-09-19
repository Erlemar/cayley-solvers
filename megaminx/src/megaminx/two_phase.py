"""Two-phase (Kociemba-style) megaminx solver: subgroup reduction + restricted finish.

Phase 2 rotates only the 6 TOP faces (12 generators). Its reachable set from the
solved state is the subgroup H. Phase 1 drives an arbitrary scramble until every
"frozen" sticker (one that no TOP face can move) is home -- the state is then in H
-- and Phase 2 finishes using only the 12 TOP generators.

Phase 1 uses a MaskedV model: the 85 TOP-movable ("don't-care") stickers are
masked to a sentinel BY VALUE, so many fully-colored states collapse to the same
masked target (the coset). Labels stay walk-depth; correctness of any produced
solution is guaranteed downstream by verify_path on the concatenated path.

Geometry (from megaminx.decomposition, U-bowl split):
  TOP_HALF_FACES = {U, F, L, R, BL, BR}  -> Phase 2 free faces (12 gens incl. inverses)
  movable stickers = F2L (face_set subset TOP) + equator (straddle)  = 85 -> masked in Phase 1
  frozen  stickers = LL  (face_set subset BOTTOM)                    = 35 -> Phase-1 goal

Why mask BY VALUE, not by position: the state is a permutation (solved == identity).
The phase-1 distance only depends on where the frozen stickers are. Masking the
movable VALUES keeps exactly the frozen-sticker placement {(slot, val) : val frozen};
masking by position would drop frozen stickers that currently sit in movable slots.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from cayley.model import ResMLPDistance
from megaminx.decomposition import (
    TOP_HALF_FACES,
    compute_face_sets,
    ll_sticker_positions,
)
from megaminx.puzzle import Megaminx

# Sentinel class id for a masked ("don't-care") sticker. The Phase-1 inner model
# therefore needs num_classes = 121 (0..119 real sticker ids + 120 = mask).
MASK_TOKEN = 120


def phase2_move_names(puzzle: Megaminx) -> tuple[str, ...]:
    """Generator names whose face is a Phase-2 free (TOP) face, in puzzle order.

    Each TOP face contributes its forward and inverse, so this is 12 names.
    """
    names = tuple(nm for nm in puzzle.move_names if nm.lstrip("-") in TOP_HALF_FACES)
    return names


def movable_frozen_positions(puzzle: Megaminx) -> tuple[list[int], list[int]]:
    """Partition sticker positions by whether any TOP face moves them.

    Returns (movable, frozen):
      movable -- positions touched by at least one TOP generator (don't-care in Phase 1)
      frozen  -- positions no TOP generator touches (must be home to enter H)

    Because the solved state is the identity, a position index doubles as the home
    sticker value at that position, so `movable` is also the set of movable VALUES.
    """
    fs = compute_face_sets(puzzle.generators)  # {pos: frozenset(face names)}
    movable: list[int] = []
    frozen: list[int] = []
    for i in range(len(puzzle.solved_state)):
        if fs[i] & TOP_HALF_FACES:
            movable.append(i)
        else:
            frozen.append(i)
    return movable, frozen


def build_phase2_puzzle(puzzle: Megaminx) -> Megaminx:
    """A restricted Megaminx exposing only the 12 TOP-face generators.

    Bypasses Megaminx.load's 24-generator assertion by constructing the dataclass
    directly. The restricted puzzle drives both restricted random walks (training)
    and a restricted beam (solving) through the existing shared infra unchanged.
    """
    names = phase2_move_names(puzzle)
    gens = {nm: puzzle.generators[nm] for nm in names}
    return Megaminx(solved_state=puzzle.solved_state, generators=gens, move_names=names)


def phase2_v0_d1_anchors(
    puzzle: Megaminx,
    n_v0: int,
    n_d1_per_move: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Phase-2 exact anchors: solved -> 0 and each 1-move child -> 1."""
    n_v0 = int(n_v0)
    n_d1_per_move = int(n_d1_per_move)
    if n_v0 <= 0 and n_d1_per_move <= 0:
        return (
            torch.empty((0, len(puzzle.solved_state)), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    states_parts = []
    depth_parts = []
    if n_v0 > 0:
        solved = torch.tensor([list(puzzle.solved_state)], dtype=torch.int64, device=device)
        states_parts.append(solved.repeat(n_v0, 1))
        depth_parts.append(torch.zeros(n_v0, dtype=torch.float32, device=device))

    if n_d1_per_move > 0:
        d1 = torch.stack([
            torch.tensor(list(puzzle.apply_move(puzzle.solved_state, nm)), dtype=torch.int64, device=device)
            for nm in puzzle.move_names
        ])
        states_parts.append(d1.repeat_interleave(n_d1_per_move, dim=0))
        depth_parts.append(torch.ones(d1.shape[0] * n_d1_per_move, dtype=torch.float32, device=device))

    return torch.cat(states_parts, dim=0), torch.cat(depth_parts, dim=0)


def phase1_h_zero_anchors(
    phase2_puzzle: Megaminx,
    n_h_samples: int,
    h_k_max: int,
    seed: int,
    device: str,
    n_back: int = 1,
    n_solved: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Phase-1 exact coset anchors: any state in H has distance 0 to H.

    H is generated by the restricted Phase-2 puzzle, so restricted walks from solved
    are true Phase-1 goal states even when they are far from solved in Phase 2.
    """
    n_h_samples = int(n_h_samples)
    n_solved = int(n_solved)
    h_k_max = max(1, int(h_k_max))
    if n_h_samples <= 0 and n_solved <= 0:
        return (
            torch.empty((0, len(phase2_puzzle.solved_state)), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    states_parts = []
    if n_solved > 0:
        solved = torch.tensor([list(phase2_puzzle.solved_state)], dtype=torch.int64, device=device)
        states_parts.append(solved.repeat(n_solved, 1))

    if n_h_samples > 0:
        from cayley.data import generate_walks_torch

        n_walks = max(1, math.ceil(n_h_samples / h_k_max))
        h_states, _ = generate_walks_torch(
            phase2_puzzle,
            n_walks=n_walks,
            k_max=h_k_max,
            seed=seed,
            device=device,
            n_back=n_back,
        )
        states_parts.append(h_states[:n_h_samples])

    states = torch.cat(states_parts, dim=0)
    depths = torch.zeros(states.shape[0], dtype=torch.float32, device=device)
    return states, depths


class MaskedV(nn.Module):
    """Wrap a distance model so movable sticker VALUES are masked before scoring.

    Drop-in for ResMLPDistance in both the shared training loop (it just calls
    `model(states)`) and KhoruzhiiSolver (it scores raw beam states; masking is
    internal, so hashing/dedup/goal-checks still see the true permutation).
    """

    def __init__(self, inner: ResMLPDistance, movable_values, mask_token: int = MASK_TOKEN):
        super().__init__()
        self.inner = inner
        self.mask_token = int(mask_token)
        # Mirror the attributes KhoruzhiiSolver / training read off a bare model.
        self.output_dim = getattr(inner, "output_dim", 1)
        self.inference_chunk_size = getattr(inner, "inference_chunk_size", None)
        self.num_classes = getattr(inner, "num_classes", self.mask_token + 1)
        self._movable_values = tuple(sorted(int(v) for v in movable_values))
        # Boolean lookup indexed by sticker VALUE: True where the value is don't-care.
        m = torch.zeros(self.num_classes, dtype=torch.bool)
        m[torch.tensor(self._movable_values, dtype=torch.long)] = True
        self.register_buffer("movable_by_value", m)

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        xl = x.long()
        is_movable = self.movable_by_value[xl]
        return torch.where(is_movable, torch.full_like(xl, self.mask_token), xl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.mask(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict:
        return {
            "model_class": "MaskedV",
            "inner_config": self.inner.get_model_config(),
            "movable_values": list(self._movable_values),
            "mask_token": self.mask_token,
        }


def build_masked_phase1_model(puzzle: Megaminx, model_cfg: dict) -> MaskedV:
    """Construct a fresh Phase-1 MaskedV from an inner ResMLP config dict.

    model_cfg is the inner ResMLPDistance config; num_classes must be MASK_TOKEN+1.
    """
    movable, _frozen = movable_frozen_positions(puzzle)
    nc = int(model_cfg.get("num_classes", MASK_TOKEN + 1))
    if nc < MASK_TOKEN + 1:
        raise ValueError(
            f"Phase-1 inner num_classes must be >= {MASK_TOKEN + 1} "
            f"(0..119 real + mask token {MASK_TOKEN}); got {nc}"
        )
    inner = ResMLPDistance(
        state_size=model_cfg.get("state_size", 120),
        num_classes=nc,
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "embedding"),
        embed_dim=model_cfg.get("embed_dim", 16),
        output_dim=model_cfg.get("output_dim", 1),
    )
    return MaskedV(inner, movable_values=movable, mask_token=MASK_TOKEN)


def _build_resmlp(cfg: dict) -> ResMLPDistance:
    return ResMLPDistance(
        state_size=cfg.get("state_size", 120),
        num_classes=cfg.get("num_classes", 120),
        hidden_dims=tuple(cfg.get("hidden_dims", [2048, 512])),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        encoding=cfg.get("encoding", "embedding"),
        embed_dim=cfg.get("embed_dim", 16),
        output_dim=cfg.get("output_dim", 1),
        inference_chunk_size=cfg.get("inference_chunk_size", 2048),
    )


def load_two_phase_model(path, device: str):
    """Load a Phase-1 (MaskedV) or Phase-2 (plain ResMLP) checkpoint.

    Dispatches on model_config["model_class"]; returns an eval-mode model on device.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    if mc.get("model_class") == "MaskedV":
        inner = _build_resmlp(mc["inner_config"])
        model = MaskedV(inner, movable_values=mc["movable_values"], mask_token=mc["mask_token"])
    else:
        model = _build_resmlp(mc)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load {path}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model.to(device).eval()


def make_frozen_goal_check(frozen_positions, device: str):
    """Return a goal_check_fn(states)->bool for 'all frozen positions are home'.

    Home value at position p is p (solved == identity), so we compare the frozen
    columns of each beam state against the frozen position indices themselves.
    """
    idx = torch.tensor(list(frozen_positions), dtype=torch.long, device=device)
    target = idx.clone()  # solved value at position p is p

    def goal_check(states: torch.Tensor) -> torch.Tensor:
        return (states[:, idx].long() == target).all(dim=1)

    return goal_check


def two_stage_solve_one(solver1, solver2, frozen_goal, full_puzzle, s0, cfg1, cfg2):
    """Run one two-phase solve and verify it.

    solver1: KhoruzhiiSolver on the FULL puzzle with the Phase-1 MaskedV model.
    solver2: KhoruzhiiSolver on the RESTRICTED (Phase-2) puzzle with the Phase-2 V.
    frozen_goal: predicate from make_frozen_goal_check (Stage-1 termination).
    Returns a dict with "status" in {ok, phase1_fail, phase2_fail, verify_fail} and,
    when ok, len1/len2/total/path. Any returned ok path is verify_path-checked.
    """
    from cayley.verify import verify_path

    found1, len1, path1 = solver1.solve(s0, cfg1, goal_check_fn=frozen_goal)
    if not found1:
        return {"status": "phase1_fail"}
    s1 = full_puzzle.apply_path(s0, path1)
    found2, len2, path2 = solver2.solve(s1, cfg2)
    if not found2:
        return {"status": "phase2_fail", "len1": len1}
    full_path = list(path1) + list(path2)
    res = verify_path(full_puzzle, s0, full_path)
    if not res.ok:
        return {"status": "verify_fail", "reason": res.reason, "len1": len1, "len2": len2}
    return {"status": "ok", "len1": len1, "len2": len2, "total": len1 + len2, "path": full_path}
