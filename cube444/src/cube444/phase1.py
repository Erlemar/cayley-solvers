"""Phase 1 of the two-phase 4x4x4 solver: reduce an arbitrary state into R.

Phase 1 uses all 24 generators and stops the moment `orbits.is_reduced` holds.
Its value function is a *masked* V: it predicts distance-to-the-SET R rather than
distance to a single goal state.

WHY THE INPUT IS 72 SLOTS, NOT 96
---------------------------------
R places no constraint on the 24 corner slots, and corners are a closed orbit under
the full group (verified in `orbits.verify_structure`). So no move can carry corner
colour into a non-corner slot, the non-corner projection evolves autonomously, and
distance-to-R is *exactly* a function of the 72 non-corner slots. Masking corners is
an exact quotient, not an approximation -- it removes 24 slots of pure nuisance
variance and makes the network 25% cheaper per forward.

THE SPARSE-Q MARGIN TERM
------------------------
Plain walk-depth MSE has a known failure mode: at large k the conditional variance
of the true distance given s is huge, so the Bayes-optimal prediction shrinks toward
the mean and *local discrimination collapses* -- the very thing a beam ranks on.
Measured on the sibling IHES cube's e6 model, the child gap V(next) - V(undo) falls
from 1.96 at pivot depth 2 to 0.37 at depth 21 (it should be 2 everywhere).

The fix, following Kuznetsov's sparse-Q rw-middle objective, is to add a term whose
*difference* has zero conditional variance: on a random-walk pivot state s_p, the
predecessor s_{p-1} and a successor s_{p+1} always differ by exactly 2 in walk depth,
whatever k is. MSE cannot buy loss by flattening that gap; the absolute level stays
free to saturate, which is fine because a beam needs ordering, not scale.

HONEST CAVEAT: at large p the walk is not geodesic, so "the true distance gap is 2"
is itself biased -- the term is low-variance but not unbiased. That is exactly why
`lambda_margin` is a knob with a documented control arm (set it to 0.0 and everything
else is unchanged), and why the acceptance gate is the beam A/B, not the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from cube444.orbits import NONCORNER_SLOTS, is_reduced, reduction_defects


class MaskedV(nn.Module):
    """Wrap a 72-slot distance model so it accepts full 96-slot states."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.register_buffer(
            "noncorner", torch.from_numpy(NONCORNER_SLOTS.astype(np.int64)), persistent=False
        )
        self.state_size = 96
        self.num_classes = getattr(inner, "num_classes", 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.inner.state_size:
            return self.inner(x)
        return self.inner(x.index_select(-1, self.noncorner))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class WalkSpec:
    """How phase-1 training states are generated.

    `k_max` must be large enough that the walk distribution actually REACHES the
    test distribution. Measured reduction-defect counts:

        k_max=30 walks   mean 33.7  p10 16
        k_max=120 walks  mean 38.4  p10 34
        real test states mean 39.7  p10 36

    At k_max=30 a large part of the training mass sits far closer to R than any
    real puzzle ever is, and the phase-1 beam then stalls at ~8 defects because it
    walks off the training distribution near the end. Hence the default of 120.

    `label_clip` bounds the walk-depth regression target. A 120-step walk lands at
    a true distance-to-R of only ~21 (the counting bound for distance-to-R is 21.5
    and the space is small), so the raw walk depth would be a wildly biased label;
    clipping keeps the DISTRIBUTION deep while keeping the LABEL sane.

    `margin_k_max` is deliberately much smaller than `k_max`: the sparse-Q margin
    term asserts that V(next) - V(prev) == 2, which is only defensible while the
    walk is still roughly geodesic. Applying it at depth 120 -- where prev and next
    are at the same true distance -- would inject pure noise.
    """

    k_max: int = 120          # max full-group walk depth away from R
    h_walk_max: int = 40      # outer-only walk length used to pick a random h in R
    n_back: int = 1           # non-backtracking radius (1 = never immediately undo)
    label_clip: float = 30.0  # cap on the walk-depth regression target
    margin_k_max: int = 20    # pivot depth cap for the margin term


class Phase1Sampler:
    """GPU random-walk sampler for phase-1 training data.

    Walks start from a RANDOM element of R (reached by an outer-only walk from
    solved), not from solved itself -- the goal is a set, so the training
    distribution has to cover it.
    """

    def __init__(self, puzzle, device: str = "cuda", seed: int = 0):
        names = list(puzzle.move_names)
        self.names = names
        self.n_gen = len(names)
        self.device = device
        self.gens = torch.tensor(
            np.stack([np.asarray(puzzle.generators[n], dtype=np.int64) for n in names]),
            dtype=torch.long, device=device,
        )                                                    # (n_gen, 96)
        inv_name = {n: (n[1:] if n.startswith("-") else "-" + n) for n in names}
        self.inv_idx = torch.tensor(
            [names.index(inv_name[n]) for n in names], dtype=torch.long, device=device
        )
        from cube444.orbits import is_outer_move
        self.outer_idx = torch.tensor(
            [i for i, n in enumerate(names) if is_outer_move(n)],
            dtype=torch.long, device=device,
        )
        self.solved = torch.tensor(
            np.asarray(puzzle.solved_state, dtype=np.int64), dtype=torch.long, device=device
        )
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)

    def _apply(self, states: torch.Tensor, move_idx: torch.Tensor) -> torch.Tensor:
        """states (B,96) long, move_idx (B,) long -> new states."""
        return torch.gather(states, 1, self.gens[move_idx])

    def random_in_R(self, batch: int, spec: WalkSpec) -> torch.Tensor:
        """A batch of random elements of R, via outer-only walks from solved."""
        s = self.solved.unsqueeze(0).expand(batch, -1).contiguous()
        steps = int(spec.h_walk_max)
        # Per-row walk lengths: mask out moves past each row's own length.
        lens = torch.randint(0, steps + 1, (batch,), generator=self.g, device=self.device)
        for t in range(steps):
            pick = self.outer_idx[
                torch.randint(len(self.outer_idx), (batch,), generator=self.g,
                              device=self.device)
            ]
            moved = self._apply(s, pick)
            s = torch.where((t < lens).unsqueeze(1), moved, s)
        return s

    def walk(self, batch: int, spec: WalkSpec):
        """Random walks away from R.

        Returns (states, depth, prev_states, next_states, prev_depth, next_depth)
        where `states` are the pivots, `prev` is one step back toward R along the
        walk and `next` is one step further away. The (prev, next) pair is the
        zero-conditional-variance margin sample: their depths always differ by 2.
        """
        s = self.random_in_R(batch, spec)
        depth = torch.randint(1, spec.k_max + 1, (batch,), generator=self.g,
                              device=self.device)
        last = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        for t in range(spec.k_max):
            pick = torch.randint(self.n_gen, (batch,), generator=self.g, device=self.device)
            if spec.n_back >= 1:
                # resample any pick that would immediately undo the previous move
                banned = torch.where(last >= 0, self.inv_idx[last.clamp(min=0)],
                                     torch.full_like(last, -1))
                clash = pick == banned
                for _ in range(8):
                    if not bool(clash.any()):
                        break
                    resample = torch.randint(self.n_gen, (batch,), generator=self.g,
                                             device=self.device)
                    pick = torch.where(clash, resample, pick)
                    clash = pick == banned
            active = (t < depth)
            moved = self._apply(s, pick)
            s = torch.where(active.unsqueeze(1), moved, s)
            last = torch.where(active, pick, last)

        # predecessor: undo the final move actually taken
        prev = self._apply(s, self.inv_idx[last.clamp(min=0)])
        # successor: any move that is not the undo
        undo = self.inv_idx[last.clamp(min=0)]
        nxt_pick = torch.randint(self.n_gen, (batch,), generator=self.g, device=self.device)
        clash = nxt_pick == undo
        for _ in range(8):
            if not bool(clash.any()):
                break
            resample = torch.randint(self.n_gen, (batch,), generator=self.g,
                                     device=self.device)
            nxt_pick = torch.where(clash, resample, nxt_pick)
            clash = nxt_pick == undo
        nxt = self._apply(s, nxt_pick)
        return s, depth.float(), prev, nxt

    def walk_harvest(self, batch: int, spec: WalkSpec):
        """One walk of length k_max, snapshotting EVERY step.

        `walk` runs k_max gathers and keeps one state per row, throwing away the
        other k_max-1. At k_max=120 that dominates the epoch. Harvesting every step
        yields batch*k_max labelled states for the same k_max gathers, which is what
        makes a deep walk distribution affordable. Returns (states, depths).
        """
        s = self.random_in_R(batch, spec)
        last = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        out_s = []
        for _ in range(spec.k_max):
            pick = torch.randint(self.n_gen, (batch,), generator=self.g,
                                 device=self.device)
            if spec.n_back >= 1:
                banned = torch.where(last >= 0, self.inv_idx[last.clamp(min=0)],
                                     torch.full_like(last, -1))
                clash = pick == banned
                for _ in range(8):
                    if not bool(clash.any()):
                        break
                    resample = torch.randint(self.n_gen, (batch,), generator=self.g,
                                             device=self.device)
                    pick = torch.where(clash, resample, pick)
                    clash = pick == banned
            s = self._apply(s, pick)
            last = pick
            out_s.append(s.to(torch.uint8))
        states = torch.cat(out_s, dim=0)
        depths = torch.arange(1, spec.k_max + 1, device=self.device,
                              dtype=torch.float32).repeat_interleave(batch)
        return states, depths

    def children(self, states: torch.Tensor) -> torch.Tensor:
        """All 24 children of each state -> (B, n_gen, 96)."""
        b = states.shape[0]
        exp = states.unsqueeze(1).expand(b, self.n_gen, states.shape[1])
        idx = self.gens.unsqueeze(0).expand(b, self.n_gen, states.shape[1])
        return torch.gather(exp, 2, idx)

    def r_anchors(self, batch: int, spec: WalkSpec):
        """Exact anchors: states IN R (label 0) and their children (label 1).

        These play the role the solved/d=1 anchors play in the single-phase pipeline
        and are what stops the Bellman bootstrap drifting away from V(R)=0.
        """
        h = self.random_in_R(batch, spec)
        pick = torch.randint(self.n_gen, (batch,), generator=self.g, device=self.device)
        d1 = self._apply(h, pick)
        return h, d1


class ReductionTest:
    """Torch-native `reduction_defects` / `is_reduced` for the training hot path.

    The Bellman target needs V_target(child)=0 for every child already in R (the
    Dirichlet boundary condition), which means testing R on B x 24 states per step.
    Round-tripping that through numpy would dominate the step, so keep it on device.
    """

    def __init__(self, device: str = "cuda"):
        from cube444.orbits import CENTER_SLOTS, CENTER_TARGET, WING_PAIRS

        self.center_slots = torch.as_tensor(CENTER_SLOTS, dtype=torch.long, device=device)
        self.center_target = torch.as_tensor(CENTER_TARGET, dtype=torch.long, device=device)
        self.wing_a = torch.as_tensor(WING_PAIRS[:, 0], dtype=torch.long, device=device)
        self.wing_b = torch.as_tensor(WING_PAIRS[:, 1], dtype=torch.long, device=device)

    def defects(self, states: torch.Tensor) -> torch.Tensor:
        """(..., 96) long -> (...) int count of reduction defects, 0..48."""
        c = (states.index_select(-1, self.center_slots) != self.center_target).sum(-1)
        w = (states.index_select(-1, self.wing_a)
             != states.index_select(-1, self.wing_b)).sum(-1)
        return c + w

    def is_reduced(self, states: torch.Tensor) -> torch.Tensor:
        return self.defects(states) == 0


def check_reduction_test(device: str = "cpu", n: int = 256, seed: int = 0) -> bool:
    """Assert the torch and numpy reduction tests agree (guards the hot path)."""
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 6, size=(n, 96))
    rt = ReductionTest(device=device)
    t = rt.defects(torch.as_tensor(states, dtype=torch.long, device=device)).cpu().numpy()
    return bool(np.array_equal(t, np.asarray(reduction_defects(states))))
