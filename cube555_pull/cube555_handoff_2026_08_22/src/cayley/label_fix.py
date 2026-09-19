"""Dedup + exact-label override for the random-walk stream (GPU / torch).

This is the training-time counterpart of `cube444/scripts/20_label_fix.py`. That script
holds the numpy REFERENCE implementation and its self-test; this module is the torch port
that runs inside the training loop, on the device the walks are already on. The two are
cross-checked against each other by `cube444/scripts/21_test_label_fix.py`.

WHY (short version -- full argument in 02_DATA.md section 3):

A non-backtracking walk forbids only the immediate inverse, so the walk index `p` is an
UPPER BOUND on the true distance `d`, never equal to it in general. On THIS puzzle the
state is a colouring, so it pins the group element only up to the 4!^6 = 191,102,976
stabiliser of solved, and `d` is a minimum over that whole coset. Labels therefore
over-estimate for two stacked reasons -- non-geodesic slack AND coset collapse -- and
duplicate colourings carrying DIFFERENT labels are common.

Two passes, both cheap because the exact ball is already resident for the anchor mixin:

  A. dedup by colouring, keep the MINIMUM label
  B. where the exact table knows the colouring, replace the label with the true distance

Pass B is one-way by construction (p >= d), so it can only lower a label. We assert that.
A table claiming a LONGER distance than the walk means the Zobrist table or the generator
convention disagrees with the one used to build the table -- a bug, not a data quirk.

USAGE

    fixer = LabelFixer.from_anchor_file("cube444/data/bfs_anchors.pt", device="cuda")
    states, labels, stats = fixer.apply(states, labels)
    print(stats.render())

Set `fixer = None` to disable; both call sites treat None as "no fix" so the control arm
costs nothing.

NOTE ON THE CONTROL. This is an objective change: it only takes effect on a fresh run.
Do not warm-start a fixed-label run from an unfixed checkpoint and attribute the
difference to the fix. See 02_DATA.md section 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# Sentinel used when reducing labels with amin over an empty group.
_BIG_LABEL = 1 << 30


def make_ztab(
    state_size: int, num_classes: int, seed: int = 0, device: str = "cuda"
) -> torch.Tensor:
    """Zobrist table, (state_size, num_classes) int64.

    Any table works as long as ONE table is used for both the exact table and the walk
    states within a run. `LabelFixer` owns a single table for exactly this reason, so
    there is no cross-process consistency problem to get wrong.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(
        -(2**63),
        2**63 - 1,
        (state_size, num_classes),
        dtype=torch.int64,
        device=device,
        generator=g,
    )


def zhash(states: torch.Tensor, ztab: torch.Tensor) -> torch.Tensor:
    """(N, S) colourings -> (N,) int64 Zobrist hash.

    `states` may be any integer dtype; values must be in [0, num_classes).
    """
    idx = states.long()
    h = torch.zeros(states.shape[0], dtype=torch.int64, device=states.device)
    for i in range(states.shape[1]):
        h = h ^ ztab[i][idx[:, i]]
    return h


def dedup_min_label(
    states: torch.Tensor, labels: torch.Tensor, ztab: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse duplicate colourings, keeping the MINIMUM label for each.

    Keeping both copies trains the net to output two different numbers for one input;
    keeping the larger trains it to the worse of two upper bounds.
    """
    if states.shape[0] == 0:
        return states, labels
    h = zhash(states, ztab)
    hu, inv = torch.unique(h, return_inverse=True)
    n_groups = hu.numel()

    min_lab = torch.full(
        (n_groups,), _BIG_LABEL, dtype=torch.int64, device=states.device
    )
    min_lab.scatter_reduce_(0, inv, labels.long(), reduce="amin", include_self=False)

    # Representative row per group: the first occurrence. Every row in a group has the
    # same hash, hence (barring a 64-bit collision) the same colouring, so any row will
    # do -- first is chosen to make the result deterministic.
    order = torch.arange(states.shape[0], device=states.device)
    first = torch.zeros(n_groups, dtype=torch.int64, device=states.device)
    first.scatter_reduce_(0, inv, order, reduce="amin", include_self=False)

    return states[first], min_lab.to(labels.dtype)


def exact_override(
    states: torch.Tensor,
    labels: torch.Tensor,
    table_hashes: torch.Tensor,
    table_depths: torch.Tensor,
    ztab: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Replace walk labels with exact distances wherever the table knows the colouring.

    `table_hashes` must be sorted ascending, with `table_depths` in the same order.
    Returns (labels, n_in_table, n_lowered).
    """
    if states.shape[0] == 0:
        return labels, 0, 0
    h = zhash(states, ztab)
    pos = torch.searchsorted(table_hashes, h).clamp_(max=table_hashes.numel() - 1)
    hit = table_hashes[pos] == h
    exact = table_depths[pos].to(labels.dtype)

    longer = hit & (exact > labels)
    if bool(longer.any()):
        n_bad = int(longer.sum())
        raise AssertionError(
            f"{n_bad} states where the exact table claims a LONGER distance than the "
            "walk. A walk length is an upper bound, so this cannot happen: the Zobrist "
            "table or the generator convention disagrees with the one used to build "
            "the table."
        )

    n_lowered = int((hit & (exact < labels)).sum())
    return torch.where(hit, exact, labels), int(hit.sum()), n_lowered


@dataclass
class FixStats:
    """Per-epoch counters. The field names match the log block in 02_DATA.md."""

    generated: int = 0
    after_dedup: int = 0
    in_table: int = 0
    lowered: int = 0
    kept: int = 0  # after the round-to-multiple truncation (see apply)

    @property
    def dup_rate(self) -> float:
        return 1.0 - self.after_dedup / max(self.generated, 1)

    def render(self) -> str:
        tail = "" if self.kept == self.after_dedup else f" -> {self.kept:,} kept"
        return (
            f"label-fix: {self.generated:,} -> {self.after_dedup:,} rows "
            f"(dup {100 * self.dup_rate:.1f}%), {self.in_table:,} in table, "
            f"{self.lowered:,} lowered{tail}"
        )


class LabelFixer:
    """Holds the Zobrist table and the sorted exact table; applies both passes.

    Construct once per run and reuse -- building the sorted table is the only expensive
    part and it does not change between epochs.
    """

    def __init__(
        self,
        table_hashes: torch.Tensor,
        table_depths: torch.Tensor,
        ztab: torch.Tensor,
        do_dedup: bool = True,
        do_override: bool = True,
    ):
        self.table_hashes = table_hashes
        self.table_depths = table_depths
        self.ztab = ztab
        self.do_dedup = do_dedup
        self.do_override = do_override

    @classmethod
    def from_anchor_file(
        cls,
        path: str | Path,
        device: str = "cuda",
        state_size: int = 96,
        num_classes: int = 6,
        zseed: int = 0,
        do_dedup: bool = True,
        do_override: bool = True,
    ) -> "LabelFixer":
        """Build from a `01_build_bfs.py` output (`{"states", "distances"}`).

        Every label in that file is EXACT -- levels up to `--max-exact` are a complete
        BFS, and the sampled d = max_exact + 1 shell is exact by construction (a state
        reachable in d+1 moves that is not in ball(<=d) is at distance exactly d+1). The
        shell being sampled rather than complete only reduces COVERAGE; it never makes a
        hit wrong.
        """
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        states = d["states"].to(device)
        depths = d["distances"].to(device)
        ztab = make_ztab(state_size, num_classes, seed=zseed, device=device)

        h = zhash(states, ztab)
        # Collapse any duplicate colourings in the table itself, keeping the smallest
        # depth, then sort for searchsorted.
        hu, inv = torch.unique(h, return_inverse=True)
        dmin = torch.full((hu.numel(),), _BIG_LABEL, dtype=torch.int64, device=device)
        dmin.scatter_reduce_(0, inv, depths.long(), reduce="amin", include_self=False)
        order = torch.argsort(hu)
        return cls(
            hu[order].contiguous(),
            dmin[order].contiguous(),
            ztab,
            do_dedup=do_dedup,
            do_override=do_override,
        )

    @property
    def table_size(self) -> int:
        return int(self.table_hashes.numel())

    def apply(
        self, states: torch.Tensor, labels: torch.Tensor, round_to: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, FixStats]:
        """Run pass A then pass B. Returns (states, labels, stats).

        `round_to`: truncate the surviving rows to a multiple of this many. PASS THE
        BATCH SIZE HERE WHEN `torch.compile` IS ON.

        Dedup removes a data-dependent number of rows, so without this the epoch's row
        count -- and therefore the size of the trailing partial batch -- changes every
        epoch. With `dynamic=False` that makes torch.compile re-trace every epoch:
        measured 0.3 s -> 8.0 s per epoch, a 27x regression. It is the training-side
        cousin of 06_GOTCHAS #7, and it bites here precisely because this change breaks
        the fixed-shape invariant training used to enjoy.

        Truncating to a multiple of the batch size makes every batch exactly
        `batch_size`, so exactly one shape is ever compiled. The cost is under one
        batch's worth of rows (<= 3.4% of an epoch) and it is not a real loss: walks are
        regenerated from scratch every epoch, so the dropped rows are resampled anyway.
        Rows are in Zobrist-hash order at this point, which is independent of depth, so
        the truncation is an unbiased subsample rather than a shallow/deep skew.
        """
        st = FixStats(generated=int(states.shape[0]))
        if self.do_dedup:
            states, labels = dedup_min_label(states, labels, self.ztab)
        st.after_dedup = int(states.shape[0])
        if self.do_override:
            labels, st.in_table, st.lowered = exact_override(
                states, labels, self.table_hashes, self.table_depths, self.ztab
            )
        if round_to:
            n = (states.shape[0] // round_to) * round_to
            states, labels = states[:n], labels[:n]
        st.kept = int(states.shape[0])
        return states, labels, st
