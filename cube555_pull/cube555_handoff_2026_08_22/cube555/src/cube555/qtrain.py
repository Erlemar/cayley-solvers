"""Sparse-Q training pieces for the 5x5x5 picture cube.

Ported from cube444/src/cube444/qtrain.py, which the playbook says to COPY, NOT REDESIGN.
Exactly three label sources survive there and only those three are here:

  1. sparse-Q rw-middle labels -- walk of length k ~ U[k_min, k_max], one pivot p inside,
     write Q(s, undo) = p-1 and Q(s, next) = p+1, masked MSE on those two columns.
  2. exact BFS anchors -- all 30 action-values exact, from a complete d<=D table.
  3. symmetry label expansion with RANDOM frames.

Everything measured and REJECTED on 444/tetraminx is absent by construction, not by a
zeroed weight: path labels (+3.07 moves/pid), sorted-profile and the whole
permutation-invariant family, qv-consistency, Bellman-from-scratch, margin terms.

THE ONE STRUCTURAL DIFFERENCE FROM cube444. This is a permutation, not a colouring, so a
frame is a conjugation `Minv[s[M]]` and there are no colour maps. `Symmetries555.verify`
checks the same transport identity `sym(s o a, M) == sym(s, M) o relabel[M, a]` that made
the 444 version trustworthy -- the slot-permutation-only form is legal-looking and wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Symmetries555:
    """The 48 verified conjugation frames, with action transport."""

    def __init__(self, data_dir: Path, device: str):
        d = Path(data_dir)
        self.M = torch.from_numpy(np.load(d / "sym_slots_48.npy")).to(device).long()
        self.Minv = (
            torch.from_numpy(np.load(d / "sym_slots_inv_48.npy")).to(device).long()
        )
        self.relabel = (
            torch.from_numpy(np.load(d / "sym_move_relabel_48.npy")).to(device).long()
        )
        self.n_sym, self.state_size = self.M.shape
        self.n_actions = self.relabel.shape[1]
        self.device = device
        ident = (self.M == torch.arange(self.state_size, device=device)).all(dim=1)
        if int(ident.sum()) != 1:
            raise AssertionError("expected exactly one identity frame")
        self.identity = int(ident.nonzero()[0])

    def conjugate(self, states: torch.Tensor, sym_ids: torch.Tensor) -> torch.Tensor:
        """sym(s, M_k)[i] = Minv_k[ s[ M_k[i] ] ], one k per row. Keeps input dtype."""
        gathered = torch.gather(states, 1, self.M.index_select(0, sym_ids))
        out = torch.gather(self.Minv.index_select(0, sym_ids), 1, gathered.long())
        return out.to(states.dtype)

    def verify(self, perms: torch.Tensor, solved: torch.Tensor, seed: int = 0) -> None:
        """sym(apply(s,a), M_k) == apply(sym(s,M_k), relabel[k,a]) on scrambled states."""
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        n = 8
        states = solved.unsqueeze(0).expand(n, -1).contiguous().clone()
        for _ in range(8):
            mv = torch.randint(self.n_actions, (n,), generator=g, device=self.device)
            states = torch.gather(states, 1, perms[mv])
        for k in range(self.n_sym):
            ks = torch.full((n,), k, dtype=torch.int64, device=self.device)
            for a in range(self.n_actions):
                av = torch.full((n,), a, dtype=torch.int64, device=self.device)
                lhs = self.conjugate(torch.gather(states, 1, perms[av]), ks)
                rhs = torch.gather(
                    self.conjugate(states, ks), 1, perms[self.relabel[k, av]]
                )
                if not torch.equal(lhs, rhs):
                    raise AssertionError(f"symmetry transport failed at k={k}, a={a}")
        print(
            f"  symmetry transport: {self.n_sym}x{self.n_actions} frames/actions OK",
            flush=True,
        )


def expand_labels_random(sym: Symmetries555, states, pivots, undo, nxt, used: int, gen):
    """Symmetry expansion with RANDOM frames per sample, not a greedy coverage prefix.

    The greedy table is keyed on (undo, nxt), so a given action pair draws identical
    frames every step for the whole run: column coverage with no state diversity. The
    cross-model comparison on 444 says state diversity is the binding constraint, so a
    small expansion budget is better spent on fresh conjugates. Row 0 is forced to the
    identity frame so `sparse_metrics` can be read off the untransformed rows.
    """
    b, dev = states.size(0), states.device
    k = torch.randint(0, sym.n_sym, (b, used), generator=gen, device=dev)
    k[:, 0] = sym.identity
    src = torch.arange(b, device=dev).view(-1, 1).expand(-1, used).reshape(-1)
    kf = k.reshape(-1)
    out_states = sym.conjugate(states.index_select(0, src), kf)
    total = out_states.size(0)
    # Q(sym(s,k), relabel[k,a]) == Q(s,a): the label rides to the transported column.
    col_u = sym.relabel[kf, undo.index_select(0, src)]
    col_n = sym.relabel[kf, nxt.index_select(0, src)]
    p = pivots.float().index_select(0, src)
    targets = torch.zeros((total, sym.n_actions), device=dev)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    rows = torch.arange(total, device=dev)
    targets[rows, col_u] = p - 1.0
    targets[rows, col_n] = p + 1.0
    mask[rows, col_u] = True
    mask[rows, col_n] = True
    return out_states, targets, mask, torch.arange(b, device=dev) * used


def sparse_metrics(pred: torch.Tensor, undo: torch.Tensor, nxt: torch.Tensor) -> dict:
    """Objective metrics on the UNTRANSFORMED (identity-frame) rows only."""
    rows = torch.arange(pred.size(0), device=pred.device)
    q_undo, q_next = pred[rows, undo], pred[rows, nxt]
    other = torch.ones_like(pred, dtype=torch.bool)
    other[rows, undo] = False
    other[rows, nxt] = False
    best_wrong = pred.masked_fill(~other, float("inf")).min(dim=1).values
    return {
        "pair_acc": q_undo.lt(q_next).float().mean().detach(),
        "top1_acc": q_undo.lt(best_wrong).float().mean().detach(),
        "gap": (q_next - q_undo).mean().detach(),
    }


class SparseQSampler:
    """k ~ U[k_min, k_max], then one pivot p inside the walk. Non-backtracking.

    NOT `every pivot of a fixed-length walk`: that makes pivot depth uniform on
    [1, k_max-1] and pushes a third of the mass past the mixing length, where the
    "gap is exactly 2" assertion is simply false. The shipped cube4 recipe is a VARIABLE
    walk length with ONE pivot inside it, and scoring the two readings against the
    author's recorded validation block picked this one by 4x on MSE.

    `pivot_tilt` biases the pivot deeper: p = (u^(1/(1+tilt))) * (k-1) + 1, so tilt 0.5
    moves E[p/(k-1)] from 0.50 to 0.60. tilt 0 is the author's uniform-inside-walk form.

    Measured on the 5x5x5 (12_mitm_oracle.py): P(exact d == k) is 1.00 for k<=3 and
    0.93-0.97 for k = 4..9, i.e. the labels carry a few per cent of noise even where they
    are cleanest. Deeper than the oracle reaches, that noise grows; k_max is set from the
    counting bound, not from this.
    """

    def __init__(self, perms, inverse_idx, solved, k_min, k_max, pivot_tilt, generator):
        if not 2 <= k_min <= k_max:
            raise ValueError(f"need 2 <= k_min <= k_max, got {k_min}, {k_max}")
        self.n_actions, self.state_size = perms.shape
        self.perms, self.inv, self.solved = perms, inverse_idx, solved
        self.k_min, self.k_max = k_min, k_max
        self.gamma = 1.0 / (1.0 + float(pivot_tilt))
        self.g = generator
        self.device = perms.device

    def sample(self, batch: int):
        dev = self.device
        lengths = torch.randint(
            self.k_min, self.k_max + 1, (batch,), generator=self.g, device=dev
        )
        u = torch.rand((batch,), generator=self.g, device=dev)
        pivots = (u.pow(self.gamma) * (lengths - 1).float()).long() + 1
        states = self.solved.unsqueeze(0).expand(batch, -1).clone()
        pivot_states = torch.empty_like(states)
        last = torch.full((batch,), -1, dtype=torch.int64, device=dev)
        undo = torch.full_like(last, -1)
        nxt = torch.full_like(last, -1)
        for step in range(self.k_max):
            active = lengths > step
            if step == 0:
                moves = torch.randint(
                    self.n_actions, (batch,), generator=self.g, device=dev
                )
            else:  # non-backtracking: skip the inverse of the previous move
                compact = torch.randint(
                    self.n_actions - 1, (batch,), generator=self.g, device=dev
                )
                forbidden = self.inv[last.clamp_min(0)]
                moves = compact + compact.ge(forbidden).long()
            successors = torch.gather(states, 1, self.perms[moves])
            at_pivot = pivots == step
            pivot_states = torch.where(at_pivot.unsqueeze(1), states, pivot_states)
            undo = torch.where(at_pivot, self.inv[last.clamp_min(0)], undo)
            nxt = torch.where(at_pivot, moves, nxt)
            states = torch.where(active.unsqueeze(1), successors, states)
            last = torch.where(active, moves, last)
        if int((undo < 0).sum()) or int((undo == nxt).sum()):
            raise AssertionError("sampler produced an unlabelled or degenerate pivot")
        return pivot_states, pivots, undo, nxt


class PivotBuffer:
    """Amortise the sampler, which is LATENCY-bound: the k_max-iteration walk loop
    dominates and barely notices the batch size. Refilling in large blocks and handing
    out slices yields the SAME states from the SAME distribution in the same order --
    only the kernel-launch count drops. At k_max 80 this matters more than it did on 444.
    """

    def __init__(self, sampler, block: int = 1 << 16):
        self.sampler, self.block = sampler, int(block)
        self._buf = None
        self._i = 0

    def take(self, n: int):
        if self._buf is None or self._i + n > self._buf[0].shape[0]:
            self._buf = self.sampler.sample(max(self.block, n))
            self._i = 0
        j = self._i
        self._i += n
        return tuple(t[j : j + n] for t in self._buf)


class ExactAnchors:
    """States at d<=D with all 30 action-values exact, plus the depth for the value head.

    Rule 9: exact anchors in every batch, or the bootstrap settles at V(solved) ~ 2.
    But do NOT scale the dose -- bigger and deeper anchor batches are 0-for-3 on 444
    (d6a, tmx, d8spec all lost the beam). 256 rows of ~8.4k is the measured-good dose.
    """

    def __init__(self, path: Path, device: str):
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        self.states = blob["states"].to(device)
        self.q = blob["q"].to(device).float()
        self.depth = blob["depth"].to(device).float()
        self.n = self.states.shape[0]
        self.max_depth = int(blob["max_depth"])
        print(
            f"  anchors: {self.n:,} states x {self.q.shape[1]} exact columns "
            f"(d<={self.max_depth})",
            flush=True,
        )

    def sample(self, n: int, g: torch.Generator):
        idx = torch.randint(self.n, (n,), generator=g, device=self.states.device)
        return (
            self.states.index_select(0, idx),
            self.q.index_select(0, idx),
            self.depth.index_select(0, idx),
        )
