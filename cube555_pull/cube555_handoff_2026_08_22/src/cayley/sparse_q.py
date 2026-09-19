"""Sparse-Q objective for cube444 -- cell B of the 2x2 in 04_TRAIN_SPARSE_Q.md.

THE OBJECTIVE. Random walk of length k from solved; pick a pivot at walk index p. Label
EXACTLY TWO of the 24 action columns on the pivot state:

    Q(s, undo_last_move) = p - 1
    Q(s, next_walk_move) = p + 1

MSE on those two entries only; the other 22 columns are unlabelled and contribute nothing.

WHY IT SHOULD BEAT WALK-DEPTH MSE. The V loss is MSE(V(s), k). At large k the conditional
variance of the true distance given s is huge, so the Bayes-optimal prediction shrinks
toward the mean and LOCAL DISCRIMINATION COLLAPSES -- measured on an IHES V model, mean
V(next) - V(undo) falls from 1.96 at depth 2 to 0.37 at depth 21 when it should be 2 at
every depth. The sparse-Q labels have ZERO conditional variance in their DIFFERENCE (it is
always exactly 2), so MSE cannot buy loss by flattening the gap. The absolute level is free
to saturate; the gap is not.

KEEP THE ABSOLUTE MSE TERM. Do not swap this for a pure pairwise ranking loss. The beam
takes a GLOBAL top-B over all (parent, action) pairs -- verified at
`khoruzhii_search.py:349`, `q_flat = q_all.reshape(-1)` -- so Q must be comparable ACROSS
parents. A ranking loss gives within-parent order and nothing else.

WHAT THIS MODULE DELIBERATELY DOES NOT PORT from the tetraminx reference
(`reference/tetraminx_51_train_sparse_q.py`):

  * The symmetry label-expansion (`coverage_table` / `expand_labels`). It conjugates the
    pivot by a symmetry frame so the two labelled columns land on different action indices,
    labelling more of the 24 per sample. It is a pure augmentation, and on a COLOUR cube it
    needs the RECOLOUR form sym(s,R) = color_map_R[s[rotation_R]] -- the reference's
    slot-permutation-only `conjugate` is silently wrong here (06_GOTCHAS #2). Left out of
    v1 so cell B measures the OBJECTIVE, not an augmentation.
  * Path labels from solver traces, the value head, and the "box"/sorted-profile loss --
    none are part of the objective the 2x2 is testing.

The margin term is implemented but defaults to 0.0, matching every config the upstream repo
ships.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cayley.data import GeneratorTable


def generate_walks_with_actions(
    puzzle,
    n_walks: int,
    k_max: int,
    seed: int,
    device: str = "cuda",
    n_back: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-backtracking walks that also return the ACTION taken at each step.

    `cayley.data.generate_walks_torch` returns (states, depths) only. Sparse-Q needs the
    actions too: the undo label is the inverse of the action that ENTERED the pivot, and
    the next label is the action that LEAVES it.

    Returns, with S = state_size:
        states  (n_walks, k_max + 1, S) int64 -- states[:, i] is the state after i moves,
                                                 so states[:, 0] is solved
        actions (n_walks, k_max)        int64 -- actions[:, i] took state i -> state i+1
        inv     (n_gen,)                int64 -- inverse action index
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(device)
    inv = torch.from_numpy(gens.inverse_idx).to(device)
    n_gen, state_size = perms.shape

    cur = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
    cur = cur.unsqueeze(0).expand(n_walks, state_size).clone()

    states = torch.empty(
        (n_walks, k_max + 1, state_size), dtype=torch.int64, device=device
    )
    actions = torch.empty((n_walks, k_max), dtype=torch.int64, device=device)
    states[:, 0] = cur

    history = torch.full(
        (n_walks, max(n_back, 1)), -1, dtype=torch.int64, device=device
    )
    for i in range(k_max):
        a = torch.randint(0, n_gen, (n_walks,), generator=g, device=device)
        if n_back > 0:
            hist_safe = torch.where(history >= 0, history, torch.zeros_like(history))
            banned = torch.where(
                history >= 0, inv[hist_safe], torch.full_like(hist_safe, -1)
            )
            for _ in range(6):
                bad = (a.unsqueeze(1) == banned).any(dim=1)
                if not bool(bad.any()):
                    break
                a = a.clone()
                a[bad] = torch.randint(
                    0, n_gen, (int(bad.sum()),), generator=g, device=device
                )
            history = torch.cat([history[:, 1:], a.unsqueeze(1)], dim=1)
        cur = torch.gather(cur, 1, perms[a])
        states[:, i + 1] = cur
        actions[:, i] = a
    return states, actions, inv


def sparse_q_batch(
    puzzle,
    n_walks: int,
    k_max: int,
    seed: int,
    device: str = "cuda",
    n_back: int = 1,
    p_min: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one sparse-Q training batch.

    A pivot needs BOTH children to exist, so p ranges over [p_min, k_max - 1]: the undo
    label refers to step p-1 and the next label to step p+1.

    Returns:
        pivots     (N, S)  int64 -- the pivot states
        col_undo   (N,)    int64 -- action column carrying the p-1 label
        col_next   (N,)    int64 -- action column carrying the p+1 label
        p          (N,)    int64 -- the pivot's walk index
    """
    states, actions, inv = generate_walks_with_actions(
        puzzle, n_walks, k_max, seed, device=device, n_back=n_back
    )
    ps = torch.arange(p_min, k_max, device=device)  # valid pivot indices
    piv = states[:, ps].reshape(-1, states.shape[2])  # (n_walks * P, S)
    # action that ENTERED pivot p is actions[:, p-1]; undoing it is its inverse
    a_in = actions[:, ps - 1].reshape(-1)
    a_out = actions[:, ps].reshape(-1)  # action that LEAVES pivot p
    col_undo = inv[a_in]
    col_next = a_out
    p = ps.unsqueeze(0).expand(states.shape[0], -1).reshape(-1)
    return piv, col_undo, col_next, p


def sparse_q_loss(
    pred: torch.Tensor,
    col_undo: torch.Tensor,
    col_next: torch.Tensor,
    p: torch.Tensor,
    margin: float = 1.0,
    margin_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Masked MSE on the two labelled columns, plus optional top-1 margin.

    `pred` is (N, 24). Only columns `col_undo` and `col_next` carry a target; the other 22
    contribute nothing to the gradient. Returns (loss, metrics).
    """
    rows = torch.arange(pred.shape[0], device=pred.device)
    q_undo = pred[rows, col_undo]
    q_next = pred[rows, col_next]
    t_undo = (p - 1).to(pred.dtype)
    t_next = (p + 1).to(pred.dtype)

    sparse = 0.5 * ((q_undo - t_undo).pow(2).mean() + (q_next - t_next).pow(2).mean())
    loss = sparse

    # Top-1 margin: the undo action should score below every UNLABELLED action, not merely
    # below `next`. Off by default (margin_weight=0.0) -- upstream ships it disabled.
    other = torch.ones_like(pred, dtype=torch.bool)
    other[rows, col_undo] = False
    other[rows, col_next] = False
    best_wrong = pred.masked_fill(~other, float("inf")).min(dim=1).values
    margin_loss = torch.relu(margin - (best_wrong - q_undo)).pow(2).mean()
    if margin_weight:
        loss = loss + margin_weight * margin_loss

    with torch.no_grad():
        met = {
            "sparse_mse": float(sparse),
            "margin_loss": float(margin_loss),
            # the two properties the beam actually depends on
            "pair_acc": float(q_undo.lt(q_next).float().mean()),
            "top1_acc": float(q_undo.lt(best_wrong).float().mean()),
            # label asserts exactly 2; watch this NOT collapse -- that is the whole point
            "gap": float((q_next - q_undo).mean()),
        }
    return loss, met


@dataclass
class QAnchors:
    """Exact 24-way Q labels: Q(s, a) = d(apply(s, a)), from the complete BFS ball.

    Anchors were load-bearing for the V recipe (rule 9: without them the Bellman bootstrap
    settles at V(solved) ~ 2). The Q analogue is stronger, because here we can label ALL
    24 columns exactly rather than just the pivot's two.

    DEPTH CAP. Children of a depth-d state are at depth <= d+1, so every child is inside a
    table that is COMPLETE to d+1. `01_build_bfs.py --max-exact 5` gives a complete ball to
    d=5 (the d=6 shell it writes is SAMPLED, ~3.4% of 63.5M, so it cannot be relied on for
    coverage). Hence anchors are drawn from d <= 4 -- 182,407 states, all 24 children
    guaranteed present. `build` asserts that rather than trusting it.
    """

    states: torch.Tensor  # (N, S) int64, depth <= max_depth
    q: torch.Tensor  # (N, 24) float32, exact

    @classmethod
    def build(
        cls, anchor_path, puzzle, fixer, device: str = "cuda", max_depth: int = 4
    ) -> "QAnchors":
        """`fixer` is a cayley.label_fix.LabelFixer holding the sorted d<=6 hash table."""
        from cayley.label_fix import zhash

        blob = torch.load(str(anchor_path), map_location="cpu", weights_only=False)
        st, di = blob["states"], blob["distances"]
        keep = di <= max_depth
        states = st[keep].to(device).long()
        gens = GeneratorTable.from_puzzle(puzzle)
        perms = torch.from_numpy(gens.perms).to(device)
        n_gen = perms.shape[0]

        q = torch.empty((states.shape[0], n_gen), dtype=torch.float32, device=device)
        for a in range(n_gen):
            child = torch.gather(
                states, 1, perms[a].unsqueeze(0).expand(states.shape[0], -1)
            )
            h = zhash(child, fixer.ztab)
            pos = torch.searchsorted(fixer.table_hashes, h).clamp_(
                max=fixer.table_hashes.numel() - 1
            )
            hit = fixer.table_hashes[pos] == h
            if not bool(hit.all()):
                raise AssertionError(
                    f"action {a}: {int((~hit).sum()):,} of {hit.numel():,} children of a "
                    f"d<={max_depth} state are missing from the exact table. The table is "
                    f"not complete to d={max_depth + 1} -- rebuild with "
                    f"01_build_bfs.py --max-exact {max_depth + 1}, or lower max_depth."
                )
            q[:, a] = fixer.table_depths[pos].float()
        return cls(states=states, q=q)

    def sample(self, n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(
            0, self.states.shape[0], (n,), generator=g, device=self.states.device
        )
        return self.states.index_select(0, idx), self.q.index_select(0, idx)
