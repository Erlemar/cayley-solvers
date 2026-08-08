"""Sparse-Q trainer for the Professor Tetraminx (all-neighbours Q head).

Port of Vlad Kuznetsov's "random-walk middle sparse-Q" objective
(github.com/AnanasClassic/cayleypy-training-core) onto our ResMLP trunk, with
four additions of our own.

THE OBJECTIVE
-------------
Walk k steps from solved (non-backtracking), pick a pivot p in the middle, and
label exactly TWO of the 24 actions on the pivot state:

    Q(s, undo_last_move) = p - 1        Q(s, next_walk_move) = p + 1

Why this beats regressing V against walk depth: our V loss is MSE(V(s), k), and
at large k the conditional variance of the true distance is big, so the
MSE-optimal prediction shrinks toward the mean and the *local* discrimination
flattens. Here the two labels always differ by exactly 2 with zero conditional
variance, so the loss cannot be reduced by flattening the gap -- only the
absolute level is free to saturate. Measured on tv0_bellman ep24, the V's
mean gap V(next)-V(undo) decays 1.91 at depth 2 to 0.75-1.03 at depth 22-28.

OUR ADDITIONS
-------------
1. Exact-Q anchors from the BFS tables. For an anchor state at d <= 5 every one
   of its 24 children is at d <= 6 and therefore present in the endgame hash
   table, so we can supervise ALL 24 outputs with TRUE distances. Fixes the
   regime where walk-index labels are most biased and where endgame decides
   path length. Costs nothing -- the data is already built.
2. Symmetry-expanded label coverage. Our 24 spatial frames map one label pair
   onto other action columns. Note the group permutes the 4 axes but NOT the 3
   layers, so an action's orbit has 8 members and a pair covers at most 16 of
   24 columns -- the script prints the coverage it actually achieved.
3. Depth-tilted pivot sampling. The plain scheme makes shallow pivots far more
   likely; tetraminx paths are 28-36 long, so we tilt toward depth.
4. The top-1 margin loss is available and OFF in every config the upstream repo
   ships (`top1_margin_weight: 0.0`); here it is a first-class knob.

The trunk keys (embedding./input_stack./res_blocks.) are byte-compatible with
`cayley.gflow_model.ResMLPGFlowNet`, and the 24-wide head has exactly the shape
of its `policy_head`, so an AZ value head can be bolted on later with no shape
change (see 21_train_az.py::warmstart_from_v_model for the mirror of this).

    python3 tetraminx/scripts/51_train_sparse_q.py \
        --config tetraminx/configs/tq0_sparse_q.yaml \
        --output tetraminx/models/tq0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.models import build_model, count_parameters
from tetraminx.puzzle import Tetraminx


# --------------------------------------------------------------------------
# symmetry
# --------------------------------------------------------------------------
class Symmetries:
    """The 24 spatial relabelings, with the conjugation and action-transport maps.

    Convention from 01_build_symmetries.py:
        conj(s)[i] = P_inv[s[P[i]]]        and       apply(conj(s), m) = conj(apply(s, sigma(m)))
    so an action `a` labelled on `s` becomes action `sigma^-1(a)` on `conj(s)`.
    """

    def __init__(self, data_dir: Path, device: str):
        P = np.load(data_dir / "tetra_symmetries.npy").astype(np.int64)
        P_inv = np.load(data_dir / "tetra_symmetries_inv.npy").astype(np.int64)
        sigma = np.load(data_dir / "tetra_move_relabel.npy").astype(np.int64)
        sigma_inv = np.zeros_like(sigma)
        rows = np.arange(sigma.shape[1])
        for k in range(sigma.shape[0]):
            sigma_inv[k, sigma[k]] = rows
        self.n_sym, self.state_size = P.shape
        self.n_actions = sigma.shape[1]
        self.P = torch.from_numpy(P).to(device)
        self.P_inv = torch.from_numpy(P_inv).to(device)
        self.sigma = torch.from_numpy(sigma).to(device)
        self.sigma_inv = torch.from_numpy(sigma_inv).to(device)
        ident = (sigma == rows).all(axis=1).nonzero()[0]
        assert ident.size == 1, "expected exactly one identity symmetry"
        self.identity = int(ident[0])

    def conjugate(self, states: torch.Tensor, sym_ids: torch.Tensor) -> torch.Tensor:
        """conj_k(s)[i] = P_inv[k][ s[ P[k][i] ] ], one k per row."""
        gathered = torch.gather(states, 1, self.P.index_select(0, sym_ids))
        return torch.gather(self.P_inv.index_select(0, sym_ids), 1, gathered)

    def verify(self, generators: torch.Tensor, seed: int = 0) -> None:
        """Assert apply(conj_k(s), sigma_inv_k(a)) == conj_k(apply(s, a)) for all k, a."""
        g = torch.Generator(device=generators.device)
        g.manual_seed(seed)
        n_probe = 8
        base = torch.randperm(self.state_size, generator=g, device=generators.device)
        states = base.unsqueeze(0).expand(n_probe, -1).contiguous()
        for _ in range(4):  # scramble the probe states a little
            mv = torch.randint(self.n_actions, (n_probe,), generator=g, device=generators.device)
            states = torch.gather(states, 1, generators[mv])
        for k in range(self.n_sym):
            ks = torch.full((n_probe,), k, dtype=torch.int64, device=generators.device)
            for a in range(self.n_actions):
                av = torch.full((n_probe,), a, dtype=torch.int64, device=generators.device)
                lhs = torch.gather(self.conjugate(states, ks), 1,
                                   generators[self.sigma_inv[k, av]])
                rhs = self.conjugate(torch.gather(states, 1, generators[av]), ks)
                assert torch.equal(lhs, rhs), f"symmetry transport failed at k={k}, a={a}"

    def coverage_table(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Greedy per-(prev,next) list of frames whose images cover fresh action columns.

        Returns (sym_ids, columns, sides, width) with
            sym_ids [A, A, W]        frame to apply for row w
            columns [A, A, W, 2]     action column to label (padded)
            sides   [A, A, W, 2]     0 = the p-1 label, 1 = the p+1 label; -1 = unused
        Row 0 is always the identity frame, so row 0 of each group is the original sample.
        """
        A, K = self.n_actions, self.n_sym
        sinv = self.sigma_inv.cpu().numpy()
        order = [self.identity] + [k for k in range(K) if k != self.identity]
        rows_by_pair: dict[tuple[int, int], list] = {}
        for prev in range(A):
            for nxt in range(A):
                if prev == nxt:
                    continue
                seen = set()
                rows = []
                for k in order:
                    fresh = [(int(sinv[k, col]), side)
                             for col, side in ((prev, 0), (nxt, 1))
                             if int(sinv[k, col]) not in seen]
                    if fresh:
                        rows.append((k, fresh))
                        seen.update(c for c, _ in fresh)
                    if len(seen) == A:
                        break
                rows_by_pair[(prev, nxt)] = rows
        width = max(len(r) for r in rows_by_pair.values())
        covered = [len({c for _, fr in r for c, _ in fr}) for r in rows_by_pair.values()]
        sym_ids = torch.full((A, A, width), self.identity, dtype=torch.int64)
        columns = torch.zeros((A, A, width, 2), dtype=torch.int64)
        sides = torch.full((A, A, width, 2), -1, dtype=torch.int64)
        for (prev, nxt), rows in rows_by_pair.items():
            for w, (k, fresh) in enumerate(rows):
                sym_ids[prev, nxt, w] = k
                for slot, (col, side) in enumerate(fresh):
                    columns[prev, nxt, w, slot] = col
                    sides[prev, nxt, w, slot] = side
        dev = self.sigma.device
        print(f"  symmetry coverage: width {width} rows/sample, "
              f"columns covered {min(covered)}-{max(covered)} of {A}", flush=True)
        return sym_ids.to(dev), columns.to(dev), sides.to(dev), width


# --------------------------------------------------------------------------
# samplers
# --------------------------------------------------------------------------
class SparseQSampler:
    """Random-walk-middle sparse-Q sampler, entirely on the training device."""

    def __init__(self, generators: torch.Tensor, inverse_idx: torch.Tensor,
                 solved: torch.Tensor, k_min: int, k_max: int, pivot_tilt: float,
                 generator: torch.Generator):
        assert 2 <= k_min <= k_max
        self.A, self.S = generators.shape
        self.gen = generators
        self.inv = inverse_idx
        self.solved = solved
        self.k_min, self.k_max = k_min, k_max
        self.gamma = 1.0 / (1.0 + float(pivot_tilt))
        self.g = generator
        self.device = generators.device

    def sample(self, batch: int):
        dev = self.device
        lengths = torch.randint(self.k_min, self.k_max + 1, (batch,), generator=self.g, device=dev)
        u = torch.rand((batch,), generator=self.g, device=dev)
        pivots = (u.pow(self.gamma) * (lengths - 1).float()).long() + 1
        states = self.solved.unsqueeze(0).expand(batch, -1).clone()
        pivot_states = torch.empty_like(states)
        last = torch.full((batch,), -1, dtype=torch.int64, device=dev)
        prev = torch.full_like(last, -1)
        nxt = torch.full_like(last, -1)
        for step in range(self.k_max):
            active = lengths > step
            if step == 0:
                moves = torch.randint(self.A, (batch,), generator=self.g, device=dev)
            else:  # non-backtracking: skip the inverse of the previous move
                compact = torch.randint(self.A - 1, (batch,), generator=self.g, device=dev)
                forbidden = self.inv[last.clamp_min(0)]
                moves = compact + compact.ge(forbidden).long()
            successors = torch.gather(states, 1, self.gen[moves])
            at_pivot = pivots == step
            pivot_states = torch.where(at_pivot.unsqueeze(1), states, pivot_states)
            prev = torch.where(at_pivot, self.inv[last.clamp_min(0)], prev)
            nxt = torch.where(at_pivot, moves, nxt)
            states = torch.where(active.unsqueeze(1), successors, states)
            last = torch.where(active, moves, last)
        assert int((prev < 0).sum()) == 0 and int((prev == nxt).sum()) == 0
        return pivot_states, pivots, prev, nxt


class PathLabels:
    """Near-optimal-path labels: Q(s, on_path_action) = remaining_moves - 1.

    This exists to fix the one measured weakness of the random-walk labels. Past the
    diameter a walk is near stationary, so (a) its index over-states the true distance
    and (b) "undo the last move" stops being reliably distance-reducing. Measured at
    epoch 250: the Q's mean gap fell to 0.216 in the pivot 30-40 band. Path samples do
    not have that problem -- every one comes from a replay-verified solving path, so
    the target is a tight upper bound on the child's true distance (our reference is
    28.2 moves/pid against a ~28 counting bound, i.e. slack under ~0.2) at exactly the
    depths the beam actually operates at.

    Deliberately supervised with MSE on the on-path entry rather than a "this move must
    rank first" penalty. Forcing the on-path move to be strictly best is what tied
    m_rank_v0 on megaminx: the frontier-regret probe showed 56.9 percent of such
    disagreements were benign ALTERNATIVE OPTIMA, so the ranking penalty just teaches
    arbitrary path preference. An accurate value on one entry opens the gap against the
    walk-labelled entries by itself, without asserting anything false. The ranking form
    is kept as `path_rank_weight`, default 0, for ablation.
    """

    def __init__(self, path: Path, n_actions: int, device: str):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.states = blob["states"].to(device)                      # uint8, keep narrow
        self.actions = blob["actions"].to(device).long()
        self.targets = blob["dists"].to(device).float() - 1.0
        assert self.actions.max() < n_actions
        meta = blob.get("meta", {})
        print(f"  path labels: {self.states.size(0):,} samples from {meta.get('floor', path.name)}"
              f" (target range {self.targets.min():.0f}..{self.targets.max():.0f})", flush=True)

    def sample(self, batch: int, g: torch.Generator):
        idx = torch.randint(self.states.size(0), (batch,), generator=g, device=self.actions.device)
        return (self.states.index_select(0, idx).long(),
                self.actions.index_select(0, idx),
                self.targets.index_select(0, idx))


class BFSAnchors:
    """Exact 24-way Q labels from the BFS endgame hash table.

    Anchor depth is capped at table_depth - 1: every child of a d <= k state is
    at d <= k+1, so all 24 must be inside a table of depth k+1 (the assert in
    `sample` enforces it).  With the d6 table that cap is 5 (1.77M states); with
    the d7 table it is 6 (27.8M).

    Sizing, because the d7 pair does not fit on a 24 GB card the naive way:
    anchor STATES live on the host as int8 (27.8M x 88 = 2.4 GB; as device
    int64 they would be 19.6 GB) and only the sampled batch is moved.  The
    TABLE stays on the device -- the per-step lookup is 24x the batch, so it is
    the hot path -- with depths as int8 (0.4 GB, not 3.5 GB as int64).
    """

    def __init__(self, train_path: Path, endgame_path: Path, generators: torch.Tensor,
                 max_anchor_depth: int, device: str):
        blob = torch.load(train_path, map_location="cpu", weights_only=False)
        states, dists = blob["states"], blob["distances"]
        keep = dists <= max_anchor_depth
        # host-resident, narrow dtype; sample() moves 512 rows per step
        self.states = states[keep].contiguous()
        self.depths = dists[keep].contiguous()
        self.device = device
        z = np.load(endgame_path)
        self.hashes = torch.from_numpy(z["hashes"]).to(device)
        self.table_depths = torch.from_numpy(z["depths"]).to(device)
        self.ztab = torch.from_numpy(z["ztab"]).to(device)
        self.max_depth = int(z["max_depth"])
        self.gen = generators
        self.A = generators.shape[0]
        assert bool(torch.all(self.hashes[1:] > self.hashes[:-1])), "endgame hashes must be sorted"
        assert max_anchor_depth < self.max_depth, (
            f"max_anchor_depth={max_anchor_depth} needs a table of depth "
            f">={max_anchor_depth + 1}, got d{self.max_depth}")
        gb = (self.hashes.numel() * 8 + self.table_depths.numel()) / 2**30
        print(f"  bfs anchors: {self.states.size(0):,} states at d<={max_anchor_depth} "
              f"(host, {self.states.numel() / 2**30:.2f} GB int8); endgame table "
              f"{self.hashes.numel():,} states at d<={self.max_depth} "
              f"({gb:.2f} GB on {device})", flush=True)

    def _hash(self, states: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(states.size(0), dtype=torch.int64, device=states.device)
        for i in range(states.size(1)):
            out = torch.bitwise_xor(out, self.ztab[i].index_select(0, states[:, i]))
        return out

    def sample(self, batch: int, g: torch.Generator):
        idx = torch.randint(self.states.size(0), (batch,), generator=g, device=g.device)
        idx_h = idx.to("cpu")
        s = self.states.index_select(0, idx_h).to(self.device).long()
        d0 = self.depths.index_select(0, idx_h).to(self.device).float()
        children = torch.gather(s.unsqueeze(1).expand(-1, self.A, -1), 2,
                                self.gen.unsqueeze(0).expand(s.size(0), -1, -1))
        flat = children.reshape(-1, s.size(1))
        h = self._hash(flat)
        pos = torch.searchsorted(self.hashes, h).clamp_max(self.hashes.numel() - 1)
        hit = self.hashes.index_select(0, pos) == h
        assert bool(hit.all()), "an anchor child was missing from the endgame table"
        targets = self.table_depths.index_select(0, pos).view(s.size(0), self.A).float()
        return s, targets, d0


class BakedAnchors:
    """Exact 24-way Q labels precomputed by 03b_build_bfs_deep.py --bake-out.

    Same `sample` contract as BFSAnchors, but there is no endgame table at train
    time -- the 24 targets are already stored per anchor. That is what makes deep
    anchors usable: the d7 hash table is 3.63 GB of a 23 GB card (a d8 one would be
    54 GB and fit nowhere), while the baked file is states+targets only.

    It also reaches a level deeper than the table it came from. The generator set is
    closed under inverse, so a child missing from a d<=k table is at exactly k+1;
    the builder resolves those, giving exact d<=7 anchors out of the d7 table.
    """

    def __init__(self, path: Path, device: str):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.states = blob["states"].contiguous()          # int8 (N, 88)
        self.q = blob["q_targets"].contiguous()            # int8 (N, 24)
        self.depths = blob["depths"].contiguous()          # int8 (N,)
        self.device = device
        assert self.q.size(0) == self.states.size(0)
        d = self.depths.to(torch.int64)
        hist = {int(k): int(v) for k, v in zip(*torch.unique(d, return_counts=True))}
        gb = (self.states.numel() + self.q.numel()) / 2**30
        print(f"  baked anchors: {self.states.size(0):,} states from a d"
              f"{int(blob['table_depth'])} table ({gb:.2f} GB host, no table needed); "
              f"per-depth {hist}", flush=True)

    def sample(self, batch: int, g: torch.Generator):
        idx = torch.randint(self.states.size(0), (batch,), generator=g, device=g.device)
        ih = idx.to("cpu")
        s = self.states.index_select(0, ih).to(self.device).long()
        t = self.q.index_select(0, ih).to(self.device).float()
        d0 = self.depths.index_select(0, ih).to(self.device).float()
        return s, t, d0


# --------------------------------------------------------------------------
# batch assembly + loss
# --------------------------------------------------------------------------
def expand_labels(sym, tables, states, pivots, prev, nxt, used):
    """Apply `used` symmetry rows per sample; return (states, targets, mask, identity_rows)."""
    sym_ids_t, columns_t, sides_t = tables
    B = states.size(0)
    sym_ids = sym_ids_t[prev, nxt, :used]                       # (B, used)
    columns = columns_t[prev, nxt, :used]                       # (B, used, 2)
    sides = sides_t[prev, nxt, :used]                           # (B, used, 2)
    src = torch.arange(B, device=states.device).view(-1, 1).expand(-1, used).reshape(-1)
    out_states = sym.conjugate(states.index_select(0, src), sym_ids.reshape(-1))
    total = out_states.size(0)
    targets = torch.zeros((total, sym.n_actions), device=states.device)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    flat_sides = sides.reshape(total, 2)
    live = flat_sides >= 0
    rows = torch.arange(total, device=states.device).view(-1, 1).expand(-1, 2)[live]
    cols = columns.reshape(total, 2)[live]
    side = flat_sides[live]
    piv = pivots.float().index_select(0, src)[rows]
    targets[rows, cols] = torch.where(side == 0, piv - 1.0, piv + 1.0)
    mask[rows, cols] = True
    identity_rows = torch.arange(B, device=states.device) * used
    return out_states, targets, mask, identity_rows


def sparse_metrics(pred, prev, nxt, margin):
    """Upstream objective.py metrics, computed on the untransformed rows."""
    rows = torch.arange(pred.size(0), device=pred.device)
    q_prev, q_next = pred[rows, prev], pred[rows, nxt]
    other = torch.ones_like(pred, dtype=torch.bool)
    other[rows, prev] = False
    other[rows, nxt] = False
    best_wrong = pred.masked_fill(~other, float("inf")).min(dim=1).values
    margin_loss = torch.relu(margin - (best_wrong - q_prev)).pow(2).mean()
    return {
        "pair_acc": q_prev.lt(q_next).float().mean(),
        "top1_acc": q_prev.lt(best_wrong).float().mean(),
        "gap": (q_next - q_prev).mean().detach(),
        "margin_loss": margin_loss,
    }


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Train an all-neighbours sparse-Q head for Tetraminx.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--set", action="append", default=[], metavar="DOTTED.PATH=JSON",
                    help="override a config field, e.g. training.batch_size=2048")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set needs dotted.path=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cursor = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    mcfg, tcfg = cfg["model"], cfg["training"]
    if args.epochs is not None:
        tcfg["n_epochs"] = args.epochs
    dev = args.device
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.get("seed", 0))
    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    puzzle.verify_inverse_pairs()
    names = list(puzzle.move_names)
    A = len(names)
    gen = torch.tensor([puzzle.generators[n] for n in names], dtype=torch.int64, device=dev)
    inv_idx = torch.tensor([names.index(puzzle.inverse_name(n)) for n in names],
                           dtype=torch.int64, device=dev)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=dev)

    print("=" * 72, flush=True)
    print(f"sparse-Q tetraminx | device {dev} | actions {A}", flush=True)
    print(f"config: {json.dumps({'model': mcfg, 'training': tcfg}, sort_keys=True)}", flush=True)

    sym = Symmetries(args.data_dir, dev)
    if tcfg.get("verify_symmetry", True):
        sym.verify(gen)
        print("  symmetry transport verified over all 24 frames x 24 actions", flush=True)
    use_sym = bool(tcfg.get("sym_coverage", True))
    if use_sym:
        sid, scol, sside, width = sym.coverage_table()
        tables = (sid, scol, sside)
        used = width if int(tcfg.get("sym_rows", 0)) <= 0 else min(int(tcfg["sym_rows"]), width)
    else:
        tables, used = None, 1
    print(f"  symmetry rows used per sample: {used}", flush=True)

    anchors = None
    n_anchor = int(tcfg.get("anchor_batch", 0))
    if n_anchor > 0:
        baked = tcfg.get("anchor_baked")
        if baked:
            anchors = BakedAnchors(args.data_dir / baked, dev)
        else:
            anchors = BFSAnchors(
                args.data_dir / tcfg.get("anchor_states", "bfs_d6_train.pt"),
                args.data_dir / tcfg.get("anchor_table", "bfs_endgame.npz"), gen,
                int(tcfg.get("max_anchor_depth", 5)), dev)

    paths = None
    n_path = int(tcfg.get("path_batch", 0))
    if n_path > 0 and tcfg.get("path_dataset"):
        p = Path(tcfg["path_dataset"])
        paths = PathLabels(p if p.is_absolute() else PROJECT / p, A, dev)
    else:
        n_path = 0

    arch = str(mcfg.get("arch", "resmlp"))
    az_head = bool(mcfg.get("az_head", False))
    if arch == "transformer":
        model = build_model(
            "transformer", az_head=az_head,
            layout_path=mcfg.get("layout_path", str(args.data_dir / "piece_layout.json")),
            state_size=mcfg["state_size"], num_classes=mcfg["num_classes"], n_actions=A,
            d_model=int(mcfg.get("d_model", 256)), nhead=int(mcfg.get("nhead", 8)),
            num_layers=int(mcfg.get("num_layers", 4)), ff_dim=int(mcfg.get("ff_dim", 1024)),
            dropout=float(mcfg.get("dropout", 0.0)),
        ).to(dev)
    elif arch == "latent_relational":
        model = build_model(
            "latent_relational", az_head=az_head,
            layout_path=mcfg.get("layout_path", str(args.data_dir / "piece_layout.json")),
            state_size=mcfg["state_size"], num_classes=mcfg["num_classes"], n_actions=A,
            d_model=int(mcfg.get("d_model", 96)), nhead=int(mcfg.get("nhead", 4)),
            num_latents=int(mcfg.get("num_latents", 8)),
            num_layers=int(mcfg.get("num_layers", 2)),
            ff_dim=int(mcfg.get("ff_dim", 192)),
            action_ff_dim=int(mcfg.get("action_ff_dim", mcfg.get("ff_dim", 192))),
            dropout=float(mcfg.get("dropout", 0.0)),
            activation=str(mcfg.get("activation", "silu")),
        ).to(dev)
    elif arch == "generator_isab":
        model = build_model(
            "generator_isab", az_head=az_head,
            layout_path=mcfg.get("layout_path", str(args.data_dir / "piece_layout.json")),
            generator_path=mcfg.get("generator_path"),
            state_size=mcfg["state_size"], num_classes=mcfg["num_classes"], n_actions=A,
            hidden_dims=tuple(mcfg.get("hidden_dims", (2048, 512))),
            num_res_blocks=int(mcfg.get("num_res_blocks", 2)),
            embed_dim=int(mcfg.get("embed_dim", 16)),
            d_model=int(mcfg.get("d_model", 128)), nhead=int(mcfg.get("nhead", 4)),
            num_latents=int(mcfg.get("num_latents", 12)),
            num_layers=int(mcfg.get("num_layers", 2)),
            ff_dim=int(mcfg.get("ff_dim", 256)),
            action_ff_dim=int(mcfg.get("action_ff_dim", mcfg.get("ff_dim", 256))),
            dropout=float(mcfg.get("dropout", 0.0)),
            activation=str(mcfg.get("activation", "silu")),
        ).to(dev)
    else:
        model = build_model(
            arch, az_head=az_head,
            state_size=mcfg["state_size"], num_classes=mcfg["num_classes"],
            hidden_dims=tuple(mcfg["hidden_dims"]), num_res_blocks=mcfg["num_res_blocks"],
            embed_dim=mcfg.get("embed_dim", 16), n_actions=A,
        ).to(dev)
    start_epoch, best = 1, float("inf")
    if args.resume is None:
        warmstart = tcfg.get("warmstart_base")
        if warmstart:
            warmstart_path = Path(warmstart)
            if not warmstart_path.is_absolute():
                warmstart_path = PROJECT / warmstart_path
            if not warmstart_path.exists():
                raise FileNotFoundError(f"warm-start checkpoint not found: {warmstart_path}")
            ck = torch.load(warmstart_path, map_location=dev, weights_only=False)
            source = ck.get("state_dict", ck)
            source = {k.removeprefix("_orig_mod."): v for k, v in source.items()}
            incompatible = model.load_state_dict(source, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"unexpected warm-start keys: {incompatible.unexpected_keys[:20]}")
            base_prefixes = tuple(f"{name}." for name in getattr(model, "_BASE_MODULES", ()))
            missing_base = [k for k in incompatible.missing_keys if k.startswith(base_prefixes)]
            if missing_base:
                raise RuntimeError(f"warm-start missed base parameters: {missing_base[:20]}")
            print(f"  warm-started base from {warmstart_path} "
                  f"({len(incompatible.missing_keys)} new-branch tensors retained)", flush=True)

    if bool(tcfg.get("freeze_base", False)):
        if not hasattr(model, "freeze_base"):
            raise ValueError(f"freeze_base is unsupported for architecture {arch!r}")
        model.freeze_base()
        print("  frozen ResMLP base; optimizing relational correction only", flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    print(f"  arch: {arch} | az_head: {az_head} | params: {total_params:,} total / "
          f"{trainable_params:,} trainable", flush=True)
    if trainable_params == 0:
        raise ValueError("model has no trainable parameters")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg.get("weight_decay", 0.0)),
                            fused=bool(tcfg.get("fused_optimizer", False)) and dev.startswith("cuda"))
    if args.resume is not None and args.resume.exists():
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()})
        opt.load_state_dict(ck["optimizer"])
        start_epoch, best = int(ck["epoch"]) + 1, float(ck.get("best", float("inf")))
        print(f"  resumed from {args.resume} at epoch {start_epoch}", flush=True)

    model.return_value = az_head          # single-trunk dual output while training
    net = model
    if bool(tcfg.get("compile_model", False)):
        net = torch.compile(model, dynamic=False)

    g_train = torch.Generator(device=dev); g_train.manual_seed(cfg.get("seed", 0))
    g_val = torch.Generator(device=dev); g_val.manual_seed(cfg.get("seed", 0) + 1_000_000)
    k_min, k_max = int(tcfg["k_min"]), int(tcfg["k_max"])
    tilt = float(tcfg.get("pivot_tilt", 0.0))
    sampler = SparseQSampler(gen, inv_idx, solved, k_min, k_max, tilt, g_train)
    val_sampler = SparseQSampler(gen, inv_idx, solved, k_min, k_max, tilt, g_val)
    base_batch = int(tcfg["batch_size"])
    steps = int(tcfg["steps_per_epoch"])
    margin = float(tcfg.get("top1_margin", 1.0))
    margin_w = float(tcfg.get("top1_margin_weight", 0.0))
    anchor_w = float(tcfg.get("anchor_weight", 1.0))
    path_mse_w = float(tcfg.get("path_mse_weight", 1.0))
    path_rank_w = float(tcfg.get("path_rank_weight", 0.0))
    # Triangle-inequality box over all 24 entries at path-label states.
    # Both default to 0.0, so an unmodified config reproduces the previous run
    # exactly -- that is the control arm.
    box_hi_w = float(tcfg.get("box_upper_weight", 0.0))
    box_lo_w = float(tcfg.get("box_lower_weight", 0.0))
    box_slack = float(tcfg.get("box_slack", 0.5))
    # Sorted-profile loss -- the box's replacement. Also defaults to 0.0.
    prof_hi_w = float(tcfg.get("profile_high_weight", 0.0))
    prof_lo_w = float(tcfg.get("profile_low_weight", 0.0))
    prof_from = int(tcfg.get("profile_high_from", 9))   # 1-indexed rank
    path_rank_margin = float(tcfg.get("path_rank_margin", 1.0))
    value_w = float(tcfg.get("value_weight", 1.0))
    amp = bool(tcfg.get("amp", True)) and dev.startswith("cuda")

    vs, vp, vprev, vnext = val_sampler.sample(int(tcfg.get("val_size", 8192)))
    hist = torch.bincount(vp, minlength=k_max + 1).tolist()
    print(f"  pivot-depth histogram (val, tilt={tilt}): "
          f"{ {d: n for d, n in enumerate(hist) if n} }", flush=True)

    log_path = args.output / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,loss,sparse_mse,anchor_mse,path_mse,margin,pair_acc,top1_acc,"
            "path_top1,gap,secs\n", encoding="utf-8")

    def run_batch(train: bool):
        gg = g_train if train else g_val
        if train:
            s, p, pv, nx = sampler.sample(base_batch)
        else:
            s, p, pv, nx = vs, vp, vprev, vnext
        if use_sym:
            xs, tgt, msk, id_rows = expand_labels(sym, tables, s, p, pv, nx, used)
        else:
            rows = torch.arange(s.size(0), device=dev)
            tgt = torch.zeros((s.size(0), A), device=dev)
            msk = torch.zeros_like(tgt, dtype=torch.bool)
            tgt[rows, pv] = p.float() - 1.0
            tgt[rows, nx] = p.float() + 1.0
            msk[rows, pv] = True
            msk[rows, nx] = True
            xs, id_rows = s, rows
        n_rw = xs.size(0)
        a_t = a_d = p_a = p_t = None
        if anchors is not None:
            a_s, a_t, a_d = anchors.sample(n_anchor, gg)
            xs = torch.cat((xs, a_s), 0)
        if paths is not None:
            p_s, p_a, p_t = paths.sample(n_path, gg)
            xs = torch.cat((xs, p_s), 0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = net(xs)
            if az_head:
                # Auxiliary AZ value head, computed from the SAME trunk pass so the
                # dual-head cells are not silently charged a second forward.
                # Target = the classic V objective: walk depth on random-walk rows,
                # EXACT distance on the BFS anchors. Distance is conjugation-invariant,
                # so every symmetry row inherits its sample's pivot.
                pred_all, v_all = out
                v_all = v_all.float()
                src_row = torch.arange(n_rw, device=dev) // used if use_sym else None
                v_tgt = p.float().index_select(0, src_row) if use_sym else p.float()
                value_loss = v_all[:n_rw].sub(v_tgt).pow(2).mean()
                if anchors is not None:
                    value_loss = value_loss + v_all[n_rw:n_rw + n_anchor].sub(a_d).pow(2).mean()
            else:
                pred_all = out
                value_loss = out.new_zeros(())
        pred = pred_all[:n_rw].float()
        sparse = pred.sub(tgt).pow(2).masked_select(msk).mean()
        anchor_mse = pred_all.new_zeros(())
        if anchors is not None:
            anchor_mse = pred_all[n_rw:n_rw + n_anchor].float().sub(a_t).pow(2).mean()
        path_mse = pred_all.new_zeros(())
        path_rank = pred_all.new_zeros(())
        path_top1 = pred_all.new_zeros(())
        box = pred_all.new_zeros(())
        box_viol = pred_all.new_zeros(())
        if paths is not None:
            pp = pred_all[n_rw + n_anchor:].float()
            rows = torch.arange(pp.size(0), device=pp.device)
            q_on = pp[rows, p_a]
            path_mse = q_on.sub(p_t).pow(2).mean()
            others = torch.ones_like(pp, dtype=torch.bool)
            others[rows, p_a] = False
            best_other = pp.masked_fill(~others, float("inf")).min(dim=1).values
            path_rank = torch.relu(path_rank_margin - (best_other - q_on)).pow(2).mean()
            path_top1 = q_on.lt(best_other).float().mean()

            # --- triangle-inequality box on ALL 24 entries -----------------
            # p_t is the on-path target d(s)-1, so d(s) = p_t + 1. For EVERY
            # action, d(s.a) is within one move of d(s):
            #     d(s) - 1  <=  Q(s, a)  <=  d(s) + 1
            # This is certain (it is the triangle inequality, measured at 0
            # violations against the exact d<=7 table) and, unlike a ranking
            # penalty, it asserts nothing about WHICH action is best -- so it
            # sidesteps the alternative-optima problem that tied m_rank_v0.
            #
            # It matters because `path_mse` supervises exactly 1 of 24 entries;
            # the other 23 are free to drift, which is how the deep-band gap
            # collapses to 0.216. The box pins all 24 to a depth-DEPENDENT
            # interval, so flattening toward a saturation constant is penalised
            # at every depth.
            #
            # Asymmetry is deliberate: path lengths are UPPER bounds on d(s)
            # (~0.2 slack on our 28.2/pid reference), so the upper hinge is
            # certain while the lower one can over-reach by the slack -- hence
            # `box_slack`.
            d_s = p_t + 1.0
            hi = torch.relu(pp - (d_s + 1.0).unsqueeze(1)).pow(2).mean()
            lo = torch.relu((d_s - 1.0 - box_slack).unsqueeze(1) - pp).pow(2).mean()
            box = box_hi_w * hi + box_lo_w * lo

            # --- SORTED-PROFILE loss (the box's replacement) ---------------
            # The box above is REJECTED: it is satisfied perfectly by the constant
            # Q(s,a) = d(s), so a two-sided hinge is silent exactly where the
            # discrimination lives and supplies only centering pressure. See
            # tetraminx/TRIANGLE_BOX_REJECTED.md.
            #
            # This constrains the SORTED vector of the 24 outputs instead, which is
            # permutation-invariant -- it never says WHICH action wins, so it keeps
            # the alternative-optima immunity that motivated the box, while a flat
            # prediction now mismatches a step-shaped target and is penalised.
            #
            # Measured mean sorted child profile at d=7 (relative to d(s)):
            #   -1.00 -0.50 +0.38 +0.49 +0.89 +0.90 +0.99 +0.99  then +1.00 x16
            # Best constant against it scores MSE 0.247 > 0, i.e. the loss IS
            # separating -- the check the box failed.
            #
            # Only the ranks robust to depth extrapolation are constrained. The
            # profile is measured at d<=7 but applied out to d~31, and the count of
            # distance-reducing children drifts up with depth (1.00 at d=1 -> 1.60
            # at d=7, p90=2, max 4). So:
            #   rank 1      -> d-1   certain at EVERY depth: a geodesic always exists
            #   ranks R..24 -> d+1   safe even if reducing+same moves doubled
            #   ranks 2..R-1         left free -- that is where the drift lives
            if prof_hi_w > 0.0 or prof_lo_w > 0.0:
                qs = pp.sort(dim=1).values
                p_lo = qs[:, 0].sub(d_s - 1.0).pow(2).mean()
                p_hi = qs[:, prof_from - 1:].sub((d_s + 1.0).unsqueeze(1)).pow(2).mean()
                box = box + prof_lo_w * p_lo + prof_hi_w * p_hi
            with torch.no_grad():
                box_viol = (((pp > (d_s + 1.0).unsqueeze(1))
                             | (pp < (d_s - 1.0).unsqueeze(1))).float().mean())
        met = sparse_metrics(pred.index_select(0, id_rows), pv, nx, margin)
        met["path_top1"] = path_top1
        met["box_viol"] = box_viol
        loss = (sparse + anchor_w * anchor_mse + margin_w * met["margin_loss"]
                + path_mse_w * path_mse + path_rank_w * path_rank
                + value_w * value_loss + box)
        met["value_mse"] = value_loss.detach()
        met["box_loss"] = box.detach()
        return loss, sparse, anchor_mse, path_mse, met

    n_epochs = int(tcfg["n_epochs"])
    ckpt_every = int(tcfg.get("checkpoint_every_epochs", 25))
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        t0 = time.time()
        acc = {k: 0.0 for k in ("loss", "sparse", "anchor", "path", "value", "margin",
                                "pair", "top1", "ptop1", "gap", "box", "bviol")}
        for _ in range(steps):
            loss, sparse, anchor_mse, path_mse, met = run_batch(True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(tcfg.get("grad_clip", 0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            acc["loss"] += float(loss.detach())
            acc["sparse"] += float(sparse.detach())
            acc["anchor"] += float(anchor_mse.detach())
            acc["path"] += float(path_mse.detach())
            acc["value"] += float(met["value_mse"])
            acc["margin"] += float(met["margin_loss"].detach())
            acc["pair"] += float(met["pair_acc"])
            acc["top1"] += float(met["top1_acc"])
            acc["ptop1"] += float(met["path_top1"])
            acc["gap"] += float(met["gap"])
            acc["box"] += float(met["box_loss"])
            acc["bviol"] += float(met["box_viol"])
        for k in acc:
            acc[k] /= steps
        secs = time.time() - t0
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{acc['loss']:.5f},{acc['sparse']:.5f},{acc['anchor']:.5f},"
                     f"{acc['path']:.5f},{acc['margin']:.5f},{acc['pair']:.5f},"
                     f"{acc['top1']:.5f},{acc['ptop1']:.5f},{acc['gap']:.4f},{secs:.1f}\n")
        print(f"epoch {epoch:5d} | loss {acc['loss']:8.4f} sparse {acc['sparse']:8.4f} "
              f"anchor {acc['anchor']:7.4f} path {acc['path']:7.4f} val {acc['value']:7.3f} | pair {acc['pair']:.4f} "
              f"top1 {acc['top1']:.4f} ptop1 {acc['ptop1']:.4f} gap {acc['gap']:.3f} "
              f"box {acc['box']:.4f} bviol {acc['bviol']:.4f} | {secs:.1f}s", flush=True)

        if epoch % ckpt_every == 0 or epoch == n_epochs:
            model.eval()
            g_val.manual_seed(cfg.get("seed", 0) + 1_000_000)   # same val anchors every time
            with torch.no_grad():
                vloss, vsparse, vanchor, vpath, vmet = run_batch(False)
            print(f"  val: loss {float(vloss):.4f} pair {float(vmet['pair_acc']):.4f} "
                  f"top1 {float(vmet['top1_acc']):.4f} ptop1 {float(vmet['path_top1']):.4f} "
                  f"gap {float(vmet['gap']):.3f} bviol {float(vmet['box_viol']):.4f}", flush=True)
            payload = {
                "epoch": epoch,
                "state_dict": {k.removeprefix("_orig_mod."): v
                               for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(),
                "loss": acc["loss"],
                "val_loss": float(vloss),
                "best": min(best, float(vloss)),
                "model_config": {**model.get_model_config()},   # round-trips via models.model_from_config
                "train_config": tcfg,
                "metrics": {k: float(v) for k, v in vmet.items()},
            }
            torch.save(payload, args.output / f"epoch_{epoch:04d}.pt")
            if float(vloss) < best:
                best = float(vloss)
                torch.save(payload, args.output / "best.pt")
            model.train()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
