"""Acceptance gate for a sparse-Q head: does it out-rank the V at the beam's decision point?

The beam only ever asks one question: of the 24 children of this state, which are
worth keeping. So the gate is the decision metric, measured per pivot depth:

  pair  -- score(undo) < score(next-walk-move)
  top1  -- score(undo) < min over the other 22 actions
  gap   -- mean score(next) - score(undo); the label says this should be 2 at EVERY
           depth, and it is the quantity our walk-depth MSE lets collapse

A V model is scored by running it on all 24 children (24 forwards per state); a Q
model by one forward on the parent. Both produce a (N, 24) score matrix, so the
numbers are directly comparable.

Caveat worth keeping in mind: "undo" is not necessarily the unique optimal move, so
absolute top1 understates both models. The comparison between them, and the shape of
`gap` across depth, are the load-bearing readings.

    python3 tetraminx/scripts/52_eval_q.py \
        --q-model tetraminx/models/tq0/best.pt \
        --v-model tetraminx/models/tv0_bellman/epoch_0024.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.model import ResMLPDistance
from tetraminx.models import model_from_config
from tetraminx.puzzle import Tetraminx


def load_model(path: Path, device: str):
    """Load either checkpoint flavour.

    Sparse-Q checkpoints written by the current 51_train_sparse_q.py carry an
    `arch` dispatch key and are built through `model_from_config`; earlier ones
    (tq0, and every V model) were saved directly as `ResMLPDistance`. The two are
    weight-compatible -- same head key names, same 5,008,280 params -- so both load
    and both report `output_dim`, which is what the Q-vs-V kind check reads.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ck["model_config"])
    mc.pop("model_class", None)
    if "arch" in mc:
        model = model_from_config(mc).to(device).eval()
        model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()})
        return model
    mc["inference_chunk_size"] = None
    model = ResMLPDistance(**mc).to(device).eval()
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()})
    return model


def sample_pivots(puzzle, gen, inv, solved, n, k_min, k_max, tilt, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = gen.size(0)
    lengths = torch.randint(k_min, k_max + 1, (n,), generator=g, device=device)
    u = torch.rand((n,), generator=g, device=device)
    pivots = (u.pow(1.0 / (1.0 + tilt)) * (lengths - 1).float()).long() + 1
    states = solved.unsqueeze(0).expand(n, -1).clone()
    piv = torch.empty_like(states)
    last = torch.full((n,), -1, dtype=torch.int64, device=device)
    prev = torch.full_like(last, -1)
    nxt = torch.full_like(last, -1)
    for step in range(k_max):
        active = lengths > step
        if step == 0:
            mv = torch.randint(A, (n,), generator=g, device=device)
        else:
            c = torch.randint(A - 1, (n,), generator=g, device=device)
            mv = c + c.ge(inv[last.clamp_min(0)]).long()
        succ = torch.gather(states, 1, gen[mv])
        at = pivots == step
        piv = torch.where(at.unsqueeze(1), states, piv)
        prev = torch.where(at, inv[last.clamp_min(0)], prev)
        nxt = torch.where(at, mv, nxt)
        states = torch.where(active.unsqueeze(1), succ, states)
        last = torch.where(active, mv, last)
    return piv, pivots, prev, nxt


@torch.no_grad()
def score_matrix(model, states, gen, chunk=8192):
    """(N, 24) score per action. Q models: one forward on the parent. V models: on children."""
    A, S = gen.shape
    if getattr(model, "output_dim", 1) == A:
        out = []
        for i in range(0, states.size(0), chunk):
            out.append(model(states[i:i + chunk]).float())
        return torch.cat(out, 0)
    kids = torch.gather(states.unsqueeze(1).expand(-1, A, -1), 2,
                        gen.unsqueeze(0).expand(states.size(0), -1, -1)).reshape(-1, S)
    out = []
    for i in range(0, kids.size(0), chunk):
        out.append(model(kids[i:i + chunk]).float().view(-1))
    return torch.cat(out, 0).view(states.size(0), A)


def report(name, scores, pivots, prev, nxt, bands):
    n, A = scores.shape
    rows = torch.arange(n, device=scores.device)
    s_prev, s_next = scores[rows, prev], scores[rows, nxt]
    other = torch.ones_like(scores, dtype=torch.bool)
    other[rows, prev] = False
    other[rows, nxt] = False
    best_wrong = scores.masked_fill(~other, float("inf")).min(dim=1).values
    print(f"\n{name}: pair {(s_prev < s_next).float().mean():.4f}  "
          f"top1 {(s_prev < best_wrong).float().mean():.4f}  "
          f"gap {(s_next - s_prev).mean():.3f}")
    print(f"  {'band':>9} {'recs':>6} {'pair':>7} {'top1':>7} {'gap':>7} {'s(undo)':>9}")
    for lo, hi in bands:
        sel = (pivots >= lo) & (pivots <= hi)
        recs = int(sel.sum())
        if recs == 0:
            continue
        print(f"  {f'{lo}-{hi}':>9} {recs:>6} {(s_prev[sel] < s_next[sel]).float().mean():>7.3f} "
              f"{(s_prev[sel] < best_wrong[sel]).float().mean():>7.3f} "
              f"{(s_next[sel] - s_prev[sel]).mean():>7.3f} {s_prev[sel].mean():>9.2f}")
    return {"pair": float((s_prev < s_next).float().mean()),
            "top1": float((s_prev < best_wrong).float().mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q-model", type=Path, default=None)
    ap.add_argument("--v-model", type=Path, default=None)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--tilt", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    args = ap.parse_args()
    if args.q_model is None and args.v_model is None:
        raise SystemExit("pass at least one of --q-model / --v-model")

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    names = list(puzzle.move_names)
    gen = torch.tensor([puzzle.generators[n] for n in names], dtype=torch.int64, device=args.device)
    inv = torch.tensor([names.index(puzzle.inverse_name(n)) for n in names],
                       dtype=torch.int64, device=args.device)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=args.device)

    piv, pivots, prev, nxt = sample_pivots(puzzle, gen, inv, solved, args.n, args.k_min,
                                           args.k_max, args.tilt, args.device, args.seed)
    print(f"probe: {args.n} rw-middle pivots, k in [{args.k_min},{args.k_max}], tilt {args.tilt}")
    bands = [(1, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, args.k_max)]

    results = {}
    for tag, path in (("Q", args.q_model), ("V", args.v_model)):
        if path is None:
            continue
        model = load_model(path, args.device)
        kind = "Q head" if getattr(model, "output_dim", 1) == len(names) else "V (per child)"
        scores = score_matrix(model, piv, gen)
        results[tag] = report(f"{tag}  {path.name}  [{kind}]", scores, pivots, prev, nxt, bands)

    if "Q" in results and "V" in results:
        print(f"\nverdict: top1 Q {results['Q']['top1']:.4f} vs V {results['V']['top1']:.4f} "
              f"({results['Q']['top1'] - results['V']['top1']:+.4f})   "
              f"pair Q {results['Q']['pair']:.4f} vs V {results['V']['pair']:.4f} "
              f"({results['Q']['pair'] - results['V']['pair']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
