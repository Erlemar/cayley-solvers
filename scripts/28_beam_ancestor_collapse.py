"""Does the beam collapse onto a few openings?  (decides whether stratifying pays)

`scripts/26_prefix_structure.py` found that the shortest path for a pid opens
differently from the crowd of longer paths (2.4% vs 9.0% shared opening mass).  The
proposed beam-side exploit is an ancestor quota -- reserve beam slots per depth-2
opening instead of taking a flat global top-B.  That is only worth building if the
flat beam actually collapses.

This measures it directly.  The beam is EXHAUSTIVE while its width exceeds the
sphere size (18 / 261 / 3,732 / 52,620 / 733,956 for depth 1..5), so at 65k the first
real cut is depth 5 and at 1M it is depth 6 -- nothing is pruned at the opening, and a
penalty applied there is a no-op.  The question is what survives further down: if by
depth 12 the beam still descends from 200+ of the 261 openings there is no collapse
and the quota idea is dead; if it descends from 15, there is headroom.

    python scripts/28_beam_ancestor_collapse.py --checkpoint models/e6/epoch_0499.pt \
        --beam 65536 --pids 30 106 240 --bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver  # noqa: E402
from cayley.puzzle import PictureCube  # noqa: E402
from cayley.search import load_model_checkpoint  # noqa: E402
from cayley.verify import load_test_states  # noqa: E402

SPHERE = {1: 18, 2: 261, 3: 3732, 4: 52620, 5: 733956, 6: 10_200_000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--beam", type=int, nargs="+", default=[65536])
    ap.add_argument("--pids", type=int, nargs="+", default=[30, 106, 240, 500, 900])
    ap.add_argument("--ancestor-depth", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--best", type=Path,
                    help="known-good submission CSV: track whether the opening of the "
                         "BEST path stays alive in the beam, which is what a quota "
                         "would have to rescue")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    base = getattr(model, "_orig_mod", model)
    use_q = getattr(base, "output_dim", 1) > 1
    solver = KhoruzhiiSolver(puzzle, model, device=args.device, use_q_function=use_q)
    print(f"checkpoint {args.checkpoint.name}  q_head={use_q}  device={args.device}")

    best = None
    if args.best:
        from cayley.verify import load_submission
        best = load_submission(args.best)

    k = args.ancestor_depth
    n_open = SPHERE[k]
    out: dict = {"ancestor_depth": k, "n_openings": n_open, "runs": []}

    for beam in args.beam:
        cfg = KhoruzhiiSearchConfig(beam_width=beam, num_steps=args.max_steps,
                                    num_attempts=1)
        per_depth: dict[int, list[tuple[int, float, float, int]]] = defaultdict(list)
        print(f"\n=== beam {beam:,}  (exhaustive through depth "
              f"{max([d for d, s in SPHERE.items() if s <= beam], default=0)}) ===")
        for pid in args.pids:
            st = {"ancestor_depth": k}
            found, plen, _ = solver.solve(states[pid], cfg, stats=st)
            steps = st.get("steps", [])
            note = ""
            if best is not None and pid in best and "ancestor_states" in st:
                want = torch.tensor(
                    list(puzzle.apply_path(states[pid], best[pid][:k])),
                    dtype=st["ancestor_states"].dtype)
                match = (st["ancestor_states"] == want).all(dim=1).nonzero()
                if match.numel() == 0:
                    note = "  [best opening absent from the depth-k beam?!]"
                else:
                    a = int(match[0].item())
                    alive = [s["depth"] for s in steps if s["counts"][a] > 0]
                    last = max(alive) if alive else 0
                    share = [s["counts"][a] / s["beam"] for s in steps
                             if s["depth"] == min(len(best[pid]), max(alive or [0]))]
                    note = (f"  best opening {'.'.join(best[pid][:k])}: alive through "
                            f"depth {last} of {len(best[pid])}"
                            + (f", holding {share[0]:.2%} of the beam there" if share else ""))
            print(f"  pid {pid}: {'solved ' + str(plen) if found else 'NOT SOLVED'}"
                  f"  ({len(steps)} steps traced){note}")
            for s in steps:
                per_depth[s["depth"]].append(
                    (s["openings"], s["top1"], s["effective"], s["beam"]))
            out["runs"].append({"beam": beam, "pid": pid, "found": found, "len": plen,
                                "steps": [{kk: vv for kk, vv in s.items() if kk != "counts"}
                                          for s in steps]})

        print(f"  {'depth':>6s} {'beam':>9s} {'openings alive':>15s} "
              f"{'effective':>10s} {'top-1 share':>12s}")
        for d in sorted(per_depth):
            rows = per_depth[d]
            n = len(rows)
            print(f"  {d:6d} {sum(r[3] for r in rows)/n:9.0f} "
                  f"{sum(r[0] for r in rows)/n:8.1f} / {n_open:<4d} "
                  f"{sum(r[2] for r in rows)/n:10.1f} {sum(r[1] for r in rows)/n:11.1%}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
