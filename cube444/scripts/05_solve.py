"""Beam-search solver for the 4x4x4 cube, with the 24-rotation sym-ensemble.

    python3 cube444/scripts/05_solve.py \
        --checkpoint cube444/models/c_bells/epoch_0299.pt \
        --out cube444/submissions/run1.csv \
        --beams 16384,65536 --max-steps 70,140 --sym-ensemble 4 --bf16 \
        --fallback cube444/community/submission_54754_merge33.csv --resume

SYM-ENSEMBLE ON A COLOR CUBE
----------------------------
Rotating the cube physically moves colors onto different faces, so unlike the
picture cube a rotation alone does NOT preserve the solved target. Each frame is

    s_k = color_maps[k][ s[rotations[k]] ]

which does send solved -> solved. A solution found in frame k is mapped back to
the original frame move-by-move through `move_relabel_inv_24` (same order, no
reversal). Every returned path is re-verified against the ORIGINAL state before
it is accepted, so a translation bug cannot silently produce a bad submission.

Frame selection follows the K_SYM / SYM_POSITIONS convention (Rule 16): the
canonical frame list is identity followed by a seeded random sample of the other
23, so two shards of a split run agree on which frame is which.

NOTE: there are no inverse frames here. The state is a coloring, not a
permutation, so `invert_state` is undefined (see cube444/src/cube444/puzzle.py).
On megaminx the inverse frames carried most of the pooled-sym win, so expect a
smaller gain from sym here than megaminx saw.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cube444.models import load_v_model
from cube444.puzzle import Cube444


def load_tests(path: Path):
    tests, rwlen = {}, {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = int(r["initial_state_id"])
            tests[pid] = np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
            m = re.match(r"generated rw, len=(\d+)$", r["comment"])
            rwlen[pid] = int(m.group(1)) if m else -1
    return tests, rwlen


def load_paths(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        return {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(f)}


def canonical_frames(k_sym: int, seed: int) -> list[int]:
    """Identity first, then the first K-1 of a seeded permutation of rotations 1..23.

    PREFIX-NESTED by construction: frames(4) is a strict prefix of frames(8), so a
    K-sweep is interpretable (K=8 can only match or beat K=4, never differ by
    which frames it happened to draw) and a shard split is trivially safe.

    This deliberately differs from the megaminx K_SYM/SYM_POSITIONS convention
    (Rule 16), where the list is `rng.choice(non_identity, size=K_SYM-1)` and so
    DEPENDS on K_SYM -- there, two shards must agree on K_SYM or their position
    indices mean different rotations. Here `seed` alone fixes the ordering, so
    --sym-positions is unambiguous regardless of --sym-ensemble.
    """
    order = [0] + [int(x) for x in np.random.default_rng(seed).permutation(np.arange(1, 24))]
    return order[: max(1, min(k_sym, 24))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beams", type=str, default="16384,65536")
    ap.add_argument("--max-steps", type=str, default="70,140")
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--sym-ensemble", type=int, default=1, help="number of rotation frames (1..24)")
    ap.add_argument("--sym-positions", type=str, default=None,
                    help="'lo:hi' slice of the canonical frame list to run in this shard")
    ap.add_argument("--sym-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--internal-batch-size", type=int, default=2 ** 15)
    ap.add_argument("--fallback", type=str, default=None,
                    help="comma-separated CSVs; per-pid min is used when the beam fails "
                         "or is longer")
    ap.add_argument("--pids", type=str, default=None, help="explicit comma-separated pid list")
    ap.add_argument("--stratified", type=int, default=None,
                    help="benchmark on every Nth pid (plus all santa pids)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip pids already in --out")
    args = ap.parse_args()

    beams = [int(x) for x in args.beams.split(",")]
    steps = [int(x) for x in args.max_steps.split(",")]
    if len(beams) != len(steps):
        raise ValueError(f"--beams has {len(beams)} entries but --max-steps has {len(steps)}")

    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    central = np.array(puz.solved_state, dtype=np.int64)
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}

    rot = np.load(PROJECT / "data" / "rotations_24.npy")
    cmap = np.load(PROJECT / "data" / "color_maps_24.npy")
    inv_relabel = np.load(PROJECT / "data" / "move_relabel_inv_24.npy")

    frames = canonical_frames(args.sym_ensemble, args.sym_seed)
    if args.sym_positions:
        lo, hi = (int(x) for x in args.sym_positions.split(":"))
        frames = frames[lo:hi]
    print(f"sym frames (K_SYM={args.sym_ensemble}, seed={args.sym_seed}): {frames}")

    tests, rwlen = load_tests(PROJECT / "data" / "test.csv")

    fallback: dict[int, str] = {}
    if args.fallback:
        for p in args.fallback.split(","):
            for pid, path in load_paths(Path(p)).items():
                cur = fallback.get(pid)
                if cur is None or len(path.split(".")) < len(cur.split(".")):
                    fallback[pid] = path
        print(f"fallback: {len(fallback)} pids, total "
              f"{sum(len(v.split('.')) for v in fallback.values()):,} moves")

    if args.pids:
        pids = [int(x) for x in args.pids.split(",")]
    elif args.stratified:
        pids = [p for p in sorted(tests) if rwlen[p] > 0 and p % args.stratified == 0]
        pids += [p for p in sorted(tests) if rwlen[p] == -1]
    else:
        pids = sorted(tests)
    if args.limit:
        pids = pids[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, str] = {}
    if args.resume and args.out.exists():
        done = load_paths(args.out)
        print(f"resume: {len(done)} pids already solved in {args.out}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_v_model(args.checkpoint, device=args.device, dtype=dtype)
    solver = KhoruzhiiSolver(puz, model, device=args.device,
                            internal_batch_size=args.internal_batch_size)
    print(f"model: {args.checkpoint}  dtype={dtype}  device={args.device}")
    print(f"beams={beams} max_steps={steps} pids={len(pids)}")

    t_start = time.time()
    # n_found  = the beam produced a verified solution (regardless of length)
    # n_used   = that solution was <= the fallback, so we kept it
    # n_improved = strictly shorter than the fallback
    # Keep these separate: a beam that solves every puzzle but 20 moves longer
    # than the community floor is a very different situation from a beam that
    # solves nothing, and collapsing them into one counter hides that.
    n_found, n_used, n_fb, n_improved = 0, 0, 0, 0
    beam_lens: list[int] = []
    for i, pid in enumerate(pids):
        if pid in done:
            continue
        s = tests[pid]
        best: list[str] | None = None
        fb = fallback.get(pid)
        fb_len = len(fb.split(".")) if fb else None

        for k in frames:
            s_k = cmap[k][s[rot[k]]]
            for B, ns in zip(beams, steps):
                cfg = KhoruzhiiSearchConfig(beam_width=B, num_steps=ns,
                                            num_attempts=args.num_attempts,
                                            internal_batch_size=args.internal_batch_size)
                found, _, path_k = solver.solve(s_k, cfg)
                if not found:
                    continue
                path = [names[int(inv_relabel[k, name_to_idx[m]])] for m in path_k]
                cur = s.copy()
                for m in path:
                    cur = cur[G[m]]
                if not np.array_equal(cur, central):
                    print(f"  pid {pid} frame {k}: TRANSLATED PATH DOES NOT SOLVE -- discarded")
                    continue
                if best is None or len(path) < len(best):
                    best = path
                break  # this frame is solved; widening further only costs wall

        if best is not None:
            n_found += 1
            beam_lens.append(len(best))

        if best is not None and (fb_len is None or len(best) <= fb_len):
            done[pid] = puz.format_path(best)
            n_used += 1
            if fb_len is not None and len(best) < fb_len:
                n_improved += 1
        elif fb is not None:
            done[pid] = fb
            n_fb += 1
        else:
            print(f"  pid {pid}: NO SOLUTION and no fallback -- omitted")
            continue

        if (i + 1) % 10 == 0 or i == len(pids) - 1:
            tot = sum(len(v.split(".")) for v in done.values())
            el = time.time() - t_start
            mb = f"{np.mean(beam_lens):.1f}" if beam_lens else "-"
            print(f"  [{i+1}/{len(pids)}] pid={pid} rw={rwlen[pid]} "
                  f"len={len(done[pid].split('.'))} | beam-found {n_found} "
                  f"(mean {mb}) used {n_used} improved {n_improved} fb {n_fb} "
                  f"| total {tot:,} | {el:.0f}s", flush=True)
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["initial_state_id", "path"])
                for p in sorted(done):
                    w.writerow([p, done[p]])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for p in sorted(done):
            w.writerow([p, done[p]])

    tot = sum(len(v.split(".")) for v in done.values())
    print(f"\nwrote {args.out}: {len(done)} pids, {tot:,} moves (mean {tot/max(1,len(done)):.2f})")
    n_att = max(1, n_found + n_fb if n_found + n_fb else 1)
    print(f"  beam found a verified solution for {n_found}/{len(pids)} attempted "
          f"({100*n_found/max(1,len(pids)):.0f}%)")
    if beam_lens:
        print(f"  beam solution length: mean {np.mean(beam_lens):.2f}  "
              f"min {min(beam_lens)}  max {max(beam_lens)}")
    print(f"  kept beam path {n_used}   beat fallback {n_improved}   used fallback {n_fb}")
    print(f"  wall {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
