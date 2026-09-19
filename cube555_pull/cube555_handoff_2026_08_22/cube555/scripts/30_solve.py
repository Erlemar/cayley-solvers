"""Beam-search solver for the 5x5x5 picture cube.

    python cube555/scripts/30_solve.py --checkpoint cube555/models/q555_a/epoch_XXXXXX.pt \
        --pids santa --beams 4194304 --max-steps 130 --bf16 --out cube555/bench/x.csv

Q PATH. The checkpoint is a 30-wide Q head, so one forward on a parent scores all 30
children and `khoruzhii_search` only materialises and hashes the top candidates. A
scalar-V beam of the same width costs 30x the network evaluations; that ratio is why a
2^24 beam is a single-GPU job here and needed a fleet in the paper.

state_dtype IS int16, NOT the solver's int8 default. Sticker values run 0..149 and int8
holds -128..127: the wrap happens to stay injective (150 < 256) so hashing and dedup
would still "work", and then `nn.Embedding` receives a negative index. Silent everywhere
until it is not.

--invert: solve s^-1 instead of s. Exactly valid on a Cayley graph -- if b1..bm solves
s^-1 then reversing the list and inverting each move solves s -- and it is a genuinely
decorrelated trajectory, which is the cheapest extra "agent" available for the min-merge.
cube444 could not do this at all. Every returned path is replayed against the ORIGINAL
state before it is accepted, so a translation bug cannot produce a bad submission.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver  # noqa: E402
from cube555.models import load_model  # noqa: E402
from cube555.puzzle import Cube555  # noqa: E402

N_SANTA = 35  # pids 0..34; pids 35..1034 are scrambles of length pid-34


def load_tests(path: Path):
    tests = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = int(r["initial_state_id"])
            tests[pid] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64
            )
    return tests


def load_paths(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        return {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(f)}


def resolve_pids(spec: str, tests: dict) -> list[int]:
    """'santa' | 'gate' | 'smoke' | explicit list | 'all'."""
    if spec == "santa":
        return list(range(N_SANTA))
    if spec == "gate":
        # 60 fully mixed scrambles, evenly spaced over rw length 300..1000. Disjoint from
        # SMOKE. Nothing is trained on any specific state, so these are held out by
        # construction; the spacing only keeps the set reproducible.
        return [34 + int(round(x)) for x in np.linspace(300, 1000, 60)]
    if spec == "smoke":
        return [34 + int(round(x)) for x in np.linspace(320, 980, 12)]
    if spec == "all":
        return sorted(tests)
    return [int(x) for x in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beams", type=str, default="1048576")
    ap.add_argument("--max-steps", type=str, default="130")
    ap.add_argument("--pids", type=str, default="smoke")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--internal-batch-size", type=int, default=1 << 18)
    ap.add_argument(
        "--inference-chunk-size",
        type=int,
        default=None,
        help="override the checkpoint's value; REQUIRED for a fair "
        "wall-clock A/B across checkpoints",
    )
    ap.add_argument("--history-depth", type=int, default=0,
                    help="0 = within-layer dedup only; N = exclude the last N layers; "
                         "-1 = UNLIMITED (closed-set graph search)")
    ap.add_argument(
        "--invert",
        action="store_true",
        help="solve s^-1 and map the path back (reverse + invert each move)",
    )
    ap.add_argument("--blend", type=Path, nargs="+", default=None,
                    help="extra Q checkpoints averaged with --checkpoint every beam step")
    ap.add_argument("--blend-weights", type=str, default=None,
                    help="comma-separated, one per member INCLUDING --checkpoint first")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the trunk with padded fixed-size chunks; "
                         "measured 1.56x on beam throughput")
    ap.add_argument("--endgame-depth", type=int, default=0,
                    help="stop the beam when it reaches ANY state within this exact-BFS "
                         "radius of solved and splice on the exact optimal tail. D=5 "
                         "widens the goal from 1 state to 10,739,017.")
    ap.add_argument("--no-backtrack", action="store_true",
                    help="mask the undo move before the top-k")
    ap.add_argument("--frames", type=str, default="0",
                    help="comma-separated conjugation frame indices (0..47) to try in "
                         "order until one solves. Each is an INDEPENDENT search "
                         "trajectory on an isomorphic state.")
    ap.add_argument("--qv-consistency", type=float, default=0.0,
                    help="lambda for Q + lam*|Q-(V-1)|, both heads from one trunk pass")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--fallback",
        type=str,
        default=None,
        help="comma-separated CSVs; per-pid min is used when the beam fails "
        "or is longer",
    )
    args = ap.parse_args()

    beams = [int(x) for x in args.beams.split(",")]
    steps = [int(x) for x in args.max_steps.split(",")]
    if len(beams) != len(steps):
        raise SystemExit("--beams and --max-steps must have the same length")

    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    central = np.array(puz.solved_state, dtype=np.int64)
    tests = load_tests(PROJECT / "data" / "test.csv")
    name_i = {n: i for i, n in enumerate(names)}
    SYM_M = np.load(PROJECT / "data" / "sym_slots_48.npy")
    SYM_INV = np.load(PROJECT / "data" / "sym_slots_inv_48.npy")
    SYM_RINV = np.load(PROJECT / "data" / "sym_move_relabel_inv_48.npy")
    pids = resolve_pids(args.pids, tests)
    frame_list = [int(x) for x in args.frames.split(",")]
    if any(not 0 <= f < SYM_M.shape[0] for f in frame_list):
        raise SystemExit(f"--frames must be in 0..{SYM_M.shape[0]-1}")

    fallback: dict[int, str] = {}
    if args.fallback:
        for p in args.fallback.split(","):
            for pid, path in load_paths(Path(p)).items():
                cur = fallback.get(pid)
                if cur is None or len(path.split(".")) < len(cur.split(".")):
                    fallback[pid] = path

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model(args.checkpoint, device=args.device, dtype=dtype)
    model.return_value = False
    if args.inference_chunk_size is not None:
        model.inference_chunk_size = (
            None if args.inference_chunk_size == 0 else args.inference_chunk_size
        )
    if args.blend:
        from cube555.blend import BlendedQ

        members = [model] + [
            load_model(b, device=args.device, dtype=dtype) for b in args.blend
        ]
        wts = [float(x) for x in args.blend_weights.split(",")] if args.blend_weights else None
        if wts is not None and len(wts) != len(members):
            raise SystemExit(
                f"--blend-weights has {len(wts)} entries but there are {len(members)} "
                "members (--checkpoint counts as the first)"
            )
        model = BlendedQ(members, wts).to(args.device).eval()
        print(f"  blend: {len(members)} members, weights "
              f"{[round(float(w), 3) for w in model.weights]}")
    if args.compile:
        if args.blend:
            for mem in model.members:
                mem.enable_compiled_inference()
        else:
            model.enable_compiled_inference()
    print(
        f"model: {args.checkpoint}  {model.num_parameters()/1e6:.1f}M  dtype={dtype}"
        f"{'  COMPILED' if args.compile else ''}\n"
        f"scorer: Q (one forward per parent, global top-B over {model.output_dim} "
        f"actions)   invert={args.invert}\n"
        f"beams={beams} max_steps={steps} pids={len(pids)} "
        f"ics={model.inference_chunk_size}"
    )

    # ---- endgame table: widen the goal from {solved} to the exact d<=D ball ----------
    goal_fn = None
    eg = None
    if args.endgame_depth > 0:
        blob = torch.load(
            PROJECT / "data" / f"anchors_d{args.endgame_depth}.pt",
            map_location="cpu", weights_only=False,
        )
        eg_states = blob["states"].to(args.device)
        eg_q = blob["q"].to(args.device)
        eg_depth = blob["depth"].to(args.device)
        hv = torch.randint(-(2**62), 2**62, (150,), dtype=torch.int64,
                           device=args.device, generator=torch.Generator(
                               device=args.device).manual_seed(555))
        eg_hash = (eg_states.long() * hv).sum(1)
        order = torch.argsort(eg_hash)
        eg_hash = eg_hash[order]
        eg_q = eg_q[order]
        eg_depth = eg_depth[order]

        eg_sorted = eg_states[order]

        def _lookup(states_t):
            # EXACT membership: a 64-bit hash hit is not proof. With 10.7M entries and
            # ~1e11 lookups over a long run a false positive is EXPECTED (~0.1), and one
            # killed a 29 h run at pid 800. Verify the state itself, not just its hash.
            h = (states_t.long() * hv).sum(1)
            pos = torch.searchsorted(eg_hash, h).clamp_max(eg_hash.numel() - 1)
            hit = eg_hash[pos] == h
            hit &= (eg_sorted[pos].to(torch.int64) == states_t.long()).all(dim=1)
            return pos, hit

        def goal_fn(states_t):
            return _lookup(states_t)[1]

        eg = (_lookup, eg_q, eg_depth)
        print(f"  endgame: goal widened to the exact d<={args.endgame_depth} ball "
              f"({eg_hash.numel():,} states); exact tail spliced on arrival")

    solver = KhoruzhiiSolver(
        puz,
        model,
        device=args.device,
        internal_batch_size=args.internal_batch_size,
        use_q_function=True,
        qv_consistency_lambda=args.qv_consistency,
        # int8 would wrap 128..149 negative and hand nn.Embedding a negative index.
        state_dtype=torch.int16,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, str] = {}
    if args.resume and args.out.exists():
        done = load_paths(args.out)
        print(f"resume: {len(done)} pids already in {args.out}")

    def write():
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["initial_state_id", "path"])
            for p in sorted(done):
                w.writerow([p, done[p]])

    t_start = time.time()
    n_found, n_fb, lens = 0, 0, []
    for i, pid in enumerate(pids):
        if pid in done:
            continue
        s = tests[pid]
        best = None
        for frame in frame_list:
            base = np.array(puz.invert_state(s), dtype=np.int64) if args.invert else s
            # sym(x, M) = Minv[x[M]]; a path solving it maps back through relabel_inv.
            start = SYM_INV[frame][base[SYM_M[frame]]] if frame else base
            if best is not None:
                break
            for B, ns in zip(beams, steps):
                cfg = KhoruzhiiSearchConfig(
                    beam_width=B,
                    num_steps=ns,
                    num_attempts=1,
                    internal_batch_size=args.internal_batch_size,
                    history_depth=args.history_depth,
                    no_backtrack=args.no_backtrack,
                )
                found, _, raw = solver.solve(start, cfg, goal_check_fn=goal_fn)
                if not found:
                    continue
                raw = list(raw)
                if eg is not None:
                    # The beam stopped at a state in the ball, not at solved. Replay to it,
                    # then descend by EXACT argmin-Q, one depth per move, until solved.
                    _lookup, eg_q, eg_depth = eg
                    cur_t = start.copy()
                    for mv in raw:
                        cur_t = cur_t[G[mv]]
                    for _ in range(args.endgame_depth + 1):
                        st = torch.tensor(cur_t, dtype=torch.int64,
                                          device=args.device).unsqueeze(0)
                        pos, hit = _lookup(st)
                        if not bool(hit[0]):
                            # Unreachable now that membership is exact, but never let one
                            # pid kill a multi-day run.
                            print(f"  pid {pid}: endgame lookup missed -- skip frame")
                            raw = None
                            break
                        if int(eg_depth[pos[0]]) == 0:
                            break
                        a = int(eg_q[pos[0]].argmin())
                        raw.append(names[a])
                        cur_t = cur_t[G[names[a]]]
                # Frame first, then inversion: `raw` solves sym(base, M), so map each
                # move through relabel_inv (same order) to get a path solving `base`;
                # only then apply the inverse-frame reversal to reach `s`.
                if raw is None:
                    continue
                if frame:
                    raw = [names[int(SYM_RINV[frame][name_i[m]])] for m in raw]
                path = puz.invert_path(raw) if args.invert else raw
                cur = s.copy()
                for m in path:
                    cur = cur[G[m]]
                if not np.array_equal(cur, central):
                    print(f"  pid {pid}: TRANSLATED PATH DOES NOT SOLVE -- discarded")
                    continue
                if best is None or len(path) < len(best):
                    best = path
                break
        if best is not None:
            n_found += 1
            lens.append(len(best))
            done[pid] = puz.format_path(best)
        fb = fallback.get(pid)
        if fb is not None and (best is None or len(fb.split(".")) < len(best)):
            done[pid] = fb
            n_fb += 1
        if pid not in done:
            print(f"  pid {pid}: NO SOLUTION and no fallback -- omitted")
            continue
        el = time.time() - t_start
        ml = f"{np.mean(lens):.2f}" if lens else "-"
        print(
            f"  [{i+1}/{len(pids)}] pid={pid} len={len(done[pid].split('.'))} "
            f"| found {n_found} mean {ml} | fb {n_fb} | {el:.0f}s "
            f"({el/max(1,i+1):.0f}s/pid)",
            flush=True,
        )
        write()

    write()
    tot = sum(len(v.split(".")) for v in done.values())
    print(
        f"\nwrote {args.out}: {len(done)} pids, {tot:,} moves "
        f"(mean {tot/max(1,len(done)):.3f})"
    )
    if lens:
        print(
            f"  beam-only: n={len(lens)} mean {np.mean(lens):.3f} "
            f"min {min(lens)} max {max(lens)}"
        )
    print(f"  wall {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
