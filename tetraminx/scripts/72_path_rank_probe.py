"""Phase 0 GO/NO-GO probe for Beam-AVI: is there deep signal to bootstrap from?

Beam-AVI (see `BEAM_AVI_METHOD.md` / `tetraminx/BEAM_AVI_PLAN.md`) backs up
`Q(s,a) <- 1 + min_a' Q_target(s.a, a')` over states the BEAM visits. Bootstrapping
cannot manufacture signal the target net does not already have, so before spending
any GPU on the loop we measure the precondition directly:

    replay a known near-geodesic path through the beam, and at every step record the
    CROSS-PARENT PERCENTILE of the on-path candidate among all `B * n_actions`
    candidate scores the beam is ranking at that step.

Null is 0.5 (no signal). cube444 measured 0.034 (z = 30.2) at its deployment width and
called that ample. Near 0.5 in the deep band means STOP -- there is nothing to
bootstrap from and the whole line is dead.

WIDTH MATTERS AND 65k IS THE WRONG ANSWER. cube444 probed at B=65536 because 65k was
still a competent solver there. On tetraminx it is not: 65k is 33.73 moves/pid against
deployment's 27.93 at 4M, i.e. +5.8/pid, so a 65k beam visits states the deployed beam
never sees. The probe defaults to B = 2^20.

The probe is READ-ONLY. It never injects the on-path state into the beam -- it scores
that one candidate on the side and compares it against the distribution the beam is
already ranking. `--verify-noperturb` proves the hook changes nothing by solving one
pid with it on and off and comparing the paths.

    .venv/Scripts/python.exe tetraminx/scripts/72_path_rank_probe.py \
        --checkpoint tetraminx/models/mx_tf_az/epoch_1500.pt \
        --blend tetraminx/models/mx_resmlp_az/best.pt --blend-weights 0.8 0.2 \
        --qv-consistency 0.3 --history-depth 1 --bf16 \
        --beam 1048576 --pids 0-39 \
        --out tetraminx/results/path_rank_probe_b20.json
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.khoruzhii_search import (KhoruzhiiSearchConfig, KhoruzhiiSolver,
                                     _q_predict, _qv_predict, _state_hash)
from tetraminx.puzzle import Tetraminx


def _load_solve_module():
    """Import `30_solve.py` for EndgameTable + load_model.

    The filename starts with a digit so it is not a legal module name; loading it by
    path is still better than copying the two helpers, because a copy is exactly the
    dual-code-path drift this repo has been bitten by before.
    """
    path = PROJECT / "tetraminx" / "scripts" / "30_solve.py"
    spec = importlib.util.spec_from_file_location("_tetra_solve30", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_pids(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-parent path-rank probe: the Beam-AVI go/no-go gate.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--blend", nargs="+", type=Path, default=None)
    ap.add_argument("--blend-weights", nargs="+", type=float, default=None)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--reference", type=Path,
                    default=PROJECT / "tetraminx" / "submissions" / "FINAL_tetraminx_28065.csv",
                    help="CSV of near-geodesic paths to replay; shortest available is best")
    ap.add_argument("--endgame", type=Path, default=None,
                    help="BFS table for the goal test; default data-dir/bfs_endgame.npz")
    ap.add_argument("--beam", type=int, default=2 ** 20)
    ap.add_argument("--max-steps", type=int, default=45)
    ap.add_argument("--history-depth", type=int, default=1)
    ap.add_argument("--qv-consistency", type=float, default=0.0)
    ap.add_argument("--pids", type=str, default="0-39")
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="solver internal_batch_size. NOT 30_solve.py's 32768 default: "
                         "the PieceTransformer's SDPA materializes a (chunk, heads, 88, 88) "
                         "score matrix, so chunk 32768 peaks at 8.79 GiB and Windows WDDM "
                         "PAGES instead of OOMing -- measured 9.2 s/step vs 1.0 s/step at "
                         "4096 on the same beam (B=65536, 4090). Same class as CLAUDE.md "
                         "rules 27 and 31. Throughput is flat from 2048 to 8192")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-beam-agg", type=int, default=65536,
                    help="aggregate only steps whose beam is at least this wide; "
                         "early steps have a tiny candidate pool and a coarse percentile")
    ap.add_argument("--verify-noperturb", type=int, default=None,
                    help="pid to solve twice (hook off, hook on) to prove the probe is inert")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    solve30 = _load_solve_module()

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    gens = np.array([puzzle.generators[nm] for nm in move_names], dtype=np.int64)
    solved = np.array(puzzle.solved_state, dtype=np.int64)

    eg_path = args.endgame or (args.data_dir / "bfs_endgame.npz")
    endgame = solve30.EndgameTable(Path(eg_path), args.device)
    print(f"endgame table: {endgame.hashes.numel():,} states <= d{endgame.max_depth}", flush=True)

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states_by_pid = {int(r["initial_state_id"]):
                         np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                         for r in csv.DictReader(f)}
    with open(args.reference, encoding="utf-8", newline="") as f:
        ref = {int(r["initial_state_id"]): r["path"].split(".")
               for r in csv.DictReader(f) if r["path"]}
    print(f"reference: {args.reference.name}, {len(ref)} pids, "
          f"{sum(len(v) for v in ref.values()):,} moves", flush=True)

    if args.blend:
        from tetraminx.models import BlendedQ
        members = [solve30.load_model(p, args.device, args.bf16, args.chunk_size)
                   for p in [args.checkpoint, *args.blend]]
        model = BlendedQ(members, weights=args.blend_weights).to(args.device).eval()
        if args.bf16:
            model = model.to(torch.bfloat16)
        print(f"blend: {len(members)} members, weights "
              + ", ".join(f"{w:.3f}" for w in model.weights.tolist()), flush=True)
    else:
        model = solve30.load_model(args.checkpoint, args.device, args.bf16, args.chunk_size)

    if args.qv_consistency != 0.0 and not getattr(model, "has_value_head", False):
        raise SystemExit("--qv-consistency needs a checkpoint trained with az_head=true")

    n_act = len(move_names)
    is_q = int(getattr(model, "output_dim", 1)) == n_act
    if not is_q:
        raise SystemExit("the probe needs a Q head (output_dim == n_actions)")

    solver = KhoruzhiiSolver(puzzle, model, device=args.device,
                             internal_batch_size=args.chunk_size,
                             use_q_function=True,
                             qv_consistency_lambda=args.qv_consistency,
                             state_dtype=torch.int8)
    dev = args.device
    gens_t = torch.from_numpy(gens).to(dev)
    lam = float(args.qv_consistency)

    def onpath_score(state_np: np.ndarray, action: int) -> float:
        """Score the single candidate (state, action) exactly as the beam scores it."""
        st = torch.as_tensor(state_np, dtype=torch.int8, device=dev).unsqueeze(0)
        if lam != 0.0:
            base = getattr(model, "_orig_mod", model)
            base.return_value = True
            q, v = _qv_predict(model, st, args.chunk_size, n_act)
            base.return_value = False
            s = q.float()
            s = s + lam * (s - (v.float() - 1.0).unsqueeze(1).expand_as(s)).abs()
        else:
            s = _q_predict(model, st, args.chunk_size, n_act).float()
        return float(s[0, action].item())

    # ---- the hook ---------------------------------------------------------
    # Filled in before each step, read inside `_do_greedy_step_q_topk`.
    ctx: dict = {}

    def hook(states, q_all, v_parent, score_flat):
        ctx["beam"] = int(states.size(0))
        ctx["n_cand"] = int(score_flat.numel())
        so = ctx.get("score_op")
        if so is not None:
            ctx["n_less"] = int((score_flat < so).sum().item())
            # Is the on-path PARENT still alive in the beam? Once the beam drops it the
            # percentile still answers the AVI question ("would the model rank the
            # on-path child well if it held the parent"), but survival is worth
            # recording separately -- cube444 found its beam beat the reference path
            # while dropping it, which reads like a bug and is not one.
            oph = ctx.get("onpath_hash")
            if oph is not None:
                h = _state_hash(states, solver.hash_vec)
                ctx["parent_alive"] = bool((h == oph).any().item())
        ctx["raw_q_min"] = float(q_all.float().min().item())

    if args.verify_noperturb is not None:
        pid = args.verify_noperturb
        s0 = states_by_pid[pid]
        cfg = KhoruzhiiSearchConfig(beam_width=min(args.beam, 65536), num_steps=args.max_steps,
                                    history_depth=args.history_depth)
        gf = lambda s: endgame.contains(s)
        solver.step_probe = None
        ok_a, _, path_a = solver.solve(s0, cfg, goal_check_fn=gf)
        ctx.clear()
        solver.step_probe = hook
        ok_b, _, path_b = solver.solve(s0, cfg, goal_check_fn=gf)
        solver.step_probe = None
        same = (ok_a == ok_b) and (path_a == path_b)
        print(f"no-perturbation check pid {pid}: solved={ok_a}/{ok_b} "
              f"len={len(path_a)}/{len(path_b)} identical={same}", flush=True)
        if not same:
            raise SystemExit("PROBE PERTURBS THE SEARCH -- fix before trusting any number")

    pids = [p for p in parse_pids(args.pids) if p in states_by_pid and p in ref]
    print(f"probing {len(pids)} pids at B={args.beam:,} "
          f"(hd={args.history_depth}, qv={lam})", flush=True)

    solver.step_probe = hook
    records: list[dict] = []
    t_start = time.time()

    for pi, pid in enumerate(pids):
        s0 = states_by_pid[pid]
        names = ref[pid]
        acts = [name_to_idx[nm] for nm in names]
        L = len(acts)

        # On-path states, and a replay check: a torn or wrong reference row would
        # silently turn the probe into noise.
        onpath = [s0]
        cur = s0
        for a in acts:
            cur = cur[gens[a]]
            onpath.append(cur)
        if not np.array_equal(cur, solved):
            print(f"[skip] pid {pid}: reference path does not solve", flush=True)
            continue

        states = torch.as_tensor(s0, dtype=torch.int8, device=dev).unsqueeze(0).clone()
        hash_log: deque[torch.Tensor] = deque(maxlen=max(4, args.history_depth))
        excluded0 = torch.empty(0, dtype=torch.int64, device=dev)
        t0 = time.time()
        steps: list[dict] = []

        for j in range(args.max_steps):
            ctx.clear()
            if j < L:
                ctx["score_op"] = onpath_score(onpath[j], acts[j])
                oh = _state_hash(torch.as_tensor(onpath[j], dtype=torch.int8,
                                                 device=dev).unsqueeze(0), solver.hash_vec)
                ctx["onpath_hash"] = oh[0]
            excluded = excluded0
            if args.history_depth > 0 and hash_log:
                excluded = torch.cat(list(hash_log)[-args.history_depth:])

            states, _y, _moves, _idx = solver._do_greedy_step(states, excluded, args.beam)
            if states.numel() == 0:
                break
            hash_log.append(_state_hash(states, solver.hash_vec))

            if "n_less" in ctx:
                steps.append(dict(
                    step=j,
                    remaining=L - j,                 # true distance of the on-path PARENT
                    beam=ctx["beam"],
                    n_cand=ctx["n_cand"],
                    pct=ctx["n_less"] / max(1, ctx["n_cand"]),
                    parent_alive=ctx.get("parent_alive"),
                ))
            if bool(endgame.contains(states).any().item()):
                break

        wall = time.time() - t0
        depth_reached = steps[-1]["step"] + 1 if steps else 0
        records.append(dict(pid=pid, ref_len=L, steps=steps, wall=wall,
                            depth_reached=depth_reached))
        full = [s for s in steps if s["beam"] >= args.min_beam_agg]
        head = np.mean([s["pct"] for s in full]) if full else float("nan")
        print(f"[{pi+1}/{len(pids)}] pid {pid:4d} ref_len={L:2d} steps={len(steps):2d} "
              f"full={len(full):2d} mean_pct(full)={head:.4f} {wall:.1f}s", flush=True)

    solver.step_probe = None

    # ---- aggregate --------------------------------------------------------
    flat = [s for r in records for s in r["steps"]]
    full = [s for s in flat if s["beam"] >= args.min_beam_agg]
    summary: dict = dict(
        beam=args.beam, pids=len(records), history_depth=args.history_depth,
        qv_consistency=lam, reference=str(args.reference),
        checkpoint=str(args.checkpoint), blend=[str(b) for b in (args.blend or [])],
        min_beam_agg=args.min_beam_agg,
        n_steps_all=len(flat), n_steps_full=len(full),
        wall=time.time() - t_start,
    )
    if full:
        p = np.array([s["pct"] for s in full], dtype=np.float64)
        n = p.size
        se = p.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        summary["mean_pct_full"] = float(p.mean())
        summary["median_pct_full"] = float(np.median(p))
        summary["z_vs_null"] = float((0.5 - p.mean()) / se) if se and se > 0 else float("nan")
        summary["frac_top1e-3"] = float((p < 1e-3).mean())
        alive = [s["parent_alive"] for s in full if s["parent_alive"] is not None]
        summary["frac_parent_alive"] = float(np.mean(alive)) if alive else float("nan")

    bands = [(28, 99), (25, 27), (22, 24), (19, 21), (16, 18), (13, 15), (7, 12), (0, 6)]
    band_rows = []
    for lo, hi in bands:
        sel = [s for s in flat if lo <= s["remaining"] <= hi]
        selfull = [s for s in sel if s["beam"] >= args.min_beam_agg]
        if not sel:
            continue
        p_all = np.array([s["pct"] for s in sel], dtype=np.float64)
        p_full = np.array([s["pct"] for s in selfull], dtype=np.float64)
        band_rows.append(dict(
            band=f"{lo}-{hi if hi < 99 else '+'}", n=len(sel), n_full=len(selfull),
            mean_pct=float(p_all.mean()),
            mean_pct_full=float(p_full.mean()) if p_full.size else float("nan"),
            median_beam=float(np.median([s["beam"] for s in sel])),
        ))
    summary["bands"] = band_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, records=records), f)

    print("\n=== cross-parent percentile of the on-path candidate (null = 0.5) ===")
    print(f"{'remaining d':>12} {'n':>7} {'n_full':>7} {'mean_pct':>10} "
          f"{'mean_full':>10} {'med_beam':>12}")
    for r in band_rows:
        print(f"{r['band']:>12} {r['n']:>7} {r['n_full']:>7} {r['mean_pct']:>10.5f} "
              f"{r['mean_pct_full']:>10.5f} {r['median_beam']:>12,.0f}")
    if "mean_pct_full" in summary:
        print(f"\nfull-width steps only: mean {summary['mean_pct_full']:.5f}  "
              f"median {summary['median_pct_full']:.5f}  "
              f"z vs 0.5 null {summary['z_vs_null']:.1f}")
        print(f"on-path parent still in beam: {summary['frac_parent_alive']:.3f} of steps")
    print(f"\nwrote {args.out}")
    print("\nVERDICT RULE: mean percentile near 0.5 in the deep bands means there is no "
          "signal to bootstrap -- STOP. cube444 saw 0.034 (z = 30.2) and proceeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
