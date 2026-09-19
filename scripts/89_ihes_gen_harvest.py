"""IHES Beam-AVI generation: run the beam with the current model and harvest its Bellman backup.

Port of `tetraminx/scripts/73_gen_harvest.py` (method: `BEAM_AVI_METHOD.md`, cube444 -4.2%
below its floor; the tetraminx port was REJECTED for Q-space flattening -- see
`tetraminx/BEAM_AVI_PLAN.md` s5b and run `92_ihes_level_check.py` after every round).

At beam step j+1 the search forwards exactly the survivors of step j, so for every survivor
(child) the backup is already on the GPU:

    row = (parent, move, 1 + min_a Q(child, a))      parent = child[gen[inv[move]]]

IHES is a permutation puzzle, so the parent is ONE inverse-move gather from the child and
every row is local to one step (`--self-test` proves it against the beam's own indices).

Streams:
  sparse  every step, `--rows-per-step` random survivors. Optimistic by construction (a
          child survives because the beam liked it).
  dense   `--full-expand P` survivors drawn WITHOUT reference to Q, all 18 children
          forwarded, all 18 targets stored. The unbiased correction.
Targets use RAW Q (no qv-consistency, no bans) and are overridden by the exact d<=6
Zobrist table wherever the child is inside it. The gate pids of `data/ihes_gate108.json`
are EXCLUDED explicitly; the scramble sample is seeded from the round number.

    python scripts/89_ihes_gen_harvest.py --checkpoint exports/ihes_tf_v1b/ihes_tf_v1b.pt \
        --round 0 --n-pids 50 --beam 262144 --out runs/ihes_avi/A/shards/r000.pt
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.khoruzhii_search import KhoruzhiiSolver, _q_predict, _state_hash  # noqa: E402
from cayley.puzzle import PictureCube  # noqa: E402


def _solve84():
    """EndgameTable + load_model from the IHES solve driver (same code the gate uses)."""
    path = PROJECT / "scripts" / "84_solve_tf.py"
    spec = importlib.util.spec_from_file_location("_ihes_solve84", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest Bellman targets from the IHES beam.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--endgame", type=Path, default=None)
    ap.add_argument("--beam", type=int, default=262144)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--history-depth", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--round", type=int, required=True,
                    help="scramble sample is seeded FROM the round (a fixed seed would "
                         "silently re-draw the same pids every round)")
    ap.add_argument("--n-pids", type=int, default=50)
    ap.add_argument("--pids", type=str, default="", help="explicit list; overrides the sample")
    ap.add_argument("--min-ref-len", type=int, default=21,
                    help="only harvest scrambles whose floor path is at least this long "
                         "(the 52 pids <= 20 are the shallow block)")
    ap.add_argument("--floor", type=Path,
                    default=PROJECT / "submissions" / "ihes_20260912_1410_verified.csv",
                    help="used ONLY to measure scramble depth for --min-ref-len")
    ap.add_argument("--exclude", type=Path, default=PROJECT / "data" / "ihes_gate108.json",
                    help="JSON with 'pids' that must never be harvested (the gate)")
    ap.add_argument("--rows-per-step", type=int, default=8192)
    ap.add_argument("--full-expand", type=int, default=256)
    ap.add_argument("--clamp", type=float, default=32.0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.self_test and args.out is None:
        raise SystemExit("--out is required unless --self-test")

    s84 = _solve84()
    dev = args.device
    bf16 = not args.no_bf16
    puzzle = PictureCube.load(args.data_dir / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    n_act = len(move_names)
    inv_idx = torch.tensor([name_to_idx[puzzle.inverse_name(nm)] for nm in move_names],
                           dtype=torch.int64, device=dev)

    endgame = s84.EndgameTable(Path(args.endgame or args.data_dir / "bfs_table_d6_hash.npz"), dev)
    print(f"endgame table: {endgame.hashes.numel():,} states <= d{endgame.max_depth}", flush=True)

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states_by_pid = {int(r["initial_state_id"]):
                         np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                         for r in csv.DictReader(f)}

    model, ck = s84.load_model(args.checkpoint, dev, bf16, args.chunk_size)
    if int(getattr(model, "output_dim", 1)) != n_act:
        raise SystemExit("Beam-AVI needs an all-actions Q checkpoint")
    solver = KhoruzhiiSolver(puzzle, model, device=dev, internal_batch_size=args.chunk_size,
                             use_q_function=True, qv_consistency_lambda=0.0,
                             state_dtype=torch.int8)
    print(f"target net: {args.checkpoint} (epoch {ck.get('epoch')}, "
          f"bellman_step {ck.get('bellman_step')}, avi_round {ck.get('avi', {}).get('round')})",
          flush=True)

    if args.self_test:
        s0 = torch.as_tensor(states_by_pid[106], dtype=torch.int8, device=dev).unsqueeze(0)
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        st = s0
        for j in range(6):
            prev = st
            st, _y, mv, idx = solver._do_greedy_step(st, empty, 4096)
            recon = solver._apply_move(st, inv_idx[mv])
            ok = bool(torch.equal(recon, prev.index_select(0, idx)))
            print(f"  step {j}: beam={st.size(0):6,d} parent-reconstruction exact={ok}", flush=True)
            if not ok:
                raise SystemExit(f"SELF-TEST FAILED at step {j}")
        print("self-test PASSED: inverse-move reconstruction is exact", flush=True)
        return 0

    # ---- scrambles -----------------------------------------------------------
    excluded = set(json.loads(args.exclude.read_text(encoding="utf-8"))["pids"]) \
        if args.exclude and str(args.exclude).lower() != "none" else set()
    if args.pids:
        pids = [int(x) for x in args.pids.replace(" ", "").split(",") if x]
        clash = sorted(set(pids) & excluded)
        if clash:
            raise SystemExit(f"--pids includes excluded gate pids {clash}")
    else:
        with open(args.floor, encoding="utf-8", newline="") as f:
            ref_len = {int(r["initial_state_id"]): len(r["path"].split("."))
                       for r in csv.DictReader(f) if r["path"]}
        pool = [p for p in sorted(states_by_pid)
                if p not in excluded and ref_len.get(p, 0) >= args.min_ref_len]
        rng = np.random.default_rng(1000 + args.round)
        pids = sorted(rng.choice(pool, size=min(args.n_pids, len(pool)), replace=False).tolist())
    print(f"round {args.round}: {len(pids)} scrambles (excluded {len(excluded)} gate pids), "
          f"B={args.beam:,}, rows/step={args.rows_per_step:,}, full-expand={args.full_expand}",
          flush=True)

    # ---- harvest ---------------------------------------------------------------
    ctx: dict = {}
    sp_child, sp_move, sp_tgt = [], [], []
    dn_state, dn_tgt = [], []
    n_exact = [0]

    def bellman_target(q_raw: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """1 + min_a Q(s, a), replaced by the exact table depth where s is in the table."""
        t = 1.0 + q_raw.float().min(dim=1).values
        d = endgame.lookup(states)
        hit = d >= 0
        n_exact[0] += int(hit.sum().item())
        return torch.where(hit, d.float(), t).clamp_(0.0, args.clamp)

    def hook(states, q_all, v_parent, score_flat):
        in_moves = ctx.get("in_moves")
        if in_moves is None:                  # seed layer: the scramble has no parent
            return
        B = states.size(0)
        tgt = bellman_target(q_all, states)
        n = min(args.rows_per_step, B)
        sel = torch.randint(0, B, (n,), device=states.device)
        sp_child.append(states.index_select(0, sel).to(torch.uint8).cpu())
        sp_move.append(in_moves.index_select(0, sel).to(torch.int8).cpu())
        sp_tgt.append(tgt.index_select(0, sel).cpu())
        p = args.full_expand
        if p > 0:
            par = states.index_select(0, torch.randint(0, B, (p,), device=states.device))
            kids = par[:, solver.all_moves].reshape(-1, solver.state_size)
            q_kids = _q_predict(solver.model, kids, args.chunk_size, n_act)
            dn_state.append(par.to(torch.uint8).cpu())
            dn_tgt.append(bellman_target(q_kids, kids).view(p, n_act).cpu())

    solver.step_probe = hook
    t_start = time.time()
    empty = torch.empty(0, dtype=torch.int64, device=dev)
    solved = 0
    for pi, pid in enumerate(pids):
        states = torch.as_tensor(states_by_pid[pid], dtype=torch.int8, device=dev).unsqueeze(0)
        hash_log: deque[torch.Tensor] = deque(maxlen=max(4, args.history_depth))
        ctx.clear()
        t0, n0 = time.time(), len(sp_child)
        hit = False
        for j in range(args.max_steps):
            excl = empty
            if args.history_depth > 0 and hash_log:
                excl = torch.cat(list(hash_log)[-args.history_depth:])
            states, _y, moves, _idx = solver._do_greedy_step(states, excl, args.beam)
            if states.numel() == 0:
                break
            ctx["in_moves"] = moves
            hash_log.append(_state_hash(states, solver.hash_vec))
            if bool((endgame.lookup(states) >= 0).any().item()):
                hit = True
                break
        # the hook sees a layer only when it is used as parents; harvest the last one here
        if states.numel() > 0 and ctx.get("in_moves") is not None:
            q_final = _q_predict(solver.model, states, args.chunk_size, n_act)
            hook(states=states, q_all=q_final, v_parent=None, score_flat=None)
        solved += int(hit)
        rows = sum(t.size(0) for t in sp_child[n0:])
        print(f"[{pi+1}/{len(pids)}] pid {pid:4d} steps={j+1:2d} reached_table={int(hit)} "
              f"rows={rows:,} {time.time()-t0:.1f}s", flush=True)
    solver.step_probe = None

    blob = {
        "sparse": {"child": torch.cat(sp_child), "move": torch.cat(sp_move),
                   "target": torch.cat(sp_tgt).float()},
        "dense": ({"state": torch.cat(dn_state), "target": torch.cat(dn_tgt).float()}
                  if dn_state else None),
        "meta": dict(round=args.round, pids=pids, beam=args.beam, bf16=bf16,
                     history_depth=args.history_depth, checkpoint=str(args.checkpoint),
                     rows_per_step=args.rows_per_step, full_expand=args.full_expand,
                     clamp=args.clamp, exact_overrides=n_exact[0], reached_table=solved,
                     excluded=str(args.exclude), wall=time.time() - t_start),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    s_t = blob["sparse"]["target"]
    print(f"\nsparse rows {s_t.numel():,}  target mean {s_t.mean():.3f} min {s_t.min():.1f} "
          f"max {s_t.max():.1f} | dense rows "
          f"{0 if blob['dense'] is None else blob['dense']['state'].size(0):,} | exact "
          f"overrides {n_exact[0]:,} | reached table {solved}/{len(pids)}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.0f} MiB, "
          f"{time.time() - t_start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
