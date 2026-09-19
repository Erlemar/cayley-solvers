"""Beam-AVI generation: run the deployed beam and harvest its Bellman backup.

The beam already computes everything the backup needs and throws it away. At step j it
forwards B parents, keeps the top B, and at step j+1 forwards exactly those survivors --
so `min_a' Q(child, a')` is on the GPU whether we want it or not:

    row  =  state  = the PARENT at step j
            action = the move that produced the survivor
            target = 1 + min_a Q(survivor, a)        from step j+1's forward

WHY THIS IS SIMPLER HERE THAN ON CUBE444. The reference implementation pairs step-j
parents to step-j+1 survivors BY POSITION and has to drop a batch whenever the driver
filtered anything in between -- a silent-mislabel hazard. Tetraminx is a permutation
puzzle whose generators all have order 3, so the parent is recovered from the child by
one gather with the inverse move:

    child = parent[gen[m]]      =>      parent = child[gen[inv[m]]]

Every row is therefore local to one step. `--self-test` proves the identity against the
beam's own parent indices before any data is written.

THREE STREAMS, and the second exists to correct the first:

  sparse  every harvested survivor. BIASED: a child survives because the beam ranked
          Q(parent, action) low, and the target then reads that same child's min Q, so
          the stream is systematically optimistic and self-confirming. Cheap and huge.
  dense   `--full-expand P` parents sampled UNIFORMLY, without reference to Q, all 24
          children expanded explicitly. Unbiased, separately weightable. Costs P*24/B.
  value   V(child) = 1 + min_a Q(child, a) -- the SAME number the sparse target uses,
          so it is free. Needed because `qv_consistency` reads |Q - (V-1)| on an
          ABSOLUTE scale: AVI moves Q's level, and a value head left behind would make
          the deployed penalty start charging correct children.

Targets use RAW Q -- before the qv-consistency adjustment and before any
non-backtracking ban. Both are search heuristics; neither changes the MDP, and the
move back to the parent is a legitimate member of the Bellman min. Training on the
adjusted score would be a self-confirming loop with no fixed point at the true distance.

EXACT OVERRIDE. Where a survivor is inside the d<=6 BFS table its distance is KNOWN, so
the target is the table value rather than a bootstrap. The beam terminates on a table
hit, so the last layers of every solve are exactly labelled -- strictly stronger
grounding than the reference method's single `is_solved` terminal case.

    .venv/Scripts/python.exe tetraminx/scripts/73_gen_harvest.py \
        --checkpoint tetraminx/models/mx_tf_az/epoch_1500.pt \
        --blend tetraminx/models/mx_resmlp_az/best.pt --blend-weights 0.8 0.2 \
        --qv-consistency 0.3 --history-depth 1 --bf16 --chunk-size 4096 \
        --beam 262144 --round 0 --n-pids 50 --full-expand 256 \
        --out tetraminx/runs/avi/shards/r000.pt
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.khoruzhii_search import KhoruzhiiSolver, _q_predict, _state_hash
from tetraminx.puzzle import Tetraminx

# The 15-pid acceptance gate. Passed as an explicit EXCLUSION LIST, never as a count:
# the reference implementation held out by its own seeded permutation and left 47 of
# its 54 gate pids inside the training set.
GATE_PIDS = (0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 990, 995, 999)


def _load_solve_module():
    path = PROJECT / "tetraminx" / "scripts" / "30_solve.py"
    spec = importlib.util.spec_from_file_location("_tetra_solve30", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest Bellman targets from the deployed beam.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--blend", nargs="+", type=Path, default=None)
    ap.add_argument("--blend-weights", nargs="+", type=float, default=None)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--endgame", type=Path, default=None)
    ap.add_argument("--beam", type=int, default=262144)
    ap.add_argument("--max-steps", type=int, default=45)
    ap.add_argument("--history-depth", type=int, default=1)
    ap.add_argument("--qv-consistency", type=float, default=0.0)
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="see the comment in 30_solve.py -- 32768 is 9.2x slower here")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--round", type=int, required=True,
                    help="round number; the scramble sample is seeded FROM IT, so a "
                         "fixed seed would silently re-draw the same pids every round")
    ap.add_argument("--n-pids", type=int, default=50)
    ap.add_argument("--pids", type=str, default="",
                    help="explicit pid list; overrides the round-seeded sample")
    ap.add_argument("--min-ref-len", type=int, default=27,
                    help="only sample scrambles at least this deep. pids 0-35 are the "
                         "SHALLOW block (mean reference 15.2 vs 28.6 elsewhere) and "
                         "harvesting them would re-weight the shard toward the one band "
                         "the model already handles")
    ap.add_argument("--reference", type=Path,
                    default=PROJECT / "tetraminx" / "submissions" / "FINAL_tetraminx_28065.csv",
                    help="used ONLY to measure scramble depth for --min-ref-len")
    ap.add_argument("--rows-per-step", type=int, default=8192)
    ap.add_argument("--full-expand", type=int, default=256)
    ap.add_argument("--clamp", type=float, default=40.0)
    ap.add_argument("--self-test", action="store_true",
                    help="assert the inverse-move parent reconstruction against the "
                         "beam's own parent indices, then exit")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    solve30 = _load_solve_module()
    dev = args.device

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    n_act = len(move_names)
    inv_idx = torch.tensor([name_to_idx[puzzle.inverse_name(nm)] for nm in move_names],
                           dtype=torch.int64, device=dev)

    eg_path = args.endgame or (args.data_dir / "bfs_endgame.npz")
    endgame = solve30.EndgameTable(Path(eg_path), dev)
    print(f"endgame table: {endgame.hashes.numel():,} states <= d{endgame.max_depth}", flush=True)

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states_by_pid = {int(r["initial_state_id"]):
                         np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                         for r in csv.DictReader(f)}

    if args.blend:
        from tetraminx.models import BlendedQ
        members = [solve30.load_model(p, dev, args.bf16, args.chunk_size)
                   for p in [args.checkpoint, *args.blend]]
        model = BlendedQ(members, weights=args.blend_weights).to(dev).eval()
        if args.bf16:
            model = model.to(torch.bfloat16)
        print(f"target net: blend of {len(members)}, weights "
              + ", ".join(f"{w:.3f}" for w in model.weights.tolist()), flush=True)
    else:
        model = solve30.load_model(args.checkpoint, dev, args.bf16, args.chunk_size)

    solver = KhoruzhiiSolver(puzzle, model, device=dev,
                             internal_batch_size=args.chunk_size, use_q_function=True,
                             qv_consistency_lambda=args.qv_consistency,
                             state_dtype=torch.int8)

    # ---- self-test: parent = apply(child, inv[move]) ----------------------
    if args.self_test:
        s0 = torch.as_tensor(states_by_pid[50], dtype=torch.int8, device=dev).unsqueeze(0)
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        st = s0
        for j in range(5):
            prev = st
            st, _y, mv, idx = solver._do_greedy_step(st, empty, 4096)
            recon = solver._apply_move(st, inv_idx[mv])
            truth = prev.index_select(0, idx)
            ok = bool(torch.equal(recon, truth))
            print(f"  step {j}: beam={st.size(0):6,d} parent-reconstruction exact={ok}", flush=True)
            if not ok:
                bad = int((recon != truth).any(dim=1).sum().item())
                raise SystemExit(f"SELF-TEST FAILED at step {j}: {bad} rows mismatch")
        print("self-test PASSED: inverse-move reconstruction is exact", flush=True)
        return 0

    # ---- pick scrambles ---------------------------------------------------
    if args.pids:
        pids = [int(x) for x in args.pids.replace(" ", "").split(",") if x]
    else:
        with open(args.reference, encoding="utf-8", newline="") as f:
            ref_len = {int(r["initial_state_id"]): len(r["path"].split("."))
                       for r in csv.DictReader(f) if r["path"]}
        pool = [p for p in sorted(states_by_pid)
                if p not in GATE_PIDS and ref_len.get(p, 0) >= args.min_ref_len]
        rng = np.random.default_rng(1000 + args.round)      # seeded FROM the round
        pids = sorted(rng.choice(pool, size=min(args.n_pids, len(pool)), replace=False).tolist())
    print(f"round {args.round}: {len(pids)} scrambles, B={args.beam:,}, "
          f"rows/step={args.rows_per_step:,}, full-expand={args.full_expand}", flush=True)

    # ---- harvest ----------------------------------------------------------
    ctx: dict = {}
    sp_child, sp_move, sp_tgt = [], [], []
    dn_state, dn_tgt = [], []
    n_exact = [0]

    def bellman_target(q_raw: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """1 + min_a Q(s,a), overridden by the exact table where it applies."""
        t = 1.0 + q_raw.float().min(dim=1).values
        d = endgame.lookup(states)                       # -1 when absent
        hit = d >= 0
        n_exact[0] += int(hit.sum().item())
        return torch.where(hit, d.float(), t).clamp_(0.0, args.clamp)

    def hook(states, q_all, v_parent, score_flat):
        in_moves = ctx.get("in_moves")
        if in_moves is None:                              # seed layer: no parent yet
            return
        B = states.size(0)
        tgt = bellman_target(q_all, states)

        n = min(args.rows_per_step, B)
        sel = torch.randint(0, B, (n,), device=states.device)
        sp_child.append(states.index_select(0, sel).to(torch.uint8).cpu())
        sp_move.append(in_moves.index_select(0, sel).to(torch.int8).cpu())
        sp_tgt.append(tgt.index_select(0, sel).cpu())

        # Unbiased stream: parents chosen WITHOUT reference to Q, all 24 expanded.
        p = args.full_expand
        if p > 0:
            par = states.index_select(0, torch.randint(0, B, (p,), device=states.device))
            kids = par[:, solver.all_moves].reshape(-1, solver.state_size)   # (p*24, S)
            q_kids = _q_predict(solver.model, kids, args.chunk_size, n_act)
            t_kids = bellman_target(q_kids, kids).view(p, n_act)
            dn_state.append(par.to(torch.uint8).cpu())
            dn_tgt.append(t_kids.cpu())

    solver.step_probe = hook
    t_start = time.time()
    empty = torch.empty(0, dtype=torch.int64, device=dev)

    for pi, pid in enumerate(pids):
        states = torch.as_tensor(states_by_pid[pid], dtype=torch.int8, device=dev).unsqueeze(0)
        hash_log: deque[torch.Tensor] = deque(maxlen=max(4, args.history_depth))
        ctx.clear()
        t0, n0 = time.time(), len(sp_child)
        for j in range(args.max_steps):
            excluded = empty
            if args.history_depth > 0 and hash_log:
                excluded = torch.cat(list(hash_log)[-args.history_depth:])
            states, _y, moves, _idx = solver._do_greedy_step(states, excluded, args.beam)
            if states.numel() == 0:
                break
            ctx["in_moves"] = moves
            hash_log.append(_state_hash(states, solver.hash_vec))
            if bool(endgame.contains(states).any().item()):
                break
        # FINAL LAYER. The hook fires from inside _do_greedy_step, i.e. one step LATE:
        # it sees a layer only when that layer is used as the parents of the next call.
        # The loop breaks the moment survivors reach the table, so without this the
        # last -- and most valuable -- layer is never harvested: those are exactly the
        # rows the d<=6 table labels EXACTLY rather than by bootstrap. Measured: 0
        # exact overrides before this, which is what exposed it. One extra forward/pid.
        if states.numel() > 0 and ctx.get("in_moves") is not None:
            q_final = _q_predict(solver.model, states, args.chunk_size, n_act)
            hook(states=states, q_all=q_final, v_parent=None, score_flat=None)
        rows = sum(t.size(0) for t in sp_child[n0:])
        print(f"[{pi+1}/{len(pids)}] pid {pid:4d} steps={j+1:2d} rows={rows:,} "
              f"{time.time()-t0:.1f}s", flush=True)

    solver.step_probe = None

    blob = {
        "sparse": {"child": torch.cat(sp_child), "move": torch.cat(sp_move),
                   "target": torch.cat(sp_tgt)},
        "dense": ({"state": torch.cat(dn_state), "target": torch.cat(dn_tgt)}
                  if dn_state else None),
        "meta": dict(round=args.round, pids=pids, beam=args.beam,
                     history_depth=args.history_depth, qv_consistency=args.qv_consistency,
                     checkpoint=str(args.checkpoint),
                     blend=[str(b) for b in (args.blend or [])],
                     rows_per_step=args.rows_per_step, full_expand=args.full_expand,
                     clamp=args.clamp, exact_overrides=n_exact[0],
                     wall=time.time() - t_start),
    }
    # The parent is NOT stored: it is one gather from the child, and dropping it halves
    # the shard. The trainer reconstructs it the same way (see 74_train_avi.py).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)

    s_t = blob["sparse"]["target"]
    print(f"\nsparse rows {s_t.numel():,}  target mean {s_t.mean():.3f} "
          f"min {s_t.min():.1f} max {s_t.max():.1f}")
    if blob["dense"] is not None:
        print(f"dense rows  {blob['dense']['state'].size(0):,} x {n_act}")
    print(f"exact table overrides: {n_exact[0]:,}")
    print(f"wrote {args.out}  ({args.out.stat().st_size/2**20:.0f} MiB, "
          f"{time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
