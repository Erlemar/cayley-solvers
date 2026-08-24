"""Matched-node beam A/B: pure all-neighbors (alpha=1) vs per-child V.

The decisive test for the all-neighbors Q-head idea (DeepCubeAQ / "predicts all 24
child distances in one forward pass"). Both arms run the SAME beam machinery at the
SAME beam width B (matched node budget); the only difference is how a child is scored:

  Arm A (per-child V):  KhoruzhiiSolver(model=V) -- score(child) = V(child).
                        24 forward-passes-worth of compute per expanded node.
  Arm B (all-neighbors): QShortlisterSolver(teacher=V, student=24-head, alpha=1.0).
                        At alpha=1, target_k = alpha*B = B, so the "shortlist" IS
                        top-B by the head's predicted child-values; the teacher
                        re-score then selects top-B-of-B (identity) and next_values
                        is discarded. => selection is 100% by the head, ZERO V
                        influence. One forward pass per parent.

We compare RAW beam path length (the beam's own search quality, before any
post-processing). matched NODES, not wall-clock: on our selection-bound TPU kernel
the forward pass is only ~8% of the step, so the throughput thesis doesn't pay off
for us -- the only question worth answering is whether the head's ranking is good
enough to not lengthen paths.

  B ~= A  => all-neighbors preserves quality; the mechanism is viable.
  B >> A  => joins the qshort regression; the competitor's sub-80k is width /
             engineering, not the head being a quality win.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/allneighbor_ab.py \
      --teacher megaminx/models/m_az_v4_v_only_e99.pt \
      --student megaminx/models/m_anv_azv4/epoch_0119.pt \
      --pids 0,200,400,600,800,900,950,990 --beam 65536 --max-steps 120 --bf16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "beam_lab"))

from beam_search import KhoruzhiiSearchConfig, KhoruzhiiSolver  # noqa: E402
from beam_search_qshort import QShortlisterSolver  # noqa: E402
from cayley.search import load_model_checkpoint  # noqa: E402
from cayley.verify import load_test_states, verify_path  # noqa: E402
from megaminx.puzzle import Megaminx  # noqa: E402


def _raw_solve(solver, state, cfg):
    """Return (found, raw_len, names, wall). raw_len = beam path length pre-postproc."""
    t0 = time.time()
    found, plen, names, _prof = solver.solve(state, cfg)
    return found, plen, names, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path,
                    default=PROJECT / "models" / "m_az_v4_v_only_e99.pt")
    ap.add_argument("--student", type=Path,
                    default=PROJECT / "models" / "m_anv_azv4" / "epoch_0119.pt")
    ap.add_argument("--pids", type=str, default="0,200,400,600,800,900,950,990")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--internal-batch", type=int, default=16384)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    pids = [int(x) for x in args.pids.split(",") if x.strip()]

    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    teacher = load_model_checkpoint(args.teacher, device=device, dtype=dtype)
    student = load_model_checkpoint(args.student, device=device, dtype=dtype)
    # Avoid double-chunking (the solver chunks at internal_batch_size).
    for m in (teacher, student):
        if hasattr(m, "inference_chunk_size"):
            m.inference_chunk_size = None
    t_dim = getattr(teacher, "output_dim", 1)
    s_dim = getattr(student, "output_dim", 1)
    print(f"teacher V: {sum(p.numel() for p in teacher.parameters()):,} params, "
          f"output_dim={t_dim}", flush=True)
    print(f"student Q (all-neighbors head): "
          f"{sum(p.numel() for p in student.parameters()):,} params, "
          f"output_dim={s_dim}", flush=True)
    if t_dim != 1 or s_dim != len(puzzle.move_names):
        print("ERROR: teacher must be V (output_dim=1) and student a 24-head", file=sys.stderr)
        return 2

    solver_a = KhoruzhiiSolver(puzzle, teacher, device=device,
                               internal_batch_size=args.internal_batch, profile=False)
    solver_b = QShortlisterSolver(puzzle, teacher=teacher, student=student,
                                  device=device, internal_batch_size=args.internal_batch,
                                  alpha=1.0)
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps,
                                num_attempts=1, internal_batch_size=args.internal_batch)

    print(f"\nbeam={args.beam} max_steps={args.max_steps}  "
          f"A=per-child V   B=all-neighbors(alpha=1, no V re-score)", flush=True)
    print(f"{'pid':>5s} {'A_len':>7s} {'B_len':>7s} {'B-A':>6s} "
          f"{'A_s':>6s} {'B_s':>6s}  note", flush=True)
    print("-" * 56, flush=True)

    sum_a = sum_b = 0
    both = 0
    a_only = b_only = neither = 0
    deltas = []
    for pid in pids:
        state = states[pid]
        fa, la, na, wa = _raw_solve(solver_a, state, cfg)
        if fa and not verify_path(puzzle, state, na).ok:
            fa = False
        fb, lb, nb, wb = _raw_solve(solver_b, state, cfg)
        if fb and not verify_path(puzzle, state, nb).ok:
            fb = False

        note = ""
        if fa and fb:
            both += 1
            sum_a += la
            sum_b += lb
            deltas.append(lb - la)
            d = lb - la
            dstr = f"{d:+d}"
            note = "tie" if d == 0 else ("B worse" if d > 0 else "B better")
        elif fa and not fb:
            a_only += 1
            dstr = "B-FAIL"
            note = "all-neighbors did not solve"
        elif fb and not fa:
            b_only += 1
            dstr = "A-FAIL"
        else:
            neither += 1
            dstr = "both-FAIL"
        astr = str(la) if fa else "-"
        bstr = str(lb) if fb else "-"
        print(f"{pid:>5d} {astr:>7s} {bstr:>7s} {dstr:>6s} "
              f"{wa:>6.1f} {wb:>6.1f}  {note}", flush=True)

    print("-" * 56, flush=True)
    print(f"solved: A={both + a_only}/{len(pids)}  B={both + b_only}/{len(pids)}  "
          f"(both={both}, A-only={a_only}, B-only={b_only}, neither={neither})", flush=True)
    if both:
        net = sum_b - sum_a
        print(f"over {both} commonly-solved pids: A={sum_a}  B={sum_b}  "
              f"net B-A = {net:+d} ({net / both:+.1f}/pid)", flush=True)
        worse = sum(1 for d in deltas if d > 0)
        better = sum(1 for d in deltas if d < 0)
        tie = sum(1 for d in deltas if d == 0)
        print(f"per-pid: {worse} B-worse, {tie} tie, {better} B-better; "
              f"max B-worse={max(deltas):+d}, max B-better={min(deltas):+d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
