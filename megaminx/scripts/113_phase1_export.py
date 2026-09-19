"""Run Phase 1 only (locally) and export the H-state for a TPU Phase-2 run.

Hybrid two-phase @ big beam: Phase 1 barely scales with beam, so we run it on the
local GPU at a modest beam, then ship its end-state (which is in subgroup H) to the
TPU to run Phase 2 at a much wider beam. This script does the Phase-1 half and emits,
per pid: the Phase-1 move list (full 24-gen names) and the H-state (the scramble after
applying Phase 1), to be concatenated with the TPU Phase-2 path and verified.

    .venv/Scripts/python.exe megaminx/scripts/113_phase1_export.py \
        --phase1-checkpoint megaminx/models/m_phase1_v0/epoch_1499.pt \
        --pids 993 --beam 131072 --bf16 \
        --out megaminx/submissions/phase1_export_pid993.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.verify import load_test_states
from megaminx.puzzle import Megaminx
from megaminx.two_phase import (
    load_two_phase_model,
    make_frozen_goal_check,
    movable_frozen_positions,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-checkpoint", required=True, type=Path)
    ap.add_argument("--pids", type=str, default="993")
    ap.add_argument("--beam", type=int, default=131072)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--internal-batch-size", type=int, default=2**14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    full = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    movable, frozen = movable_frozen_positions(full)
    states = load_test_states(PROJECT / "data" / "test.csv")
    pids = [int(x) for x in args.pids.split(",") if x.strip()]

    v1 = load_two_phase_model(args.phase1_checkpoint, args.device)
    if args.bf16 and args.device == "cuda":
        v1 = v1.to(torch.bfloat16)
    solver1 = KhoruzhiiSolver(full, v1, device=args.device,
                              internal_batch_size=args.internal_batch_size)
    cfg1 = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps,
                                 num_attempts=args.num_attempts,
                                 internal_batch_size=args.internal_batch_size)
    frozen_goal = make_frozen_goal_check(frozen, args.device)
    frozen_set = set(frozen)

    out = []
    for pid in pids:
        s0 = states[pid]
        t0 = time.time()
        found1, len1, path1 = solver1.solve(s0, cfg1, goal_check_fn=frozen_goal)
        wall = time.time() - t0
        if not found1:
            print(f"pid {pid}: PHASE-1 FAIL ({wall:.1f}s)", flush=True)
            out.append({"pid": pid, "found": False})
            continue
        h_state = list(full.apply_path(s0, path1))
        # sanity: every frozen sticker must be home (state is in H)
        frozen_ok = all(h_state[p] == p for p in frozen)
        print(f"pid {pid}: phase1 len {len1}  frozen_home={frozen_ok}  {wall:.1f}s", flush=True)
        out.append({"pid": pid, "found": True, "len1": len1, "path1": list(path1),
                    "h_state": h_state, "frozen_home": bool(frozen_ok)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=0)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
