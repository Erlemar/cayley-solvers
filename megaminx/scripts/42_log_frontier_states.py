"""Collect beam-search frontier states for DAgger-style Bellman replay (Idea 3 v0).

Runs vanilla beam search (V-only, m05) on N pids; hooks `_do_greedy_step` to
capture the surviving states at every beam step; deduplicates across all pids
via state-bytes; writes the resulting tensor to `data/frontier_states.pt`.

The output is STATES ONLY — no labels. The Bellman training loop computes its
target on the fly (`1 + min_a V_target(apply(s, a))`) for the frontier-state
mixin, identical to the random-walk path. The lever is *state distribution*
(visited frontier vs random walk), not *label source*. This is structurally
different from the m43 solver-trace mixin, which used realized-suffix-length
labels and went OOD-catastrophic.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/42_log_frontier_states.py \\
        --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \\
        --out megaminx/data/frontier_states.pt \\
        --n-pids 100 --beams 16384 --max-steps 80 --bucket-from 4 --bucket-to 7
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

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def load_v_model(ckpt_path: Path, device: str) -> ResMLPDistance:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    cfg = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=cfg.get("state_size", 120),
        num_classes=cfg.get("num_classes", 120),
        hidden_dims=tuple(cfg.get("hidden_dims", (2048, 512))),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        encoding=cfg.get("encoding", "embedding"),
        embed_dim=cfg.get("embed_dim", 16),
    ).to(device).eval()
    model.load_state_dict(sd)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="V-model to drive the beam (e.g. m05).")
    ap.add_argument("--out", required=True, type=Path,
                    help="output path for frontier_states.pt")
    ap.add_argument("--n-pids", type=int, default=100,
                    help="number of pids to collect from")
    ap.add_argument("--bucket-from", type=int, default=4,
                    help="lowest bucket to sample (inclusive). bucket = pid // 100.")
    ap.add_argument("--bucket-to", type=int, default=7,
                    help="highest bucket to sample (inclusive)")
    ap.add_argument("--beams", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--max-states-per-pid", type=int, default=200_000,
                    help="cap on raw states logged per pid (before dedup) to bound memory")
    ap.add_argument("--final-cap", type=int, default=300_000,
                    help="cap on the final unique-state set (after dedup, before save)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    model = load_v_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint}: "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # Pid selection: rng.choice from each bucket, mid-difficulty.
    rng = np.random.default_rng(args.seed)
    candidate_pids: list[int] = []
    for b in range(args.bucket_from, args.bucket_to + 1):
        candidate_pids.extend(range(b * 100, min((b + 1) * 100, 1001)))
    if args.n_pids >= len(candidate_pids):
        chosen = sorted(candidate_pids)
    else:
        chosen = sorted(rng.choice(candidate_pids, size=args.n_pids, replace=False).tolist())
    print(f"selected {len(chosen)} pids from buckets {args.bucket_from}..{args.bucket_to}: "
          f"first 10 = {chosen[:10]}")

    # Load test states.
    test_states: dict[int, list[int]] = {}
    with open(PROJECT / "data" / "test.csv") as f:
        for row in csv.DictReader(f):
            pid = int(row["initial_state_id"])
            test_states[pid] = [int(x) for x in row["initial_state"].split(",")]

    # Build solver. V-only (no Q-shortlister, no macros).
    solver = KhoruzhiiSolver(
        puzzle, model, device=device,
        internal_batch_size=args.internal_batch_size,
        random_seed=args.seed, state_dtype=torch.int8,
    )

    # Monkey-patch _do_greedy_step to capture next_states per step.
    captured: list[torch.Tensor] = []
    cap_per_pid_count = [0]
    cap_limit = args.max_states_per_pid
    orig_step = solver._do_greedy_step

    def _logging_step(states, states_bad_hashed, B):
        ns, vals, mvs, idx = orig_step(states, states_bad_hashed, B)
        if cap_per_pid_count[0] < cap_limit and ns.numel() > 0:
            need = cap_limit - cap_per_pid_count[0]
            take = ns[:need].detach().cpu()
            captured.append(take)
            cap_per_pid_count[0] += take.size(0)
        return ns, vals, mvs, idx

    solver._do_greedy_step = _logging_step

    # Run solves.
    cfg = KhoruzhiiSearchConfig(beam_width=args.beams, num_steps=args.max_steps,
                                num_attempts=1, internal_batch_size=args.internal_batch_size)
    n_solved = 0
    n_total_states = 0
    t_start = time.time()
    for k, pid in enumerate(chosen):
        if pid not in test_states:
            print(f"  WARNING: pid {pid} not in test.csv; skipping", file=sys.stderr)
            continue
        cap_per_pid_count[0] = 0  # reset per-pid cap
        captured_before = sum(t.size(0) for t in captured)
        t0 = time.time()
        found, plen, _path = solver.solve(test_states[pid], cfg)
        captured_after = sum(t.size(0) for t in captured)
        n_states_pid = captured_after - captured_before
        n_total_states += n_states_pid
        if found:
            n_solved += 1
        if (k + 1) % 5 == 0 or k == 0:
            elapsed = time.time() - t_start
            eta = (len(chosen) - (k + 1)) * (elapsed / (k + 1)) if k > 0 else 0
            print(f"  [{k+1}/{len(chosen)}] pid={pid:4d} found={int(found)} "
                  f"plen={plen} states={n_states_pid:,} (cum {n_total_states:,}) "
                  f"({time.time()-t0:.1f}s) elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone. {n_solved}/{len(chosen)} solved, "
          f"collected {n_total_states:,} raw states in {elapsed:.0f}s.", flush=True)

    # Concat + dedupe via bytes.
    print("concat + dedupe...", flush=True)
    all_states = torch.cat(captured, dim=0)  # (N, 120) int8 cpu
    print(f"  raw: {all_states.size(0):,} states")
    # numpy uint8 view -> tobytes per row -> set
    arr = all_states.numpy().astype(np.uint8)
    seen: dict[bytes, int] = {}
    for i in range(arr.shape[0]):
        b = arr[i].tobytes()
        if b not in seen:
            seen[b] = i
    unique_idx = np.array(sorted(seen.values()), dtype=np.int64)
    unique_states = all_states[unique_idx]
    print(f"  unique: {unique_states.size(0):,} states")

    if unique_states.size(0) > args.final_cap:
        # Random-subsample so the dataset fits the cap (reproducible via numpy seed).
        rng2 = np.random.default_rng(args.seed + 1)
        sel = rng2.choice(unique_states.size(0), size=args.final_cap, replace=False)
        sel.sort()
        unique_states = unique_states[torch.from_numpy(sel)]
        print(f"  subsampled to cap: {unique_states.size(0):,} states")

    # Save.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "n_pids": len(chosen),
        "n_solved": n_solved,
        "beams": args.beams,
        "max_steps": args.max_steps,
        "bucket_range": [args.bucket_from, args.bucket_to],
        "raw_states": int(all_states.size(0)),
        "unique_states": int(unique_states.size(0)),
        "elapsed_s": elapsed,
        "seed": args.seed,
    }
    torch.save({"states": unique_states, "metadata": metadata}, args.out)
    print(f"wrote {args.out}: {unique_states.size(0):,} states")
    print(f"  metadata: {metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
