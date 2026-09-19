"""Probe whether a known solution prefix survives a V-only beam.

For one pid and one target path, run the same Khoruzhii V-only beam step by
step. At each depth it checks whether the target prefix state is still present
in the beam, and compares the target next-child value with the selected beam
cutoff value. This is a cheap local diagnostic for "why did a known shorter
path not survive search?"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_test_states, verify_path
from megaminx.bridge import compute_prefix_states
from megaminx.puzzle import Megaminx


def _load_path(path_csv: Path, pid: int) -> list[str]:
    with open(path_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_pid = int(row.get("initial_state_id", row.get("id", -1)))
            if row_pid == pid:
                path = row["path"].strip()
                return path.split(".") if path else []
    raise KeyError(f"pid {pid} not found in {path_csv}")


def _state_pos(states: torch.Tensor, target: torch.Tensor) -> int | None:
    hits = torch.nonzero((states == target).all(dim=1), as_tuple=True)[0]
    if hits.numel() == 0:
        return None
    return int(hits[0].item())


@torch.no_grad()
def _model_value(model: torch.nn.Module, state: torch.Tensor, device: str) -> float:
    x = state.unsqueeze(0).to(device=device, dtype=torch.long)
    if device.startswith("cuda"):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return float(model(x).flatten()[0].float().item())
    return float(model(x).flatten()[0].float().item())


@torch.no_grad()
def _local_child_stats(
    solver: KhoruzhiiSolver,
    parent: torch.Tensor,
    move_idx: int,
) -> tuple[int, float, float, str]:
    child = torch.gather(
        parent.unsqueeze(0).expand(solver.n_actions, -1),
        1,
        solver.all_moves,
    )
    vals = torch.empty(solver.n_actions, dtype=torch.float32, device=solver.device)
    for i in range(0, solver.n_actions, solver.internal_batch_size):
        pred = solver.model(child[i : i + solver.internal_batch_size].long()).flatten()
        vals[i : i + solver.internal_batch_size] = pred.float()
    chosen = vals[move_idx]
    rank = int((vals < chosen).sum().item()) + 1
    best_idx = int(torch.argmin(vals).item())
    return rank, float(chosen.item()), float(vals[best_idx].item()), solver.move_names[best_idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--target-csv", required=True, type=Path)
    ap.add_argument("--pid", type=int, default=992)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=90)
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states_by_pid = load_test_states(PROJECT / "data" / "test.csv")
    initial = states_by_pid[args.pid]
    target_path = _load_path(args.target_csv, args.pid)
    vr = verify_path(puzzle, initial, target_path)
    if not vr.ok:
        raise SystemExit(f"target path does not verify: {vr.reason}")

    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    solver = KhoruzhiiSolver(
        puzzle,
        model,
        device=args.device,
        internal_batch_size=args.internal_batch_size,
        state_dtype=torch.int8,
    )
    move_to_idx = {name: i for i, name in enumerate(solver.move_names)}
    prefix_states = compute_prefix_states(initial, target_path, puzzle)
    prefix_t = [
        torch.tensor(s, dtype=solver.state_dtype, device=args.device)
        for s in prefix_states
    ]

    state0 = torch.tensor(list(initial), dtype=solver.state_dtype, device=args.device)
    states = state0.unsqueeze(0).clone()
    states_bad_hashed = torch.empty(0, dtype=torch.int64, device=args.device)

    rows: list[dict] = []
    first_drop = None
    t0 = time.time()
    print(
        f"checkpoint={args.checkpoint.name} pid={args.pid} beam={args.beam:,} "
        f"target_len={len(target_path)} device={args.device}",
        flush=True,
    )
    print(
        "step parent_pos child_pos local_rank target_v best_sibling_v "
        "cutoff_v margin_to_cutoff best_sibling_move",
        flush=True,
    )

    for step in range(min(args.max_steps, len(target_path))):
        parent_pos = _state_pos(states, prefix_t[step])
        move_idx = move_to_idx[target_path[step]]
        rank, target_v, best_sibling_v, best_move = _local_child_stats(
            solver, prefix_t[step], move_idx
        )

        states, y_pred, _moves, _idx = solver._do_greedy_step(
            states, states_bad_hashed, args.beam
        )
        if states.numel() == 0:
            print(f"beam emptied at step {step + 1}", flush=True)
            break

        child_pos = _state_pos(states, prefix_t[step + 1])
        cutoff_v = float(y_pred.float().max().item()) if y_pred.numel() else float("nan")
        margin = target_v - cutoff_v
        if child_pos is None and first_drop is None:
            first_drop = step + 1

        row = {
            "step": step + 1,
            "remaining_after": len(target_path) - step - 1,
            "move": target_path[step],
            "parent_pos": parent_pos,
            "child_pos": child_pos,
            "local_rank": rank,
            "target_v": target_v,
            "best_sibling_v": best_sibling_v,
            "best_sibling_move": best_move,
            "cutoff_v": cutoff_v,
            "margin_to_cutoff": margin,
            "beam_size": int(states.size(0)),
        }
        rows.append(row)

        print(
            f"{step + 1:4d} "
            f"{str(parent_pos):>10} {str(child_pos):>9} "
            f"{rank:10d} {target_v:8.3f} {best_sibling_v:14.3f} "
            f"{cutoff_v:8.3f} {margin:16.3f} {best_move}",
            flush=True,
        )

        # Once a prefix has fallen out, print a few more rows to see whether it
        # reappears through an alternate word, then stop.
        if first_drop is not None and step + 1 >= first_drop + 5:
            break

    elapsed = time.time() - t0
    print(f"\nfirst_drop={first_drop} wall={elapsed:.1f}s rows={len(rows)}", flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(args.checkpoint),
                    "target_csv": str(args.target_csv),
                    "pid": args.pid,
                    "beam": args.beam,
                    "first_drop": first_drop,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
