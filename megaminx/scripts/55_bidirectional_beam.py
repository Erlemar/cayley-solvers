"""Bidirectional beam: forward neural beam + sampled-BFS backward beam from V0.

Existing MitmKhoruzhiiSolver does forward neural beam + EXACT BFS-d6 set as backward.
This prototype EXTENDS that with deeper backward via sampled BFS — backward expands
to a configured depth (e.g., d=10-15), keeping at most B_bwd states per layer (random
sample when over budget).

Key design:
  - Backward beam built ONCE per V0 (not per pid). Reused across all pids.
  - At each forward step, check `torch.isin(forward_hashes, backward_hashes)`.
  - On match: forward_path + reversed(backward_path) = full solve.

Memory budget on L4 (24 GB):
  - Backward beam at B=131k × depth 15: ~2M states × 120 bytes = 240MB
  - Forward beam at B=131k: 16MB
  - Plus model, hash buffers
  - Comfortable.

Usage:
    python megaminx/scripts/55_bidirectional_beam.py \\
        --v-checkpoint megaminx/models/m_curr_v3/epoch_0499.pt \\
        --pids 0,1,2,3,4 \\
        --fwd-beam 131072 --bwd-beam 131072 --bwd-depth 12 --max-steps 80 \\
        --out megaminx/submissions/bidir_smoke.csv
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

from cayley.khoruzhii_search import (
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    _state_hash,
    _model_predict,
)
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def build_backward_beam(
    puzzle, V0_state, hash_vec, all_moves, inv_idx,
    depth: int, beam_width: int, device: str,
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    """Sampled BFS from V0, expanding outward up to `depth` layers.

    Returns:
        backward_hashes: (N,) int64, sorted, unique. For fast `torch.isin`.
        path_dict: hash_int -> list[int] (sequence of forward action indices to apply
                   to V0 to reach that state). When forward beam matches a state,
                   to complete the solve, apply REVERSED(inv(actions)) to get from
                   the matched state back to V0.

    Memory: roughly O(depth * beam_width * (state_size + sizeof(path))).
    """
    state_size = V0_state.size(0)
    n_gen = all_moves.size(0)

    # Layer 0: V0 itself. Path from V0 to V0 is empty.
    layer_states = V0_state.unsqueeze(0).clone()                  # (1, state_size) int8
    layer_paths: list[list[int]] = [[]]
    all_hashes: list[torch.Tensor] = [_state_hash(layer_states, hash_vec)]
    all_path_dict: dict[int, list[int]] = {int(all_hashes[0][0].item()): []}

    print(f"[bidir-build] layer 0: 1 state", flush=True)
    rng = np.random.default_rng(seed=42)

    for d in range(1, depth + 1):
        # Expand: from each state in layer, apply each generator.
        n_parents = layer_states.size(0)
        # children: (n_parents * n_gen, state_size)
        # Use gather: state[i, all_moves[a, j]]
        parent_states_long = layer_states.long()                  # (n_parents, state_size)
        child_states_all = torch.empty((n_parents * n_gen, state_size),
                                        dtype=torch.int8, device=device)
        child_paths_all: list[list[int]] = []
        for p_idx in range(n_parents):
            parent = parent_states_long[p_idx]
            parent_path = layer_paths[p_idx]
            for a in range(n_gen):
                child_states_all[p_idx * n_gen + a] = parent[all_moves[a]].to(torch.int8)
                child_paths_all.append(parent_path + [a])

        # Hash children
        child_hashes = _state_hash(child_states_all, hash_vec)

        # Dedup against all previously seen states
        seen_mask = torch.zeros(child_states_all.size(0), dtype=torch.bool, device=device)
        # Build a tensor of all seen hashes for batch isin.
        # all_hashes is a list of (n_i,) tensors; concatenate.
        all_hashes_flat = torch.cat(all_hashes)
        seen_mask = torch.isin(child_hashes, all_hashes_flat)
        # Among the new states, also dedup within this layer
        unseen_idx = torch.nonzero(~seen_mask, as_tuple=True)[0]
        if unseen_idx.numel() == 0:
            print(f"[bidir-build] layer {d}: no new states (saturated)", flush=True)
            break
        unique_states = child_states_all[unseen_idx]
        unique_hashes = child_hashes[unseen_idx]
        unique_paths = [child_paths_all[i] for i in unseen_idx.cpu().tolist()]

        # Local dedup within this layer (can have duplicates from multiple parents)
        sorted_h, sort_idx = torch.sort(unique_hashes)
        is_first = torch.cat([
            torch.tensor([True], device=device),
            sorted_h[1:] != sorted_h[:-1],
        ])
        first_in_sorted = sort_idx[is_first]
        unique_states = unique_states[first_in_sorted]
        unique_hashes = unique_hashes[first_in_sorted]
        unique_paths = [unique_paths[int(i.item())] for i in first_in_sorted]

        # Cap at beam_width via random sample
        if unique_states.size(0) > beam_width:
            sample_idx = torch.from_numpy(
                rng.choice(unique_states.size(0), beam_width, replace=False)
            ).to(device)
            unique_states = unique_states[sample_idx]
            unique_hashes = unique_hashes[sample_idx]
            unique_paths = [unique_paths[int(i.item())] for i in sample_idx]

        # Add to all sets
        all_hashes.append(unique_hashes)
        for h, p in zip(unique_hashes.tolist(), unique_paths):
            all_path_dict[int(h)] = p
        layer_states = unique_states
        layer_paths = unique_paths
        print(f"[bidir-build] layer {d}: {layer_states.size(0):>7,} states, "
              f"total {len(all_path_dict):>7,}", flush=True)

    backward_hashes_concat = torch.cat(all_hashes)
    backward_hashes_sorted = torch.sort(backward_hashes_concat).values
    return backward_hashes_sorted, all_path_dict


def load_model(path, device, output_dim=1):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        output_dim=mc.get("output_dim", output_dim),
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-checkpoint", required=True, type=Path)
    ap.add_argument("--pids", type=str, default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fwd-beam", type=int, default=131072)
    ap.add_argument("--bwd-beam", type=int, default=131072)
    ap.add_argument("--bwd-depth", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))

    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    print(f"Bidirectional beam: {len(pids)} pids, fwd_beam={args.fwd_beam:,}, "
          f"bwd_beam={args.bwd_beam:,}, bwd_depth={args.bwd_depth}", flush=True)

    v_model = load_model(args.v_checkpoint, args.device)
    if args.bf16 and args.device == "cuda":
        v_model = v_model.to(torch.bfloat16)

    n_gen = len(puzzle.move_names)
    state_size = len(puzzle.solved_state)
    all_moves = torch.zeros((n_gen, state_size), dtype=torch.int64, device=args.device)
    for i, name in enumerate(puzzle.move_names):
        all_moves[i] = torch.tensor(puzzle.generators[name], dtype=torch.int64)
    V0 = torch.tensor(puzzle.solved_state, dtype=torch.int8, device=args.device)

    # Inverse generator index mapping (for converting backward path back to forward path).
    inv_names = [puzzle.inverse_name(n) for n in puzzle.move_names]
    inv_idx = np.array(
        [puzzle.move_names.index(n) for n in inv_names], dtype=np.int64
    )

    # Hash vector — must match KhoruzhiiSolver's
    gen_t = torch.Generator(device=args.device)
    gen_t.manual_seed(0)
    hash_vec = torch.randint(
        0, int(1e15), (state_size,), dtype=torch.int64, device=args.device, generator=gen_t,
    )

    # Build backward beam (ONE-TIME)
    print("[bidir] building backward beam...", flush=True)
    t0 = time.time()
    bwd_hashes_sorted, bwd_path_dict = build_backward_beam(
        puzzle, V0, hash_vec, all_moves, inv_idx,
        depth=args.bwd_depth, beam_width=args.bwd_beam, device=args.device,
    )
    bwd_build_wall = time.time() - t0
    print(f"[bidir] backward built: {len(bwd_path_dict):,} states in {bwd_build_wall:.1f}s",
          flush=True)

    # Forward solver — standard KhoruzhiiSolver but with custom V0 check (matching backward set)
    solver = KhoruzhiiSolver(
        puzzle=puzzle, model=v_model, device=args.device,
        internal_batch_size=2**14,
    )

    # Override goal: forward state in backward set
    def goal_check_fn(states):
        # states: (B, state_size) int8
        h = _state_hash(states, hash_vec)
        return torch.isin(h, bwd_hashes_sorted)

    cfg = KhoruzhiiSearchConfig(beam_width=args.fwd_beam, num_steps=args.max_steps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.out, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["initial_state_id", "path"])

    n_solved = 0
    total_moves = 0
    print(f"\n{'pid':>4} | {'path_len':>8} {'fwd':>4} {'bwd':>4} | {'wall':>8} | {'verify':>6}",
          flush=True)
    print("-" * 60, flush=True)

    for pid in pids:
        s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
        t0 = time.time()
        found, fwd_len, fwd_path = solver.solve(s0, cfg, goal_check_fn=goal_check_fn)
        wall = time.time() - t0

        if not found:
            print(f"{pid:>4} | {'N/F':>8} {'-':>4} {'-':>4} | {wall:>8.1f}s | NOT FOUND",
                  flush=True)
            writer.writerow([pid, ""])
            continue

        # Get the matched backward state's hash, look up its path
        # Re-apply forward path to s0 to get the meeting state
        cur = list(s0)
        for name in fwd_path:
            gen = puzzle.generators[name]
            cur = [cur[g] for g in gen]
        meeting_state_h = int(_state_hash(
            torch.tensor(cur, dtype=torch.int8, device=args.device).unsqueeze(0), hash_vec
        )[0].item())

        if meeting_state_h not in bwd_path_dict:
            # Hash collision or beam reached V0 directly (V0_h is in bwd set)
            # Check: cur == V0 means we already solved, no backward append needed.
            if tuple(cur) == puzzle.solved_state:
                bwd_segment = []
            else:
                print(f"{pid:>4} | hash {meeting_state_h:x} not in bwd dict (collision)",
                      flush=True)
                writer.writerow([pid, ""])
                continue
        else:
            # bwd_path_dict[h] = forward action indices applied to V0 to reach this state.
            # To go from this state to V0, apply INVERSE actions in REVERSE order.
            fwd_actions_from_v0 = bwd_path_dict[meeting_state_h]
            bwd_segment = [puzzle.move_names[int(inv_idx[a])]
                           for a in reversed(fwd_actions_from_v0)]

        full_path = fwd_path + bwd_segment

        # Verify
        cur = list(s0)
        for name in full_path:
            gen = puzzle.generators[name]
            cur = [cur[g] for g in gen]
        verify_ok = (tuple(cur) == puzzle.solved_state)

        if verify_ok:
            print(f"{pid:>4} | {len(full_path):>8} {fwd_len:>4} {len(bwd_segment):>4} | "
                  f"{wall:>8.1f}s | OK", flush=True)
            writer.writerow([pid, ".".join(full_path)])
            n_solved += 1
            total_moves += len(full_path)
        else:
            print(f"{pid:>4} | {len(full_path):>8} {fwd_len:>4} {len(bwd_segment):>4} | "
                  f"{wall:>8.1f}s | FAIL", flush=True)
            writer.writerow([pid, ""])
        f_csv.flush()

    f_csv.close()

    print(f"\n== summary ==", flush=True)
    print(f"solved: {n_solved}/{len(pids)}", flush=True)
    print(f"total moves: {total_moves:,}", flush=True)
    print(f"avg per solved: {total_moves/max(1,n_solved):.1f}", flush=True)
    print(f"backward build wall: {bwd_build_wall:.1f}s (one-time)", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
