"""PDB-augmented beam search test.

Wraps the existing KhoruzhiiSolver with a `pdb_lookup` callable that gives EXACT
distance for states ≤ d=6 (from bfs_bytes_d6.pkl, 19M states). Combined with
V_neural via max(V_neural, pdb_distance) — this fixes V_neural's overestimate
near V0 (V_full(V0) ≈ 0.91 in m_curr_v3, but actual distance is 0).

Hypothesis: tighter heuristic near V0 → better last-few-step navigation →
shorter paths.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/57_pdb_beam.py \\
        --v-checkpoint megaminx/models/m_curr_v3/epoch_0499.pt \\
        --pdb-table megaminx/data/bfs_bytes_d6.pkl \\
        --pids 0,100,200,500,800 \\
        --beam 65536 --max-steps 120 \\
        --out megaminx/submissions/pdb_smoke.csv
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

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver, _state_hash
from cayley.model import ResMLPDistance
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def build_pdb_lookup(bfs_table: BfsBytesTable, hash_vec: torch.Tensor,
                     state_size: int, device: str, sentinel: float = 1e6):
    """Build a pdb_lookup callable from a BfsBytesTable.

    Returns: callable (states (B, state_size) int8) -> (B,) float16 distance.
    For states in the BFS table (≤ d), returns exact distance (0..d).
    For states outside, returns `sentinel` (large value; KhoruzhiiSolver detects this
    via `in_pdb_set = pdb_val < 1e5` and falls back to neural V).

    Memory: 2 × N × 8 bytes for sorted hashes + 1 byte depths = ~3GB for N=19M on GPU.
    """
    print(f"[pdb] indexing {len(bfs_table.table):,} states...", flush=True)
    N = len(bfs_table.table)
    # Compute hashes of all keys + depths
    hash_vec_np = hash_vec.cpu().numpy().astype(np.int64)

    # Vectorize: build array of states, hash all at once on GPU
    all_states = np.empty((N, state_size), dtype=np.int8)
    depths = np.empty(N, dtype=np.uint8)
    for i, (key_bytes, path_bytes) in enumerate(bfs_table.table.items()):
        all_states[i] = np.frombuffer(key_bytes, dtype=np.int8)
        depths[i] = len(path_bytes)
    all_states_t = torch.from_numpy(all_states).to(device)
    hashes = torch.empty(N, dtype=torch.int64, device=device)
    bs = 2 ** 16
    for i in range(0, N, bs):
        chunk = all_states_t[i : i + bs].long()
        hashes[i : i + bs] = (chunk * hash_vec).sum(dim=1)
    depths_t = torch.from_numpy(depths).to(device)

    # Sort hashes for binary search
    sort_idx = torch.argsort(hashes)
    sorted_hashes = hashes[sort_idx]
    sorted_depths = depths_t[sort_idx]

    print(f"[pdb] indexed: {N:,} states, depths {sorted_depths.min().item()}..{sorted_depths.max().item()}",
          flush=True)

    sentinel_tensor = torch.tensor(sentinel, dtype=torch.float16, device=device)

    def lookup(states: torch.Tensor) -> torch.Tensor:
        # states: (B, state_size) int8
        h = (states.long() * hash_vec).sum(dim=1)  # (B,) int64
        # Binary search for h in sorted_hashes
        idx = torch.searchsorted(sorted_hashes, h)
        idx = torch.clamp(idx, max=N - 1)
        in_set = sorted_hashes[idx] == h
        depth = torch.where(in_set, sorted_depths[idx].to(torch.float16),
                            sentinel_tensor.expand_as(idx).to(torch.float16))
        return depth

    return lookup


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
    ap.add_argument("--pdb-table", type=Path, default=None,
                    help="path to bfs_bytes_d*.pkl. If unset, runs without PDB (control).")
    ap.add_argument("--pids", type=str, default="0,100,200,500,800")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    print(f"PDB-augmented beam: {len(pids)} pids, beam={args.beam:,}, "
          f"PDB={'ON' if args.pdb_table else 'OFF'}", flush=True)

    v_model = load_model(args.v_checkpoint, args.device)
    if args.bf16 and args.device == "cuda":
        v_model = v_model.to(torch.bfloat16)

    state_size = len(puzzle.solved_state)

    # Setup hash vec (must match KhoruzhiiSolver's; it uses seed=0)
    gen_t = torch.Generator(device=args.device)
    gen_t.manual_seed(0)
    hash_vec = torch.randint(
        0, int(1e15), (state_size,), dtype=torch.int64, device=args.device, generator=gen_t,
    )

    # Build PDB lookup if requested
    pdb_lookup = None
    if args.pdb_table:
        print(f"[pdb] loading {args.pdb_table}...", flush=True)
        t0 = time.time()
        bfs_table = BfsBytesTable.load(args.pdb_table)
        print(f"[pdb] loaded in {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()
        pdb_lookup = build_pdb_lookup(bfs_table, hash_vec, state_size, args.device)
        print(f"[pdb] indexed in {time.time()-t0:.1f}s", flush=True)

    solver = KhoruzhiiSolver(
        puzzle=puzzle, model=v_model, device=args.device,
        internal_batch_size=2**14,
        pdb_lookup=pdb_lookup,
    )
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.out, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["initial_state_id", "path"])

    print(f"\n{'pid':>4} | {'path_len':>8} | {'wall':>8} | verify", flush=True)
    print("-" * 60, flush=True)

    n_solved = 0
    total_moves = 0
    for pid in pids:
        s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
        t0 = time.time()
        found, path_len, path = solver.solve(s0, cfg)
        wall = time.time() - t0
        if not found:
            print(f"{pid:>4} | {'N/F':>8} | {wall:>8.1f}s | N/F", flush=True)
            writer.writerow([pid, ""])
            continue
        cur = list(s0)
        for name in path:
            gen = puzzle.generators[name]
            cur = [cur[g] for g in gen]
        verify_ok = (tuple(cur) == puzzle.solved_state)
        print(f"{pid:>4} | {path_len:>8} | {wall:>8.1f}s | {'OK' if verify_ok else 'FAIL'}",
              flush=True)
        if verify_ok:
            writer.writerow([pid, ".".join(path)])
            n_solved += 1
            total_moves += path_len
        else:
            writer.writerow([pid, ""])
        f_csv.flush()
    f_csv.close()

    print(f"\nsolved {n_solved}/{len(pids)} | total {total_moves} | avg {total_moves/max(1,n_solved):.1f}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
