"""A/B benchmark: cayleypy beam_search vs our KhoruzhiiSolver on the 12 lab samples.

Tests several cayleypy beam_mode configs, each with our trained m05 model:
  - simple (no extras)
  - iterated, history_depth=0, no MITM
  - iterated, history_depth=10, no MITM
  - iterated, history_depth=0, hashed_neigbourhood=BFS-d6 (MITM)
  - iterated, history_depth=10, hashed_neigbourhood=BFS-d6 (combined)

Times each per-puzzle and reports path lengths. Compare against our KhoruzhiiSolver
results from local_baseline.csv / local_compile.csv.

Usage:
    python run_cayleypy_benchmark.py --checkpoint models/m05_bellman_warm/epoch_0499.pt \\
        --beam 131072 --max-steps 180 --bf16 \\
        --out results/cayleypy_iterated.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import load_checkpoint  # noqa: E402
from puzzle import Megaminx  # noqa: E402

from cayleypy import CayleyGraph, CayleyGraphDef, Predictor  # noqa: E402


def load_samples(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pid = int(row["initial_state_id"])
            state = tuple(int(x) for x in row["initial_state"].split(","))
            rows.append((pid, state))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--puzzle-info", type=Path, default=HERE / "data" / "puzzle_info.json")
    ap.add_argument("--samples", type=Path, default=HERE / "data" / "sample_puzzles.csv")
    ap.add_argument("--beam", type=int, default=131072)
    ap.add_argument("--max-steps", type=int, default=180)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true",
                    help="wrap model with torch.compile (dynamic=False) before passing "
                         "to Predictor; pre-warm at the graph.batch_size shape")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    # Load puzzle definition.
    puzzle = Megaminx.load(args.puzzle_info)
    samples = load_samples(args.samples)
    print(f"samples: {len(samples)} puzzles")

    # Build cayleypy CayleyGraph from our generators + central_state.
    gens_names = list(puzzle.move_names)
    graph_def = CayleyGraphDef.create(
        generators=[list(puzzle.generators[n]) for n in gens_names],
        generator_names=list(gens_names),
        central_state=list(puzzle.solved_state),
    )
    graph = CayleyGraph(
        graph_def,
        dtype=torch.int8,
        bit_encoding_width=None,
        batch_size=2**14,       # 16384 — match our KhoruzhiiSolver internal_batch_size
        hash_chunk_size=2**14,  # 16384 — match
        device=device,
    )
    print(f"cayleypy graph: state_size={graph.definition.state_size}, n_gen={graph.definition.n_generators}")

    # Load m05 model + cast to bf16.
    print(f"loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(str(args.checkpoint), device=device)
    if args.bf16 and device == "cuda":
        model = model.to(torch.bfloat16)
    model.eval()

    if args.compile:
        from beam_search import setup_model_for_inference, setup_model_for_compile
        setup_model_for_inference(model)
        print(f"compiling model with torch.compile (pre-warm at graph.batch_size={graph.batch_size})...")
        t0 = time.time()
        model = setup_model_for_compile(model, batch_size=graph.batch_size)
        print(f"compile + warmup done in {time.time() - t0:.1f}s")

    # Wrap as cayleypy Predictor.
    predictor = Predictor(graph, model)

    # Optionally precompute BFS-d6 shell (for hashed_neigbourhood / MITM tests).
    print("precomputing BFS-d6 shell for MITM tests...")
    t0 = time.time()
    bfs_result = graph.bfs(max_diameter=6, return_all_hashes=True)
    print(f"  BFS-d6 done in {time.time() - t0:.1f}s, {len(bfs_result.layer_sizes)} layers, "
          f"total states {sum(bfs_result.layer_sizes):,}")

    # Configurations to test.
    configs = [
        {"name": "simple",                       "beam_mode": "simple", "history_depth": 0, "use_mitm": False},
        {"name": "iterated_hd0",                 "beam_mode": "iterated", "history_depth": 0, "use_mitm": False},
        {"name": "iterated_hd10",                "beam_mode": "iterated", "history_depth": 10, "use_mitm": False},
        {"name": "iterated_hd0_mitm",            "beam_mode": "iterated", "history_depth": 0, "use_mitm": True},
        {"name": "iterated_hd10_mitm",           "beam_mode": "iterated", "history_depth": 10, "use_mitm": True},
    ]

    rows = []
    for cfg in configs:
        print(f"\n=== {cfg['name']} ===")
        print(f"  beam_mode={cfg['beam_mode']}  history_depth={cfg['history_depth']}  use_mitm={cfg['use_mitm']}")
        for pid, state in samples:
            t_start = time.time()
            try:
                result = graph.beam_search(
                    start_state=state,
                    beam_mode=cfg["beam_mode"],
                    predictor=predictor,
                    beam_width=args.beam,
                    max_steps=args.max_steps,
                    history_depth=cfg["history_depth"],
                    return_path=True,
                    path_device="cpu",
                    hashed_neigbourhood=bfs_result if cfg["use_mitm"] else None,
                    verbose=0,
                )
                wall = time.time() - t_start
                found = bool(result.path is not None)
                path_len = len(result.path) if found else -1
                rows.append({
                    "config": cfg["name"], "pid": pid,
                    "found": int(found), "path_len": path_len, "wall_s": wall,
                })
                print(f"  pid={pid:4d}  found={int(found)}  path={path_len}  wall={wall:.1f}s",
                      flush=True)
            except Exception as e:
                rows.append({"config": cfg["name"], "pid": pid,
                             "found": 0, "path_len": -1, "wall_s": time.time() - t_start})
                print(f"  pid={pid:4d}  ERROR: {type(e).__name__}: {e}", flush=True)

    # Save CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "pid", "found", "path_len", "wall_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")

    # Summary table
    print("\n=== aggregate by config ===")
    by_cfg: dict[str, list] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    print(f"{'config':>22s} {'solved':>8s} {'path-sum':>9s} {'wall':>8s}")
    for name, sub in by_cfg.items():
        solved = sum(1 for r in sub if r["found"])
        psum = sum(r["path_len"] for r in sub if r["found"])
        wall = sum(r["wall_s"] for r in sub)
        print(f"{name:>22s} {solved:>3d}/{len(sub):<3d} {psum:>9d} {wall:>7.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
