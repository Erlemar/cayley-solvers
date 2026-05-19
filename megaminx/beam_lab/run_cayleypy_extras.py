"""cayleypy + history_depth and cayleypy + hashed_neighbourhood (MITM) experiments.

Runs on the 3-puzzle subset with compile dynamic=True for fair comparison.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from beam_search import setup_model_for_inference
from model import load_checkpoint
from puzzle import Megaminx

from cayleypy import CayleyGraph, CayleyGraphDef, Predictor

device = "cuda"
puzzle = Megaminx.load(HERE / "data" / "puzzle_info.json")
gens_names = list(puzzle.move_names)
graph_def = CayleyGraphDef.create(
    generators=[list(puzzle.generators[n]) for n in gens_names],
    generator_names=list(gens_names),
    central_state=list(puzzle.solved_state),
)
graph = CayleyGraph(
    graph_def, dtype=torch.int8, bit_encoding_width=None,
    batch_size=2**14, hash_chunk_size=2**14, device=device,
)

model = load_checkpoint(str(HERE / "models" / "m05_bellman_warm" / "epoch_0499.pt"), device=device)
model = model.to(torch.bfloat16)
setup_model_for_inference(model)
print("compiling dynamic=True...", flush=True)
t0 = time.time()
model = torch.compile(model, dynamic=True, fullgraph=False)
with torch.inference_mode():
    model(torch.zeros((16384, 120), dtype=torch.int64, device=device))
    model(torch.zeros((16353, 120), dtype=torch.int64, device=device))
print(f"compile + warmup: {time.time() - t0:.1f}s", flush=True)
predictor = Predictor(graph, model)

print("\nprecomputing BFS-d6 shell for MITM tests...", flush=True)
t0 = time.time()
bfs_result = graph.bfs(max_diameter=6, return_all_hashes=True)
print(f"  done in {time.time() - t0:.1f}s", flush=True)

with open(HERE / "data" / "sample_3.csv") as f:
    rows = [(int(r["initial_state_id"]), tuple(int(x) for x in r["initial_state"].split(",")))
            for r in csv.DictReader(f)]

configs = [
    {"name": "iterated_hd10",  "mode": "iterated", "hd": 10, "mitm": False},
    {"name": "simple_mitm",    "mode": "simple",   "hd": 0,  "mitm": True},
]

for cfg in configs:
    name, mode, hd, mitm = cfg["name"], cfg["mode"], cfg["hd"], cfg["mitm"]
    print(f"\n=== {name} (mode={mode} hd={hd} mitm={mitm}) ===", flush=True)
    for pid, state in rows:
        t0 = time.time()
        try:
            result = graph.beam_search(
                start_state=state, beam_mode=mode, predictor=predictor,
                beam_width=131072, max_steps=180, history_depth=hd,
                return_path=True, path_device="cpu",
                hashed_neigbourhood=bfs_result if mitm else None,
                verbose=0,
            )
            wall = time.time() - t0
            plen = len(result.path) if result.path else -1
            print(f"  pid={pid:4d}  path={plen}  wall={wall:.1f}s", flush=True)
        except Exception as e:
            print(f"  pid={pid:4d}  ERROR: {type(e).__name__}: {e}", flush=True)
