"""Bench AZ-style dual-head model on beam search using only the V head.

Per the AZ v4 finding (memory az_v4_breakthrough.md), the policy head doesn't
compose with KhoruzhiiSolver — use V-only inference. This wraps ResMLPGFlowNet
so its `forward(x) -> (B,)` returns just the value head, making it drop-in
compatible with KhoruzhiiSolver.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.gflow_model import ResMLPGFlowNet
from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from megaminx.puzzle import Megaminx


def parse_checkpoint_spec(spec: str) -> tuple[Path, str]:
    direct = Path(spec)
    if direct.exists():
        return direct, direct.stem
    path_s, label = (spec.rsplit(":", 1) + [None])[:2]
    path = Path(path_s)
    return path, label or path.stem


class VOnlyWrapper(nn.Module):
    def __init__(self, gflow_model: ResMLPGFlowNet, inference_chunk_size: int = 4096):
        super().__init__()
        self.gflow = gflow_model
        self.inference_chunk_size = inference_chunk_size
        self.output_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x.size(0), self.inference_chunk_size):
            chunk = x[i:i + self.inference_chunk_size]
            h = self.gflow.trunk(chunk)
            v = self.gflow.value_head(h).squeeze(-1)
            outs.append(v)
        return torch.cat(outs, dim=0)


def load_az_v_only(path: Path, device: str, dtype=torch.bfloat16):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    mc = ck["model_config"]
    m = ResMLPGFlowNet(
        state_size=mc["state_size"], num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]),
        num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_actions=mc.get("n_actions", 24),
    )
    m.load_state_dict(sd)
    m = m.to(device=device, dtype=dtype).eval()
    return VOnlyWrapper(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="AZ-style ckpt paths (model.pt[:label] form)")
    ap.add_argument("--pids", default="0,100")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    args = ap.parse_args()
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
    pids = [int(x) for x in args.pids.split(",")]

    results = {}
    for spec in args.checkpoints:
        path, label = parse_checkpoint_spec(spec)
        print(f"\n--- {label} ({path}) ---", flush=True)
        m = load_az_v_only(path, device="cuda")
        solver = KhoruzhiiSolver(
            puzzle=puzzle, model=m, device="cuda",
            internal_batch_size=2**14,
        )
        cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)
        n_solved = 0
        total = 0
        for pid in pids:
            s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
            t0 = time.time()
            found, path_len, p = solver.solve(s0, cfg)
            wall = time.time() - t0
            ok = False
            if found and p:
                cur = list(s0)
                for name in p:
                    g = puzzle.generators[name]
                    cur = [cur[gi] for gi in g]
                ok = (tuple(cur) == puzzle.solved_state)
            print(f"  pid {pid:4d}: {path_len if ok else '-':>5} moves  "
                  f"({'OK' if ok else 'N/F'}, {wall:.1f}s)", flush=True)
            if ok:
                n_solved += 1
                total += int(path_len)
        avg = total / max(1, n_solved)
        print(f"  bench: {n_solved}/{len(pids)} solved | total {total} | avg {avg:.1f}",
              flush=True)
        results[label] = (n_solved, total, len(pids))
        del m, solver
        torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for label, (s, t, n) in results.items():
        print(f"  {label}: {s}/{n} solved, total {t}, avg {t/max(1,s):.1f}")


if __name__ == "__main__":
    main()
