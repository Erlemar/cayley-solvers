"""Quick-bench TB or AZ dual-head model in beam search.

Wraps `ResMLPGFlowNet` so its value head exposes a `model(states) -> (B,) distance`
interface compatible with `KhoruzhiiSolver`. For TB: distance = -value (since value
is log F at optimum, and distance = -log F + const). For AZ: distance = +value
(since value head is trained to predict remaining distance directly).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/65_bench_dual_head.py \\
        --gflow-checkpoint megaminx/models/m_tb_v0_full/epoch_0199.pt \\
        --mode tb \\
        --pids 0,50,100,150,200,250,300,350,400,450 \\
        --beam 65536 --max-steps 120 \\
        --out megaminx/submissions/m_tb_v0_full_bench.csv
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


class GFlowValueAdapter(nn.Module):
    """Wraps ResMLPGFlowNet so calling it returns just the distance scalar.

    sign=-1: value head outputs log F(s) (TB regime); distance ~ -log F (up to const)
    sign=+1: value head outputs predicted distance directly (AZ regime)
    """

    def __init__(self, gflow_model: ResMLPGFlowNet, sign: float = 1.0,
                 inference_chunk_size: int = 4096):
        super().__init__()
        self.gflow = gflow_model
        self.sign = sign
        self.state_size = gflow_model.state_size
        self.num_classes = gflow_model.num_classes
        self.inference_chunk_size = inference_chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= self.inference_chunk_size:
            _logits, value = self.gflow(x)
            return self.sign * value
        outs = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            _logits, value = self.gflow(x[i : i + self.inference_chunk_size])
            outs.append(self.sign * value)
        return torch.cat(outs, dim=0)


def load_gflow_model(path: Path, device: str) -> ResMLPGFlowNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPGFlowNet(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_actions=mc.get("n_actions", 24),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    if unexpected:
        print(f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}",
              flush=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gflow-checkpoint", required=True, type=Path)
    ap.add_argument("--mode", choices=["tb", "az"], required=True,
                    help="tb = TB (sign=-1, value is log F); az = AZ (sign=+1, value is distance)")
    ap.add_argument("--pids", type=str, default="0,50,100,150,200,250,300,350,400,450")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    sign = -1.0 if args.mode == "tb" else 1.0

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    print(f"Bench {args.mode.upper()}: {len(pids)} pids, beam={args.beam:,}, "
          f"sign={sign:+.0f}", flush=True)

    gflow = load_gflow_model(args.gflow_checkpoint, args.device)
    if args.bf16 and args.device == "cuda":
        gflow = gflow.to(torch.bfloat16)
    adapter = GFlowValueAdapter(gflow, sign=sign).to(args.device).eval()
    print(f"adapter ready ({sum(p.numel() for p in gflow.parameters()):,} params)", flush=True)

    solver = KhoruzhiiSolver(
        puzzle=puzzle, model=adapter, device=args.device,
        internal_batch_size=2**14,
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

    print(f"\nsolved {n_solved}/{len(pids)} | total {total_moves} | "
          f"avg {total_moves/max(1,n_solved):.1f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
