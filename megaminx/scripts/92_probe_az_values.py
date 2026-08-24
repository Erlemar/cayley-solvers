"""Probe AZ value-head scale on anchors and path-labeled states."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable
from cayley.gflow_model import ResMLPGFlowNet
from megaminx.puzzle import Megaminx


def load_az(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["state_dict"].items()}
    mc = ckpt["model_config"]
    model = ResMLPGFlowNet(
        state_size=mc["state_size"],
        num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]),
        num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_actions=mc.get("n_actions", 24),
    )
    model.load_state_dict(sd)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def value_head(model: ResMLPGFlowNet, states: torch.Tensor) -> torch.Tensor:
    h = model.trunk(states)
    return model.value_head(h).squeeze(-1).float()


def describe(name: str, values: torch.Tensor):
    values = values.float()
    print(
        f"{name}: mean={values.mean().item():.4f} "
        f"min={values.min().item():.4f} max={values.max().item():.4f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset", type=Path, default=PROJECT / "data" / "az_dataset_73614.pt")
    ap.add_argument("--n-path", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    gens = torch.from_numpy(GeneratorTable.from_puzzle(puzzle).perms).to(args.device)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.long, device=args.device).unsqueeze(0)
    d1 = torch.gather(solved.expand(gens.size(0), -1), 1, gens)

    model = load_az(args.checkpoint, args.device)
    data = torch.load(args.dataset, map_location=args.device, weights_only=False)
    states = data["states"][: args.n_path].to(args.device).long()
    targets = data["values"][: args.n_path].to(args.device).float()

    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else None
    if ctx is None:
        pred_solved = value_head(model, solved)
        pred_d1 = value_head(model, d1)
        pred_path = value_head(model, states)
    else:
        with ctx:
            pred_solved = value_head(model, solved)
            pred_d1 = value_head(model, d1)
            pred_path = value_head(model, states)

    print(f"checkpoint: {args.checkpoint}", flush=True)
    describe("solved_pred", pred_solved)
    describe("d1_pred", pred_d1)
    describe("path_pred", pred_path)
    describe("path_target", targets)
    print(f"path_mse={torch.mean((pred_path - targets) ** 2).item():.4f}", flush=True)
    print(f"path_corr={torch.corrcoef(torch.stack([pred_path, targets]))[0, 1].item():.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
