"""m06: distill m07's V-head into a Q-head (24-output) for cheaper beam expansion.

    python megaminx/scripts/06_train_qdistill.py \
        --config megaminx/configs/m06_qdistill_k80.yaml \
        --output megaminx/models/m06_qdistill_k80

Teacher (V): scalar prediction = predicted distance.
Student (Q): 24-vector = [V(apply(s, a_g)) for g in generators].

For each batch of states, compute teacher V on all 24 children — student trains
to match those. Body weights initialized from teacher; head random-init. At
inference, beam search calls Q once per parent and gets all 24 children scores.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import generate_walks_torch
from cayley.model import ResMLPDistance
from cayley.search import load_model_checkpoint
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    distill_cfg = cfg["distill"]

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = args.device

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)
    state_size = len(puzzle.solved_state)
    assert model_cfg["output_dim"] == n_gen, f"output_dim must equal n_gen={n_gen}"

    print(f"loading teacher from {distill_cfg['teacher_path']}", flush=True)
    teacher_path = (PROJECT.parent / distill_cfg["teacher_path"]
                    if not Path(distill_cfg["teacher_path"]).is_absolute()
                    else Path(distill_cfg["teacher_path"]))
    teacher = load_model_checkpoint(teacher_path, device=device, dtype=torch.float32)
    teacher_base = getattr(teacher, "_orig_mod", teacher)
    for p in teacher_base.parameters():
        p.requires_grad = False
    teacher.eval()
    print(f"teacher params: {teacher_base.num_parameters():,}", flush=True)

    student = ResMLPDistance(
        state_size=model_cfg["state_size"], num_classes=model_cfg["num_classes"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "embedding"),
        embed_dim=model_cfg.get("embed_dim", 16),
        output_dim=n_gen,
    ).to(device)

    # Copy body weights from teacher (skip head; shape mismatches naturally caught).
    teacher_sd = teacher_base.state_dict()
    student_sd = student.state_dict()
    copied = 0
    for k, v in teacher_sd.items():
        if k.startswith("head."):
            continue
        if k in student_sd and student_sd[k].shape == v.shape:
            student_sd[k] = v
            copied += 1
    student.load_state_dict(student_sd)
    print(f"student params: {student.num_parameters():,} (copied {copied} body tensors from teacher)",
          flush=True)

    if train_cfg.get("compile_model", False) and device == "cuda":
        student = torch.compile(student, dynamic=False)

    optim_kwargs = dict(lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0))
    if train_cfg.get("fused_optimizer", True) and device == "cuda":
        optim_kwargs["fused"] = True
    optimizer = torch.optim.Adam(student.parameters(), **optim_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["n_epochs"])

    seed = cfg.get("seed", 0)
    gen = torch.Generator(device=device).manual_seed(seed)
    generators = torch.tensor(
        [puzzle.generators[n] for n in puzzle.move_names], dtype=torch.int64, device=device
    )
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)

    @torch.no_grad()
    def teacher_q_targets(states: torch.Tensor) -> torch.Tensor:
        """(B, state_size) -> (B, n_gen) teacher V on each child state."""
        B = states.shape[0]
        children = torch.gather(
            states.unsqueeze(1).expand(B, n_gen, state_size),
            2,
            generators.unsqueeze(0).expand(B, n_gen, state_size),
        ).reshape(B * n_gen, state_size)
        is_solved_child = (children == solved).all(dim=1)
        chunk = 4096
        out_v = torch.empty(B * n_gen, dtype=torch.float32, device=device)
        for i in range(0, B * n_gen, chunk):
            out_v[i : i + chunk] = teacher(children[i : i + chunk]).flatten().to(torch.float32)
        out_v = torch.where(is_solved_child, torch.zeros_like(out_v), out_v)
        return out_v.view(B, n_gen)

    print(f"training: {train_cfg}", flush=True)
    t0 = time.time()
    for epoch in range(train_cfg["n_epochs"]):
        n_walks = max(1, train_cfg["samples_per_epoch"] // train_cfg["k_max"])
        states, _ = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=train_cfg["k_max"],
            seed=seed + epoch, device=device, n_back=train_cfg.get("n_back", 1),
        )
        q_targets = teacher_q_targets(states)

        student.train()
        N = states.shape[0]
        perm = torch.randperm(N, generator=gen, device=device)
        total_loss, n_batches = 0.0, 0
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if train_cfg.get("amp", False) and device == "cuda"
            else _NullCtx()
        )
        for i in range(0, N, train_cfg["batch_size"]):
            idx = perm[i : i + train_cfg["batch_size"]]
            with autocast_ctx:
                pred = student(states[idx])
                loss = F.mse_loss(pred, q_targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg = total_loss / max(n_batches, 1)
        print(f"epoch {epoch:4d} | loss {avg:.4f} | "
              f"lr {float(scheduler.get_last_lr()[0]):.2e} | {time.time() - t0:.1f}s elapsed",
              flush=True)
        t0 = time.time()

        if (epoch + 1) % train_cfg.get("checkpoint_every_epochs", 25) == 0 or epoch == train_cfg["n_epochs"] - 1:
            ckpt_path = out / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "loss": avg,
                "model_config": {
                    **{k: model_cfg[k] for k in ("state_size", "num_classes", "num_res_blocks", "encoding", "embed_dim")},
                    "hidden_dims": list(model_cfg["hidden_dims"]),
                    "output_dim": n_gen,
                },
                "teacher_path": str(teacher_path),
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)
    return 0


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    sys.exit(main())
