from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import JewelTransformer, TransformerConfig


def _make_tensors(data: np.lib.npyio.NpzFile, select: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(data["edge_perm"][select]),
        torch.from_numpy(data["edge_ori"][select]),
        torch.from_numpy(data["ring_ori"][select]),
        torch.from_numpy(data["distance"][select]),
        torch.from_numpy(data["action_mask"][select]),
        torch.from_numpy(data["complete_mask"][select]),
        torch.from_numpy(data["sample_weight"][select]),
    )


def _targets_from_mask(mask: torch.Tensor) -> torch.Tensor:
    bits = torch.arange(12, device=mask.device)
    return ((mask[:, None].long() >> bits[None, :]) & 1).float()


def compute_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict]:
    _, _, _, distance, action_mask, complete_mask, sample_weight = batch
    target_actions = _targets_from_mask(action_mask)
    log_probs = F.log_softmax(outputs["policy_logits"].float(), dim=-1)
    masked = log_probs.masked_fill(target_actions == 0, -torch.inf)
    policy_per = -torch.logsumexp(masked, dim=-1)
    policy_loss = (policy_per * sample_weight).sum() / sample_weight.sum()

    exact = complete_mask.bool()
    if exact.any():
        geo_loss = F.binary_cross_entropy_with_logits(
            outputs["geodesic_logits"][exact].float(), target_actions[exact]
        )
        regret_target = 2.0 * (1.0 - target_actions[exact])
        regret_loss = F.smooth_l1_loss(outputs["regret"][exact].float(), regret_target)
        value_loss = F.smooth_l1_loss(outputs["distance"][exact].float(), distance[exact].float())
        bins = torch.arange(outputs["cdf_logits"].shape[1], device=distance.device)
        cdf_target = (distance[exact, None] <= bins[None, :]).float()
        cdf_loss = F.binary_cross_entropy_with_logits(outputs["cdf_logits"][exact].float(), cdf_target)
    else:
        zero = outputs["distance"].sum() * 0.0
        geo_loss = regret_loss = value_loss = cdf_loss = zero

    demo = ~exact
    upper_loss = torch.relu(outputs["distance"][demo].float() - distance[demo].float()).square().mean() if demo.any() else 0.0
    total = policy_loss + 0.25 * geo_loss + 0.5 * regret_loss + 0.1 * value_loss + 0.1 * cdf_loss + 0.02 * upper_loss
    metrics = {
        "loss": float(total.detach()),
        "policy": float(policy_loss.detach()),
        "geo": float(geo_loss.detach()),
        "regret": float(regret_loss.detach()),
        "value": float(value_loss.detach()),
        "cdf": float(cdf_loss.detach()),
    }
    return total, metrics


@torch.no_grad()
def evaluate(model: JewelTransformer, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict:
    model.eval()
    totals = {"count": 0, "exact": 0, "top1_opt": 0, "top2_any": 0, "set_mass": 0.0, "value_abs": 0.0, "loss": 0.0}
    for batch_i, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_i >= max_batches:
            break
        batch = tuple(x.to(device, non_blocking=True) for x in cpu_batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(batch[0], batch[1], batch[2])
            loss, _ = compute_loss(outputs, batch)
        targets = _targets_from_mask(batch[4])
        exact = batch[5].bool()
        top2 = outputs["policy_logits"].topk(2, dim=-1).indices
        top1_hit = targets.gather(1, top2[:, :1]).squeeze(1) > 0
        top2_hit = targets.gather(1, top2).amax(dim=1) > 0
        probs = outputs["policy_logits"].float().softmax(dim=-1)
        n = len(batch[0])
        totals["count"] += n
        totals["loss"] += float(loss) * n
        if exact.any():
            ne = int(exact.sum())
            totals["exact"] += ne
            totals["top1_opt"] += int(top1_hit[exact].sum())
            totals["top2_any"] += int(top2_hit[exact].sum())
            totals["set_mass"] += float((probs[exact] * targets[exact]).sum())
            totals["value_abs"] += float((outputs["distance"][exact].float() - batch[3][exact].float()).abs().sum())
    return {
        "loss": totals["loss"] / max(totals["count"], 1),
        "exact_top1": totals["top1_opt"] / max(totals["exact"], 1),
        "exact_top2": totals["top2_any"] / max(totals["exact"], 1),
        "exact_set_mass": totals["set_mass"] / max(totals["exact"], 1),
        "exact_value_mae": totals["value_abs"] / max(totals["exact"], 1),
        "count": totals["count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="jewel/artifacts/train_mixed_v1.npz")
    parser.add_argument("--out", default="jewel/models/transformer_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    data = np.load(args.data)
    is_validation = data["is_validation"]
    train_ds = _make_tensors(data, ~is_validation)
    val_ds = _make_tensors(data, is_validation)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    config = TransformerConfig(args.d_model, args.heads, args.layers, args.ffn, args.dropout)
    raw_model = JewelTransformer(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        raw_model.load_state_dict(initial["model"])
        print(json.dumps({"initialized_from": args.init_checkpoint, "initial_epoch": initial.get("epoch")}), flush=True)
    model = torch.compile(raw_model, mode="reduce-overhead") if args.compile else raw_model
    params = sum(p.numel() for p in raw_model.parameters())
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda")
    train_batches = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
    total_steps = train_batches * args.epochs

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_top1 = -1.0
    global_step = 0
    print(json.dumps({"device": str(device), "params": params, "config": config.to_dict(), "train": len(train_ds), "validation": len(val_ds)}), flush=True)

    for epoch in range(args.epochs):
        model.train()
        started = perf_counter()
        running = 0.0
        seen = 0
        for batch_i, cpu_batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_i >= args.max_train_batches:
                break
            batch = tuple(x.to(device, non_blocking=True) for x in cpu_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(batch[0], batch[1], batch[2])
                loss, _ = compute_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += float(loss.detach()) * len(batch[0])
            seen += len(batch[0])
        validation = evaluate(raw_model, val_loader, device)
        row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": running / max(seen, 1),
            "seconds": perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
            **validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {"model": raw_model.state_dict(), "config": config.to_dict(), "epoch": epoch, "step": global_step, "metrics": row}
        torch.save(checkpoint, out / "last.pt")
        if validation["exact_top1"] > best_top1:
            best_top1 = validation["exact_top1"]
            torch.save(checkpoint, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
