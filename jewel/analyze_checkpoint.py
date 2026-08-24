from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import load_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--out")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_transformer(args.checkpoint, device)
    data = np.load(args.data)
    indices = np.flatnonzero(data["is_validation"])
    stats: dict[tuple[str, int], dict[str, float]] = {}
    for start in range(0, len(indices), args.batch_size):
        idx = indices[start : start + args.batch_size]
        ep = torch.from_numpy(data["edge_perm"][idx]).to(device)
        eo = torch.from_numpy(data["edge_ori"][idx]).to(device)
        ro = torch.from_numpy(data["ring_ori"][idx]).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(ep, eo, ro)
        logits = outputs["policy_logits"].float().cpu().numpy()
        top2 = np.argsort(-logits, axis=1)[:, :2]
        masks = data["action_mask"][idx]
        distances = data["distance"][idx]
        complete = data["complete_mask"][idx]
        probs = torch.from_numpy(logits).softmax(dim=-1).numpy()
        for i in range(len(idx)):
            kind = "exact" if complete[i] else "demo"
            key = (kind, int(distances[i]))
            row = stats.setdefault(key, {"count": 0, "top1": 0, "top2": 0, "mass": 0.0})
            row["count"] += 1
            row["top1"] += bool(int(masks[i]) & (1 << int(top2[i, 0])))
            row["top2"] += any(int(masks[i]) & (1 << int(a)) for a in top2[i])
            row["mass"] += sum(probs[i, a] for a in range(12) if int(masks[i]) & (1 << a))
    report = {}
    for (kind, depth), row in sorted(stats.items()):
        report.setdefault(kind, {})[str(depth)] = {
            "count": int(row["count"]),
            "top1": row["top1"] / row["count"],
            "top2": row["top2"] / row["count"],
            "target_mass": float(row["mass"] / row["count"]),
        }
    rendered = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
