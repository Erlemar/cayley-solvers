from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .gflownet import build_jewel_env, load_gflownet, rollout_paths
from .official import OfficialPuzzle, parse_official_state
from .puzzle import JewelState, apply_path, rank_states


def _action_targets(masks: np.ndarray) -> np.ndarray:
    bits = np.arange(12, dtype=np.uint16)
    return ((masks[:, None].astype(np.uint16) >> bits[None, :]) & 1).astype(bool)


@torch.no_grad()
def exact_policy_metrics(
    model,
    official: OfficialPuzzle,
    data_path: str | Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    data = np.load(data_path)
    selected = data["is_validation"].astype(bool) & data["complete_mask"].astype(bool)
    ranks = rank_states(
        data["edge_perm"][selected], data["edge_ori"][selected], data["ring_ori"][selected]
    )
    _, unique_indices = np.unique(ranks, return_index=True)
    variants = {
        "row_weighted": np.arange(int(selected.sum())),
        "unique_state_weighted": np.sort(unique_indices),
    }
    ep = data["edge_perm"][selected]
    eo = data["edge_ori"][selected]
    ro = data["ring_ori"][selected]
    masks = data["action_mask"][selected]
    result: dict[str, dict] = {}
    for name, indices in variants.items():
        top1_hits = 0
        top2_hits = 0
        set_mass = 0.0
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            states = [JewelState(ep[i], eo[i], ro[i]) for i in batch_indices]
            stickers = np.stack([official.from_structured(state) for state in states]).astype(np.int64)
            tensor = torch.from_numpy(stickers).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                backward_logits, _ = model(tensor)
            probabilities = backward_logits.float().softmax(dim=-1).cpu().numpy()
            targets = _action_targets(masks[batch_indices])
            order = np.argsort(-probabilities, axis=1)
            top1_hits += int(targets[np.arange(len(targets)), order[:, 0]].sum())
            top2_hits += int(targets[np.arange(len(targets))[:, None], order[:, :2]].any(axis=1).sum())
            set_mass += float((probabilities * targets).sum())
        result[name] = {
            "count": int(len(indices)),
            "top1_descending": top1_hits / max(len(indices), 1),
            "top2_contains_descending": top2_hits / max(len(indices), 1),
            "probability_mass_on_descending": set_mass / max(len(indices), 1),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--test", default="jewel/data/test.csv")
    parser.add_argument("--exact-data", default="jewel/artifacts/train_mixed_v4_public.npz")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--out", default="jewel/results/gflownet_evaluation.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official = OfficialPuzzle.load(args.puzzle_info)
    env = build_jewel_env(official, device)
    model, config, checkpoint = load_gflownet(args.checkpoint, device)

    started = perf_counter()
    exact_metrics = exact_policy_metrics(model, official, args.exact_data, device, args.batch_size)
    with open(args.test, encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    official_states = np.stack(
        [parse_official_state(row["initial_state"]) for row in test_rows]
    ).astype(np.int64)
    done, paths = rollout_paths(
        model,
        env,
        torch.from_numpy(official_states).to(device),
        max_steps=args.max_steps,
    )
    done_cpu = done.cpu().numpy()
    verified = 0
    lengths: list[int] = []
    rows: list[dict] = []
    for index, (row, path, solved) in enumerate(zip(test_rows, paths, done_cpu)):
        initial = parse_official_state(row["initial_state"])
        valid_official = bool(solved) and np.array_equal(
            official.apply_path(initial, path), official.central_state
        )
        structured = official.to_structured(initial)
        valid_internal = bool(solved) and apply_path(structured, path).rank() == 0
        valid = valid_official and valid_internal
        if valid:
            verified += 1
            lengths.append(len(path))
        rows.append(
            {
                "initial_state_id": int(row["initial_state_id"]),
                "solved": bool(solved),
                "verified": valid,
                "length": len(path) if valid else None,
            }
        )
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "config": config.to_dict(),
        "exact_validation": exact_metrics,
        "competition_greedy": {
            "count": len(rows),
            "solved_and_verified": verified,
            "score_on_solved": int(sum(lengths)),
            "mean_length_on_solved": float(np.mean(lengths)) if lengths else None,
            "max_steps": args.max_steps,
        },
        "seconds": perf_counter() - started,
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
