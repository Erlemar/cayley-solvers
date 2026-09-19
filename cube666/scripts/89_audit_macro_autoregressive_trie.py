"""Audit verified short-macro recall from autoregressive trie decoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_autoregressive import load_macro_autoregressive_checkpoint  # noqa: E402


@dataclass
class TrieNode:
    children: dict[int, int] = field(default_factory=dict)
    actions: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--minimum-value", type=float, default=0.0)
    parser.add_argument("--decode-beam", type=int, default=512)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=89666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def build_trie(tokens: np.ndarray, eos: int) -> list[TrieNode]:
    nodes = [TrieNode()]
    for action, row in enumerate(tokens):
        if row[0] < 0:
            continue
        node = 0
        for raw_token in row:
            token = int(raw_token)
            if token < 0:
                break
            if token == eos:
                nodes[node].actions.append(action)
                break
            child = nodes[node].children.get(token)
            if child is None:
                child = len(nodes)
                nodes[node].children[token] = child
                nodes.append(TrieNode())
            node = child
    return nodes


@torch.inference_mode()
def decode(
    model: torch.nn.Module,
    state: np.ndarray,
    trie: list[TrieNode],
    *,
    beam_width: int,
    topk: int,
    device: torch.device,
) -> list[int]:
    state_tensor = torch.from_numpy(state[None]).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state_hidden = model.encode(state_tensor)
    hidden = model.decoder_initial(state_hidden)
    node_ids = [0]
    last_tokens = torch.full(
        (1,), model.config.bos_token, dtype=torch.long, device=device
    )
    scores = torch.zeros(1, device=device)
    completed: list[tuple[float, int]] = []
    for _ in range(model.config.maximum_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, next_hidden = model.decode_step(last_tokens, hidden)
        logp = torch.log_softmax(logits.float(), dim=1)
        candidate_scores: list[torch.Tensor] = []
        candidate_nodes: list[int] = []
        candidate_tokens: list[int] = []
        candidate_parents: list[int] = []
        for parent, node_id in enumerate(node_ids):
            node = trie[node_id]
            if node.actions:
                terminal_score = float(
                    scores[parent] + logp[parent, model.config.eos_token]
                )
                completed.extend((terminal_score, action) for action in node.actions)
            for token, child in node.children.items():
                candidate_scores.append(scores[parent] + logp[parent, token])
                candidate_nodes.append(child)
                candidate_tokens.append(token)
                candidate_parents.append(parent)
        if not candidate_scores:
            break
        stacked = torch.stack(candidate_scores)
        keep = min(beam_width, len(candidate_scores))
        kept_scores, positions = stacked.topk(keep)
        selected = positions.cpu().numpy()
        parent_tensor = torch.as_tensor(
            [candidate_parents[position] for position in selected], device=device
        )
        hidden = next_hidden[:, parent_tensor]
        scores = kept_scores
        node_ids = [candidate_nodes[position] for position in selected]
        last_tokens = torch.as_tensor(
            [candidate_tokens[position] for position in selected], device=device
        )
    completed.sort(reverse=True)
    output: list[int] = []
    seen: set[int] = set()
    for _, action in completed:
        if action in seen:
            continue
        seen.add(action)
        output.append(action)
        if len(output) == topk:
            break
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model, checkpoint = load_macro_autoregressive_checkpoint(args.checkpoint, device)
    action_tokens = np.asarray(checkpoint["action_tokens"], dtype=np.int16)
    action_costs = np.asarray(checkpoint["action_costs"])
    trie = build_trie(action_tokens, model.config.eos_token)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    eligible: list[int] = []
    short_labels: dict[int, set[int]] = {}
    for row in range(len(states)):
        valid = {
            int(action)
            for action in labels[row, : int(counts[row])]
            if action >= 0 and action_costs[int(action)] <= 14
        }
        if (
            valid
            and int(groups[row]) % args.folds == args.fold
            and values[row] >= args.minimum_value
        ):
            eligible.append(row)
            short_labels[row] = valid
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.samples:
        eligible = rng.choice(eligible, size=args.samples, replace=False).tolist()
    hits = {1: 0, 16: 0, 128: 0, 512: 0}
    hits_high = {1: 0, 16: 0, 128: 0, 512: 0}
    high_count = 0
    proposal_counts: list[int] = []
    started = time.perf_counter()
    for number, row in enumerate(eligible, 1):
        proposed = decode(
            model,
            states[row],
            trie,
            beam_width=args.decode_beam,
            topk=args.topk,
            device=device,
        )
        proposal_counts.append(len(proposed))
        correct = short_labels[row]
        is_high = values[row] >= 80
        high_count += int(is_high)
        for width in hits:
            found = bool(correct.intersection(proposed[:width]))
            hits[width] += int(found)
            if is_high:
                hits_high[width] += int(found)
        if number == 1 or number % 32 == 0:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "samples": number,
                        "top16": hits[16] / number,
                        "top512": hits[512] / number,
                    }
                ),
                flush=True,
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "decode_beam": args.decode_beam,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "high_value_samples": high_count,
        "mean_proposals": float(np.mean(proposal_counts)),
        "samples": len(eligible),
        **{f"top{width}_recall": hits[width] / len(eligible) for width in hits},
        **{
            f"high_value_top{width}_recall": hits_high[width] / max(high_count, 1)
            for width in hits_high
        },
        "trie_nodes": len(trie),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
