"""Build AZ v2 dataset with off-path sibling augmentation.

For each (state_i, action_i, value_i) tuple from the original AZ v1 dataset
(state along solver path, action taken, remaining moves), also expand 23
sibling states by applying each non-taken action to state_i. Label each
sibling with `V_teacher(sibling)` from a teacher V model (e.g., m_dd_v0 50ep).

Output schema:
  states_p: (N, 120) int8        — original path states (have policy target)
  actions_p: (N,) int8           — action taken at each path state
  values_p: (N,) float32         — remaining-distance from solver path
  states_v: (N + 23N, 120) int8  — path + siblings (have value target)
  values_v: (N + 23N,) float32   — value target for each (path: realized, sibling: V_teacher)

Usage:
    .venv/Scripts/python.exe megaminx/scripts/69_build_az_dataset_v2.py \\
        --base-dataset megaminx/data/az_dataset_78029.pt \\
        --teacher-v megaminx/models/m_dd_v0/epoch_0049.pt \\
        --out megaminx/data/az_dataset_v2.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def load_v_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        output_dim=1,
    )
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", required=True, type=Path)
    ap.add_argument("--teacher-v", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--inference-batch-size", type=int, default=16384)
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)

    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(args.device).long()  # (n_gen, S)

    print(f"loading base dataset: {args.base_dataset}", flush=True)
    base = torch.load(args.base_dataset, map_location=args.device, weights_only=False)
    states_p = base["states"].to(args.device).long()      # (N, S) int8 → long
    actions_p = base["actions"].to(args.device).long()    # (N,)
    values_p = base["values"].to(args.device)              # (N,) float32
    N = states_p.size(0)
    print(f"  {N:,} path tuples", flush=True)

    print(f"loading teacher V: {args.teacher_v}", flush=True)
    v_model = load_v_model(args.teacher_v, args.device)

    # For each path state, generate 24 children (one per action).
    # The taken action's child is on-path; the other 23 are siblings.
    # For "states_v", we keep original path state + 23 siblings (NOT the on-path child;
    # that's already at the next step in the path).
    print(f"generating siblings (23 per path state) and labeling...", flush=True)
    t0 = time.time()
    all_sibling_states = []
    all_sibling_values = []
    bs = args.inference_batch_size
    for i in range(0, N, bs):
        chunk_states = states_p[i : i + bs]   # (Bc, S)
        chunk_actions = actions_p[i : i + bs] # (Bc,)
        Bc = chunk_states.size(0)

        # Compute all 24 children: (Bc, n_gen, S)
        # For each (b, g), child[b, g] = chunk_states[b][perms[g]]
        children = chunk_states.unsqueeze(1).expand(Bc, n_gen, state_size).clone()
        gen_idx = perms.unsqueeze(0).expand(Bc, n_gen, state_size)
        children = torch.gather(children, 2, gen_idx)  # (Bc, n_gen, S)

        # Mask out the taken action: build mask (Bc, n_gen) bool with True for siblings
        sibling_mask = torch.ones((Bc, n_gen), dtype=torch.bool, device=args.device)
        sibling_mask.scatter_(1, chunk_actions.unsqueeze(1), False)
        # Flatten to (Bc * 23, S)
        siblings = children[sibling_mask]  # (Bc*23, S)

        # Evaluate V on siblings
        with torch.no_grad():
            sib_values = v_model(siblings).flatten().float()  # (Bc*23,)

        all_sibling_states.append(siblings.to(torch.int8).cpu())
        all_sibling_values.append(sib_values.cpu())

        if (i // bs) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+Bc:,}/{N:,} states processed in {elapsed:.1f}s", flush=True)

    siblings_states = torch.cat(all_sibling_states, dim=0)   # (23N, S) int8
    siblings_values = torch.cat(all_sibling_values, dim=0)   # (23N,) float32
    print(f"\n  siblings: {siblings_states.shape[0]:,} states, "
          f"value range [{siblings_values.min():.2f}, {siblings_values.max():.2f}]",
          flush=True)

    # Combine path states + sibling states for value training
    states_v = torch.cat([states_p.to(torch.int8).cpu(), siblings_states], dim=0)
    values_v = torch.cat([values_p.cpu(), siblings_values], dim=0)
    print(f"\n  total value-training set: {states_v.shape[0]:,} states", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states_p": states_p.to(torch.int8).cpu(),
        "actions_p": actions_p.cpu().to(torch.int8),
        "values_p": values_p.cpu(),
        "states_v": states_v,
        "values_v": values_v,
        "source_base": str(args.base_dataset),
        "source_teacher": str(args.teacher_v),
    }, args.out)
    print(f"\nwrote {args.out}: states_v {tuple(states_v.shape)} "
          f"({states_v.nbytes / 1e6:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
