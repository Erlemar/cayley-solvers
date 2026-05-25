"""Harvest frontier-regret pairwise triples from verified paths (doc 13.2).

The distinguishing feature vs m_rank_v0 (child-rank on random-walk children, which
TIED): we mine the states where the V model ACTUALLY MISRANKS relative to a verified
near-optimal path. At each step S_i -> S_{i+1} along a verified best path, the on-path
child S_{i+1} is the "good" child (verified remaining suffix = L-i-1). Any sibling c
(another of the 24 generators applied to S_i) that the V model scores LOWER than the
on-path child is a real misrank -- V would steer the beam toward c. Those become
"bad" children.

Output triples (parent, good, bad) feed a pairwise loss (want V(good) < V(bad)) as a
small auxiliary term in a Bellman fine-tune. The KEY DIAGNOSTIC printed here is the
misrank RATE: if the V rarely misranks on verified paths, frontier-regret has little
signal to learn (and likely ties, like m_rank_v0). If it misranks often, there is a
real mistake distribution to train on.

NOTE (v0 honesty): a "bad" sibling's TRUE distance is not verified -- it is only
"off the best-known path while V prefers it". On a best-known path the on-path child
is near-optimal, so this is a reasonable (soft) label; the pairwise loss tolerates
some noise. A future --confirm-resolve would re-solve each bad sibling to verify its
realized suffix > suffix_good (not implemented in v0; harvest is GPU-light without it).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/84_harvest_frontier_regret.py \\
        --checkpoint megaminx/models/m_dd_v0/epoch_0049.pt \\
        --paths-csv megaminx/submissions/merge_v12_az_v4_plus_our.csv \\
        --out megaminx/data/frontier_regret_triples.pt \\
        --bf16
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.model import ResMLPDistance
from megaminx.bridge import compute_prefix_states
from megaminx.puzzle import Megaminx


def load_v_model(ckpt_path: Path, device: str) -> ResMLPDistance:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    cfg = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=cfg.get("state_size", 120),
        num_classes=cfg.get("num_classes", 120),
        hidden_dims=tuple(cfg.get("hidden_dims", (2048, 512))),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        encoding=cfg.get("encoding", "embedding"),
        embed_dim=cfg.get("embed_dim", 16),
    ).to(device).eval()
    model.load_state_dict(sd)
    return model


def load_test_states(test_csv: Path) -> dict[int, np.ndarray]:
    states = {}
    with open(test_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["initial_state_id"])
            states[pid] = np.array([int(x) for x in row["initial_state"].split(",")], dtype=np.int64)
    return states


def load_paths(paths_csv: Path) -> dict[int, list[str]]:
    paths = {}
    with open(paths_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["initial_state_id"])
            raw = row["path"].strip()
            paths[pid] = raw.split(".") if raw else []
    return paths


@torch.inference_mode()
def v_predict(model, states_t: torch.Tensor, chunk: int, use_bf16: bool) -> torch.Tensor:
    """states_t: (N, S) long. Returns (N,) float32."""
    out = torch.empty(states_t.size(0), dtype=torch.float32, device=states_t.device)
    for i in range(0, states_t.size(0), chunk):
        c = states_t[i : i + chunk]
        if use_bf16:
            with torch.autocast(device_type=states_t.device.type, dtype=torch.bfloat16):
                v = model(c).float().flatten()
        else:
            v = model(c).float().flatten()
        out[i : i + c.size(0)] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "m_dd_v0" / "epoch_0049.pt",
                    help="V model whose misranks we harvest (train this same model to fix them).")
    ap.add_argument("--paths-csv", type=Path,
                    default=PROJECT / "submissions" / "merge_v12_az_v4_plus_our.csv",
                    help="verified best paths (full-1001 submission CSV).")
    ap.add_argument("--test-csv", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "frontier_regret_triples.pt")
    ap.add_argument("--pids", type=str, default=None, help="comma list; default all.")
    ap.add_argument("--max-path-len", type=int, default=200,
                    help="skip pids whose path exceeds this (excludes long fallback paths "
                         "that aren't near-optimal, which would inject bad 'good' labels).")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="a sibling counts as 'bad' if V(sibling) < V(on-path) - margin. "
                         "0 = any strict misrank; >0 focuses on clearer mistakes.")
    ap.add_argument("--max-bad-per-step", type=int, default=3,
                    help="cap emitted bad siblings per misrank step (lowest-V first).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    solved = tuple(puzzle.solved_state)
    move_names = list(puzzle.move_names)
    n_gen = len(move_names)
    move_to_idx = {m: i for i, m in enumerate(move_names)}
    gen_perms = torch.tensor(
        [list(puzzle.generators[m]) for m in move_names], dtype=torch.long, device=args.device
    )  # (n_gen, S)

    model = load_v_model(args.checkpoint, args.device)
    test_states = load_test_states(args.test_csv)
    paths = load_paths(args.paths_csv)

    if args.pids:
        pid_list = [int(p) for p in args.pids.split(",")]
    else:
        pid_list = sorted(paths.keys())

    print(f"harvesting from {len(pid_list)} pids; V={args.checkpoint.name}; "
          f"paths={args.paths_csv.name}; max_path_len={args.max_path_len}; margin={args.margin}",
          flush=True)

    parents_all, goods_all, bads_all = [], [], []
    meta_pid, meta_depth, meta_suffix, meta_vgood, meta_vbad = [], [], [], [], []

    n_steps_examined = 0
    n_misrank_steps = 0
    rank_good_sum = 0  # sum of (#siblings V prefers over on-path child) across steps
    n_skipped_len = 0
    n_skipped_verify = 0
    n_pids_used = 0
    depth_bucket_misrank = Counter()
    depth_bucket_steps = Counter()
    pid_bucket_triples = Counter()
    sanity_checked = False

    t0 = time.time()
    for pid in pid_list:
        if pid not in test_states or pid not in paths:
            continue
        path = paths[pid]
        L = len(path)
        if L == 0 or L > args.max_path_len:
            n_skipped_len += 1
            continue
        if any(m not in move_to_idx for m in path):
            n_skipped_verify += 1
            continue
        prefix = compute_prefix_states(test_states[pid], path, puzzle)  # L+1 tuples
        if tuple(prefix[-1]) != solved:
            n_skipped_verify += 1
            continue
        n_pids_used += 1

        parents = torch.tensor(np.array(prefix[:-1]), dtype=torch.long, device=args.device)  # (L,S)
        children = parents[:, gen_perms]  # (L, n_gen, S)
        Lc = children.shape[0]
        v_children = v_predict(
            model, children.reshape(Lc * n_gen, -1), args.chunk, args.bf16
        ).reshape(Lc, n_gen)  # (L, n_gen)

        move_idxs = torch.tensor([move_to_idx[m] for m in path], device=args.device)  # (L,)
        rows = torch.arange(Lc, device=args.device)
        v_good = v_children[rows, move_idxs]  # (L,)

        if not sanity_checked:
            on_path_child = children[rows, move_idxs]  # (L,S)
            expected = torch.tensor(np.array(prefix[1:]), dtype=torch.long, device=args.device)
            assert torch.equal(on_path_child, expected), "child[move_idx] != S_{i+1} (convention bug)"
            sanity_checked = True

        # misrank: siblings with V strictly below the on-path child (minus margin)
        better = v_children < (v_good.unsqueeze(1) - args.margin)  # (L, n_gen)
        better[rows, move_idxs] = False  # exclude the on-path child itself
        rank_good_per_step = better.sum(dim=1)  # (L,)

        n_steps_examined += Lc
        misrank_steps_mask = rank_good_per_step > 0
        n_misrank_steps += int(misrank_steps_mask.sum().item())
        rank_good_sum += int(rank_good_per_step.sum().item())

        for i in range(Lc):
            db = min(i // 10, 12)
            depth_bucket_steps[db] += 1
            k = int(rank_good_per_step[i].item())
            if k == 0:
                continue
            depth_bucket_misrank[db] += 1
            # pick up to max_bad_per_step worst (lowest-V) bad siblings
            bad_idx = torch.nonzero(better[i], as_tuple=True)[0]
            bad_v = v_children[i, bad_idx]
            order = torch.argsort(bad_v)[: args.max_bad_per_step]
            for a in bad_idx[order].tolist():
                parents_all.append(parents[i].to(torch.int8).cpu())
                goods_all.append(children[i, move_idxs[i]].to(torch.int8).cpu())
                bads_all.append(children[i, a].to(torch.int8).cpu())
                meta_pid.append(pid)
                meta_depth.append(i)
                meta_suffix.append(L - i - 1)
                meta_vgood.append(float(v_good[i].item()))
                meta_vbad.append(float(v_children[i, a].item()))
                pid_bucket_triples[min(pid // 100, 10)] += 1

    n_triples = len(parents_all)
    elapsed = time.time() - t0

    print(f"\n=== harvest stats ({elapsed:.0f}s) ===", flush=True)
    print(f"pids used: {n_pids_used}  (skipped: {n_skipped_len} too-long/empty, "
          f"{n_skipped_verify} unverified/badmove)")
    print(f"on-path steps examined: {n_steps_examined}")
    if n_steps_examined:
        print(f"misrank steps: {n_misrank_steps} "
              f"({100.0 * n_misrank_steps / n_steps_examined:.1f}% of steps)")
        print(f"mean siblings-V-prefers-over-correct per step: "
              f"{rank_good_sum / n_steps_examined:.3f}")
    print(f"triples emitted: {n_triples}")
    print("\nmisrank rate by depth bucket (step index // 10):")
    for db in sorted(depth_bucket_steps):
        s = depth_bucket_steps[db]
        m = depth_bucket_misrank[db]
        lo = db * 10
        hdr = f"[{lo},{lo+10})" if db < 12 else "[120+)"
        print(f"  {hdr:>8}  steps={s:>6}  misrank={m:>6}  ({100.0*m/max(s,1):.1f}%)")
    print("\ntriples by pid bucket (pid // 100):")
    for pb in sorted(pid_bucket_triples):
        print(f"  bucket {pb:>2} (pids {pb*100}-{pb*100+99}): {pid_bucket_triples[pb]}")

    if n_triples == 0:
        print("\nNO TRIPLES -- V does not misrank on these verified paths. "
              "Frontier-regret has no signal here (consistent with the 6M ceiling).")
        return 0

    out = {
        "parents": torch.stack(parents_all),       # (N,S) int8
        "goods": torch.stack(goods_all),           # (N,S) int8
        "bads": torch.stack(bads_all),             # (N,S) int8
        "pid": torch.tensor(meta_pid, dtype=torch.int32),
        "depth": torch.tensor(meta_depth, dtype=torch.int32),
        "suffix_good": torch.tensor(meta_suffix, dtype=torch.int32),
        "v_good": torch.tensor(meta_vgood, dtype=torch.float32),
        "v_bad": torch.tensor(meta_vbad, dtype=torch.float32),
        "meta": {
            "checkpoint": str(args.checkpoint),
            "paths_csv": str(args.paths_csv),
            "max_path_len": args.max_path_len,
            "margin": args.margin,
            "max_bad_per_step": args.max_bad_per_step,
            "n_pids_used": n_pids_used,
            "n_steps_examined": n_steps_examined,
            "n_misrank_steps": n_misrank_steps,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\nwrote {n_triples} triples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
