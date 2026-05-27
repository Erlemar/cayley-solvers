"""Depth-stratified Q-shortlister recall: does recall sag at deep parents?

Companion to 09_eval_q_recall.py. That script reports ONE global recall number
over a mixed-depth parent pool. This one buckets parents by random-walk depth
and reports recall@alpha SEPARATELY per depth bucket.

Why this matters (GT-Q saturation diagnostic):
  The deployed qshort (QShortlisterSolver) selects GLOBAL top-alpha*B children
  across all parents in the beam. At beam step j every parent is at ~the same
  depth, so the load-bearing question is: "at depth d, does student top-alpha*B
  recall teacher top-B?" A student whose implied value (min_a Q) fails to
  SATURATE at the puzzle diameter (the failure mode that killed the GraphTransformer
  V model, see gt_v_no_saturation memory / CLAUDE rule 23) gives deep parents
  systematically inflated Q -> their genuinely-good children get dropped from the
  global shortlist -> recall collapses at depth even when the pooled number looks
  fine. Flat-across-depth recall == calibration OK. Sagging-at-depth recall == the
  bipartite action-node structure (or a saturation fix) is needed.

Per bucket we hold #parents and B fixed so buckets are directly comparable.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/09b_eval_q_recall_by_depth.py \
        --teacher megaminx/models/m_az_v4_v_only_e99.pt \
        --student megaminx/models/m_gt_q_v0/epoch_0119.pt \
        --alpha 1,1.5,2,3,4 --bf16
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

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.search import load_model_checkpoint
from megaminx.puzzle import Megaminx


@torch.inference_mode()
def _score_children_teacher(teacher, children_flat, chunk, device):
    n = children_flat.shape[0]
    out = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(0, n, chunk):
        out[i : i + chunk] = teacher(children_flat[i : i + chunk]).flatten().to(torch.float32)
    return out


@torch.inference_mode()
def _score_parents_student(student, parents, n_gen, chunk, device):
    P = parents.shape[0]
    out = torch.empty((P, n_gen), dtype=torch.float32, device=device)
    for i in range(0, P, chunk):
        out[i : i + chunk] = student(parents[i : i + chunk]).to(torch.float32)
    return out


def _recall_for_pool(teacher, student, parents, generators, n_gen, B, alpha_list, chunk, device):
    """Returns dict alpha -> recall for one parent pool (parents all same-ish depth)."""
    P, S = parents.shape
    children = torch.gather(
        parents.unsqueeze(1).expand(P, n_gen, S),
        2,
        generators.unsqueeze(0).expand(P, n_gen, S),
    )
    children_flat = children.reshape(P * n_gen, S)
    n_total = children_flat.shape[0]

    teacher_v = _score_children_teacher(teacher, children_flat, chunk, device)
    student_q = _score_parents_student(student, parents, n_gen, chunk, device).reshape(-1)

    b = min(B, n_total)
    _, teacher_topB = torch.topk(teacher_v, b, largest=False, sorted=False)
    teacher_set = set(teacher_topB.cpu().tolist())

    out = {}
    for alpha in alpha_list:
        k = min(int(round(alpha * b)), n_total)
        _, student_topk = torch.topk(student_q, k, largest=False, sorted=False)
        recall = len(teacher_set & set(student_topk.cpu().tolist())) / max(b, 1)
        out[alpha] = recall
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True, help="V-model (output_dim=1)")
    ap.add_argument("--student", type=Path, required=True, help="Q-head (output_dim=n_gen)")
    ap.add_argument("--depth-buckets", type=str, default="10,20,30,40,50,60,70,80",
                    help="comma-separated bucket centers (random-walk depth)")
    ap.add_argument("--bucket-halfwidth", type=int, default=5,
                    help="a parent at depth d joins bucket c if |d-c| <= halfwidth")
    ap.add_argument("--parents-per-bucket", type=int, default=2048,
                    help="parents sampled per bucket (B = this value); buckets with "
                         "fewer available parents are reported with the count they have")
    ap.add_argument("--k-max", type=int, default=85, help="max random-walk depth for the pool")
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--alpha", type=str, default="1,1.5,2,3,4")
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(device)

    dt = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    teacher = load_model_checkpoint(args.teacher, device=device, dtype=dt)
    student = load_model_checkpoint(args.student, device=device, dtype=dt)
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params, "
          f"output_dim={getattr(teacher, 'output_dim', '?')}", flush=True)
    print(f"student: {sum(p.numel() for p in student.parameters()):,} params, "
          f"output_dim={getattr(student, 'output_dim', '?')}", flush=True)
    if getattr(student, "output_dim", 1) != n_gen:
        print(f"ERROR: student output_dim != n_gen ({n_gen})", file=sys.stderr)
        return 2

    centers = [int(x) for x in args.depth_buckets.split(",")]
    alpha_list = [float(x) for x in args.alpha.split(",")]
    hw = args.bucket_halfwidth
    per = args.parents_per_bucket

    # One big walk pool, then bucket by depth. Generate enough walks that each
    # bucket can be filled. Each walk contributes ~k_max states across depths.
    n_walks = max(1, (per * len(centers) * 4) // args.k_max)
    pool, depths = generate_walks_torch(
        puzzle, n_walks=n_walks, k_max=args.k_max, seed=args.seed, device=device, n_back=1,
    )
    print(f"walk pool: {pool.shape[0]:,} states over depths "
          f"[{int(depths.min())}, {int(depths.max())}]", flush=True)

    chunk = 8192
    print(f"\nRecall@alpha by parent depth (B = parents-per-bucket = {per}; "
          f"lower V/Q = better). Target {args.target_recall:.2f}.", flush=True)
    header = f"{'depth':>7s} {'#par':>6s} " + " ".join(f"a={a:<4g}" for a in alpha_list)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    pooled_parents = []
    for c in centers:
        mask = (depths - c).abs() <= hw
        idx = torch.nonzero(mask, as_tuple=True)[0]
        if idx.numel() == 0:
            print(f"{c:>7d} {0:>6d}  (no parents in bucket)", flush=True)
            continue
        idx = idx[torch.randperm(idx.numel(), device=device)[:per]]
        parents = pool[idx]
        pooled_parents.append(parents)
        B = parents.shape[0]
        t0 = time.time()
        rec = _recall_for_pool(teacher, student, parents, generators, n_gen, B, alpha_list, chunk, device)
        cells = []
        for a in alpha_list:
            mark = "*" if rec[a] >= args.target_recall else " "
            cells.append(f"{rec[a]:.3f}{mark}")
        print(f"{c:>7d} {B:>6d} " + " ".join(f"{cell:>6s}" for cell in cells)
              + f"   ({time.time()-t0:.1f}s)", flush=True)

    # Pooled (mixed-depth) row == what 09_eval_q_recall.py roughly reports.
    if pooled_parents:
        parents = torch.cat(pooled_parents, dim=0)
        B = min(args.parents_per_bucket, parents.shape[0])
        rec = _recall_for_pool(teacher, student, parents, generators, n_gen, B, alpha_list, chunk, device)
        cells = []
        for a in alpha_list:
            mark = "*" if rec[a] >= args.target_recall else " "
            cells.append(f"{rec[a]:.3f}{mark}")
        print("-" * len(header), flush=True)
        print(f"{'POOLED':>7s} {parents.shape[0]:>6d} " + " ".join(f"{cell:>6s}" for cell in cells), flush=True)

    print("\n* = meets target recall. Read DOWN each alpha column: flat = calibration OK; "
          "falling with depth = value non-saturation (bipartite/saturation-fix indicated).",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
