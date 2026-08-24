"""Validation gate for Q-shortlister: does student top-α*B contain teacher top-B?

Run this BEFORE deploying the Q-head into beam search. If recall < 0.99 at α=4,
the shortlister will silently lose ranking and beam quality will drop.

Procedure:
1. Sample N representative parent states (random walks at varying depths).
2. For each parent, expand all 24 children, dedup.
3. Score children with TEACHER → identify teacher's top-B.
4. Score parents with STUDENT (one forward, 24-vector) → identify student's top-α*B.
5. Measure: recall = |teacher_top_B ∩ student_top_αB| / |teacher_top_B|.

Sweep α to find the smallest α where recall ≥ target (default 0.99).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/09_eval_q_recall.py \
        --teacher megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --student megaminx/models/m23_q_shortlister/epoch_0499.pt \
        --alpha 1,2,3,4,6,8 \
        --target-recall 0.99
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


def load_v_model(path: Path, device: str):
    """Polymorphic loader — handles ResMLP and GraphTransformer checkpoints."""
    return load_model_checkpoint(path, device=device, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True, help="V-model (output_dim=1)")
    ap.add_argument("--student", type=Path, required=True, help="Q-head (output_dim=n_gen)")
    ap.add_argument("--n-parents", type=int, default=4096,
                    help="parent states to sample. Total candidates per α-test = n_parents * n_gen.")
    ap.add_argument("--k-max", type=int, default=80,
                    help="random-walk depth for parent sampling")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--alpha", type=str, default="1,2,3,4,6,8",
                    help="comma-separated α values to test. α*B candidates from student vs B from teacher.")
    ap.add_argument("--effective-B", type=int, default=131072,
                    help="effective beam size B for the recall test. The 'top-B' is "
                         "computed across all n_parents*n_gen candidates.")
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)

    teacher = load_v_model(args.teacher, device)
    if args.bf16 and device == "cuda":
        teacher = teacher.to(torch.bfloat16)
    student = load_v_model(args.student, device)
    if args.bf16 and device == "cuda":
        student = student.to(torch.bfloat16)
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params, output_dim={teacher.output_dim}")
    print(f"student: {sum(p.numel() for p in student.parameters()):,} params, output_dim={student.output_dim}")
    if student.output_dim != n_gen:
        print(f"ERROR: student output_dim {student.output_dim} != n_gen {n_gen}", file=sys.stderr)
        return 2

    # Sample parents from random walks.
    n_walks = max(1, args.n_parents // args.k_max)
    parents, depths = generate_walks_torch(
        puzzle, n_walks=n_walks, k_max=args.k_max,
        seed=args.seed, device=device, n_back=1,
    )
    parents = parents[: args.n_parents]
    print(f"sampled {parents.shape[0]} parent states (random-walk seed={args.seed})")

    # Build all children: (n_parents, n_gen, S)
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(device)
    B_par = parents.shape[0]
    children = torch.gather(
        parents.unsqueeze(1).expand(B_par, n_gen, parents.shape[1]),
        2,
        generators.unsqueeze(0).expand(B_par, n_gen, parents.shape[1]),
    )  # (B_par, n_gen, S)
    children_flat = children.reshape(B_par * n_gen, parents.shape[1])
    n_total = children_flat.shape[0]
    print(f"total candidates: {n_total:,} ({B_par} parents * {n_gen} children)")

    # Teacher V on every child.
    print("scoring children with teacher V...")
    t0 = time.time()
    chunk = 8192
    teacher_v = torch.empty(n_total, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for i in range(0, n_total, chunk):
            teacher_v[i : i + chunk] = teacher(children_flat[i : i + chunk]).flatten().to(torch.float32)
    print(f"  done in {time.time() - t0:.1f}s")

    # Student Q on every parent.
    print("scoring parents with student Q (one forward per parent)...")
    t0 = time.time()
    student_q = torch.empty((B_par, n_gen), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for i in range(0, B_par, chunk):
            student_q[i : i + chunk] = student(parents[i : i + chunk]).to(torch.float32)
    student_q_flat = student_q.view(-1)  # parallel to children_flat ordering
    print(f"  done in {time.time() - t0:.1f}s")

    # Compute recall sweep.
    B = args.effective_B if args.effective_B <= n_total else n_total
    target_recall = args.target_recall
    alpha_list = [float(x) for x in args.alpha.split(",")]
    print(f"\neffective B = {B:,}  (top-B by teacher) ; total candidates = {n_total:,}")

    # Teacher top-B (lowest values = best).
    _, teacher_topB = torch.topk(teacher_v, B, largest=False, sorted=False)
    teacher_set = set(teacher_topB.cpu().tolist())

    print(f"\n{'alpha':>6s} {'top-αB':>10s} {'recall':>8s} {'meets gate':>11s}")
    print("-" * 40)
    for alpha in alpha_list:
        k = min(int(round(alpha * B)), n_total)
        _, student_topαB = torch.topk(student_q_flat, k, largest=False, sorted=False)
        student_set = set(student_topαB.cpu().tolist())
        intersection = teacher_set & student_set
        recall = len(intersection) / max(B, 1)
        gate = "PASS" if recall >= target_recall else ""
        print(f"{alpha:>6.1f} {k:>10d} {recall:>8.4f} {gate:>11s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
