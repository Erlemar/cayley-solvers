"""Build a distillation dataset for the all-neighbors V-head experiment.

For N random-walk parent states across depths [1, k_max], compute the teacher V
of all 24 children:

    q_target[i] = V_teacher(apply(parent, move_i))   (solved child pinned to 0)

This is the clean "all-neighbors V" target: distilling the PRODUCTION V's
child-ranking into a single 24-output forward pass, with BROAD off-path coverage
(random walks span the manifold; the #1 lesson for beam-usable heads). Unlike the
oracle dataset (29_*, exact but d<=5 only) and m23 (KL+shortlist confound,
distilled from m05), this isolates the question: can a 24-head reproduce the
production teacher's child-ranking at DEPTH?

Output: data/distill_q_<tag>.pt with
    {"states": (N,120) int8, "q_targets": (N,24) float16, "depths": (N,) int8}
Consumed unchanged by 30_train_oracle_q.py (masked MSE; all targets >= 0 so the
mask is full).

Run:
  PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/30b_build_distill_q_dataset.py \
      --teacher megaminx/models/m_az_v4_v_only_e99.pt \
      --n-parents 1200000 --k-max 80 --out megaminx/data/distill_q_azv4.pt --bf16
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
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path,
                    default=PROJECT / "models" / "m_az_v4_v_only_e99.pt")
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "data" / "distill_q_azv4.pt")
    ap.add_argument("--n-parents", type=int, default=1_200_000)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--walks-per-chunk", type=int, default=2000,
                    help="random walks generated per memory chunk (k_max states each)")
    ap.add_argument("--parent-batch", type=int, default=8192,
                    help="parents per teacher-forward batch")
    ap.add_argument("--teacher-chunk", type=int, default=16384,
                    help="children per teacher inner forward chunk")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)
    state_size = len(puzzle.solved_state)
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(device)  # (n_gen, S)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)

    dt = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    teacher = load_model_checkpoint(args.teacher, device=device, dtype=dt)
    teacher.eval()
    out_dim = getattr(teacher, "output_dim", 1)
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params, "
          f"output_dim={out_dim}", flush=True)
    if out_dim != 1:
        print(f"ERROR: teacher must be a V model (output_dim=1), got {out_dim}",
              file=sys.stderr)
        return 2

    # 1. Generate broad-coverage parent states (random walks, depths [1, k_max]).
    # generate_walks_torch returns states in STEP-MAJOR order (all walks at depth 1,
    # then depth 2, ...), so we collect WHOLE chunks then shuffle+trim -- truncating
    # mid-chunk would bias toward shallow depths.
    print(f"\ngenerating >={args.n_parents:,} parents (RW depths [1,{args.k_max}]) ...",
          flush=True)
    t0 = time.time()
    chunk_states, chunk_depths = [], []
    total = 0
    chunk_i = 0
    while total < args.n_parents:
        st, dp = generate_walks_torch(
            puzzle, n_walks=args.walks_per_chunk, k_max=args.k_max,
            seed=args.seed + chunk_i, device=device, n_back=1,
        )
        chunk_states.append(st.to(torch.int8).cpu())
        chunk_depths.append(dp.to(torch.int8).cpu())
        total += st.shape[0]
        chunk_i += 1
    states_all = torch.cat(chunk_states, dim=0)
    depths_all = torch.cat(chunk_depths, dim=0)
    perm = torch.randperm(states_all.shape[0],
                          generator=torch.Generator().manual_seed(args.seed))[:args.n_parents]
    states_cpu = states_all[perm].contiguous()
    depths_cpu = depths_all[perm].contiguous()
    del states_all, depths_all, chunk_states, chunk_depths
    print(f"  parents ready: {states_cpu.shape[0]:,} "
          f"(depths {int(depths_cpu.min())}..{int(depths_cpu.max())}) "
          f"in {time.time()-t0:.1f}s", flush=True)

    # 2. For each parent, teacher-V all 24 children -> q_targets (B, 24).
    print(f"\nscoring 24 children/parent with teacher ({args.n_parents * n_gen:,} "
          f"forwards) ...", flush=True)
    t0 = time.time()
    q_targets = torch.empty((args.n_parents, n_gen), dtype=torch.float16)
    n_solved_children = 0
    for start in range(0, args.n_parents, args.parent_batch):
        end = min(start + args.parent_batch, args.n_parents)
        parents = states_cpu[start:end].to(device).long()  # (B, S)
        B = parents.shape[0]
        children = torch.gather(
            parents.unsqueeze(1).expand(B, n_gen, state_size),
            2,
            generators.unsqueeze(0).expand(B, n_gen, state_size),
        ).reshape(B * n_gen, state_size)  # (B*n_gen, S)
        is_solved = (children == solved_state).all(dim=1)
        vals = torch.empty(B * n_gen, dtype=torch.float32, device=device)
        for i in range(0, B * n_gen, args.teacher_chunk):
            vals[i:i + args.teacher_chunk] = (
                teacher(children[i:i + args.teacher_chunk]).flatten().float()
            )
        vals = torch.where(is_solved, torch.zeros_like(vals), vals)
        n_solved_children += int(is_solved.sum().item())
        q_targets[start:end] = vals.view(B, n_gen).to(torch.float16).cpu()
        if (start // args.parent_batch) % 25 == 0:
            print(f"  {end:>9,} / {args.n_parents:,} ({time.time()-t0:.1f}s)",
                  flush=True)
    print(f"  done in {time.time()-t0:.1f}s; solved children pinned to 0: "
          f"{n_solved_children:,}", flush=True)

    # Sanity stats
    qf = q_targets.float()
    print(f"\nq_target stats: min={qf.min():.2f} max={qf.max():.2f} "
          f"mean={qf.mean():.2f}", flush=True)
    for d in (5, 20, 40, 60, 80):
        m = (depths_cpu.long() == d)
        if m.any():
            print(f"  depth {d:>2d}: mean child-V = {qf[m].mean():.2f} "
                  f"(n={int(m.sum())})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states_cpu,
        "q_targets": q_targets,
        "depths": depths_cpu,
        "teacher": str(args.teacher),
    }, args.out)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
