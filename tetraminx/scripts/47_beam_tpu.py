"""Run the pure-TorchTPU beam search (tetraminx.beam_tpu) and verify every path.

No JAX anywhere -- this is the "fully TorchTPU" beam. It also runs unchanged on
CUDA/CPU, which is how the TPU result is checked: the same seed and beam width
must produce the same solution lengths on both.

    python tetraminx/scripts/47_beam_tpu.py \
        --checkpoint tetraminx/models/mx_tf_az/epoch_1500.pt \
        --device tpu --beam 4096 --pids 0-7 --compile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.beam_tpu import TpuBeamConfig, TpuBeamSearch
from tetraminx.puzzle import Tetraminx


def parse_pids(spec: str, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [p for p in out if 0 <= p < n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Pure-TorchTPU beam search.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--endgame", type=Path, default=None)
    ap.add_argument("--beam", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--model-chunk", type=int, default=2048)
    ap.add_argument("--check-every", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=2,
                    help="send-bucket overshoot; each rank receives alpha*B_local")
    ap.add_argument("--pids", type=str, default="0-7")
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the step body (dynamic=False, required on TPU)")
    ap.add_argument("--model-loop", action="store_true",
                    help="Ablation: unroll model chunks instead of the default compiled TPU scan")
    ap.add_argument("--half-sdpa", action="store_true",
                    help="Allow bf16 attention reduction on TPU; faster in the benchmark, "
                         "but changes scores and can change solution paths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-impl", default=None, choices=["sdpa", "einsum"],
                    help="override the checkpoint's attention path; benchmark "
                         "the choice for your model and runtime")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    if args.half_sdpa and dev != "tpu":
        ap.error("--half-sdpa is only supported by this driver on --device tpu")
    if dev == "tpu":
        # TorchTPU reads this PyTorch context flag despite its CUDA namespace.
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(args.half_sdpa)

    # --- distributed -------------------------------------------------------
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        _ = torch.device(dev)
        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist" if dev == "tpu" else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    is_main = rank == 0
    if not is_main:
        # Only rank 0 narrates. Every rank runs the IDENTICAL compute below --
        # they all hold the gathered trees, so they all reconstruct the same
        # path. Diverging would risk a mesh deadlock at the next collective.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    gens_np = np.stack([puzzle.generators[m] for m in move_names]).astype(np.int64)
    gens = torch.from_numpy(gens_np)
    name_to_idx = {m: i for i, m in enumerate(move_names)}

    endgame_path = args.endgame or (args.data_dir / "bfs_endgame.npz")
    z = np.load(endgame_path, allow_pickle=True)
    ztab = torch.from_numpy(z["ztab"])
    table_hashes = torch.from_numpy(z["hashes"])
    table_depths = torch.from_numpy(z["depths"])
    print(f"endgame table: {table_hashes.numel():,} states at d<={int(z['max_depth'])}",
          flush=True)

    # --- model -------------------------------------------------------------
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = dict(ck["model_config"])
    lp = Path(mcfg.get("layout_path", ""))
    if lp and not lp.is_absolute():
        mcfg["layout_path"] = str(PROJECT / lp)
    if args.attn_impl is not None and mcfg.get("arch") == "transformer":
        mcfg["attn_impl"] = args.attn_impl
    from tetraminx.models import model_from_config
    model = model_from_config(mcfg).to(dev).eval()
    model.load_state_dict({k.removeprefix("_orig_mod."): v
                           for k, v in ck["state_dict"].items()})
    model.return_value = False              # beam reads the Q head only
    if args.bf16:
        model = model.to(torch.bfloat16)
    print(f"model: {mcfg.get('arch')} attn_impl={mcfg.get('attn_impl', 'sdpa')} "
          f"epoch {ck.get('epoch')}", flush=True)

    # --- scrambles ---------------------------------------------------------
    import csv
    states_by_pid: dict[int, np.ndarray] = {}
    with open(args.data_dir / "test.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = int(row["initial_state_id"])
            states_by_pid[pid] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int64)
    pids = parse_pids(args.pids, len(states_by_pid))

    cfg = TpuBeamConfig(beam_width=args.beam, max_steps=args.max_steps,
                        model_chunk=args.model_chunk, check_every=args.check_every,
                        alpha=args.alpha, bf16=args.bf16,
                        model_scan=args.compile and dev == "tpu" and not args.model_loop)
    beam = TpuBeamSearch(model, gens, ztab, table_hashes, table_depths, cfg,
                         device=dev, seed=args.seed, rank=rank, world=world)
    print(f"beam: global {cfg.beam_width:,} = {beam.b_local:,}/rank x {world} "
          f"ranks | alpha {cfg.alpha} -> {beam.recv_n:,} candidates/rank/step",
          flush=True)

    step_fn = None
    if args.compile:
        # dynamic=False is mandatory on TPU; every shape in step() is fixed.
        if dev == "tpu":
            from torch._functorch import config as ft_config
            from torch_tpu._internal.compile._backend import TpuBackend
            # The AOT graph cache reused a different global precision variant
            # in controlled tests. Keep the backend's compiled-kernel cache,
            # but bypass serialized AOT graphs for every precision setting.
            ft_config.enable_autograd_cache = False
            ft_config.enable_remote_autograd_cache = False
            step_fn = torch.compile(beam.step,
                backend=TpuBackend(enable_serialization=False), dynamic=False)
        else:
            step_fn = torch.compile(beam.step, dynamic=False)
        print("step body compiled (dynamic=False)", flush=True)

    # Host-side replay: the ONLY thing that decides whether a path is real.
    solved = np.asarray(puzzle.solved_state, dtype=np.int64)

    def replay(s0: np.ndarray, names: list[str]) -> np.ndarray:
        cur = s0
        for nm in names:
            cur = cur[gens_np[name_to_idx[nm]]]
        return cur

    def descend(state: np.ndarray) -> list[str]:
        """Exact optimal tail from an endgame-table state down to solved."""
        cur = torch.from_numpy(state).to(dev)
        ztab_d, th, td = beam.ztab, beam.table_hashes, beam.table_depths

        def lookup(x):
            from tetraminx.beam_tpu import _zobrist
            h = _zobrist(x, ztab_d)
            pos = torch.searchsorted(th, h).clamp(max=th.numel() - 1)
            hit = th.index_select(0, pos) == h
            d = td.index_select(0, pos).to(torch.int64)
            return torch.where(hit, d, torch.full_like(d, -1))

        d = int(lookup(cur[None, :])[0].item())
        assert d >= 0, "descend() called on a state outside the table"
        out: list[str] = []
        gens_d = gens.to(dev)
        while d > 0:
            children = cur[gens_d]
            cd = lookup(children)
            nxt = int(torch.argmax((cd == d - 1).to(torch.int32)).item())
            out.append(move_names[nxt])
            cur = children[nxt]
            d -= 1
        return out

    results: dict[int, str] = {}
    n_solved = 0
    total_len = 0
    t_all = time.time()
    for pid in pids:
        s0 = states_by_pid[pid]
        t0 = time.time()
        hit_rank, hit_step, hit_pos, hit_depth, tree = beam.solve(
            torch.from_numpy(s0), compiled_step=step_fn)
        wall = time.time() - t0

        if hit_step is None:
            print(f"pid {pid:4d}  NO HIT in {cfg.max_steps} steps  {wall:6.1f}s",
                  flush=True)
            continue

        path = TpuBeamSearch.walk_back(tree, hit_rank, hit_step, hit_pos, move_names)
        mid = replay(s0, path)
        tail = descend(mid)
        full = path + tail

        final = replay(s0, full)
        ok = bool(np.array_equal(final, solved))
        n_solved += int(ok)
        total_len += len(full)
        results[pid] = ".".join(full)
        print(f"pid {pid:4d}  len {len(full):3d} "
              f"(beam {len(path)} + tail {len(tail)})  "
              f"hit@step {hit_step} r{hit_rank} d={hit_depth}  {wall:6.1f}s  "
              f"{'VERIFIED' if ok else 'REPLAY FAILED'}", flush=True)
        if not ok:
            raise SystemExit(f"pid {pid}: replay did not reach solved")

    n = len(pids)
    print(f"\n{n_solved}/{n} solved and replay-verified; "
          f"total {total_len} moves, mean {total_len / max(n_solved, 1):.2f}/pid; "
          f"{time.time() - t_all:.1f}s wall", flush=True)

    if args.out and is_main:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in results.items()}, fh)
        print(f"wrote {args.out}", flush=True)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
