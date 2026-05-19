"""Minimal reproducer: multi-call xm.all_gather alignment on torch_xla / TPU v3-8.

CLAIM UNDER TEST
================
When a single XLA program performs N separate `xm.all_gather` calls (one per
tensor), the gathered tensors should be consistently ordered across calls:
row k of every gathered output should correspond to the same source
(rank, local_idx) pair.

Each rank constructs two local tensors of different rank/dtype/shape that BOTH
encode the same source identity (rank * ID_MULT + local_idx) at row i. After
two `xm.all_gather` calls in the same XLA program, we decode the source
identity from each gathered output and compare elementwise.

EXPECTED (correct behavior):
    All trials report mismatches=0.

OBSERVED IN PARENT PROJECT (megaminx beam-search SPMD, 2026-05-09):
    5 versions of cross-rank beam search (v2-v5), each with a different
    cross-rank architecture, all hit a false-positive walkback bug. v2-v3
    used multi-call xm.all_gather of 5 separate tensors (state-2D int8,
    parent-1D int32, move-1D int8, hash-1D int64, score-1D float32). At the
    walkback step, applying the recovered (move, parent) tuple for
    chosen_states[k] (where chosen_states[k] equals the goal state V0 by
    elementwise tensor equality) did NOT reproduce V0 - meaning the gathered
    tensors were misaligned across calls. The fingerprint:
    `found_step = pid_index_within_batch - 1` for sequential pids 1..4.

ENVIRONMENT WHERE PROJECT BUG WAS OBSERVED
==========================================
    Kaggle TPU v3-8 (PJRT runtime)
    torch == 2.8.x
    torch_xla == 2.8.x
    python 3.10
    PJRT_DEVICE=TPU (set by Kaggle kernel)

RELATED UPSTREAM ISSUES
=======================
    pytorch/xla #3824 - Mismatched rank in collective ops in PJRT runtime
                        (pin_layout=False scrambles rank order to [0,1,4,5,6,7,2,3])
    pytorch/xla #3510 - all_gather memory regression vs all_reduce-based path
                        (workaround: revert to padded all_reduce. We tried this
                         in v4; it still produced misaligned outputs.)

USAGE
=====
On a Kaggle kernel with Accelerator = TPU 1x v3-8:
    !pip install -q torch==2.8.0 torch_xla==2.8.0  # if not pre-installed
    !python repro.py

On a TPU VM:
    PJRT_DEVICE=TPU python repro.py
"""
from __future__ import annotations

import os

os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


NUM_TRIALS = 10
LOCAL_K = 4096            # rows per rank per trial
STATE_DIM = 120           # mirrors megaminx packed state size (60 stickers x 2)
ID_MULT = 1_000_000       # encoding multiplier; (rank, idx) -> rank*ID_MULT + idx


def _mp_fn(rank: int) -> None:
    device = xm.xla_device()
    world_size = xm.xrt_world_size()

    if rank == 0:
        print("=" * 72)
        print("Multi-call xm.all_gather alignment reproducer")
        print("=" * 72)
        print(f"torch        = {torch.__version__}")
        print(f"torch_xla    = {getattr(torch_xla, '__version__', 'unknown')}")
        print(f"PJRT_DEVICE  = {os.environ.get('PJRT_DEVICE', '<unset>')}")
        print(f"world_size   = {world_size}")
        print(f"device       = {device}")
        print(f"NUM_TRIALS   = {NUM_TRIALS}")
        print(f"LOCAL_K      = {LOCAL_K}  (rows per rank per trial)")
        print(f"STATE_DIM    = {STATE_DIM}")
        print()
        print("Each rank builds T1=(K, STATE_DIM) int32 and T2=(K,) int32, both")
        print("filled so that encoded[i] = rank*ID_MULT + i. After TWO separate")
        print("xm.all_gather calls in the same XLA program, T1_global[k, 0] and")
        print("T2_global[k] should be equal for every k. Mismatches mean the two")
        print("gathers placed source-identity rows in different positions.")
        print()

    total_mismatches = 0

    for trial in range(NUM_TRIALS):
        local_idx = torch.arange(LOCAL_K, device=device, dtype=torch.int32)
        rank_t = torch.full((LOCAL_K,), rank, device=device, dtype=torch.int32)
        encoded = rank_t * ID_MULT + local_idx                               # (K,) int32

        # T1: 2D state-shaped tensor; every column equals encoded.
        t1_local = encoded.unsqueeze(1).expand(LOCAL_K, STATE_DIM).contiguous()
        # T2: 1D scalar tensor; same encoding, different shape/rank.
        t2_local = encoded.clone()

        # TWO SEPARATE all_gather calls in the same XLA program.
        t1_global = xm.all_gather(t1_local, dim=0)   # (W*K, STATE_DIM) int32
        t2_global = xm.all_gather(t2_local, dim=0)   # (W*K,) int32

        xm.mark_step()

        if rank != 0:
            continue

        id_from_t1 = t1_global[:, 0].cpu()
        id_from_t2 = t2_global.cpu()

        n_total = id_from_t1.shape[0]
        diff_mask = id_from_t1 != id_from_t2
        n_mismatch = int(diff_mask.sum().item())
        total_mismatches += n_mismatch

        line = f"trial {trial:2d}: total={n_total} mismatches={n_mismatch}"
        if n_mismatch:
            bad = diff_mask.nonzero(as_tuple=True)[0][:5]
            samples = "; ".join(
                f"row {int(k)}: t1={int(id_from_t1[k])} t2={int(id_from_t2[k])}"
                for k in bad
            )
            line += f"\n           samples: {samples}"
        print(line)

    if rank == 0:
        print()
        print("-" * 72)
        if total_mismatches == 0:
            print("PASS: multi-call xm.all_gather is consistently ordered across calls.")
            print()
            print("Note: if you came here for the megaminx SPMD beam-search bug, the")
            print("misalignment must be in a different code path. The next thing to")
            print("instrument is per-call buffer reuse inside the per-pid worker")
            print("function (allocations of `tree_*`, `min_v_log`, `found_step` via")
            print("torch.full(...) at function entry).")
        else:
            print(f"FAIL: {total_mismatches} total mismatches across {NUM_TRIALS} trials.")
            print()
            print("Multi-call xm.all_gather does NOT preserve cross-rank ordering in")
            print("this configuration. Workaround: pack all per-row payloads into a")
            print("single buffer, do ONE all_gather, then unpack on receive.")


if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())
