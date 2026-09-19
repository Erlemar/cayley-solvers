"""Check a cube555 checkpoint against the MultiGPUBeamSearch stream1 MLP contract.

The engine is a CUDA/C++ project we cannot clone, compile or run locally (no outbound
network from this shell, no T4). So the only way to know whether a checkpoint will load
BEFORE burning a Kaggle GPU session is to restate the engine's constraints here and
check them offline. Every rule below was read off the pinned commit
f679504baddbab3765c91af526e57ec9360cf309:

  tools/export_stream1_mlp.py         export_resmlp_layernorm()
  tools/stream1_weight_io.hpp         the compiled manifest gate

RULES, and where each comes from
  1. keys                `embedding.weight`, `input_stack.{0,1,3,4}.{weight,bias}`,
                         `res_blocks.{n}.{lin1,ln1,lin2,ln2}.{weight,bias}`,
                         `head.weight`, `head.bias`
                         -- the exporter reads these literal strings.
  2. embed_dim == 16     `if embed_dim != 16: raise ValueError("runtime folded input
                         expects embed_dim=16; got {embed_dim}")`
                         The FOLD itself (`embedding @ block.t()`) is valid for any
                         embed_dim -- this is a guard, not arithmetic. But it also
                         divides: `state_len = input_stack.0.weight.shape[1] //
                         embed_dim`, so a wrong value silently mis-derives state_len.
  3. TWO input levels    `hidden2, hidden1_in = sd["input_stack.3.weight"].shape`.
                         Unconditional. This is the rule cube555's shipped ResMLPQ
                         fails: it has ONE level (`input_stack.0/1` only), because it
                         is built from `d_model`, not from a `hidden_dims` pair.
  4. hidden1 >= hidden2  C++: "hidden1 must be >= hidden2 because Stream1 reuses
                         hidden1 scratch for residual output".
  5. output_dim ==       C++: `if (model.num_classes < STATE_LEN || (model.output_dim
     move_count          != MOVE_COUNT && model.output_dim != STREAM1_SINGLE_SCORE_
                         OUTPUT_DIM))` -> throw. 30 for cube555.
  6. num_classes >=      same C++ line. 150 >= 150 holds.
     state_len
  7. contiguous res      the exporter regex-matches `res_blocks.(\\d+)` and validates
     block indices       contiguity from 0.

Usage:
  .venv/Scripts/python.exe cube555/gpu/test_export_contract.py            # all shipped ckpts
  .venv/Scripts/python.exe cube555/gpu/test_export_contract.py --checkpoint <path>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

STATE_LEN = 150
NUM_CLASSES = 150
MOVE_COUNT = 30
REQUIRED_EMBED_DIM = 16
MAX_PUBLIC_STATE_LEN = 120   # tools/cayleypy_public/data.py, State128 payload

ARTIFACTS = Path("C:/Users/and-l/cayley/cube555/tpu/kaggle_dataset")


def check(sd: dict, label: str) -> list[str]:
    """Return the list of contract violations; empty means the engine will load it."""
    bad: list[str] = []      # structural: the CUDA's shape, cannot be patched away
    soft: list[str] = []     # a Python-side guard the notebook patches
    keys = set(sd)

    def shape(k):
        return tuple(sd[k].shape) if k in sd else None

    # --- rule 1/3: the literal keys the exporter reads -------------------------
    for k in ("embedding.weight", "head.weight", "head.bias"):
        if k not in keys:
            alt = ""
            if k.startswith("head") and f"q_{k}" in keys:
                alt = f"  (present as 'q_{k}' -- rename it)"
            bad.append(f"missing key {k!r}{alt}")
    for idx in (0, 1, 3, 4):
        for suf in ("weight", "bias"):
            k = f"input_stack.{idx}.{suf}"
            if k not in keys:
                bad.append(f"missing key {k!r}")

    # --- rule 7: contiguous residual blocks ------------------------------------
    rb = sorted({int(m.group(1)) for k in keys
                 if (m := re.match(r"res_blocks\.(\d+)\.", k))})
    if rb and rb != list(range(len(rb))):
        bad.append(f"res_blocks indices not contiguous from 0: {rb}")
    for n in rb:
        for part in ("lin1", "ln1", "lin2", "ln2"):
            for suf in ("weight", "bias"):
                k = f"res_blocks.{n}.{part}.{suf}"
                if k not in keys:
                    bad.append(f"missing key {k!r}")

    # --- rule 2: embed_dim -----------------------------------------------------
    emb = shape("embedding.weight")
    embed_dim = emb[1] if emb else None
    if emb is None:
        bad.append("no embedding.weight: engine's resmlp-layernorm path needs "
                   "encoding='embedding' (one-hot would route to batchnorm-folded)")
    else:
        if emb[0] != NUM_CLASSES:
            bad.append(f"embedding rows {emb[0]} != num_classes {NUM_CLASSES}")
        if embed_dim != REQUIRED_EMBED_DIM:
            # PATCHABLE, not structural. The guard sits in Python; the fold it guards,
            # `(embedding @ block.t()).t()`, is valid for any embed_dim, and
            # `state_len = input_stack.0.weight.shape[1] // embed_dim` still derives
            # 150 correctly at 24. embed_dim is absent from the manifest and never
            # reaches the CUDA -- the runtime consumes the FOLDED
            # (state_len x num_classes x hidden1) table. The notebook patches the one
            # line, exactly as the reference notebook patches its own contract dicts.
            soft.append(f"embed_dim {embed_dim} != {REQUIRED_EMBED_DIM} "
                        f"-- exporter guard 'runtime folded input expects embed_dim=16'; "
                        f"patch that line (fold math is embed_dim-agnostic)")

    # --- rules 3/4/5/6: shapes -------------------------------------------------
    s0 = shape("input_stack.0.weight")
    s3 = shape("input_stack.3.weight")
    hd = shape("head.weight")
    if s0 and embed_dim:
        derived = s0[1] // embed_dim
        if s0[1] % embed_dim:
            bad.append(f"input_stack.0.weight in-dim {s0[1]} not divisible by "
                       f"embed_dim {embed_dim}")
        elif derived != STATE_LEN:
            bad.append(f"derived state_len {derived} != {STATE_LEN} "
                       f"(in-dim {s0[1]} / embed_dim {embed_dim})")
    if s0 and s3:
        hidden1, hidden2 = s0[0], s3[0]
        if s3[1] != hidden1:
            bad.append(f"input_stack.3 in-dim {s3[1]} != hidden1 {hidden1}")
        if hidden1 < hidden2:
            bad.append(f"hidden1 {hidden1} < hidden2 {hidden2} "
                       f"(C++: Stream1 reuses hidden1 scratch for residual output)")
        if hd and hd[1] != hidden2:
            bad.append(f"head in-dim {hd[1]} != hidden2 {hidden2}")
    if hd and hd[0] != MOVE_COUNT:
        bad.append(f"output_dim {hd[0]} != move_count {MOVE_COUNT}")
    if NUM_CLASSES < STATE_LEN:
        bad.append(f"num_classes {NUM_CLASSES} < state_len {STATE_LEN}")

    # --- rule 8: THE ONE THAT ENDS IT -----------------------------------------
    # tools/cayleypy_public/data.py:
    #     if not 1 <= state_len <= 120:
    #         raise ValueError(
    #             "public runner requires 1 <= state_len <= 120 for the State128
    #              logical payload")
    # cube555 is 150. This is the engine's packed state representation, not a
    # contract dict -- raising it is CUDA/C++ core work, which is exactly what the
    # reference notebook's method depends on NOT having to do. No model surgery
    # helps: the cap is on the puzzle, not the scorer.
    if STATE_LEN > MAX_PUBLIC_STATE_LEN:
        bad.append(f"state_len {STATE_LEN} > {MAX_PUBLIC_STATE_LEN} -- public runner "
                   f"cap ('State128 logical payload', tools/cayleypy_public/data.py). "
                   f"BLOCKS cube555 on this engine regardless of the checkpoint.")

    return bad, soft


def describe(sd: dict) -> str:
    def sh(k):
        return tuple(sd[k].shape) if k in sd else None
    emb, s0, s3 = sh("embedding.weight"), sh("input_stack.0.weight"), sh("input_stack.3.weight")
    head = sh("head.weight") or sh("q_head.weight")
    nrb = len({int(m.group(1)) for k in sd if (m := re.match(r"res_blocks\.(\d+)\.", k))})
    return (f"embedding {emb}  input_stack.0 {s0}  input_stack.3 {s3}  "
            f"res_blocks {nrb}  head {head}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, nargs="*", default=None)
    args = ap.parse_args()

    import torch

    cks = args.checkpoint or sorted(ARTIFACTS.glob("q555_*.pt"))
    if not cks:
        raise SystemExit(f"no checkpoints found in {ARTIFACTS}")

    n_ok = 0
    for ck in cks:
        blob = torch.load(str(ck), map_location="cpu", weights_only=False)
        sd = blob.get("model", blob.get("state_dict", blob))
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
              for k, v in sd.items()}
        print(f"\n=== {ck.name}")
        print(f"    {describe(sd)}")
        bad, soft = check(sd, ck.name)
        for b in bad:
            print(f"      HARD  - {b}")
        for b in soft:
            print(f"      PATCH - {b}")
        if not bad:
            n_ok += 1
            print("    CONTRACT OK"
                  + ("  (after the notebook's exporter patch)" if soft else ""))

    print(f"\n{n_ok}/{len(cks)} checkpoint(s) satisfy the engine contract")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
