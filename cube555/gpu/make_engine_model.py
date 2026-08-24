"""Rewrite cube555's `ResMLPQ` into the engine's two-level shape EXACTLY, no training.

THE PROBLEM. MultiGPUBeamSearch's stream1 MLP path hardcodes a TWO-level input stack
(`hidden1`/`hidden2`, with the C++ requiring hidden1 >= hidden2). cube555's ResMLPQ has
ONE level. See cube555/gpu/test_export_contract.py for the full contract.

THE FIRST THING I TRIED, AND WHY IT FAILED. Distil a new two-level front end onto the
teacher's post-input-stack activation. Activation MSE fell 12x (0.405 -> 0.034) while
argmin agreement against the teacher stayed at chance (1/30). Switching the objective to
Q-space with a centered ranking term did not help either -- the centered-Q error stayed
about as large as the entire centered-Q signal. The reason is scale: the teacher's
within-parent action spread is ~0.6 against a Q level of ~44, i.e. ~1.4% relative, so a
front end has to reproduce the activation to well under a percent before the RANKING --
the only thing a beam consumes -- survives. Gradient descent was nowhere near that.

THE CONSTRUCTION THAT WORKS. The rewrite is exact algebra, so no fitting is involved.
Write the teacher's front end as v = W0 e + b0, h = ReLU(LN(v)). Build the student with
hidden1 = 2 * hidden2 and set

    W0' = [ W0 ; -W0 ]      b0' = [ b0 ; -b0 ]      so   u = [v ; -v]
    LN1: gamma = 1, beta = 0
    W3  = [ I , -I ]        b3  = 0
    LN2: gamma, beta        copied from the teacher's LayerNorm

then, step by step:

    mean(u) = 0 exactly, because u is antisymmetric by construction, so
    LN1(u)            = [v ; -v] / s          with s = sqrt(mean(v^2) + eps)
    ReLU(LN1(u))      = [v+ ; v-] / s         (v+ = max(v,0), v- = max(-v,0))
    W3 @ that         = (v+ - v-) / s = v / s
    LN2(v / s)        = LN2(v)                LayerNorm is invariant to a positive
                                              rescale of its input
    ReLU(LN2(v))      = h                     the teacher's activation

The intermediate ReLU is exactly what the +/- pair is for: splitting v into its positive
and negative parts loses nothing, and W3 = [I, -I] puts them back together. Everything
downstream -- 10 residual blocks and the head -- is copied verbatim, so the student is
the teacher.

THE ONE INEXACTNESS, and it is measured below rather than asserted. LayerNorm's epsilon
is not scale-invariant: the teacher computes (v-mu)/sqrt(sigma^2 + eps) while the student
computes (v-mu)/sqrt(sigma^2 + eps*s^2). The relative difference is of order
eps*(s^2-1)/(2*sigma^2) with eps=1e-5, so it is ~1e-5 and shows up in the reported
max|dQ|. If s were ever tiny it would matter; it is not, and the check prints it.

EMBED DIM. The student keeps the teacher's embedding VERBATIM at embed_dim=24, because
the per-position table W0[:, 24i:24(i+1)] @ E^T has rank up to 24 and an embed_dim=16
student could only reach rank 16 -- the low-rank floor is why no amount of training fixes
this either. The exporter has a guard `if embed_dim != 16: raise`, but the fold it
guards, `(embedding @ block.t()).t()`, is valid for any embed_dim, and `state_len =
input_stack.0.weight.shape[1] // embed_dim` = 3600 // 24 = 150 comes out right. embed_dim
does not appear in the manifest and never reaches the CUDA -- the runtime consumes the
FOLDED (state_len x num_classes x hidden1) table. So the notebook patches that one line,
in the same spirit as the reference notebook's own contract patches.

    .venv/Scripts/python.exe cube555/gpu/make_engine_model.py
    .venv/Scripts/python.exe cube555/gpu/make_engine_model.py --checkpoint <in> --out <out>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
HANDOFF = PROJECT / "cube555_pull" / "cube555_handoff_2026_08_22" / "cube555"
ARTIFACTS = PROJECT / "cube555" / "tpu" / "kaggle_dataset"

STATE_SIZE, NUM_CLASSES, N_GEN = 150, 150, 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "q555_2k_BEST.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--w3-scale", type=float, default=32.0,
                    help="gain k on W3 = k*[I,-I]. LN2 normalises k away, so any k>0 is "
                         "algebraically identical -- but it rescales what LN2's epsilon "
                         "is compared against: the student's effective eps becomes "
                         "eps*s^2/k^2 where s=sqrt(mean(v^2))~6-9 here. k=1 leaves "
                         "eps*s^2 ~ 8e-4 against the teacher's 1e-5 and costs ~3e-3 of "
                         "max|dQ|; k>=32 drives both sides into the eps-negligible "
                         "regime. Swept, not guessed -- see the RUNBOOK.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = args.out or (HERE / "engine_models" /
                       args.checkpoint.name.replace(".pt", "_engine.pt"))

    import torch

    sys.path.insert(0, str(PROJECT / "src"))
    sys.path.insert(0, str(HANDOFF / "src"))
    from cayley.model import ResMLPDistance
    from cube555.models import load_model

    dev = args.device
    teacher = load_model(args.checkpoint, device=dev, dtype=torch.float32)
    teacher.return_value = False
    teacher.inference_chunk_size = None
    d = teacher.d_model
    n_rb = teacher.num_res_blocks
    emb_dim = teacher.embed_dim
    print(f"teacher {args.checkpoint.name}: d_model={d} res_blocks={n_rb} "
          f"embed_dim={emb_dim} params={teacher.num_parameters():,}")

    student = ResMLPDistance(
        state_size=STATE_SIZE, num_classes=NUM_CLASSES,
        hidden_dims=(2 * d, d), num_res_blocks=n_rb,
        encoding="embedding", embed_dim=emb_dim, output_dim=N_GEN,
        inference_chunk_size=None,
    ).to(dev)
    print(f"student ResMLPDistance hidden_dims=({2 * d}, {d}) embed_dim={emb_dim}: "
          f"{sum(p.numel() for p in student.parameters()):,} params")

    t = teacher.state_dict()
    s = student.state_dict()
    W0, b0 = t["input_stack.0.weight"], t["input_stack.0.bias"]
    ln_g, ln_b = t["input_stack.1.weight"], t["input_stack.1.bias"]

    with torch.no_grad():
        s["embedding.weight"] = t["embedding.weight"].clone()
        # level 1: duplicate with a sign flip  ->  u = [v ; -v]
        s["input_stack.0.weight"] = torch.cat([W0, -W0], dim=0)
        s["input_stack.0.bias"] = torch.cat([b0, -b0], dim=0)
        # LN1 must be a pure normalisation: identity gain, no shift
        s["input_stack.1.weight"] = torch.ones(2 * d)
        s["input_stack.1.bias"] = torch.zeros(2 * d)
        # level 2: W3 = [I, -I] recombines the positive and negative parts
        k = float(args.w3_scale)
        s["input_stack.3.weight"] = torch.cat(
            [torch.eye(d), -torch.eye(d)], dim=1) * k
        s["input_stack.3.bias"] = torch.zeros(d)
        # LN2 carries the teacher's affine, and is scale-invariant to the 1/s left over
        s["input_stack.4.weight"] = ln_g.clone()
        s["input_stack.4.bias"] = ln_b.clone()
        # trunk + head verbatim
        for key in s:
            if key.startswith("res_blocks."):
                s[key] = t[key].clone()
        s["head.weight"] = t["q_head.weight"].clone()
        s["head.bias"] = t["q_head.bias"].clone()
    student.load_state_dict(s)
    student.eval()

    # ---- verify on REAL test states + shallow walks --------------------------
    info = json.loads((ARTIFACTS / "puzzle_info.json").read_text(encoding="utf-8"))
    G = np.stack([np.asarray(v, dtype=np.int64) for v in info["generators"].values()])
    solved = np.asarray(info["central_state"], dtype=np.int64)
    states = []
    with open(HANDOFF / "data" / "test.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= args.n_eval // 2:
                break
            states.append([int(x) for x in r["initial_state"].split(",")])
    rng = np.random.default_rng(555)
    while len(states) < args.n_eval:
        st = solved.copy()
        for _ in range(int(rng.integers(1, 40))):
            st = st[G[rng.integers(0, len(G))]]
        states.append(st.tolist())
    x = torch.tensor(states, dtype=torch.long, device=dev)

    with torch.no_grad():
        qt = teacher(x).double()
        qs = student(x).double()
        ht = teacher.input_stack(teacher.encode(x)).double()
        hs = student.input_stack(student.encode(x)).double()
        v = (teacher.encode(x) @ W0.T + b0).double()
        s_scale = torch.sqrt((v ** 2).mean(1) + 1e-5)

    dq = (qs - qt).abs()
    dh = (hs - ht).abs()
    argmin_same = int((qs.argmin(1) == qt.argmin(1)).sum())
    order_same = int((qs.argsort(1) == qt.argsort(1)).all(1).sum())
    n = x.shape[0]
    print(f"\nverification on {n} states ({args.n_eval // 2} real test.csv + walks):")
    print(f"  LN1 scale s: min {float(s_scale.min()):.3f} "
          f"max {float(s_scale.max()):.3f}  (the eps-inexactness scales with s^2-1)")
    print(f"  hidden  max|d| {float(dh.max()):.3e}  mean {float(dh.mean()):.3e}")
    print(f"  Q       max|d| {float(dq.max()):.3e}  mean {float(dq.mean()):.3e}")
    print(f"  argmin  agreement {argmin_same}/{n}")
    print(f"  full 30-action ordering identical {order_same}/{n}")

    # Gate on the RANKING, which is all the beam consumes, plus a Q tolerance far
    # below the fp16/bf16 the engine itself runs in on a T4. Demanding bit-identical
    # full ordering would fail on near-ties that the teacher's own fp32-vs-bf16 gap
    # reorders anyway (the JAX port measured 255/256 at 2.7e-4).
    # Gate on the RANKING -- all the beam consumes -- plus a Q tolerance an order of
    # magnitude below the fp16/bf16 the engine runs in on a T4. The floor here is fp32
    # round-off from the extra 2*d-wide matmul, not error: max|dQ| is bit-identical at
    # --w3-scale 32 and 128 (1.755e-4), i.e. the epsilon term is already gone.
    ok = (argmin_same == n and order_same == n and float(dq.max()) < 1e-3)
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "state_dict": {k: v.cpu() for k, v in student.state_dict().items()},
            "model_config": {
                "model_class": "ResMLPDistance", "state_size": STATE_SIZE,
                "num_classes": NUM_CLASSES, "hidden_dims": [2 * d, d],
                "num_res_blocks": n_rb, "encoding": "embedding",
                "embed_dim": emb_dim, "output_dim": N_GEN,
            },
            "provenance": (f"exact algebraic rewrite of {args.checkpoint.name} into the "
                           f"engine's two-level input stack; +/- duplication with "
                           f"W3=[I,-I]. No training."),
        }
        torch.save(blob, str(out))
        print(f"\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print("VERDICT :", "PASS" if ok else "FAIL -- rewrite is not equivalent")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
