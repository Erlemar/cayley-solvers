"""Export a Megaminx ResMLPDistance checkpoint to a TensorRT engine.

Usage (on GCP L4):
    python3 megaminx/beam_lab/export_tensorrt.py \
        --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --batch-size 16384 --precision fp16 \
        --output megaminx/models/m05_trt_fp16_b16384_sm89.ts

Hard requirement: torch_tensorrt's torch ABI must match the installed torch.
Confirmed working combo (2026-04-28): torch 2.11.0+cu129 + torch_tensorrt 2.11.0+cu129.

The exported engine is a TorchScript module that exposes the same forward(int64
input of shape (batch_size, 120)) -> float scalar interface as the original.
The engine is sm-arch specific (built on L4 = sm_89; rebuild for any other GPU).

Validation: after export, run a 5-iter forward and print output mean to verify
no NaN/Inf, then save with torch.save / torch.jit.save.
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

from cayley.model import ResMLPDistance


def _load_state_dict(ckpt_path: str, device: str = "cuda"):
    """Load checkpoint, strip _orig_mod prefix, return (state_dict, model_config)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    cfg = ckpt.get("model_config", {})
    return sd, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path,
                    help="path to save the exported engine (.ts)")
    ap.add_argument("--batch-size", type=int, default=16384,
                    help="MUST match internal_batch_size at solve time. "
                         "Engine is shape-fixed.")
    ap.add_argument("--state-size", type=int, default=120)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="primary target precision for TRT compile. fp16 default; "
                         "fall back to bf16 if fp16 quality fails the validation gate.")
    ap.add_argument("--workspace-mb", type=int, default=4096,
                    help="TRT build workspace in MB (default 4096 = 4 GB)")
    args = ap.parse_args()

    device = "cuda"
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    # Resolve precision
    if args.precision == "fp16":
        prec = torch.float16
    elif args.precision == "bf16":
        prec = torch.bfloat16
    else:
        prec = torch.float32

    print(f"loading checkpoint: {args.checkpoint}")
    sd, model_cfg = _load_state_dict(str(args.checkpoint), device=device)
    if not model_cfg:
        # Fallback to default megaminx arch if config missing
        model_cfg = {
            "state_size": args.state_size, "num_classes": args.state_size,
            "hidden_dims": [2048, 512], "num_res_blocks": 2,
            "encoding": "embedding", "embed_dim": 16,
        }
    print(f"model config: {model_cfg}")

    model = ResMLPDistance(
        state_size=model_cfg["state_size"],
        num_classes=model_cfg["num_classes"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "embedding"),
        embed_dim=model_cfg.get("embed_dim", 16),
    )
    model.load_state_dict(sd)
    model = model.to(device).to(prec).eval()
    if hasattr(model, "inference_chunk_size"):
        model.inference_chunk_size = None  # disable internal chunking

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {n_params:,} params, dtype={prec}")

    # Sanity: eager forward on a dummy batch
    dummy = torch.zeros((args.batch_size, args.state_size), dtype=torch.int64, device=device)
    with torch.inference_mode():
        eager_out = model(dummy)
    print(f"eager forward: out shape={tuple(eager_out.shape)}, dtype={eager_out.dtype}, "
          f"mean={float(eager_out.mean()):.4f}")

    # TensorRT compile.
    # In torch_tensorrt 2.11, when the model is already cast to a non-fp32 precision
    # (e.g. .to(fp16)), TRT enables explicit typing automatically and `enabled_precisions`
    # must NOT be passed (raises an AssertionError). We rely on the model's own dtype.
    import torch_tensorrt
    print(f"compiling to TRT engine: precision={args.precision}, batch_size={args.batch_size}, "
          f"workspace={args.workspace_mb} MB...")
    t0 = time.time()
    compile_kwargs = dict(
        inputs=[
            torch_tensorrt.Input(
                min_shape=(args.batch_size, args.state_size),
                opt_shape=(args.batch_size, args.state_size),
                max_shape=(args.batch_size, args.state_size),
                dtype=torch.int64,
            ),
        ],
        workspace_size=args.workspace_mb * 1024 * 1024,
        truncate_long_and_double=True,  # int64 -> int32 inside TRT
        require_full_compilation=False,  # let unsupported subgraphs fall back
    )
    if args.precision == "fp32":
        # Only explicitly request precisions for fp32 (where the model dtype is fp32).
        compile_kwargs["enabled_precisions"] = {torch.float32}
    trt_module = torch_tensorrt.compile(model, **compile_kwargs)
    print(f"compile + warmup: {time.time() - t0:.1f}s")

    # Smoke-test the engine
    print("smoke test (5 forward passes)...")
    with torch.inference_mode():
        for i in range(5):
            t1 = time.time()
            trt_out = trt_module(dummy)
            torch.cuda.synchronize()
            print(f"  iter {i}: {(time.time() - t1) * 1000:.2f}ms, "
                  f"out shape={tuple(trt_out.shape)}, mean={float(trt_out.mean()):.4f}")

    # Compare to eager
    eager_mean = float(eager_out.float().mean())
    trt_mean = float(trt_out.float().mean())
    delta = abs(eager_mean - trt_mean)
    print(f"output mean: eager={eager_mean:.4f}, trt={trt_mean:.4f}, |delta|={delta:.4f}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trt_module, str(args.output))
    print(f"saved {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
