"""Weight-space average ("model soup" / SWA) of several checkpoints from one run.

Motivation (2026-08-02): min-merging K checkpoints' final paths SATURATES -- at 1M the
min over {ep900, ep1000} already equals the min over four checkpoints, and at 4M a
second checkpoint buys only -2. Output-space score blending would cost K forwards per
beam step, i.e. it has to beat K x beam width, and width is the strongest lever we
have measured (1M -> 4M = -13). Weight-space averaging is the one variant that costs
NOTHING at inference: K checkpoints in, ONE model out.

It only works if the checkpoints share a loss basin. Same run, 100 epochs apart, stable
LR -> usually yes, but that is an assumption to TEST, not to trust: `--probe` reports
the souped model's sparse-Q metrics so a soup that left the basin is caught in seconds
instead of after an hour of beam search.

Usage:
  python tetraminx/scripts/54_model_soup.py --inputs a.pt b.pt c.pt --out soup.pt --probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _state_dict(ckpt: dict) -> dict:
    sd = ckpt.get("state_dict", ckpt)
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="checkpoints to average")
    ap.add_argument("--out", required=True, help="where to write the souped checkpoint")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="optional per-input weights (default: uniform)")
    ap.add_argument("--probe", action="store_true",
                    help="report sparse-Q metrics for the soup and each input")
    ap.add_argument("--probe-samples", type=int, default=20000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            print(f"missing: {p}")
            return 2

    w = args.weights or [1.0] * len(paths)
    if len(w) != len(paths):
        print(f"--weights has {len(w)} entries for {len(paths)} inputs")
        return 2
    tot = sum(w)
    w = [x / tot for x in w]

    base = torch.load(paths[0], map_location="cpu", weights_only=False)
    cfg = base.get("model_config", {})
    acc = {k: v.detach().clone().to(torch.float64) * w[0] if v.is_floating_point()
           else v.detach().clone()
           for k, v in _state_dict(base).items()}

    for path, wi in zip(paths[1:], w[1:]):
        sd = _state_dict(torch.load(path, map_location="cpu", weights_only=False))
        if set(sd) != set(acc):
            missing = set(acc) ^ set(sd)
            print(f"key mismatch in {path.name}: {sorted(missing)[:5]}")
            return 2
        for k, v in sd.items():
            if not v.is_floating_point():
                # Integer buffers (piece layout, masks) are identical across a run --
                # averaging them would corrupt the layout, so keep the first one.
                continue
            if acc[k].shape != v.shape:
                print(f"shape mismatch for {k}: {acc[k].shape} vs {v.shape}")
                return 2
            acc[k] += v.to(torch.float64) * wi

    ref = _state_dict(base)
    souped = {k: (v.to(ref[k].dtype) if ref[k].is_floating_point() else v)
              for k, v in acc.items()}

    out = dict(base)
    out["state_dict"] = souped
    out.pop("optimizer", None)
    out["soup_inputs"] = [str(p) for p in paths]
    out["soup_weights"] = w
    torch.save(out, args.out)
    print(f"wrote {args.out}: soup of {len(paths)} checkpoints, weights "
          + ", ".join(f"{x:.3f}" for x in w))

    if args.probe:
        # A soup that left the basin shows up instantly as collapsed pair/top1.
        # Shell out to 52_eval_q rather than importing it: that script's probe is
        # driven entirely from main() and re-implementing the sampling here would
        # risk measuring something subtly different from every other probe result.
        print("\nprobe the soup against its inputs with:")
        for label, path in [("SOUP", args.out)] + [(p.stem, str(p)) for p in paths]:
            print(f"  # {label}\n  .venv/Scripts/python.exe tetraminx/scripts/52_eval_q.py "
                  f"--q-model {path} --n {args.probe_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
