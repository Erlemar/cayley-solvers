"""Drive N rounds of Beam-AVI: generate -> train -> refresh the target net -> repeat.

Round k generates with the CURRENT weights and trains them into round k+1:

    target net for round k = blend(avi/r{k-1}, mx_resmlp_az)   (r-1 = the incumbent)
    student for round k    = avi/r{k-1}                        -> avi/r{k}

The ResMLP blend partner is NOT trained -- it is frozen, and it is in the loop only
because the deployed stack blends it, so leaving it out would generate from a different
distribution than the one this model is going to be deployed in. That distribution IS
the method (cube444 arm C20: same steps, anchors, lr, seed and init, random-walk states
instead of beam states, and it lost 46 of 50 pids against its own starting checkpoint).

Resumable: a round whose shard and checkpoint already exist is skipped, so a killed run
picks up where it stopped rather than regenerating hours of beam.

    .venv/Scripts/python.exe tetraminx/scripts/75_avi_loop.py --rounds 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "tetraminx" / "scripts"
PY = sys.executable


def run(cmd: list[str], tag: str) -> None:
    print(f"\n{'='*72}\n{tag}\n{'='*72}", flush=True)
    print(" ".join(str(c) for c in cmd), flush=True)
    t = time.time()
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"{tag} FAILED with exit {r.returncode}")
    print(f"[{tag} ok, {time.time()-t:.0f}s]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--incumbent", type=Path,
                    default=PROJECT / "tetraminx" / "models" / "mx_tf_az" / "epoch_1500.pt")
    ap.add_argument("--partner", type=Path,
                    default=PROJECT / "tetraminx" / "models" / "mx_resmlp_az" / "best.pt")
    ap.add_argument("--shard-dir", type=Path,
                    default=PROJECT / "tetraminx" / "runs" / "avi" / "shards")
    ap.add_argument("--model-dir", type=Path,
                    default=PROJECT / "tetraminx" / "models" / "avi")
    ap.add_argument("--beam", type=int, default=262144)
    ap.add_argument("--n-pids", type=int, default=20)
    ap.add_argument("--rows-per-step", type=int, default=8192)
    ap.add_argument("--full-expand", type=int, default=256)
    ap.add_argument("--qv-consistency", type=float, default=0.3)
    ap.add_argument("--history-depth", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--solo", action="store_true",
                    help="generate with the TRANSFORMER ALONE as the target net, not the "
                         "0.8/0.2 blend. The first 5-round run (2026-08-26) used the "
                         "blend as target while training the transformer alone, so the "
                         "student was fit to a target it cannot represent -- distillation "
                         "toward the blend -- and the deployed stack then re-blends the "
                         "partner, double-counting it. Beam-AVI specifies target net == "
                         "the model being trained; this flag is that fix. Result of the "
                         "blend-target run: every arm +4 to +6 moves, flattened "
                         "(deep gap01 0.438 -> 0.270). See BEAM_AVI_PLAN.md s5b")
    args = ap.parse_args()

    args.shard_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    for k in range(args.start_round, args.rounds):
        prev = args.incumbent if k == 0 else args.model_dir / f"r{k-1:03d}.pt"
        shard = args.shard_dir / f"r{k:03d}.pt"
        out = args.model_dir / f"r{k:03d}.pt"
        if out.exists():
            print(f"[round {k}] {out.name} exists -- skipping", flush=True)
            continue
        if not prev.exists():
            raise SystemExit(f"round {k}: warm-start {prev} missing")

        if not shard.exists():
            gen = [PY, SCRIPTS / "73_gen_harvest.py", "--checkpoint", prev]
            if not args.solo:
                gen += ["--blend", args.partner, "--blend-weights", "0.8", "0.2"]
            gen += ["--qv-consistency", args.qv_consistency,
                    "--history-depth", args.history_depth,
                    "--bf16", "--chunk-size", args.chunk_size,
                    "--beam", args.beam, "--round", k, "--n-pids", args.n_pids,
                    "--rows-per-step", args.rows_per_step,
                    "--full-expand", args.full_expand,
                    "--out", shard]
            who = prev.name if args.solo else f"{prev.name} + partner"
            run(gen, f"round {k}: GENERATE (target net = {who})")
        else:
            print(f"[round {k}] shard exists -- reusing {shard.name}", flush=True)

        run([PY, SCRIPTS / "74_train_avi.py",
             "--shard", shard, "--init", prev, "--out", out,
             "--steps", args.steps, "--batch", args.batch, "--lr", args.lr],
            f"round {k}: TRAIN ({prev.name} -> {out.name})")

    print(f"\nALL ROUNDS DONE in {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
