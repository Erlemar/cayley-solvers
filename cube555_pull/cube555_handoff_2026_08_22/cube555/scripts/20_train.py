"""Train the ResMLP 30-way Q scorer for the 5x5x5 picture cube.

    python cube555/scripts/20_train.py --config cube555/configs/q555_a.yaml \
        --output cube555/models/q555_a

BATCH = two fixed-size row groups, so shapes never vary:

  rw      base_batch pivots x sym_rows conjugation frames. Masked MSE on the two
          labelled columns; the frame transports the label onto a different column, so
          coverage over a run reaches all 30.
  anchor  n_anchor states at d<=D with ALL 30 columns exact. Rule 9.

The value head is trained on the same trunk pass: walk index on rw rows, exact depth on
anchors. It is a diagnostic, not a scorer.

THE BUDGET RULE, which decided the 444 project: fix (weight updates x fresh states)
FIRST, then choose sym_rows and n_anchor to fit inside it. The failed 444 attempt chose
augmentation richness first and bought 1/10th the updates at the same wall clock.
This script prints both numbers at startup -- read them before letting it run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube555.models import build_model  # noqa: E402
from cube555.puzzle import Cube555  # noqa: E402
from cube555.qtrain import (  # noqa: E402
    ExactAnchors,
    PivotBuffer,
    SparseQSampler,
    Symmetries555,
    expand_labels_random,
    sparse_metrics,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument(
        "--fresh-schedule",
        action="store_true",
        help="with --resume: keep optimizer moments, rebuild the LR schedule from THIS "
        "config (the warm-restart merge member)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mc, tc, qc = cfg["model"], cfg["training"], cfg["sparse_q"]
    dev = args.device
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(cfg.get("seed", 555)))
    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    perms = torch.tensor(
        [puz.generators[n] for n in names], dtype=torch.int64, device=dev
    )
    inv_idx = torch.tensor(
        [names.index(puz.inverse_name(n)) for n in names], dtype=torch.int64, device=dev
    )
    solved = torch.tensor(puz.solved_state, dtype=torch.uint8, device=dev)

    model = build_model(mc).to(dev)
    n_par = model.num_parameters()
    print(
        f"model: ResMLPQ {mc.get('encoding','embedding')}{mc.get('embed_dim',24)} "
        f"d{mc.get('d_model',1024)}x{mc.get('num_res_blocks',10)}  "
        f"num_classes={mc['num_classes']}  output_dim={model.output_dim}  "
        f"{n_par/1e6:.1f}M params"
    )
    if model.output_dim != len(names):
        raise SystemExit(f"output_dim {model.output_dim} != {len(names)} generators")

    sym = Symmetries555(PROJECT / "data", dev)
    if bool(tc.get("verify_symmetry", True)):
        sym.verify(perms, solved.long())
    sym_rows = int(tc.get("sym_rows", 4))
    use_sym = bool(tc.get("sym_random", True)) and sym_rows > 0

    anchors = ExactAnchors(
        PROJECT / "data" / f"anchors_d{int(qc.get('anchor_max_depth', 4))}.pt", dev
    )
    n_anchor = int(qc.get("n_anchor", 256))
    anchor_w = float(qc.get("anchor_weight", 1.0))
    value_w = float(qc.get("value_weight", 1.0))

    g = torch.Generator(device=dev)
    g.manual_seed(int(cfg.get("seed", 555)))
    sg = torch.Generator(device=dev)
    sg.manual_seed(int(cfg.get("seed", 555)) + 1)
    sampler = SparseQSampler(
        perms,
        inv_idx,
        solved,
        int(tc.get("k_min", 2)),
        int(tc.get("k_max", 80)),
        float(tc.get("pivot_tilt", 0.5)),
        sg,
    )
    pivots = PivotBuffer(sampler, int(tc.get("pivot_block", 1 << 16)))

    base_batch = int(tc["base_batch"])
    steps = int(tc["steps_per_epoch"])
    n_epochs = int(tc["n_epochs"])
    updates = n_epochs * steps
    fresh = updates * base_batch
    rows = base_batch * (sym_rows if use_sym else 1) + n_anchor
    print(
        f"BUDGET  updates {updates:,}  fresh states {fresh:,}  rows/step {rows:,}\n"
        f"        (444 winner: 2,732,544 / 1,399,062,528.  paper: 8B states, 4M params)"
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc.get("lr", 3.0e-4)),
        weight_decay=float(tc.get("weight_decay", 3.0e-3)),
        fused=bool(tc.get("fused_optimizer", True)) and str(dev).startswith("cuda"),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=0.0)

    start_epoch = 0
    if args.resume is not None:
        ck = torch.load(str(args.resume), map_location=dev, weights_only=False)
        model.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        )
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck and not args.fresh_schedule:
            sched.load_state_dict(ck["scheduler"])
            start_epoch = int(ck.get("epoch", 0)) + 1
        print(
            f"resumed {args.resume} at epoch {start_epoch}"
            f"{' (LR schedule REBUILT)' if args.fresh_schedule else ''}"
        )

    grad_clip = float(tc.get("grad_clip", 1.0))
    use_amp = bool(tc.get("amp", True)) and str(dev).startswith("cuda")
    ckpt_every = int(tc.get("checkpoint_every_epochs", 400))
    log_path = args.output / "train.jsonl"

    t_start = time.time()
    for epoch in range(start_epoch, n_epochs):
        model.train()
        model.return_value = True
        acc = dict(
            loss=0.0, sparse=0.0, anchor=0.0, value=0.0, pair=0.0, top1=0.0, gap=0.0
        )
        t0 = time.time()
        for _ in range(steps):
            piv, p, col_u, col_n = pivots.take(base_batch)
            if use_sym:
                xs, tgt, msk, id_rows = expand_labels_random(
                    sym, piv, p, col_u, col_n, sym_rows, g
                )
            else:
                r = torch.arange(piv.shape[0], device=dev)
                tgt = torch.zeros((piv.shape[0], model.output_dim), device=dev)
                msk = torch.zeros_like(tgt, dtype=torch.bool)
                tgt[r, col_u] = p.float() - 1.0
                tgt[r, col_n] = p.float() + 1.0
                msk[r, col_u] = True
                msk[r, col_n] = True
                xs, id_rows = piv, r
            n_rw = xs.shape[0]
            a_s, a_q, a_d = anchors.sample(n_anchor, g)
            x = torch.cat([xs, a_s], dim=0)

            ctx = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else torch.autocast("cpu", enabled=False)
            )
            with ctx:
                out, v_all = model(x)
            out = out.float()
            v_all = v_all.float()

            pred_rw = out[:n_rw]
            sparse = pred_rw.sub(tgt).pow(2).masked_select(msk).mean()
            anchor_mse = out[n_rw:].sub(a_q).pow(2).mean()
            src = torch.arange(n_rw // (sym_rows if use_sym else 1), device=dev)
            if use_sym:
                src = src.repeat_interleave(sym_rows)
            v_tgt_rw = p.float().index_select(0, src)
            value_loss = 0.5 * (
                v_all[:n_rw].sub(v_tgt_rw).pow(2).mean()
                + v_all[n_rw:].sub(a_d).pow(2).mean()
            )
            loss = sparse + anchor_w * anchor_mse + value_w * value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

            met = sparse_metrics(pred_rw.index_select(0, id_rows), col_u, col_n)
            acc["loss"] += float(loss)
            acc["sparse"] += float(sparse)
            acc["anchor"] += float(anchor_mse)
            acc["value"] += float(value_loss)
            acc["pair"] += float(met["pair_acc"])
            acc["top1"] += float(met["top1_acc"])
            acc["gap"] += float(met["gap"])
        for k in acc:
            acc[k] /= steps

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            el = time.time() - t_start
            done = epoch - start_epoch + 1
            eta = el / done * (n_epochs - start_epoch - done) / 3600.0
            print(
                f"epoch {epoch:6d} | loss {acc['loss']:8.4f} | sparse {acc['sparse']:7.3f} "
                f"| anchor {acc['anchor']:6.4f} | val {acc['value']:7.3f} "
                f"| pair {acc['pair']:.4f} top1 {acc['top1']:.4f} gap {acc['gap']:+.3f} "
                f"| lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s/ep "
                f"| eta {eta:.1f}h",
                flush=True,
            )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        **{k: round(v, 6) for k, v in acc.items()},
                        "lr": sched.get_last_lr()[0],
                    }
                )
                + "\n"
            )

        if (epoch + 1) % ckpt_every == 0 or epoch == n_epochs - 1:
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model.get_model_config(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "epoch": epoch,
                    "config": cfg,
                    "metrics": acc,
                },
                args.output / f"epoch_{epoch:06d}.pt",
            )
    print(f"done in {(time.time()-t_start)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
