"""Build a residual-distance dataset for the §3.5 saturation probe.

The §3.5 Relative-state / Bridge Transformer wants a model D(s_from, s_to) that
estimates the distance between two arbitrary states. Under the repo convention,
the residual X = make_residual(s_i, s_j) = inv_s_j[s_i] is itself a valid state
whose true distance-to-solved equals the bridge length needed to go s_i -> s_j.
So a residual is just a state, and D is just a ResMLPDistance trained on residuals.

The core open question (CLAUDE rule 23 / 25): the deployed bridge scores residuals
with the distance-to-solved V, which SATURATES at d~=25-30 -> on deep residuals it
predicts ~25 for everything and the predicted-save signal collapses to false
positives. Does a model trained DIRECTLY on residuals with exact path-window
labels escape that ceiling, or inherit it?

This script builds the probe dataset. From verified solution paths we replay
prefix states, enumerate windows (i, j), and label residual X_ij with j - i
(the window length = an upper bound on, and for near-optimal paths ~=, the true
distance between s_i and s_j). We balance across window-length buckets so deep
windows (the saturation-relevant regime) are not drowned out, and split by pid so
the val residuals come from scrambles the model never trained on.

We also score every val residual with a production V model (default m_dd_v0) and
store V's prediction, so the trainer can print the head-to-head per-bucket
calibration table (true vs D vs V) with no checkpoint needed on the train host.

Run (local; corpus CSVs live here):
    .venv/Scripts/python.exe megaminx/scripts/85_build_bridge_distance_data.py \
        --base megaminx/submissions/merge_v14_plus_min_count_v4.csv \
               megaminx/submissions/merge_v12_az_v4_plus_our.csv \
        --v-checkpoint megaminx/models/m_dd_v0/epoch_0049.pt \
        --out megaminx/data/bridge_distance_dataset.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.bridge import make_residual
from megaminx.puzzle import Megaminx


def residuals_for_path(prefix_np: np.ndarray, wlen: int) -> np.ndarray:
    """All residuals of a fixed window length for one path (vectorized).

    prefix_np: (n+1, S) int64 prefix states. Returns (m, S) int64 where
    row r == make_residual(prefix[r], prefix[r+wlen]) = inv_prefix[r+wlen][prefix[r]].
    """
    n1, S = prefix_np.shape
    if wlen >= n1:
        return np.empty((0, S), dtype=np.int64)
    rows = np.arange(n1)[:, None]
    inv_prefix = np.empty((n1, S), dtype=np.int64)
    inv_prefix[rows, prefix_np] = np.arange(S)[None, :]   # scatter -> inverse perms
    src = prefix_np[: n1 - wlen]            # (m, S)  = prefix[i]
    dstinv = inv_prefix[wlen:]              # (m, S)  = inv_prefix[i+wlen]
    return np.take_along_axis(dstinv, src, axis=1)


@torch.no_grad()
def score_v(model, residuals_u8: np.ndarray, device: str, batch: int = 8192) -> np.ndarray:
    """Production-V prediction on each residual (the saturating baseline)."""
    n = residuals_u8.shape[0]
    x = torch.from_numpy(residuals_u8.astype(np.int64)).to(device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(0, n, batch):
        pred = model(x[i : i + batch])
        out[i : i + batch] = (pred.squeeze(-1) if pred.dim() > 1 else pred).float()
    return out.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, nargs="+", required=True,
                    help="One or more verified solution CSVs (the path corpus)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--v-checkpoint", type=Path, default=None,
                    help="Production V model; scores val residuals for the D-vs-V table")
    ap.add_argument("--test-csv", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--window-min", type=int, default=5)
    ap.add_argument("--window-max", type=int, default=80)
    ap.add_argument("--train-per-wlen", type=int, default=12000,
                    help="Target residuals per integer window length (train)")
    ap.add_argument("--val-per-wlen", type=int, default=2000)
    ap.add_argument("--val-pid-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=85)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    puzzle = Megaminx.load(args.puzzle_info)
    S = len(puzzle.solved_state)
    identity = np.arange(S, dtype=np.int64)
    test_states = load_test_states(args.test_csv)
    wlens = list(range(args.window_min, args.window_max + 1))

    # pid -> train/val split (a pid's windows go entirely to one side; no leakage).
    all_pids = sorted(test_states.keys())
    perm = rng.permutation(len(all_pids))
    n_val = int(round(args.val_pid_frac * len(all_pids)))
    val_pid_set = {all_pids[i] for i in perm[:n_val]}
    print(f"pids: {len(all_pids)} total, {n_val} val / {len(all_pids) - n_val} train", flush=True)

    # Collect residuals into per-(split, wlen) lists, then subsample for balance.
    train_b: dict[int, list[np.ndarray]] = {w: [] for w in wlens}
    val_b: dict[int, list[np.ndarray]] = {w: [] for w in wlens}
    n_paths = n_skip = 0
    t0 = time.time()
    for base in args.base:
        paths = load_submission(base)
        print(f"corpus {base.name}: {len(paths)} pids", flush=True)
        for pid, path in paths.items():
            if pid not in test_states:
                continue
            init = test_states[pid]
            if not verify_path(puzzle, init, path).ok:
                n_skip += 1
                continue
            # replay -> prefix states (n+1, S)
            cur = np.asarray(init, dtype=np.int64)
            pref = [cur]
            for m in path:
                cur = cur[np.asarray(puzzle.generators[m], dtype=np.int64)]
                pref.append(cur)
            prefix_np = np.stack(pref, axis=0)
            n_paths += 1
            if n_paths == 1:   # one-time end-to-end correctness check
                for _ in range(6):
                    w = int(rng.integers(args.window_min, min(args.window_max, len(path) - 1) + 1))
                    i = int(rng.integers(0, len(path) - w + 1))
                    r_vec = residuals_for_path(prefix_np, w)[i]
                    r_ref = make_residual(prefix_np[i], prefix_np[i + w])
                    assert np.array_equal(r_vec, r_ref), "vectorized residual != make_residual"
                    assert tuple(puzzle.apply_path(tuple(int(x) for x in r_vec), path[i : i + w])) \
                        == puzzle.solved_state, "residual not solved by its own window path"
                print("  self-check OK: residuals match make_residual + bridge guarantee", flush=True)
            dst = val_b if pid in val_pid_set else train_b
            for w in wlens:
                res = residuals_for_path(prefix_np, w)
                if res.shape[0] == 0:
                    continue
                keep = ~(res == identity).all(axis=1)     # drop closed-loop (identity) residuals
                res = res[keep]
                if res.shape[0]:
                    dst[w].append(res.astype(np.uint8))
    print(f"replayed {n_paths} paths ({n_skip} unverifiable) in {time.time() - t0:.1f}s", flush=True)

    def assemble(buckets: dict[int, list[np.ndarray]], target: int):
        res_parts, wlen_parts = [], []
        for w in wlens:
            if not buckets[w]:
                continue
            arr = np.concatenate(buckets[w], axis=0)
            if arr.shape[0] > target:
                sel = rng.choice(arr.shape[0], size=target, replace=False)
                arr = arr[sel]
            res_parts.append(arr)
            wlen_parts.append(np.full(arr.shape[0], w, dtype=np.int16))
        res = np.concatenate(res_parts, axis=0)
        wlen = np.concatenate(wlen_parts, axis=0)
        order = rng.permutation(res.shape[0])
        return res[order], wlen[order]

    train_res, train_wlen = assemble(train_b, args.train_per_wlen)
    val_res, val_wlen = assemble(val_b, args.val_per_wlen)
    print(f"dataset: train {train_res.shape[0]:,} | val {val_res.shape[0]:,} "
          f"(wlen {args.window_min}..{args.window_max})", flush=True)
    # per-bucket coverage at the saturation-relevant lengths
    for w in (10, 20, 30, 40, 50, 60, 70):
        tr = int((train_wlen == w).sum())
        va = int((val_wlen == w).sum())
        print(f"  wlen {w:3d}: train {tr:6d}  val {va:5d}", flush=True)

    out = {
        "train_residuals": torch.from_numpy(train_res),          # uint8 (N,120)
        "train_wlen": torch.from_numpy(train_wlen),              # int16 (N,)
        "val_residuals": torch.from_numpy(val_res),
        "val_wlen": torch.from_numpy(val_wlen),
        "meta": {
            "state_size": S, "window_min": args.window_min, "window_max": args.window_max,
            "train_per_wlen": args.train_per_wlen, "val_per_wlen": args.val_per_wlen,
            "corpus": [str(b) for b in args.base], "seed": args.seed,
            "n_train_pids": len(all_pids) - n_val, "n_val_pids": n_val,
        },
    }

    if args.v_checkpoint is not None:
        print(f"scoring residuals with production V {args.v_checkpoint.name} "
              f"(the saturating baseline)", flush=True)
        v = load_model_checkpoint(args.v_checkpoint, device=args.device, dtype=torch.float32)
        out["val_v_pred"] = torch.from_numpy(score_v(v, val_res, args.device))
        out["train_v_pred"] = torch.from_numpy(score_v(v, train_res, args.device))
        out["meta"]["v_checkpoint"] = str(args.v_checkpoint)
        # quick preview of V saturation on val residuals
        print("  V(residual) mean by true window length (expect flatline past ~30):", flush=True)
        vp = out["val_v_pred"].numpy()
        for w in (10, 20, 30, 40, 50, 60, 70):
            mask = val_wlen == w
            if mask.any():
                print(f"    wlen {w:3d}: V_mean {vp[mask].mean():5.1f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    sz = args.out.stat().st_size / 1e6
    print(f"saved {args.out} ({sz:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
