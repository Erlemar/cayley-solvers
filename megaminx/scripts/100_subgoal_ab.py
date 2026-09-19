"""Direction-3 falsifier: hierarchical subgoal search (macro-jump beam) v0.

Tests the ONE mechanism the literature credits for value-noise robustness
(kSubS/AdaSubS; Zawalski et al. 2406.03361): query the value V to select the
beam every k-th step instead of every step, pruning by the POLICY in between.

Arms (all share the SAME width B, the SAME AZ v4 policy proposer + AZ v4 V):
  sg_k1   baseline -- V every step (policy-shortlist -> V-select). The honest
          baseline: identical to the subgoal arms EXCEPT the V-gating period.
  sg_k{2,3,..}  subgoal -- V every k steps, cumulative-policy pruning between.
  qshort  (optional, --student) production reference: m23 Q-shortlist + V.

Because our real solver is a WIDE beam (not best-first), the binding question
is whether reducing V-decision frequency helps PATH LENGTH (our objective) --
the literature only ever reports solve-rate. So the verdict is on POST-PROCESSED
length at matched width (sg_k>1 also uses FEWER V-evals -> if it's shorter, it
wins at less compute). --matched-budget instead widens sg_k to k*B (equal V-evals).

FALSIFIED if no k>1 beats sg_k1 on total post-processed length: then our wide
beam already absorbs the V-noise (the clean-Rubik regime of 2406.03361) and the
structural-search chapter closes.

Smoke (correctness -- every path must verify):
  .venv/Scripts/python.exe megaminx/scripts/100_subgoal_ab.py \
      --pids 300,500 --beam 4096 --ks 1,2 --bf16 --no-qshort

Real falsifier:
  .venv/Scripts/python.exe megaminx/scripts/100_subgoal_ab.py \
      --pids 300,500,700,850,950 --beam 16384 --ks 1,2,3 --bf16 \
      --out-json megaminx/results/subgoal_ab.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "beam_lab"))

from beam_search import KhoruzhiiSearchConfig, setup_model_for_inference  # noqa: E402
from beam_search_qshort import QShortlisterSolver  # noqa: E402
from beam_search_subgoal import SubgoalSolver  # noqa: E402
from cayley.search import load_model_checkpoint  # noqa: E402
from cayley.verify import load_submission, load_test_states, verify_path  # noqa: E402
from megaminx.post_process import full_post_process  # noqa: E402
from megaminx.puzzle import Megaminx  # noqa: E402


def _pp_len(puzzle, state, names):
    """Post-process (cheap move-string normalization) + verify. Returns (raw, pp)
    lengths or (raw, None) if the post-processed path fails to verify."""
    raw = len(names)
    pp = full_post_process(names, puzzle=puzzle)
    return raw, (len(pp) if verify_path(puzzle, state, pp).ok else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "m_az_v4_v_only.pt",
                    help="teacher V")
    ap.add_argument("--policy", type=Path, default=PROJECT / "models" / "m_az_v4_pi_only.pt",
                    help="policy proposer (AZ v4 pi, aligned with the V)")
    ap.add_argument("--student", type=Path, default=PROJECT / "models" / "m23_q_shortlister" / "epoch_0499.pt",
                    help="m23 Q-shortlister for the production qshort reference arm")
    ap.add_argument("--no-qshort", action="store_true", help="skip the qshort reference arm")
    ap.add_argument("--best-csv", type=Path,
                    default=PROJECT / "submissions" / "merge_v14_plus_min_count_v4.csv")
    ap.add_argument("--pids", type=str, default="300,500,700,850,950")
    ap.add_argument("--ks", type=str, default="1,2,3", help="V-gating periods; 1 = baseline")
    ap.add_argument("--beam", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=170)
    ap.add_argument("--alpha", type=float, default=8.0, help="V-step shortlist = alpha*B")
    ap.add_argument("--policy-temp", type=float, default=1.0)
    ap.add_argument("--matched-budget", action="store_true",
                    help="widen sg_k to k*B so V-evals match k=1 (default: same width)")
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    assert 1 in ks, "include k=1 (the baseline arm)"

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    floor_len = {pid: len(p) for pid, p in load_submission(args.best_csv).items()}

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    V = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    PI = load_model_checkpoint(args.policy, device=args.device, dtype=dtype)
    setup_model_for_inference(V); setup_model_for_inference(PI)
    sg = SubgoalSolver(puzzle, V, PI, device=args.device,
                       internal_batch_size=args.internal_batch_size, random_seed=args.seed,
                       alpha=args.alpha, policy_temp=args.policy_temp)

    qsolver = None
    if not args.no_qshort:
        try:
            STU = load_model_checkpoint(args.student, device=args.device, dtype=dtype)
            setup_model_for_inference(STU)
            qsolver = QShortlisterSolver(puzzle, V, STU, device=args.device,
                                         internal_batch_size=args.internal_batch_size,
                                         random_seed=args.seed, alpha=2.0)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] qshort reference unavailable: {type(e).__name__}: {e}")

    print(f"V={args.checkpoint.name} policy={args.policy.name} dtype={dtype} beam={args.beam} "
          f"alpha={args.alpha} temp={args.policy_temp} matched_budget={args.matched_budget}")
    print(f"pids={pids}  ks={ks}\n")

    results = []
    for pid in pids:
        s = states[pid]
        row = {"pid": pid, "L_floor": floor_len.get(pid), "arms": {}}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if qsolver is not None:
            t0 = time.time()
            f, _l, names, prof = qsolver.solve(
                s, KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps, num_attempts=1))
            raw, pp = (_pp_len(puzzle, s, names) if f else (None, None))
            row["arms"]["qshort"] = {"found": f, "raw": raw, "pp": pp,
                                     "wall_s": round(time.time() - t0, 1)}

        for k in ks:
            width = args.beam * (k if args.matched_budget else 1)
            t0 = time.time()
            f, _l, names, prof = sg.solve(
                s, KhoruzhiiSearchConfig(beam_width=width, num_steps=args.max_steps), k=k)
            raw, pp = (_pp_len(puzzle, s, names) if f else (None, None))
            row["arms"][f"sg_k{k}"] = {
                "found": f, "raw": raw, "pp": pp, "width": width,
                "n_v_evals": getattr(prof, "n_v_evals", None),
                "n_v_steps": getattr(prof, "n_v_steps", None),
                "n_steps": prof.n_steps, "wall_s": round(time.time() - t0, 1),
            }

        base = row["arms"].get("sg_k1", {})
        parts = []
        if qsolver is not None:
            q = row["arms"]["qshort"]
            parts.append(f"qshort={q['pp']}({q['raw']})")
        for k in ks:
            a = row["arms"][f"sg_k{k}"]
            tag = "base" if k == 1 else ""
            d = (f" d={a['pp'] - base['pp']:+d}" if (k != 1 and a["pp"] and base.get("pp")) else "")
            parts.append(f"k{k}={a['pp']}({a['raw']}){tag}{d}[{a['wall_s']}s,V{a['n_v_evals']//1000 if a['n_v_evals'] else 0}k]")
        print(f"pid={pid:4d} floor={row['L_floor']}  " + "  ".join(parts), flush=True)
        results.append(row)

    # ---- summary: each k>1 vs k=1 (baseline), on POST-PROCESSED length
    print("\n==== summary (post-processed length; sg_k1 = baseline) ====")
    base_solved = {r["pid"]: r["arms"]["sg_k1"]["pp"] for r in results
                   if r["arms"]["sg_k1"].get("pp") is not None}
    for k in [x for x in ks if x != 1]:
        both = [r for r in results
                if r["arms"]["sg_k1"].get("pp") is not None
                and r["arms"][f"sg_k{k}"].get("pp") is not None]
        if not both:
            print(f"  k={k}: no pids solved by both arms"); continue
        d_base = sum(r["arms"]["sg_k1"]["pp"] for r in both)
        d_k = sum(r["arms"][f"sg_k{k}"]["pp"] for r in both)
        wins = sum(1 for r in both if r["arms"][f"sg_k{k}"]["pp"] < r["arms"]["sg_k1"]["pp"])
        ties = sum(1 for r in both if r["arms"][f"sg_k{k}"]["pp"] == r["arms"]["sg_k1"]["pp"])
        ve_base = sum(r["arms"]["sg_k1"]["n_v_evals"] for r in both)
        ve_k = sum(r["arms"][f"sg_k{k}"]["n_v_evals"] for r in both)
        sr_k = sum(1 for r in results if r["arms"][f"sg_k{k}"].get("found"))
        sr_b = sum(1 for r in results if r["arms"]["sg_k1"].get("found"))
        print(f"  k={k}: over {len(both)} both-solved  sum_pp base={d_base} k{k}={d_k} "
              f"delta={d_k - d_base:+d}  (k{k} wins {wins}, ties {ties}, "
              f"losses {len(both) - wins - ties})  V-evals {ve_k / max(ve_base,1):.2f}x  "
              f"solve-rate {sr_k}/{len(results)} vs base {sr_b}/{len(results)}")
    print("\n  VERDICT: subgoal helps our objective only if some k>1 has delta<0 "
          "(shorter) at <=1.0x V-evals, or clearly shorter at matched budget.")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "results": results},
                      f, indent=1)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
