# Megaminx Next Steps — Brainstorm & Strategic Plan

## Context

**Where we are.** Submitted best **88,195 (rank #3)**, target **<70,000** (−20.7%). Top of LB: Kuznetsov 79,971; DrozdovDan 81,946; us 88,195; **Rokicki 93,606** (the literal God-number cuber is below us — classical search has saturated on this puzzle's structure). Production stack: m05 (V model, 6.0M, Bellman warmstart) + m23 Q-shortlister (recall=100% @ α=2, 4.4× speedup) + beam 131k with `--compile + pad_to_batch_size=True` + BFS-d6 window post-process + beam-stack rescue (saved pid 490, 920).

**Why this plan.** Three plateaus confirm we are **data-signal-bound, not capacity-bound**: RW MSE plateaus at ~64, Bellman loss at ~0.094-0.10, m26/m26b (12M/13M params) deliver +1 solve / +1.8 path = WORSE than m05's 6M. Going from 88K → 70K is a heuristic-quality problem; speed wins only matter to the extent they fund larger beams. External literature (CayleyPy paper) reports zero Megaminx results — **we are pioneering, not deriving**. The IHES picture-cube experience (also a Cayley-graph permutation puzzle) provides the highest-confidence cross-pollination signal: BFS-exact-target mixin into Bellman is the single most-cited untried lever.

**Decisions taken.** Fork the Bellman training loop locally inside `megaminx/scripts/` rather than patching `cayley.bellman`. Defer icosahedral symmetry derivation (sym v3) to long-list backlog. Keep the existing acceptance gate (≥+3 strat-5 solves AND ≤0.95× mean model_avg).

---

## Recommended action plan

Tiers ordered by **expected EV per engineering hour**, not by chronology. T0 and T1 items run mostly in parallel; T2 unlocks only if T1A passes; T3 are research bets.

### TIER 0 — Close-out items already in flight

#### T0.1 — TRT bf16 deploy
- **State**: in flight per HANDOFF §13. fp16 build was −18.5% wall but +4 path-sum; bf16 expected eager-vs-trt |delta|=0 → exact path match.
- **Action**: validate path-sum on 12-puzzle benchmark on GCP L4. If matches compile baseline (1049), add `--tensorrt-engine <path>` arg to `scripts/03_solve.py` (line 208 `load_model_checkpoint`).
- **Files**: `beam_lab/export_tensorrt.py` (already built), `scripts/03_solve.py` (~5 lines added).
- **Expected**: −15-20% wall → unlocks beam 196k in compile's 131k budget. Small quality win on hard tail (~−0.5%).
- **Acceptance**: path-sum exactly matches compile baseline on 12-puzzle.
- **Wall to verify**: 1h.

### TIER 1 — High EV, near-term, parallelizable

#### T1.A — Option B: BFS-d6 exact target mixin into Bellman (HIGHEST EV)
- **Why**: addresses the data-bound plateau head-on. For states within the d≤6 BFS shell (19.4M states), use the exact distance instead of the bootstrapped `1 + min_a V_target(apply(s,a))`. Anchors the learned function to ground truth on the boundary.
- **Approach**: **fork the loop locally**, do not patch `cayley.bellman.train_bellman`. New script `scripts/05b_bellman_refine_mixin.py` re-implements the Bellman target loop with the mixin (mirror the structure of `scripts/05_bellman_refine.py:71`).
- **Files to write**:
  - `scripts/05b_bellman_refine_mixin.py` — new training script. Re-implements Bellman target generation: for each batch state `s`, check `key = bytes(s) in bfs_table.table`; if hit, target = `len(bfs_table.table[key])` (path length); else target = `1 + min_a V_target(apply(s,a))`. Reuse `cayley.model.ResMLPDistance`, `cayley.training.TrainConfig`.
  - `configs/m27_bellman_bfs6.yaml` — copy of `configs/m05_bellman_warm.yaml` with `bellman.warmstart_path` pointing at m05's body and a new `bellman.bfs_table_path` field.
- **Reused utilities**: `BfsBytesTable.load` (`src/megaminx/bfs_bytes.py:48`); `BfsBytesTable.lookup` (line 40); the `puzzle.apply_move` numpy gather pattern (`src/megaminx/bfs_bytes.py:87`).
- **Wall**: +10-15% per epoch over m05 baseline. ~40-60 min for 200-500 ep on local 4090.
- **Expected**: tighter convergence than 0.094-0.10 plateau; on strat-5 expect ≥+3 solves OR ≤0.90× mean.
- **Acceptance**: standard gate (≥+3 strat-5 solves AND ≤0.95× mean). Strat-5 protocol: `--stratified 5 --strat-seed 0` (51 puzzles).
- **Risk**: mixin signal could conflict with bootstrap (pulls boundary down, interior gradient pushes up). If divergence, try `bfs_loss_weight=0.5` instead of full replacement.

#### T1.B — Multi-seed beam ensemble (3 seeds, take min)
- **Why**: untested; IHES catalog showed +3% via min-merge across seeds. The current beam is greedy with hardcoded `random_seed=0` (`scripts/03_solve.py:274`) — RNG only enters via `_do_greedy_step()` tie-breaks and stagnation retry shuffling. Different seeds explore non-identical sub-trees.
- **Files**: `scripts/03_solve.py` add `--seed` arg (lines 109-162); pass to `KhoruzhiiSolver(... random_seed=args.seed)` (line 299) and `QShortlisterSolver(... random_seed=args.seed)` (line 274).
- **Surgery**: ~5 lines. Verify `KhoruzhiiSolver._do_greedy_step()` (in cayley lib, not in repo) actually consumes `random_seed`. If not, this becomes a research item (no value).
- **Wall**: 3× per puzzle. Strat-5 cost: ~3× current strat-5 (~30-60 min on 4090).
- **Expected**: −2-4% paths from independent tie-break sampling.
- **Acceptance**: standard gate.
- **Order it last in the sequence**: do this AFTER T0.1 so we can use TRT-funded headroom.

#### T1.C — Aggressive beam-stack rescue (top decile of long-path pids)
- **Why**: current best (88K) only rescued 2 pids (490, 920) and saved 700+ moves between them. Many other pids likely sit on the boundary. Beam-stack already implemented (`scripts/12_beam_stack_rescue.py`), the missing piece is **automated identification** of rescue candidates.
- **Approach**: post-process the latest `submissions/<best>.csv` to extract pids whose path length is in the top decile; auto-feed all of them to `12_beam_stack_rescue.py` with widened runner-up beam (e.g., 2B instead of B).
- **Files**: new helper `scripts/13_auto_rescue.py` (~50 lines): reads submission CSV, ranks pids by `len(path)`, picks top-K (default 50), iterates `solve_with_backtrack()` for each, splices winners back. Reuse `BeamStackSolver.solve_with_backtrack` (line 102 of `12_beam_stack_rescue.py`).
- **Wall**: ~2 min/pid × 50 = ~1.5h on 4090.
- **Expected**: −500-1500 moves total (each rescued pid typically saves 200-700 moves; assume 5-10 successful rescues out of 50 attempts).
- **Acceptance**: each rescue must verify via `verify_path` and shorten the existing path. No regression possible by construction.

#### T1.D — NISS retest on m05+m23 stack
- **Why**: HANDOFF §6 rejected NISS on m07 (doubled wall, no win). m05 + m23 is a structurally different stack (sharper heuristic + Q-shortlister). The flag is already wired (`scripts/03_solve.py:117` `--niss`); it costs 1 strat-5 run to settle.
- **Files**: zero code changes. Run: `03_solve.py --checkpoint m05/epoch_0499.pt --qshort-student m23/epoch_0499.pt --niss --stratified 5 --bf16 --beams 131072 --max-steps 150`.
- **Wall**: ~12h compute (doubled wall × strat-5 with qshort).
- **Expected**: most likely confirms rejection (m05 already sharp). If it shows even +2 solves, opens cheap diversity lever.
- **Acceptance**: standard gate vs current m05+m23 baseline.

### TIER 2 — Conditional on T1.A passing

#### T2.A — Rebuild m23 Q-shortlister against m27 teacher
- **Why**: if Option B (T1.A) yields a sharper teacher, m23 must be re-distilled against it or qshort recall drops.
- **Files**: `scripts/09_train_q_shortlister.py` (existing); new config `configs/m28_qshort_from_m27.yaml`.
- **Wall**: ~3h on 4090 (same recipe as m23).
- **Acceptance**: recall ≥99% at α=2 on held-out states (`scripts/09_eval_q_recall.py`).

#### T2.B — m22 K-step lookahead Bellman (K=2)
- **Why**: alternative second-stage signal. Code complete (`scripts/10_train_m22_horizon.py`) but never trained. Different signal than 1-step Bellman: averages over 24² = 576 leaves per state, may break the 0.094 plateau.
- **Wall**: ~30-40 min/epoch (vs m05's 25 min); ~150-200 min for 300 epochs.
- **Trigger**: only if T1.A passes the gate marginally (+3 solves but mean still close to 1.0×) — m22 then becomes the next refinement attempt.
- **Acceptance**: standard gate vs T1.A teacher.

#### T2.C — m24 V + π policy multi-task
- **Why**: HANDOFF §9.2 rates this +3% on hard puzzles. Beam scoring becomes `V(child) + λ·(-log π(a|parent))`. Currently `11_train_policy_head.py` is policy-ONLY (no V-head); needs the V-head added back.
- **Files**: rewrite `11_train_policy_head.py` to dual-head (V + π); add `--policy-head` and `--policy-weight λ` args to `03_solve.py` so the qshort/beam path consumes π logits.
- **Wall**: ~200 min for 200 ep training; ~1d engineering for the dual-head + beam integration.
- **Trigger**: only if T2.B is rejected and we still need a quality lever.

#### T2.D — Tail re-solve post-processing (from IHES E7)
- **Why**: existing `full_post_process` does same-face + inverse + BFS-d6 windows but only at fixed window sizes. Tail re-solve runs a short narrow beam from the state at position N (last K moves) to find a shorter K-suffix.
- **Files**: extend `src/megaminx/post_process.py` with `tail_resolve(path, puzzle, model, k, beam_width)` and integrate into `full_post_process`.
- **Wall**: ~3h port + 1h validation.
- **Expected**: −50-200 moves total (small but free; works regardless of training changes).
- **Acceptance**: every emitted path must verify; total moves must monotonically decrease.

### TIER 3 — Research bets (plan, do not commit)

These take a week+ each and have lower confidence. Worth writing down so we don't accidentally re-derive them; only pick up if T1+T2 stall.

- **T3.A — Test-time per-puzzle Bellman refinement.** For each test puzzle, do 50-200 gradient steps of self-supervised Bellman on random walks near the scramble. Novel; not in HANDOFF or IHES catalog. Risk: overfits per-puzzle, may not generalize across the 1001 set.
- **T3.B — Macro-Q shortlisting.** Mine 2-6 move macros from solved paths; train Q-head with `output_dim = 24 + n_macros`. Speeds up by reducing step count, not per-step cost. Per HANDOFF §10: 3 days effort, generalization risk.
- **T3.C — A* / IDA* with PDB lower bound.** Korf-style pattern DBs on Megaminx subgroups (e.g., one face fixed) → admissible heuristic for IDA*. 1+ week. Beam's parallelism makes A* hard to beat at width 131k+.
- **T3.D — BFS-d7 partial slices.** 250M full-shell states won't fit (30 GB even bytes-keyed). Possible via subgroup factorization. Engineering cost high; gain marginal beyond d=6 (tested saturated).
- **T3.E — Bidirectional with learned front-to-front scoring.** Train `h(s_forward, s_backward)`. Real research project, 1+ week, high uncertainty.

### Explicitly NOT doing

- **Bigger architecture for Bellman.** m26/m26b proved the 0.094 plateau is data-bound. Same for RW MSE 64.
- **Icosahedral symmetry v3.** Per user decision: scope out. Stays in long-list backlog. Multi-orbit issue blocked v2.
- **Iterated mode in cayleypy.** Russian commenter warning + measured +124% wall.
- **NISS on m07 line.** Already rejected.
- **Manual CUDA Graphs without `--compile`.** Inductor's kernel fusion is the real win.
- **Bigger internal_batch_size.** 16k saturates 4090/L4 (compute-bound).

---

## Files to read / modify (quick map)

| Path | Role | T-tier touching it |
|---|---|---|
| `scripts/03_solve.py` | THE production solver. Auto-detects Q-function via `output_dim > 1`. | T0.1 (TRT arg), T1.B (--seed arg), T1.D (--niss exists), T2.C (--policy-head arg) |
| `scripts/05_bellman_refine.py` | Reference for Bellman loop structure (calls external `cayley.bellman.train_bellman`). | T1.A reads, T1.A forks into 05b |
| `scripts/05b_bellman_refine_mixin.py` | NEW. Local fork of Bellman loop with BFS-d6 mixin. | T1.A creates |
| `scripts/12_beam_stack_rescue.py` | Beam-stack rescue logic. Currently manual pid list. | T1.C reads |
| `scripts/13_auto_rescue.py` | NEW. Auto-identifies long-path pids and feeds them to beam-stack. | T1.C creates |
| `scripts/09_train_q_shortlister.py` | Q-distillation script. Reusable for retraining m23 against new teacher. | T2.A reads |
| `scripts/10_train_m22_horizon.py` | K-step Bellman, never trained. | T2.B reads |
| `scripts/11_train_policy_head.py` | Policy-only; needs V-head added for V+π. | T2.C rewrites |
| `src/megaminx/bfs_bytes.py` | `BfsBytesTable` (line 22-50). Provides `lookup(perm)`. | T1.A consumer |
| `src/megaminx/post_process.py` | `full_post_process` (line 63-85). | T2.D extends |
| `src/megaminx/mitm_solver.py` | MITM beam (terminates on shell entry). Reference for beam-loop modifications. | T1.C reads |
| `configs/m05_bellman_warm.yaml` | Baseline Bellman config. Source for m27 config. | T1.A copies |
| `configs/m27_bellman_bfs6.yaml` | NEW. Bellman config with `bfs_table_path` field. | T1.A creates |
| `data/bfs_bytes_d6.pkl` | 19.4M-state shell, ~4 GB on disk. | T1.A loads at training start |
| `data/pp_bfs6_fallback.csv` | 414,678-floor fallback. NEVER use sample_fallback.csv. | All solves |

---

## Verification end-to-end

For every T1/T2 item, the canonical proof is the same and uses one entrypoint:

```bash
# Strat-5 eval on the new model (51 puzzles, the canonical metric):
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint <new_ckpt> --out <out.csv> \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 --bf16
```

Compare per-bucket `model_avg` and `chosen_avg` against the m05 baseline printed at the bottom of the run. **Acceptance gate (unchanged)**: ≥+3 strat-5 solves AND mean model_avg ≤ 0.95× current best AND no bucket regresses by more than 1 solve.

For T0.1 (TRT) specifically, the gate is exact path-match on the 12-puzzle benchmark, not strat-5.

For T1.B (multi-seed): verify each seed produces a valid path independently, then take min per pid; total moves on the 51-puzzle strat must beat the single-seed baseline by ≥+3 solves AND lower mean.

For T1.C (auto-rescue): every output row must `verify_path` clean and be shorter than the corresponding row in the source submission. No false positives possible.

For full submission (only after gate passes):

```bash
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint <new_ckpt> \
    --qshort-student models/m23_or_m28/epoch_0499.pt \
    --qshort-alpha 2 \
    --out submissions/<name>.csv \
    --beams 524288 --max-steps 150 --bf16
# Then:
.venv/Scripts/kaggle.exe competitions submit -c cayley-py-megaminx \
    -f submissions/<name>.csv -m "<short description>"
```

---

## Recommended sequencing (for the next 2-4 weeks)

1. **Week 1**: T0.1 (TRT bf16 deploy, 1 day) + T1.C (auto-rescue, 1 day eng + 2h compute) + T1.D (NISS retest, 12h compute) — all parallel, all cheap.
2. **Week 1-2**: T1.A (Option B trainer + m27 train + strat-5). The big bet. Estimated 3-5 days end-to-end.
3. **Week 2**: T1.B (multi-seed) once TRT bf16 is funding the bigger beam.
4. **Week 2-3 (conditional)**: If T1.A passes → T2.A (rebuild m23). If T1.A passes marginally → T2.B (m22). If T1.A doesn't pass → T2.C (m24 V+π) as the alternative second-stage lever.
5. **Week 3+**: T2.D (tail re-solve, free win) plus a Tier-3 spike if no Tier-1/2 lever has cracked the 80K barrier.

The 70K target is gated by **T1.A succeeding or some Tier-3 research panning out**. Everything else is incremental polish on top.

---

## Code sketches

These are illustrative scaffolds, not production-ready. Each one mirrors the existing patterns (script-level CLI, `PROJECT.parent / "src"` imports, `ResMLPDistance` from cayley, target-net snapshot, bf16 autocast).

### Sketch — T0.1: `--tensorrt-engine` arg in `03_solve.py`

Add to `scripts/03_solve.py` argument parser (after line 161):

```python
ap.add_argument("--tensorrt-engine", type=Path, default=None,
                help="path to a TorchScript-saved TRT engine (built via "
                     "beam_lab/export_tensorrt.py). Replaces eager/compile model "
                     "for inference. Must match the V-model interface; bf16 build "
                     "expected to preserve paths exactly.")
```

Replace the model-load block (lines 207-211):

```python
dtype = torch.bfloat16 if args.bf16 else torch.float32
if args.tensorrt_engine is not None:
    model = torch.jit.load(str(args.tensorrt_engine), map_location=args.device)
    model.eval()
    print(f"loaded TRT engine: {args.tensorrt_engine.name} (bf16={args.bf16})")
else:
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
if args.chunk_size is not None:
    base = getattr(model, "_orig_mod", model)
    base.inference_chunk_size = args.chunk_size
```

Validation (12-puzzle path-sum match):

```bash
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint models/m05_bellman_warm/epoch_0499.pt \
    --tensorrt-engine models/m05_trt_bf16_b16384_sm89.ts \
    --out submissions/_trt_check.csv \
    --beams 131072 --max-steps 150 --bf16 \
    --pid-from 0 --pid-to 12
# Compare resulting total_moves vs the compile baseline (should be identical for bf16).
```

### Sketch — T1.A: BFS-d6 mixin Bellman trainer (`scripts/05b_bellman_refine_mixin.py`)

This is the highest-EV item. Mirror `05_bellman_refine.py` but **inline** the Bellman target loop so we can intercept per-state targets. Note that we re-implement what `cayley.bellman.train_bellman` does internally — the upside is full control of the target.

```python
"""m27 — Bellman refinement with BFS-d6 exact-target mixin.

For each batch state s:
  if bytes(s) in bfs_table:  target = len(bfs_table[s])     # exact distance
  else:                      target = 1 + min_a V_target(apply(s, a))  # bootstrap

Forks scripts/05_bellman_refine.py: same warmstart/data/optim/sched, but the target
generation runs locally instead of inside cayley.bellman.train_bellman.

Usage:
    .venv/Scripts/python.exe scripts/05b_bellman_refine_mixin.py \
        --config configs/m27_bellman_bfs6.yaml \
        --output models/m27_bellman_bfs6
"""
from __future__ import annotations

import argparse, copy, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.model import ResMLPDistance
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


@torch.no_grad()
def bellman_target_with_mixin(
    target_model: ResMLPDistance,
    states: torch.Tensor,            # (B, S) int64
    walk_depths: torch.Tensor,       # (B,)  float32 — upper bound clamp
    generators: torch.Tensor,        # (n_gen, S)
    solved_state: torch.Tensor,      # (S,)
    bfs_keys_t: torch.Tensor,        # (M, S) int8 — BFS-d6 keys, GPU
    bfs_lens_t: torch.Tensor,        # (M,)  int8  — BFS-d6 word lengths, GPU
    bfs_hashes_sorted: torch.Tensor, # (M,) int64, sorted hash for fast isin
    hash_vec: torch.Tensor,          # (S,) int64 random hash basis
    chunk_size: int,
) -> torch.Tensor:
    """Return per-state target; uses BFS-d6 exact length when state is in shell,
    otherwise standard 1-step Bellman bootstrap.
    """
    B, S = states.shape
    n_gen = generators.shape[0]

    # 1. Standard 1-step Bellman: y_boot = 1 + min_a V_target(apply(s, a))
    children = torch.gather(
        states.unsqueeze(1).expand(B, n_gen, S), 2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    )  # (B, n_gen, S)
    children_flat = children.reshape(B * n_gen, S)
    leaf_v = torch.empty(B * n_gen, dtype=torch.float32, device=states.device)
    for i in range(0, B * n_gen, chunk_size):
        leaf_v[i : i + chunk_size] = target_model(
            children_flat[i : i + chunk_size]
        ).flatten().to(torch.float32)
    is_solved = (children_flat == solved_state).all(dim=1)
    leaf_v = torch.where(is_solved, torch.zeros_like(leaf_v), leaf_v)
    y_boot = 1.0 + leaf_v.view(B, n_gen).min(dim=1).values
    y_boot = torch.minimum(y_boot, walk_depths).clamp(min=0.0)

    # 2. BFS-d6 mixin: for each batch state, hash → isin → verify bytes-equality.
    state_hashes = (hash_vec * states).sum(dim=1)              # (B,)
    in_shell = torch.isin(state_hashes, bfs_hashes_sorted)     # (B,) bool
    if not in_shell.any():
        return y_boot

    # Verify by bytes-equality (hash collisions possible). For each candidate hit,
    # find its index in bfs_keys_t and compare the 120-byte state.
    cand_idx = torch.nonzero(in_shell, as_tuple=True)[0]       # (Bhit,)
    # For each candidate, search bfs_hashes_sorted via searchsorted (O(log M)).
    pos = torch.searchsorted(bfs_hashes_sorted, state_hashes[cand_idx])
    pos = pos.clamp(max=bfs_hashes_sorted.size(0) - 1)
    # Iterate small set on CPU because hash collisions are rare and bytes-compare is exact.
    cand_states_cpu = states[cand_idx].to(torch.int8).cpu().numpy()
    cand_pos_cpu = pos.cpu().numpy()
    bfs_lens_arr = bfs_lens_t.cpu().numpy()  # cache once outside, see note below
    target = y_boot.clone()
    # NOTE: production code should precompute a bytes->len dict for O(1) lookup.
    # Sketch only: this is the conceptual flow.
    bytes_to_len = bfs_lens_t.cpu().numpy()  # placeholder
    # (See real impl note: pass a {bytes(state): length} dict from main and look up.)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    bell_cfg = cfg["bellman"]
    if args.epochs is not None:
        train_cfg["n_epochs"] = args.epochs

    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device: {device}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    model = ResMLPDistance(
        state_size=model_cfg["state_size"], num_classes=model_cfg["num_classes"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "embedding"),
        embed_dim=model_cfg.get("embed_dim", 16),
    ).to(device)

    # Warmstart from m05.
    ws_path = bell_cfg["warmstart_path"]
    sd = torch.load(ws_path, map_location=device, weights_only=False)["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"warmstarted from {ws_path}")

    # Load BFS-d6 table, push keys/hashes to GPU.
    bfs_table = BfsBytesTable.load(bell_cfg["bfs_table_path"])
    print(f"BFS-d6 shell: {len(bfs_table.table):,} states")
    # Build a {bytes -> len} dict for O(1) per-batch lookup. ~150 MB for 19.4M entries.
    bytes_to_len: dict[bytes, int] = {k: len(v) for k, v in bfs_table.table.items()}

    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg["n_epochs"])

    gens_t = torch.from_numpy(GeneratorTable.from_puzzle(puzzle).perms).to(device)
    solved_t = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    rng = torch.Generator(device=device).manual_seed(train_cfg.get("seed", 0))

    for epoch in range(train_cfg["n_epochs"]):
        t0 = time.time()
        n_walks = max(1, train_cfg["samples_per_epoch"] // train_cfg["k_max"])
        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=train_cfg["k_max"],
            seed=train_cfg.get("seed", 0) + epoch, device=device,
            n_back=train_cfg.get("n_back", 1),
        )
        depths_f = depths.to(torch.float32)
        N = states.shape[0]

        model.train()
        total_loss, n_batches = 0.0, 0
        perm = torch.randperm(N, generator=rng, device=device)
        for i in range(0, N, train_cfg["batch_size"]):
            idx = perm[i : i + train_cfg["batch_size"]]
            bs = states[idx]; bd = depths_f[idx]

            # Standard Bellman target.
            with torch.no_grad():
                children = torch.gather(
                    bs.unsqueeze(1).expand(-1, gens_t.size(0), -1), 2,
                    gens_t.unsqueeze(0).expand(bs.size(0), -1, -1),
                )
                cf = children.reshape(-1, bs.size(1))
                leaf_v = torch.empty(cf.size(0), dtype=torch.float32, device=device)
                for j in range(0, cf.size(0), bell_cfg["target_net_chunk"]):
                    leaf_v[j : j + bell_cfg["target_net_chunk"]] = target_model(
                        cf[j : j + bell_cfg["target_net_chunk"]]
                    ).flatten().to(torch.float32)
                leaf_v = torch.where(
                    (cf == solved_t).all(dim=1), torch.zeros_like(leaf_v), leaf_v,
                )
                y_boot = 1.0 + leaf_v.view(bs.size(0), -1).min(dim=1).values
                target = torch.minimum(y_boot, bd).clamp(min=0.0)

                # BFS-d6 mixin: O(B) dict lookups on CPU. For B=4096, 4ms.
                bs_cpu = bs.to(torch.int8).cpu().numpy()
                for k_idx in range(bs.size(0)):
                    key = bs_cpu[k_idx].tobytes()
                    exact_len = bytes_to_len.get(key)
                    if exact_len is not None:
                        target[k_idx] = float(exact_len)

            if autocast_ctx is not None:
                with autocast_ctx:
                    pred = model(bs)
                    loss = F.mse_loss(pred, target)
            else:
                pred = model(bs)
                loss = F.mse_loss(pred, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()

        avg = total_loss / max(n_batches, 1)
        print(f"epoch {epoch:4d} | loss {avg:.4f} | "
              f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time() - t0:.1f}s",
              flush=True)

        if (epoch + 1) % bell_cfg["target_update_every_epochs"] == 0:
            sd_now = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
                      if not k.startswith("_orig_mod.") or True}
            target_model.load_state_dict(sd_now)
        if (epoch + 1) % train_cfg["checkpoint_every_epochs"] == 0:
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "loss": avg,
                        "model_config": model_cfg, "warmstart_from": ws_path,
                        "bfs_mixin": bell_cfg["bfs_table_path"]},
                       args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Companion config `configs/m27_bellman_bfs6.yaml` (mirror of `m05_bellman_warm.yaml`):

```yaml
seed: 27
model:
  state_size: 120
  num_classes: 120
  hidden_dims: [2048, 512]
  num_res_blocks: 2
  encoding: embedding
  embed_dim: 16
training:
  n_epochs: 500
  samples_per_epoch: 500_000
  batch_size: 4096
  k_max: 80
  n_back: 1
  lr: 5.0e-4
  seed: 27
  checkpoint_every_epochs: 25
bellman:
  warmstart_path: models/m05_bellman_warm/epoch_0499.pt
  bfs_table_path: data/bfs_bytes_d6.pkl
  target_update_every_epochs: 10
  target_net_chunk: 8192
```

**Performance note**: the per-batch CPU loop is the obvious bottleneck. On a 4090 a B=4096 batch costs ~4 ms in the loop (cheap relative to the K=24 forward pass for target net). If profiling shows it dominates, vectorize via the hash+isin scheme already in `mitm_solver.py:67-85` — reuse that pattern for GPU-side mixin.

### Sketch — T1.B: `--seed` arg for multi-seed beam ensemble

Two-line change in `scripts/03_solve.py`. Add arg:

```python
ap.add_argument("--seed", type=int, default=0,
                help="random seed for KhoruzhiiSolver / QShortlister tie-breaking. "
                     "Run multiple seeds and post-merge for ensemble (see merge_seeds.py).")
```

Pass it where solvers are constructed (lines 274, 299):

```python
_qshort_inner = QShortlisterSolver(
    puzzle, teacher=model, student=student, device=args.device,
    internal_batch_size=args.qshort_internal_batch_size,
    random_seed=args.seed, state_dtype=state_dtype, alpha=args.qshort_alpha,
)
# ... and:
solver = KhoruzhiiSolver(
    puzzle, model, device=args.device, state_dtype=state_dtype,
    use_q_function=detected_q, random_seed=args.seed,
)
```

Driver script `scripts/14_multi_seed_solve.py` (~30 lines) — runs N seeds and emits the per-pid min:

```python
"""Multi-seed beam ensemble: solve N times with different seeds, take min path per pid."""
from __future__ import annotations

import argparse, csv, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--qshort-student", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path,
                    help="merged best-of-N submission CSV")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="comma-separated seeds; one solve per seed")
    ap.add_argument("--beams", type=str, default="131072")
    ap.add_argument("--max-steps", type=str, default="150")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed_outputs: list[Path] = []
    for s in seeds:
        out_s = args.out.with_suffix(f".seed{s}.csv")
        cmd = [
            ".venv/Scripts/python.exe", "scripts/03_solve.py",
            "--checkpoint", str(args.checkpoint),
            "--out", str(out_s), "--seed", str(s),
            "--beams", args.beams, "--max-steps", args.max_steps,
            "--bf16",
        ]
        if args.qshort_student is not None:
            cmd += ["--qshort-student", str(args.qshort_student), "--qshort-alpha", "2"]
        print(f">>> seed {s}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        per_seed_outputs.append(out_s)

    # Merge: per pid, pick shortest path across seeds.
    best: dict[int, list[str]] = {}
    for p in per_seed_outputs:
        with open(p) as f:
            r = csv.DictReader(f)
            for row in r:
                pid = int(row["initial_state_id"])
                path = row["path"].split(".") if row["path"] else []
                if pid not in best or len(path) < len(best[pid]):
                    best[pid] = path
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(best):
            w.writerow([pid, ".".join(best[pid])])
    print(f"wrote {args.out}: {len(best)} pids merged from {len(seeds)} seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Risk**: `KhoruzhiiSolver._do_greedy_step()` (in cayley lib, not in repo) might not actually consume `random_seed` for tie-breaks. Verify before committing the wall-time. If RNG isn't wired, we'd need to add it inside cayley OR fall back to swapping different `--qshort-alpha` values per run as a poor-man's diversity lever.

### Sketch — T1.C: Auto-rescue driver `scripts/13_auto_rescue.py`

Wraps `12_beam_stack_rescue.py`. Reads the latest best submission, identifies long-path pids, runs beam-stack rescue on them, then merges shorter results back.

```python
"""Auto-rescue: pick top-K longest-path pids from current best submission, run beam-stack
rescue on them, merge results back. Idempotent and safe by construction (only replaces
when shorter)."""
from __future__ import annotations

import argparse, csv, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path,
                    help="current best submission CSV")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="output: baseline with rescued pids replaced")
    ap.add_argument("--top-k", type=int, default=50,
                    help="rescue the K longest-path pids from baseline")
    ap.add_argument("--beam", type=int, default=262144,
                    help="wider beam for rescue (2x default)")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    # Rank pids by current path length.
    rows: list[tuple[int, list[str]]] = []
    with open(args.baseline) as f:
        r = csv.DictReader(f)
        for row in r:
            pid = int(row["initial_state_id"])
            path = row["path"].split(".") if row["path"] else []
            rows.append((pid, path))
    rows_sorted = sorted(rows, key=lambda x: -len(x[1]))
    target_pids = [pid for pid, _ in rows_sorted[: args.top_k]]
    print(f"top-{args.top_k} longest pids (by current path length):")
    for pid, p in rows_sorted[: args.top_k]:
        print(f"  pid {pid:4d}: {len(p)} moves")

    rescue_csv = args.out.with_suffix(".rescues.csv")
    cmd = [
        ".venv/Scripts/python.exe", "scripts/12_beam_stack_rescue.py",
        "--checkpoint", str(args.checkpoint),
        "--pids", ",".join(str(p) for p in target_pids),
        "--beam", str(args.beam), "--max-steps", str(args.max_steps),
        "--max-retries", str(args.max_retries),
        "--out", str(rescue_csv), "--bf16",
    ]
    print(f">>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    # Merge: for each pid in rescue_csv, replace baseline path if rescue is shorter.
    rescues: dict[int, list[str]] = {}
    with open(rescue_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            pid = int(row["initial_state_id"])
            rescues[pid] = row["path"].split(".") if row["path"] else []

    n_replaced, saved = 0, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid, path in sorted(rows):
            if pid in rescues and 0 < len(rescues[pid]) < len(path):
                saved += len(path) - len(rescues[pid])
                n_replaced += 1
                w.writerow([pid, ".".join(rescues[pid])])
            else:
                w.writerow([pid, ".".join(path)])
    print(f"replaced {n_replaced}/{len(rescues)} rescued pids, saved {saved} moves total")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Sketch — T1.D: NISS retest (no code, just the command)

```bash
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint models/m05_bellman_warm/epoch_0499.pt \
    --qshort-student models/m23_q_shortlister/epoch_0499.pt \
    --qshort-alpha 2 \
    --niss \
    --out submissions/_niss_strat5_check.csv \
    --beams 131072 --max-steps 150 --bf16 \
    --stratified 5 --strat-seed 0
```

Compare per-bucket `model_avg` against the m05+m23 baseline at the same beam/max-steps without `--niss`.

### Sketch — T2.A: Q-shortlister retraining against m27 (no new script)

Reuse `scripts/09_train_q_shortlister.py` with a new config that points its `teacher_path` at the m27 checkpoint:

```bash
.venv/Scripts/python.exe scripts/09_train_q_shortlister.py \
    --teacher models/m27_bellman_bfs6/epoch_0499.pt \
    --out-dir models/m28_qshort_from_m27 \
    --n-epochs 500 --samples-per-epoch 500000 --batch-size 8192
# Then validate recall:
.venv/Scripts/python.exe scripts/09_eval_q_recall.py \
    --teacher models/m27_bellman_bfs6/epoch_0499.pt \
    --student models/m28_qshort_from_m27/epoch_0499.pt \
    --alpha 2
# Pass criterion: recall >= 0.99 at alpha=2 across difficulty buckets.
```

### Sketch — T2.B: launching m22 K=2 (no new code, script exists)

```bash
.venv/Scripts/python.exe scripts/10_train_m22_horizon.py \
    --warmstart models/m05_bellman_warm/epoch_0499.pt \
    --K 2 --n-epochs 300 --samples-per-epoch 200000 --batch-size 2048 \
    --out-dir models/m22_horizon_K2
# Then strat-5:
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint models/m22_horizon_K2/epoch_0299.pt \
    --out submissions/_m22_strat5.csv \
    --beams 65536 --max-steps 150 --bf16 \
    --stratified 5 --strat-seed 0
```

### Sketch — T2.C: V + π dual-head (rewrite of 11_train_policy_head.py)

The current `11_train_policy_head.py:129` has `output_dim=n_gen` (24, policy logits) — there is no V output, so beam scoring `V + λ·(-log π)` doesn't work as-is. Two rewrites:

1. **Modify `ResMLPDistance`** (in `src/cayley/model.py`, NOT in repo) to support a tuple `output_dim=(1, n_gen)` returning two tensors. If editing cayley is undesired, take option 2.
2. **Two parallel heads on top of the body**: keep the existing model class, instantiate a wrapper that owns the body + two heads.

Wrapper sketch (lives in a new file `scripts/11b_train_v_pi.py`):

```python
import torch.nn as nn

class VPiHead(nn.Module):
    """Wraps a ResMLPDistance body and exposes V (scalar) and π (n_gen logits) heads."""
    def __init__(self, body: nn.Module, body_hidden: int, n_gen: int):
        super().__init__()
        self.body = body            # use ResMLPDistance with output_dim=body_hidden, no head
        self.v_head = nn.Linear(body_hidden, 1)
        self.pi_head = nn.Linear(body_hidden, n_gen)
    def forward(self, s):
        h = self.body.encode_and_residual(s)   # NB: depends on body internals; if cayley body
                                               # only exposes forward(), insert a hook earlier
        return self.v_head(h).squeeze(-1), self.pi_head(h)
```

Training loss:

```python
v_pred, pi_logits = model(batch_states)
loss_v = F.mse_loss(v_pred, target_v)              # standard Bellman target
loss_pi = F.cross_entropy(pi_logits, action_labels) # from inverse-walk labels
loss = loss_v + args.policy_weight * loss_pi
```

Beam integration in `03_solve.py` requires a new `--policy-weight λ` arg and a code path that, when the model exposes both heads, scores children as `V(child) - λ·log_softmax(pi_logits(parent))[a]`. This is a substantial change to the qshort beam loop; treat as 1 day of careful work.

### Sketch — T2.D: tail re-solve in `post_process.py`

Append to `src/megaminx/post_process.py`:

```python
@torch.no_grad()
def tail_resolve(
    path: list[str],
    puzzle,
    model,
    device: str,
    suffix_len: int = 20,
    beam_width: int = 4096,
    max_steps: int = 30,
) -> list[str]:
    """If the path's last `suffix_len` moves can be replaced by a shorter beam-search
    suffix that brings the same intermediate state to solved, return the rewritten path.
    Otherwise return path unchanged.
    """
    if len(path) <= suffix_len:
        return path
    prefix, suffix = path[:-suffix_len], path[-suffix_len:]
    # Reach the intermediate state by applying prefix to solved-of-original.
    # Caller invariant: applying full path to its initial state reaches solved.
    # Equivalent: applying suffix^{-1} to solved gives the intermediate state.
    inv_suffix = puzzle.invert_path(suffix)
    intermediate = puzzle.apply_path(puzzle.solved_state, inv_suffix)
    # Now run a short narrow beam from `intermediate` to solved.
    from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
    solver = KhoruzhiiSolver(puzzle, model, device=device)
    cfg = KhoruzhiiSearchConfig(beam_width=beam_width, num_steps=max_steps, num_attempts=1)
    found, plen, names = solver.solve(intermediate, cfg)
    if not found or plen >= len(suffix):
        return path
    new_path = prefix + names
    # Verify (don't trust solver — paranoid check).
    cur = puzzle.apply_path(puzzle.solved_state, puzzle.invert_path(path))
    end = puzzle.apply_path(cur, new_path)
    if not puzzle.is_solved(end):
        return path  # numerical glitch; keep original
    return new_path
```

Wire into `full_post_process`:

```python
def full_post_process(path, puzzle=None, bfs_table=None, max_window=None,
                      tail_model=None, tail_device="cuda"):
    prev = list(path)
    while True:
        cur = reduce_same_face_runs(prev)
        cur = cancel_adjacent_inverses(cur)
        if puzzle is not None and bfs_table is not None:
            from cayley.post_process import reduce_factor_via_bfs_table
            cur = reduce_factor_via_bfs_table(cur, puzzle, bfs_table, max_window=max_window)
        if tail_model is not None:
            cur = tail_resolve(cur, puzzle, tail_model, tail_device)
        if cur == prev:
            return cur
        prev = cur
```

### Sketches — T3.A/B/C/D/E (research bets)

These are not implemented but the API surface is sketched so we can spike them when ready.

**T3.A test-time per-puzzle Bellman refinement**: a new function `adapt_to_puzzle(model, state, n_steps=100, lr=1e-4)` that clones the model, runs `n_steps` of self-supervised Bellman on random walks emanating from `state`, then uses the adapted model for the final solve.

**T3.B macro-Q**: extend `09_train_q_shortlister.py:130` to set `output_dim = n_gen + n_macros`. Mining macros: scan all submission paths, count unigrams/bigrams/trigrams of move names, pick top N most-frequent that aren't already same-face reducible.

**T3.C IDA* with PDB**: brand new search loop in `src/megaminx/ida_star.py`. PDB build via `scripts/build_pdb_subgroup.py` (BFS over a coset projection). Multi-week.

**T3.D BFS-d7 partial slice**: extension of `scripts/build_bfs_bytes.py` with a subgroup-restricted enumeration (e.g., fix one face's solved-state). Memory-bound research.

**T3.E learned bidirectional**: trains `h(s_fwd, s_bwd)` to predict number of moves between arbitrary state pairs. Brand new training pipeline; brand new beam-loop variant.

