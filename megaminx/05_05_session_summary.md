# Session 2026-05-04 / 2026-05-05 — summary

## Net change

**78,408 → 78,408** (no change from session work alone; user submitted 78,408 mid-session via community-pushed kernel merge).

Score progression observed during this window:
- 79,522 (start of 2026-05-04)
- 79,486 (in-flight ensemble min-merge — NOT submitted, too small)
- **78,408** (community-pushed kernel merge — user-driven, 2026-05-05)

## Training-side track: empirically airtight closed

This session conclusively closed the training-side experimental track. Across **44× param range** (1.12M → 49.6M planned) and every recipe family we've tried:

| Recipe family | Models | Result |
|---|---|---|
| 14 Bellman recipe variants | m17→m43 (6M-13M) | All cluster (88-102 mean) |
| Tiny scale | m44 (1.12M) | Cluster (47/51 / 102) |
| Wide arch | m26b, m39a, m45 (pending) | Cluster (m45 still running) |
| Distributional V | m42 (median + lower-q25) | Cluster (48/51 / 95-97) |
| Solver-trace primary | m37 | **Catastrophic** (3/51) |
| Solver-trace MIXIN at 25% | m43 | **Catastrophic** (0/51) |
| Soft-Bellman | m34 | **Catastrophic** (0/51) |
| Listwise rank loss | m38 | Cluster (48/51 / 99) |
| L_upper penalty | m40 | Cluster (48/51 / 99) |
| TTT v1 (Akyurek-style) | T1.3 | Null (within noise) |
| TTT v2 (DAGGER-style) | T1.3 v2 | Null (mechanism N/A on hard tail) |

**Conclusion**: cluster ceiling is information-bound at this data scale. No single-model recipe variant breaks through. Path to <70K must come from **inference compute** (sym-ensemble, multi-agent T1.2) or **search-space changes** (T1.1 macros, T2.4 Minkwitz).

## Macros revisited (T1.1 Phase 2)

Phase 1 had 5 surviving macros; Phase 2 broadened the scrape to 54 algorithms across 9 categories → **45 unique surviving perms** (4% collapse rate vs Phase 1's 50%, validating broader-source approach).

Hamming distribution: 28 at h=18 (4-move adjacent commutators, same regime as h18 filter we already rejected), 3 at h=12 (sledge-3x family — minimum disruption), 14 at h=25-54 (Sune/Antisune/T-perm/Y-perm/CP).

A/B test on top-50 longest pids of 79,946:
- Curated A: 24/50 improved, **−79 moves**, macro_insert 1.2% accept rate
- Brute-force B (killed early at 10/50): 2/10 improved, −2 moves, 0% macro_insert

**Curated DOES outperform brute-force**, but absolute gain is modest (-1.6 moves/pid) at 17h+ wall. Not the −10K to −20K the tier doc promised. **T1.1 mechanism kept in production but accepts modest contribution; don't deepen the multi-day scrape.**

## In-flight at session end

**m45 (49.6M large V) on GCP** — pretrain DONE (final loss 63.26), Bellman warmstart in progress at ~146s/epoch on L4. ETA ~17h Bellman + ~100 min strat-5. Final capacity-headroom test. **Picks up next session.**

| Decision branch | Action |
|---|---|
| m45 lands at cluster | Information-bound theory bulletproof across 44× param. Pivot fully to T1.2 multi-agent + T2.4 Minkwitz + curated macro deepening |
| m45 breaks cluster (≥+3 solves AND mean ≤84.9) | Investigate scaling further, possibly m46 transformer |

m46 transformer (~30M params, GCP, 3-4 days) is **decision-gated on m45**. Plan filed at `m46_transformer_plan.md`.

## Files added/modified this session

**New scripts**:
- `megaminx/scripts/37_ttt_solve.py` — TTT v1 (Akyurek-style)
- `megaminx/scripts/38_train_distributional_v.py` — m42 QR-DQN training
- `megaminx/scripts/39_curated_to_table.py` — Phase 2 macros → SA-compatible format
- `megaminx/scripts/40_ttt_self_distill.py` — TTT v2 (DAGGER-style)
- `megaminx/scripts/run_after_m43.sh` — chain runner
- `megaminx/scripts/run_m44_chain.sh`, `run_m45_chain_gcp.sh`, `watch_and_launch_*.sh` — auto-launch infra

**New configs**:
- `m40_upper_penalty.yaml` (L_upper)
- `m42_distributional.yaml` (QR-DQN)
- `m43_solver_trace_mixin.yaml` (25% mixin)
- `m44_pretrain.yaml` + `m44_bellman.yaml` (tiny 1.12M)
- `m45_pretrain.yaml` + `m45_bellman.yaml` (large 49.6M)

**New library code**:
- `src/cayley/distributional.py` — quantile-Huber loss + distributional Bellman target
- `src/cayley/bellman.py` — added `lambda_upper`, `solver_trace_path` + `solver_trace_fraction` fields
- `megaminx/scripts/03_solve.py` — added `--strat-buckets` flag (hard-tail eval) + `--quantile-reduce {median,mean,lower-q25,...}` for distributional V

**New plan/journal docs**:
- `m40_plan.md` — L_upper plan and decision matrix
- `05_03_priority_plan.md` — T2.4/T1.3/T2.5/etc. ranking
- `05_04_next_phase_plan.md` — T1.4/T2.4/T1.3 v2 plans after m37 catastrophe
- `m46_transformer_plan.md` — transformer scoping
- `data/named_macros_phase2.yaml` (54 algorithms)
- `data/curated_macros_phase2.pkl` (45 surviving perms)
- `data/curated_table_phase2.pkl` (commutator-table format for SA)

**New checkpoints (cluster fodder for ensembling)**:
- `models/m40_upper_penalty/epoch_0499.pt` (~6M)
- `models/m42_distributional/epoch_0499.pt` (~6M, 32-quantile output)
- `models/m43_solver_trace_mixin/epoch_0499.pt` (~6M, but V is broken — don't use)
- `models/m44/{pretrain,bellman}/*.pt` (~1.12M)
- `models/m45/{pretrain,bellman}/*.pt` (~49.6M, m45 still training)

## Recommended next-session priorities

1. **Pull m45 result first** — single most informative remaining experiment.
2. **If m45 clusters**: pivot fully to T1.2 multi-agent ensemble (we have 9+ cluster-V members; CayleyPy paper validated -2K to -5K). Hard-tail-only at ~17h on 4090 OR multi-platform.
3. **T2.4 Minkwitz proper port** — 3-5 day commitment, additive-only. Worth doing in parallel with multi-agent.
4. **T1.6 SA full-1001 sweep** with curated phase-2 macros — full coverage of 79,522/78,408 base. ~5-7 days L4.
5. Continue periodic TPU runs whenever quota refreshes (next Mon).

## Tasks still pending at session end

- #6 Alexander's beam-frontier Bellman TTT — never started
- #15 T1.1 macro scrape — Phase 2 done, modest contribution; deepening deferred
- #37 T2.4 Minkwitz — needs multi-day port
- #41 m45 — RUNNING on GCP
- #42 m46 transformer — gated on m45

All other m-series tasks (m34→m44) marked completed.
