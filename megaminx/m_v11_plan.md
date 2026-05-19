# m_v11 — bigger V trunk scale-up (Track B of the v5 experiment)

**Launched**: 2026-05-18 19:25 UTC on cayley-gpu (L4), tmux session `mv11`.

## Hypothesis

The 6M cluster ceiling at strat-5 mean 87-89 may be capacity-bound, not
recipe-bound. Rule 14 falsified the m_dd_v0 recipe at 20.5M. This run
tests an intermediate size (11.8M, 2× baseline) under a recipe that
breaks the m_dd_v0 pattern via Rule 14's explicit advice:
**lower lr + no PDB penalty**.

## Recipe diffs vs m_dd_v_big

| param | m_dd_v_big (regressed) | m_v11_bellman | rationale |
|---|---|---|---|
| hidden_dims | (4096, 1024) | (3072, 768) | smaller scale-up — 2× baseline, not 3.4× |
| num_res_blocks | 4 | 3 | matches the trunk-width reduction |
| params | 20.5M | 11.8M | within user's 10-12M target |
| lr (Bellman) | 5.0e-4 | 2.0e-4 | slower learning at bigger capacity (Rule 14) |
| lambda_pdb | 5.0 | 0.0 | no PDB penalty (Rule 14 — V0 anchor still pins V(V0)=0) |
| seed | 47 | 80 | independent run |

Everything else matches m_dd_v_big: 200 epochs, batch 8192, k_max 80,
target update every 10, BFS-d6 10%, frontier 25%, V0 anchor 32, d=1 anchor 4.

## Two-stage chain

1. **Stage 1a** — `02_train.py --config m_v11_pretrain.yaml` — pure
   random-walk MSE pretrain at the bigger arch (50 ep). Output:
   `models/m_v11_pretrain/epoch_0049.pt`.
2. **Stage 1b** — `60_train_admissible.py --config m_v11_bellman.yaml` —
   Bellman + anchor refine, warm from Stage 1a (200 ep). Output:
   `models/m_v11_bellman/epoch_*.pt` every 25 ep.

Chained with `&&` in the tmux session — Stage 1b only runs if 1a exits 0.

Stage 2 (AZ-style fine-tune on the 75,200 dataset, warm from Stage 1b)
is a **separate, later** launch after Stage 1b's best checkpoint is
identified by beam bench.

## Revised wall-clock estimate

Observed at startup: Stage 1a runs at **1.2s/epoch** post-compile (no
Bellman target net forward). Revised estimates:

- Stage 1a: ~90s total (was 2-3h estimate — off by 100×)
- Stage 1b: ~40-50s/epoch × 200 = **~2.5-3h** total
- Stage 2 (future): ~1-2h at 25-ep cap

Total Stage 1: ~3 hours, not 36 hours.

## Decision gate (after Stage 1b)

Validate every 25-ep checkpoint with `61_eval_v_at_solved.py` (V0/d=1/d=6
calibration) + 10-pid beam bench (per Rule 12 — loss alone is unreliable
for beam quality).

| Outcome | Action |
|---|---|
| Bench ≥ AZ v4 baseline (10/10, ≤874 total on the 10-pid set used in az_v4_breakthrough.md) at any checkpoint | Promote that ckpt as Stage 2 warmstart. Launch Stage 2. |
| Best bench worse than AZ v4 but better than m_dd_v0 baseline | Capacity-bound diagnosis partially confirmed. Investigate larger sweep before Stage 2. |
| Best bench regresses vs AZ v4 at every checkpoint | Capacity-bound diagnosis FAILED at 11.8M. Combined with Rule 14 (20.5M failed), conclude 6M is genuinely the right scale for this puzzle. Document and pivot to other levers (symmetry training, beam search improvements). |

## Abort criteria during Stage 1b

- V calibration drifts: |V(V0)| > 0.5 or V(d=1) outside [0.5, 1.5]
- Training loss explodes (>10× initial) for 5 consecutive epochs
- Bench at ep 50 is more than 2× worse than m_dd_v0 baseline (catastrophic regression)

Kill via `tmux kill-session -t mv11` on GCP.

## Monitoring commands

```bash
# Status check
gcloud compute ssh cayley-gpu --zone=us-east1-b --command='
  tmux ls
  tail -5 ~/cayley/megaminx/models/m_v11_pretrain_training.log
  tail -5 ~/cayley/megaminx/models/m_v11_bellman_training.log 2>/dev/null
  ls -la ~/cayley/megaminx/models/m_v11_bellman/*.pt 2>/dev/null
'

# Attach to tmux (live tail)
gcloud compute ssh cayley-gpu --zone=us-east1-b -- -t 'tmux attach -t mv11'

# Done marker
gcloud compute ssh cayley-gpu --zone=us-east1-b --command='cat /tmp/mv11_done 2>/dev/null'
```

## Files

- Local: `megaminx/configs/m_v11_pretrain.yaml`, `megaminx/configs/m_v11_bellman.yaml`
- GCP: `~/cayley/megaminx/configs/m_v11_*.yaml`, output at `~/cayley/megaminx/models/m_v11_*/`
- Plan: `megaminx/m_v11_plan.md` (this file)
