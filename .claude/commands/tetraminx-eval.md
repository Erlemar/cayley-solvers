---
description: Run the standard tetraminx 15-pid TPU beam eval (guarded, resumable) and score it against FINAL_28561
---

Run one tetraminx beam evaluation on the TPU with the house protocol, then score it
against the standing best. Arguments: `$ARGUMENTS` (e.g. `ep1500 4M 0i` or
`ep1200 16M 0f,0i,1f,1i +cons +blend`).

## Protocol (do not vary these unless asked — comparability depends on it)

- **pids**: `0,50,100,200,300,400,500,600,700,800,900,950,990,995,999` (the 15-pid set)
- **history-depth 1** (measured ceiling: h4 is byte-identical, h0 is +1)
- **num-steps 60**, endgame table on, every path replay-verified
- default frame is `0f` ONLY for comparability with the ep700..ep1500 checkpoint curve;
  for a *quality* run use `--frame-spec 0i` — the inverse frame measured -2 at 4M and is
  ~9% faster (CLAUDE.md / HANDOFF "Frames" section)

## Do this

1. **Resolve the checkpoint.** If it is not already on the TPU, copy it from A100-b via
   local (checkpoints live at `~/cayley/tetraminx/models/mx_tf_az/epoch_NNNN.pt`, land on
   the TPU as `~/tetra/mx_tfaz_epNNNN.pt`). Verify with `sha256sum` on both ends.

2. **Check for kernel drift before any run that changes width class** — local is the
   source of truth and has been AHEAD of the TPU before:
   ```
   sha256sum tetraminx/kaggle_notebooks/tpu_beam_tetraminx/{jax_beam_spmd_v_only,jax_model}.py
   ssh tpu "sha256sum ~/tetra/{jax_beam_spmd_v_only,jax_model}.py"
   ```
   Re-scp if they differ. A stale kernel silently drops history + endgame above 16M.

3. **Launch in tmux with the MUTUAL busy-guard** so it cannot collide with the auto-eval
   watcher or another beam (both sides need the guard — see `jax_tpu_gotchas`):
   ```bash
   while ps -eo cmd | grep -q '[g]cp_beam_tetraminx'; do sleep 60; done
   ~/tpu-env/bin/python -u ~/tetra/gcp_beam_tetraminx.py \
     --checkpoint ~/tetra/<ckpt>.pt --tag <tag> --data-dir ~/tetra \
     --b-global <B> --frame-spec <frames> --num-steps 60 --history-depth 1 \
     --pids "$PIDS" --out ~/out/<tag>.json --tree-dir /dev/shm/trees
   ```
   Optional, both measured positive and near-free:
   `--qv-consistency 0.3` (needs an az_head checkpoint; -4, additive with history)
   `--blend ~/tetra/mx_resmlp_az.pt --blend-weights 0.8 0.2` (~1.07x cost, -2)

4. **Report totals AND averages** — the driver prints `total / avg / avg>1`; `avg>1`
   excludes the trivial pid 0 and is the honest per-puzzle number.

5. **Score against the standing best**, never a remembered floor (rule 26):
   ```
   scp tpu:~/out/<tag>.json <scratch>/json/ && \
   .venv/Scripts/python.exe tetraminx/scripts/56_compare_vs_final.py <scratch>/json/<tag>.json
   ```
   Read the **merge** column, not `diff`: `FINAL_28561` is a min-merge of mostly
   8M x 4-frame ResMLP runs, so a small positive `diff` at 1 frame is parity per unit
   compute, not a regression. `merge` is what would actually bank into a submission.

## Timing (v6e-8, transformer Q head, per pid per frame)

| B | ~s/pid/frame |
|---|---|
| 1M | 40 |
| 4M | 184 |
| 8M | 370 |
| 16M | 740 |

15 pids x 4 frames at 16M is ~12 h. Say the ETA before launching anything over ~2 h, and
check whether it will block a queued checkpoint eval.

## Do NOT

- Compare a local PyTorch run to a TPU number — different implementations differ by ~3
  moves at the same config. Run the matched control (CLAUDE.md rule 28).
- Raise `--history-depth` above 1 expecting gains; h4 measured byte-identical to h1.
- Trust probe metrics (pair/top1/gap) to pick a checkpoint — they have failed to predict
  beam quality four times now. Only a replay-verified beam total counts.
