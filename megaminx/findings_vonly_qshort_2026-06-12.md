# Findings — 8-chip v6e + qshort-hurts / V-only win (2026-06-12)

## TL;DR
- **qshort was inflating every TPU beam.** Running the V-only kernel (no Q model) on pid 992
  gives **78 moves** vs the V+qshort kernel's **96** — beating the merged floor of **85 by −7**.
- Banked: `submissions/merge_v15_vonly_pid992.csv` = **75,193** (was 75,200; pid 992 85→78, verified).
- **Implication:** our entire TPU-derived submission used qshort → a V-only re-run likely improves
  many hard pids, not just 992. This is the score lever, not wider fresh beams.

## 1. 8-chip v6e provisioning — FIXED via Queued Resources API
- `gcloud compute instances create --machine-type=ct6e-standard-8t --provisioning-model=FLEX_START`
  is **backend-broken**: reaches STAGING, runs ~14 min, then `INTERNAL_ERROR` / HTTP 503 /
  gRPC-13 `BACKEND_ERROR` (code -759774172771137012) and self-deletes with no retry. Reproduced
  **3/3** (europe-west4-a, us-east5-a, us-east5-b) → it's the single-host 8-chip topology, NOT
  zone/quota (CT6E quota in us-east5 is now 16; capacity reached STAGING).
- **Working path:** the Cloud TPU **Queued Resources API**, which queues + auto-retries the
  bring-up:
  ```
  gcloud components install alpha --quiet
  gcloud alpha compute tpus queued-resources create mm-v6e8-qr \
    --project=gen-lang-client-0977634337 --zone=us-east5-a \
    --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e \
    --node-id=mm-v6e8 --provisioning-model=flex-start --max-run-duration=24h
  ```
  Came up HEALTHY (`jax.device_count()=8`, "TPU v6 lite") in ~5 min. Yields a **TPU-VM**
  (1.4 TB RAM, 180 vCPU; use `gcloud compute tpus tpu-vm ssh/scp`, not `gcloud compute`).
  jax NOT preinstalled on the runtime; tree on `/dev/shm` tmpfs.
- MIG flex-start resize-requests do NOT support TPU machine types (GPU/H4D only) → QR is the
  TPU equivalent. Teardown: `gcloud alpha compute tpus queued-resources delete mm-v6e8-qr
  --zone=us-east5-a --force --quiet`.

## 2. Root cause of the bad path: qshort, not "hard tail"
- First run: `gcp_beam_v6e.py` (V **+ qshort**, loads m23_v3 Q, `beam_solve_v_qshort_spmd_packed`),
  96M, k-sym1 → **96 moves** (verified). Initially mis-read as a hard-tail ceiling.
- Ruled out post-processing: same-face reduction, adjacent-inverse cancellation, AND the d6
  BFS-window pass each recover **0 moves** → the 96 is globally suboptimal, not locally redundant.
- The only param differing from the production config (B=48M, alpha=2) was **K_SYM** — but the
  real culprit is **qshort**: m23_v3 Q misranks children → prunes good candidates → longer paths.
  This is **Rule 15** (qshort regresses on the AZ v4 V) manifesting in the TPU kernel. Repo
  corroboration: `m_az_v4_strat5_qshort_only.csv` pid992 = 101 (worst of all).

## 3. The fix: V-only kernel
- `gcp_beam_v_only.py` → `jax_beam_spmd_v_only.beam_solve_v_only_spmd_packed` (no Q, no meta,
  parent_chunk streaming; backptr packing already 8-rank-safe = 24-bit parent / 3-bit rank / 5-bit move).
- pid 992 results (single-view, no symmetries):

  | config | pid 992 | wall |
  |---|---|---|
  | qshort 96M k-sym1 | 96 | — |
  | qshort tpu historical (v16/v17 merges) | 88 | — |
  | merged floor (merge_v9..v14) | 85 | — |
  | **V-only 48M** | **78** (verify=True) | 2.4 h |
  | **V-only 64M** | **78** (verify=True) | 3.6 h |

- **Width plateaus at 78** (48M = 64M). Reaching the expected ~73-75 would need **symmetries**
  (best-of-N rotations) or an **alpha** bump — not yet tested (ran out of box time).

## 4. Banked
- `submissions/merge_v15_vonly_pid992.csv` — base = `merge_v14_plus_min_count_v4.csv` (75,200),
  pid 992 replaced 85 → 78. New total **75,193 (−7)**. pid 992 verified solving in the merged file.
- Result JSONs saved: `megaminx/pid992_vonly_48m.json`, `megaminx/pid992_vonly_64m.json`,
  `megaminx/pid992_96m_ksym1.json` (the qshort 96).

## 5. Open / next (NOT done)
- **Submit?** `merge_v15` (75,193) not yet submitted to Kaggle.
- **V-only harvest** (the real opportunity): re-run hard-tail (or full) pids V-only; every shipped
  TPU beam was qshort-built. Needs a fresh QR box + a faster config (48M V-only is ~2.4 h/pid →
  too slow for a big batch as-is).
- **pid 992 follow-up (2026-06-15):** AZ-v5 orbit V-only at 128M global beam, alpha=2, k-sym=1 on
  v6e-8 found a verified 76-move path. This improves the old V-only plateau (78 -> 76), but does
  not beat the newer `half_split_rank201_400_b16k.csv` floor for pid 992 (72). Banked as
  `submissions/pid992_azv5_orbit_vonly_128m_a2_v2.csv`; no merge.
- **Confirm 73-75 remains unproven:** the 128M identity-only run did not reach it. A true test
  would need symmetries and/or a different alpha, but pid 992 is no longer an attractive target
  because half-split repair already has 72.

## State
- No GCP TPU resources running (QR + node deleted, meter stopped).
- Memory updated: `megaminx-v6e-beam-port`, `tpu-builders-program-gcp-setup`, and the
  `/megaminx-tpu-provision` skill (QR 8-chip recipe + qshort-hurts correction + pkill bracket-trick).
- Gotcha logged: `pkill -f "python3 gcp_beam"` matches the SSH session's own wrapper → kills its
  shell; use `pkill -9 -f "[g]cp_beam_v6e.py"` (bracket trick), and never put the pkill and the
  python launch in the same SSH command.
