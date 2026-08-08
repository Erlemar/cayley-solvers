---
name: megaminx-tpu-provision
description: Provision a GCP TPU-Builders v6e flex-start VM, set it up (mount/create data disk, install jax[tpu]+torch, pull megaminx beam artifacts from GCS, verify device count), and run the SPMD beam. Use when resuming the megaminx v6e beam work (e.g. after the 8-chip backend bug is fixed) or spinning up any ct6e flex-start VM. Encodes the flex-start status/fault/stockout handling, the Windows plink-noise filter, and the diagnostics-first discipline learned 2026-06.
---

# Megaminx v6e TPU provision + setup + run

Project `gen-lang-client-0977634337`. Artifacts staged in `gs://mm-tpu-staging-0977634337/v6e/`
(edited kernel `jax_beam_spmd_v_qshort.py`, sym driver `gcp_beam_v6e.py`, `jax_model.py`, both
models, `rotations_720.npy`, `puzzle_info.json`, `test.csv`, `hashtest.py`, `vtest.py`). See
memory `[[megaminx-v6e-beam-port]]` + `[[tpu-builders-program-gcp-setup]]` for the full backstory.

**Every `gcloud compute ssh` call** must end with this filter (Windows plink noise):
`2>&1 | grep -vE "host key|fingerprint|cache|trust|connecting|Return|guarantee|Store key|Warning|ssh-ed25519|troubleshoot"`

## 1. Pick zone + machine type, check quota

- v6e hardware + CT6E quota=48 zones: **europe-west4-a** (capacity, but 8-chip backend-faults as of 2026-06), asia-northeast1-b, asia-east1-c, asia-southeast1-b, asia-south1-b. The program zones us-east5-a / us-central1-a have **0 CT6E quota** (don't use).
- `ct6e-standard-4t` = 4 chips (reliable), `ct6e-standard-8t` = 8 chips (2x ranks → super-linearly faster wide beams, but GCP-blocked in europe as of 2026-06).
- Quota check: `gcloud beta quotas info describe TPUS-PER-TPU-FAMILY-per-project-region --service=compute.googleapis.com --project=gen-lang-client-0977634337 --flatten="dimensionsInfos[]" --format="csv[no-heading](dimensionsInfos.dimensions.region,dimensionsInfos.dimensions.tpu_family,dimensionsInfos.details.value)" | grep CT6E`

## 2. Match the BPTR field to the rank count (edit kernel before staging)

In `jax_beam_spmd_v_qshort.py`: **4 ranks → 25/2/5** (`BPTR_PL_BITS=25, BPTR_RANK_BITS=2`); **8 ranks → 24/3/5** (`BPTR_PL_BITS=24, BPTR_RANK_BITS=3`). Both give B_global ≤ 134M. Wrong rank bits = silent walkback corruption (the validation raises if `world_size > 1<<BPTR_RANK_BITS`). Re-stage after editing: `gcloud storage cp <kernel> gs://mm-tpu-staging-0977634337/v6e/`.

## 2b. For 8 chips: use the Queued Resources API, NOT `instances create`

Single-host `ct6e-standard-8t` FLEX_START via `gcloud compute instances create` is backend-broken: it reaches STAGING, grinds ~14 min, then fails `INTERNAL_ERROR`/503/`BACKEND_ERROR` and self-deletes with no retry (reproduced 3/3 across europe-west4-a + us-east5-a/b, 2026-06-11 — it's the 8-chip topology, not zone/quota). **Use the Cloud TPU Queued Resources API instead** — it queues + auto-retries the bring-up:
```
gcloud components install alpha --quiet   # one-time
gcloud alpha compute tpus queued-resources create mm-v6e8-qr --project=gen-lang-client-0977634337 \
  --zone=us-east5-a --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e \
  --node-id=mm-v6e8 --provisioning-model=flex-start --max-run-duration=24h
```
Watch `queued-resources describe mm-v6e8-qr --format="value(state.state)"`: `WAITING_FOR_RESOURCES`→`PROVISIONING`→`ACTIVE` (~5 min; came up HEALTHY in us-east5-a, CT6E quota now 16). This yields a **TPU-VM** — use `gcloud compute tpus tpu-vm ssh/scp/describe`, not `gcloud compute instances`. jax is NOT pre-installed on `v2-alpha-tpuv6e` (pip `jax[tpu]`). First `tpu-vm ssh` on Windows: pipe `printf 'y\n' |` to cache the plink host key. Teardown: `gcloud alpha compute tpus queued-resources delete mm-v6e8-qr --zone=us-east5-a --force --quiet`. (MIG flex-start resize-requests are GPU/H4D-only — no TPU; QR is the TPU equivalent.) The §3/4 machine-type path below stays correct for 1t/4t.

## 3. Provision (async) + watch

```
gcloud compute instances create mm-v6e-<tag> --project=gen-lang-client-0977634337 --zone=<ZONE> \
  --machine-type=<ct6e-standard-4t|8t> --provisioning-model=FLEX_START \
  --request-valid-for-duration=2h --max-run-duration=24h --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE --image-project=ubuntu-os-accelerator-images \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --async
```
- `--request-valid-for-duration` MAX is **7200s (2h)**. FLEX_START is **$0 while PENDING**.
- **`--max-run-duration` goes up to 7 DAYS -- `7d` is accepted (verified 2026-07-30).**
  The `24h` used elsewhere in this file is a cost backstop we chose, NOT a platform
  limit; do not treat it as a ceiling. Size it to the job: a 46h tetraminx beam was
  launched at `48h` and came within ~2h of being killed mid-run with 4 pids unfinished,
  purely because 24h was assumed to be the max and 48h was picked as a timid step up.
  For any run over ~12h use `7d` and delete the VM when done -- the duration is a
  backstop against forgetting, not a budget.
- Watch in a `run_in_background` Bash loop polling `gcloud compute instances describe ... --format="value(status)"` every 30s. Status semantics: `PENDING`=queued for capacity; `STAGING`=booting; → `RUNNING` (success). **`STAGING`→gone = INTERNAL_ERROR backend fault** (check `gcloud compute operations describe <op>`); **immediate "not found" / `ZONE_RESOURCE_POOL_EXHAUSTED`** = stockout (try another zone). Record whether STAGING was reached to tell fault from stockout.

## 4. Setup (once RUNNING)

Data disk: v6e/ct6e **rejects pd-\* disks — must be hyperdisk**. Either create+attach `--type=hyperdisk-balanced` and `mkfs.ext4`, or attach an existing one at create with `--disk=name=...,auto-delete=no` (survives the flex VM). Then, in ONE ssh (nohup the slow installs):
```
sudo mkdir -p /mnt/data && sudo mount -o discard,defaults /dev/disk/by-id/google-<devname> /mnt/data && sudo chmod -R a+rwx /mnt/data
sudo apt-get update -qq && sudo apt-get install -y -qq python3.10-venv python3-pip   # update BEFORE venv or ensurepip fails
python3 -m venv ~/tpu-env
~/tpu-env/bin/pip install -q -U pip 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
~/tpu-env/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu   # for load_params_from_pt
# glibc guard: this image family is Ubuntu 22.04 (glibc 2.34), so wheels run. On ANY older base
# (e.g. a shared 20.04 box, glibc 2.31) check `ldd --version` FIRST: torch_xla >=2.5 needs glibc
# 2.34, jaxlib only ~2.27 -> JAX installs+runs but modern torch_xla won't import. See [[tpu-v4-research-box]].
mkdir -p /mnt/data/v6e /mnt/data/trees /mnt/data/out
gcloud storage cp 'gs://mm-tpu-staging-0977634337/v6e/*' /mnt/data/v6e/
~/tpu-env/bin/python -c 'import jax,torch; print("dev",jax.device_count(),jax.devices()[0].device_kind)'   # expect 4 or 8, "TPU v6 lite"
```
Note: v6e exposes chips via **vfio**, so `/dev/accel*` is EMPTY — that's normal; trust `jax.device_count()`.

## 5. Validate, then run

ALWAYS run the two diagnostics first (catches config-gap-vs-real-bug in minutes):
`~/tpu-env/bin/python /mnt/data/v6e/hashtest.py` (host==device int64 hash) and `vtest.py` (V(V0)≈0, monotonic). Then the beam — **default to the V-ONLY driver**:
```
cd /mnt/data/v6e && nohup ~/tpu-env/bin/python gcp_beam_v_only.py \
  --b-global <B> --num-steps 100 --start-pid <a> --end-pid <b> --parent-chunk 131072 \
  --tree-dir /dev/shm/trees --out /mnt/data/out/run.json > /mnt/data/out/run.log 2>&1 &
```
- **USE `gcp_beam_v_only.py` (V-only), NOT `gcp_beam_v6e.py` (V+qshort), for the AZ v4 V.** qshort (m23_v3 Q) MISRANKS children and INFLATES paths: confirmed 2026-06-12 pid 992 = **96 (qshort 96M) vs 78 (V-only 48M)** — V-only at half the width + no sym beat the merged floor (85) by 7, qshort was +18 worse. This is Rule 15 (qshort regresses on AZ v4 V) in the TPU kernel. `gcp_beam_v_only.py` is NOT in the GCS bucket by default — `scp` it from `megaminx/kaggle_notebooks/tpu_beam_spmd_jax/`. Its kernel `jax_beam_spmd_v_only.py` IS staged (8-rank-safe packing). No `--k-sym`/`--no-meta` flags (single-view, no meta; uses parent_chunk streaming).
- V-only single-view **plateaus on width** (pid 992: 48M=64M=78); to push lower needs sym diversity or higher `--alpha`. Per-step ~110s at 48M streaming (~2.4 h/pid), ~170s at 64M.
- **FAST-SOLVING puzzles (cube444/555/666, many easy pids) MUST enable JAX's persistent compilation cache** or a run is mostly recompile. The beam rebuilds its step_fn closure per `run_beam` call, so the in-memory jit cache MISSES every call and re-compiles (~90–110 s at 2^20–2^22 on v6e-8). megaminx never notices (hour-long pids amortize it), but a puzzle that solves in seconds spends ~98% of wall recompiling — a full sym-4 run balloons to ~5 days. Fix in the driver, right after `import jax`: `jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")` + `jax_persistent_cache_min_entry_size_bytes=0` + `jax_persistent_cache_min_compile_time_secs=0`. It's keyed by HLO hash (not callable identity), so a fresh jit that lowers to identical HLO HITS the on-disk cache: first call ~105 s, every later call ~2 s (~50×). Confirmed 2026-07-24 on cube444 (`cube444/tpu/gcp_beam_cube444.py` + RUNBOOK.md). Keep the cache dir across pids AND reuse it across runs at the same width (warm-starts the first compile too).
- The qshort+sym driver `gcp_beam_v6e.py --k-sym 4 ... --no-meta` is retained ONLY if a future re-distilled Q is shown to help; until then it's a path-inflating trap. Wall is super-linear in **B_local** (~`B_local^1.8`); 8-chip 96M needs meta (no-meta ceiling ≈64M) at ~222 s/step.

## 6. Teardown

`gcloud compute instances delete <name> --zone=<ZONE> --quiet` stops the meter (flex VMs can't be "stopped", only deleted). Detach/keep the hyperdisk with `auto-delete=no` (or delete it for $0 — GCS is the source of truth). Verify `gcloud compute instances list` shows nothing of ours running.
