# Megaminx TPU beam search — AZ v4 V-only (shareable)

Sibling of `tpu_beam_b2m_k2` (the original `cayleypy-megaminx-beam-shareable`)
but built around the **AZ v4 V-only** model (`m_az_v4_v_only.pt`, 6M params)
and with **no qshort**. Multi-collaborator fan-out over Kaggle TPU quotas.

## Why a new variant

The original shareable runs `m05` V teacher + `m23_v2` Q-shortlister
(qshort). `m23_v2` was distilled from `m05`'s V landscape, so its
top-k action ordering is calibrated to `m05`, not AZ v4. Strat-5
measurements (2026-05-11):

| stack | total | mean |
|---|---|---|
| AZ v4 V-only (KhoruzhiiSolver) | **4,465** | **87.5** ← clean win |
| AZ v4 V + AZ π (λ=0.05) + m23_v2 qshort | 4,621 | 90.6 |
| AZ v4 V + m23_v2 qshort, no policy | 4,705 | 92.3 |

`m23_v2` qshort regresses AZ v4 V by ~240 moves. So this notebook drops
qshort and uses AZ v4 V to rank ALL `B·N_GEN` neighbors per step.

## Files

- `build_notebook.py` — generates `cayleypy-tpu-beam-az-v4-v-only.ipynb`.
  Embeds all 1001 test pid states.
- `cayleypy-tpu-beam-az-v4-v-only.ipynb` — the generated notebook (~428 KB).
- `kernel-metadata.json` — Kaggle kernel id + dataset link.

## Running locally to update the notebook

```bash
.venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_az_v4_v_only/build_notebook.py
```

Re-run any time `build_notebook.py` changes. Then `kaggle kernels push -p .`
from inside the directory to update the Kaggle copy.

## Required dataset state

The notebook reads from `artgor/megaminx-tpu-artifacts`. **That dataset
must contain `m_az_v4_v_only.pt`** — uploaded as a new version with the
existing files plus this checkpoint. Cell 5 will SystemExit with a
helpful message if it's missing.

To upload (one-time, owner-side):

```bash
# Pull current files into a staging dir, add m_az_v4_v_only.pt,
# write a dataset-metadata.json with id=artgor/megaminx-tpu-artifacts,
# then:
.venv/Scripts/kaggle.exe datasets version -p /path/to/staging \
  -m "add m_az_v4_v_only (AZ v4 V-only, breakthrough 51/51 standalone)"
```

## Workflow for collaborators

### 1. Fork the notebook on Kaggle

Open **https://www.kaggle.com/code/artgor/cayleypy-megaminx-beam-az-v4-shareable**
and click "Copy and Edit".

### 2. Switch accelerator to TPU v3-8 (REQUIRED)

The pushed kernel may be in CPU mode (owner pushes with `enable_tpu: False`
when TPU quota is exhausted). **Each collaborator must manually flip the
accelerator** in their fork:

- Right sidebar → "Accelerator" → select **"TPU v3-8"**.
- If it's grayed out, the collaborator's own TPU quota is also exhausted.

Without this step the notebook will crash on `import torch_xla` or hang
trying to find TPU devices.

### 3. Edit the config cell (Cell 1)

```python
START_PID = 0       # inclusive
END_PID = 50        # exclusive
K_SYM = 4           # 1, 2, 4, or 8 — rotations per pid
B_TEST = 131072     # 65536, 131072, 262144, or 524288 — beam width
```

Cell 2 prints an estimated wall on launch. If it warns "exceeds 8.5h
target", reduce the pid range, K, or B.

Recommended scope per kernel at default `B_TEST=131072` (~120s/pair cached):

| K | recommended pids | per-rank pairs | wall |
|---|---|---|---|
| 2 | 750 | 188 | 0.5h + 6.3h = 6.8h |
| **4** | **375** | **188** | **0.5h + 6.3h = 6.8h** ← default |
| 8 | 188 | 188 | 0.5h + 6.3h = 6.8h |

The wall is more conservative than the original shareable's ~8.3h because
V-only's per-pair wall at B=131k is somewhat below qshort's at B=1M
(measured anchor TBD; current projection extrapolates from B=1M qshort).

Suggested splits for full-1001 at default K=4 B=131k across 3 collaborators:

| collaborator | START_PID | END_PID | pids | est. wall |
|---|---|---|---|---|
| A | 0 | 334 | 334 | 0.5h + 5.6h = 6.1h |
| B | 334 | 667 | 333 | 0.5h + 5.6h = 6.1h |
| C | 667 | 1001 | 334 | 0.5h + 5.6h = 6.1h |

If a collaborator has time to spare, bumping `B_TEST` to 262144 doubles
per-pair wall but gives the strongest V wider search.

### 4. Verify accelerator

In the right sidebar: Accelerator = "TPU v3-8". If it says GPU or None,
change it before running.

### 5. Run All

Per-iteration partial saves to `/kaggle/working/rank_<r>_partial.json`
recover everything if Kaggle hits the 9h kill mid-run.

### 6. Send results back

After the kernel completes (or hits the 9h kill), download these from
`/kaggle/working/`:

- `tpu_az_v4_results.json` — full per-pair details
- `tpu_az_v4_submission.csv` — best path per pid in this chunk

Send them to the owner for merging via `/megaminx-tpu-merge` (or the
manual aggregate flow).

## Configuration constants

| constant | value | reason |
|---|---|---|
| `B_TEST` | configurable | 65536 / 131072 (default) / 262144 / 524288 |
| `K_SYM` | configurable | 1, 2, 4 (default), or 8 rotations per pid |
| `INTERNAL_BS` | 4096 | keeps `B·N_GEN/4096` chunks/forward in the 500-2000 sweet spot for B≤524k |
| `NUM_STEPS` | 120 | depth at which paths plateau |
| `SYM_SEED` | 0 | deterministic — all collaborators choose the same rotation set |

`INTERNAL_BS=4096` matches v16's measured-safe chunk size. At B=524k
the chunks/forward = 524k·24/4096 = 3072, slightly above the 2000
upper bound — should still be fine but the per-pair wall projection
(linear in B) is the right place to start if it slows down.

## How this differs from the original shareable

| aspect | original (`tpu_beam_b2m_k2`) | this (`tpu_beam_az_v4_v_only`) |
|---|---|---|
| V model | m05 (6M) | AZ v4 V-only (6M) |
| Q shortlister | m23_v2 (12.4M, output_dim=24) | none |
| beam loop | qshort (student picks αB, teacher reranks to B) | V-only (V over all B·N_GEN, pick top B) |
| forwards / step | ~3B (B student + αB teacher) | ~24B (B·N_GEN through V) |
| default B | 1,048,576 | 131,072 |
| INTERNAL_BS | 32,768 | 4,096 |
| Kaggle kernel slug | `artgor/cayleypy-megaminx-beam-shareable` | `artgor/cayleypy-megaminx-beam-az-v4-shareable` |

Same XLA infra (env var pop, xmp.spawn, partial saves), same sym
ensemble, same dataset attachment.

## Troubleshooting

- **"missing artifact: m_az_v4_v_only.pt"**: the dataset attachment is
  an older version. Detach and re-attach the latest version of
  `artgor/megaminx-tpu-artifacts`.
- **"Expected 8 worker addresses, got 1"**: env var pop in Cell 2 didn't
  run. Restart kernel and run all from top.
- **"Runtime is already initialized"**: torch_xla submodule was imported
  in the parent process before xmp.spawn — only happens if you run
  cells out of order. Restart and run all.
- **Kernel hits 9h kill before finishing**: expected for large pid
  chunks. All completed pairs are saved. Send the partial JSONs anyway.
- **TPU sits idle for >2h with no output**: Kaggle kills idle TPU
  kernels. Healthy beam search produces a log line every ~5 min — if
  logs go silent, restart.

For deeper context, see:
- `megaminx/kaggle_notebooks/tpu_beam_smoke/RUNBOOK.md` — operational
  runbook for the original private sweep notebook
- `~/.claude/projects/.../memory/reference_kaggle_tpu_pjrt.md` —
  technical reference on Kaggle TPU PJRT quirks
- `~/.claude/projects/.../memory/az_v4_breakthrough.md` — context on
  why AZ v4 V is the model of choice
