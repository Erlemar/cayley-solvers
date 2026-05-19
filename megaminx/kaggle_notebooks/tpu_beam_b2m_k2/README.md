# Megaminx TPU beam search — B=1M, K configurable (shareable)

> **Version history**:
> - v1: B=2M K=2 → OOM at runtime (480 MB shortlist alloc)
> - v3: B=1.5M K=2 (untested before v4 push)
> - v4: B=1.5M K configurable → OOM at XLA program load (5.58 GB workspace
>   doesn't fit when 2 ranks share a chip's HBM via xmp.spawn)
> - **v5 (current)**: B=1M, K configurable in Cell 1 (default K=4) —
>   the only verified-safe size, matching v19b's known-good config.

Multi-collaborator TPU beam-search notebook for the
[CayleyPy Megaminx Kaggle competition](https://www.kaggle.com/competitions/cayley-py-megaminx).
Each collaborator runs a different chunk of pids on their own Kaggle TPU
quota; the owner unions the results back via per-pid min.

## Files

- `build_notebook.py` — generates `cayleypy-tpu-beam-b2m-k2.ipynb`.
  Embeds all 1001 test pid states; the notebook slices into them at runtime
  via the config cell.
- `cayleypy-tpu-beam-b2m-k2.ipynb` — the generated notebook (425 KB).
- `kernel-metadata.json` — Kaggle kernel ID + dataset link.

## Running locally to update the notebook

```bash
.venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_b2m_k2/build_notebook.py
```

Re-run any time you change `build_notebook.py`. Then `kaggle kernels push -p .`
from inside the directory to update the Kaggle copy.

## Workflow for collaborators

### 1. Dataset access

The notebook reads from the **public** dataset
`artgor/megaminx-tpu-artifacts` (contains `m05_epoch_0499.pt`,
`m23_v2_epoch_0499.pt`, `rotations.npy`). No special access needed —
Kaggle attaches it automatically via the kernel metadata.

### 2. Fork the notebook on Kaggle

Open **https://www.kaggle.com/code/artgor/cayleypy-megaminx-beam-shareable**
and click "Copy and Edit". This creates a fork on the collaborator's
account that runs against their TPU quota.

### 2.5. Switch accelerator to TPU v3-8 (REQUIRED)

The pushed kernel is in **CPU mode** because the owner's TPU quota was
exhausted at push time. **Each collaborator must manually flip the
accelerator** in their fork:

- Right sidebar → "Accelerator" → select **"TPU v3-8"**.
- If it's grayed out, the collaborator's TPU quota is also exhausted.

Without this step, the notebook will crash on `import torch_xla` or hang
trying to find TPU devices.

### 3. Edit the config cell (Cell 1)

```python
START_PID = 0    # inclusive
END_PID = 50     # exclusive
K_SYM = 4        # 1, 2, 4, or 8 — number of rotations per pid
```

The notebook prints an estimated wall time when Cell 2 runs — if it warns
"exceeds 8.5h target", reduce the pid range or lower K.

Recommended scope per kernel (each ~8.3h wall):

| K | recommended pids | full-1001 needs |
|---|---|---|
| 2 | 750 | 2 collaborators |
| **4** | **375** | **3 collaborators** (default) |
| 8 | 188 | 6 collaborators |

**Note on K choice for diversity**: K=4 B=1M is exactly v19b's config;
those wins are already in our merge. **Prefer K=2 or K=8** to add new
diversity. K=4 is fine as a sanity check but expects mostly duplicate
results.

Suggested splits for K=4 (default), 3 collaborators:

| collaborator | START_PID | END_PID | pids | est. wall |
|---|---|---|---|---|
| A | 0 | 334 | 334 | 7.5h |
| B | 334 | 667 | 333 | 7.5h |
| C | 667 | 1001 | 334 | 7.5h |

For K=2 (untried slice — recommended for new diversity), 2 collaborators:

| collaborator | START_PID | END_PID | pids | est. wall |
|---|---|---|---|---|
| A | 0 | 501 | 501 | 5.7h |
| B | 501 | 1001 | 500 | 5.7h |

### 4. Verify accelerator

In the right sidebar: Accelerator = "TPU v3-8". If it says GPU or None,
change it before running.

### 5. Run All

Per-iteration partial saves to `/kaggle/working/rank_<r>_partial.json` mean
even if Kaggle hits the 9h kill mid-run, every completed (pid, rotation)
pair is recovered.

### 6. Send results back

After the kernel completes (or hits the 9h kill), download these from
`/kaggle/working/`:

- `tpu_qshort_sym_results.json` — full per-pair details
- `tpu_qshort_sym_submission.csv` — best path per pid in this chunk

Send them to the owner. Owner runs `/megaminx-tpu-merge` to union with the
current best CSV.

## Configuration constants (don't change unless you know what you're doing)

| constant | value | reason |
|---|---|---|
| `B_TEST` | 1,048,576 (1M) | only verified-safe size; B=1.5M+ OOMs with two-ranks-per-chip xmp.spawn |
| `K_SYM` | configurable in Cell 1 | 1, 2, 4 (default), or 8 rotations per pid |
| `INTERNAL_BS` | 65536 | B*N_GEN/65536 = 768 chunks/forward, in sweet spot |
| `NUM_STEPS` | 120 | depth at which paths plateau in our V model |
| `ALPHA_QSHORT` | 2.0 | student picks 2B-best, teacher reranks to B |
| `SYM_SEED` | 0 | deterministic — all collaborators choose the same rotation |

## Why this configuration

Untried slice of the (K, B) plane vs prior sweeps. Existing TPU runs:
- v16: K=4, B=131k
- v17: K=8, B=262k
- v19a-fix / v19b: K=4, B=1M

Same-compute alternatives at this budget:
- B=1.5M K=2 (this notebook) — under HBM ceiling, minimal sym diversity
- B=1M K=4 — already covered (v19a-fix, v19b)
- B=2M K=2 — OOM'd on first compile (HBM ceiling is below 2M)
- B=4M K=1 — exceeds HBM single-core, would need SPMD or hybrid CPU+TPU

The bucket-niche pattern from prior sweeps suggests B=1.5M K=2 finds wins
in different pid ranges than the K=4/K=8 configs already in our merged
best.

## Wall projection

Per-pair cached wall at B=1M is ~150s (measured in v19a-fix / v19b);
first-pair compile is ~30 min per rank, paid once.

Wall ≈ 0.5h compile + (N_pids × K / 8 ranks) × 150s

| K | recommended pids | per-rank pairs | wall |
|---|---|---|---|
| 2 | 750 | 188 | 0.5h + 7.8h = 8.3h |
| **4** | **375** | **188** | **0.5h + 7.8h = 8.3h** ← default |
| 8 | 188 | 188 | 0.5h + 7.8h = 8.3h |

(All three rows are 188 per-rank-pairs by design — that's the budget
that fits ~8.3h. K only changes how many pids you cover.)

## Troubleshooting

- **"megaminx-tpu-artifacts not found"**: dataset not attached. Right
  sidebar → "+ Add Data" → search for it. If you can't find it, you don't
  have access yet.
- **"Expected 8 worker addresses, got 1"**: env var pop in Cell 2 didn't
  run. Check that Cell 2 ran before Cell 9 (the spawn cell). If it did,
  restart the kernel and run all from top.
- **"Runtime is already initialized"**: torch_xla submodule was imported
  in the parent process before xmp.spawn. The notebook is structured to
  avoid this — only happens if you manually run cells out of order.
- **Kernel hits 9h kill before finishing**: expected for chunks > 340 pids.
  All completed pairs are saved. Send the partial JSONs anyway.
- **TPU sits idle for >2h with no output**: Kaggle kills idle TPU kernels.
  Healthy beam search produces a log line every ~5 min — if logs go
  silent, restart.

For deeper context, see:
- `megaminx/kaggle_notebooks/tpu_beam_smoke/RUNBOOK.md` — operational
  runbook for the original (private) sweep notebook
- `~/.claude/projects/.../memory/reference_kaggle_tpu_pjrt.md` — technical
  reference on Kaggle TPU PJRT quirks
