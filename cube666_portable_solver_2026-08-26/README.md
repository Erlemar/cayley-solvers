# Portable CUBE666 research and solver bundle

This folder contains the updated CUBE666 model research plan and two complete solver
lanes for the 216-sticker CayleyPy supercube:

1. **Autonomous neural beam** — exact corner/parity setup followed by the accepted
   factorized neural macro beam. It includes the trained checkpoint and solves every
   normalized competition state, but its paths are much longer than the best classical
   paths.
2. **KMC classical/hybrid solver** — the patched KMCoders Rust solver used to generate
   competitive paths, plus the local Python wrapper that independently replays every
   result. A prebuilt Windows executable is included; Rust source is included for other
   operating systems.

The research plan is in
`docs/CUBE666_MODEL_UPDATED_ANALYSIS_AND_PLAN_2026-08-24.md`.

## Bundle contents

```text
docs/                         research plan and background
src/cube666/                  exact-sticker puzzle, algebra and neural search code
cube_nnn/src/cube_nnn/        verified 48-frame cube symmetry support
solver/solve_neural.py        submission-shaped autonomous neural solver
cube666/scripts/06_*.py       replay-verified KMC solver wrapper
cube666/scripts/07_*.py       strict merge and independent verification
models/                       accepted neural checkpoint
cube666/training/             neural action library
cube666/artifacts/            exact 24,288 three-cycle library
cayley-py-666-cube/           puzzle definition and 1,012 test states
external/third_party/         KMC Rust source and Windows executable
submissions/                  verified 171,019-move reference file
```

The research plan describes proposed residual and orbit-ladder models. Those proposed
models do not yet exist; this bundle supplies the exact baseline solver substrate needed
to implement and evaluate them.

## Requirements

- 64-bit Python 3.10 or newer.
- PyTorch and NumPy.
- An NVIDIA GPU is strongly recommended for neural beam search. CPU is sufficient for
  validation and the smoke test but is much slower.
- Rust stable is required only to rebuild the KMC executable.

The source machine used Python 3.14, NumPy 2.4.4 and PyTorch 2.11.0+cu128. The checkpoint
uses ordinary PyTorch tensors and is loaded with `weights_only=True`.

## Setup

### Windows PowerShell

```powershell
Set-Location cube666_portable_solver_2026-08-26
.\setup.ps1
.\.venv\Scripts\python.exe verify_manifest.py
.\.venv\Scripts\python.exe smoke_test.py
```

For an NVIDIA machine, install the appropriate CUDA-enabled PyTorch wheel before or
after running `setup.ps1` if the default wheel does not match the installed driver.

### Linux/macOS

```bash
cd cube666_portable_solver_2026-08-26
sh setup.sh
.venv/bin/python verify_manifest.py
.venv/bin/python smoke_test.py
```

## Run the autonomous neural solver

One puzzle:

```powershell
.\.venv\Scripts\python.exe run_solver.py neural --indices 0
```

Several puzzle row indices:

```powershell
.\.venv\Scripts\python.exe run_solver.py neural --indices 0,200,440,879
```

All 1,012 puzzles with the accepted automatic rescue configuration:

```powershell
.\.venv\Scripts\python.exe run_solver.py neural --all-puzzles `
  --output outputs\neural_all1012.csv `
  --report outputs\neural_all1012.json
```

The CSV contains complete primitive paths. Every emitted path is replayed on all 216
stickers before it is written.

## Run the KMC solver

The bundled Windows executable can be used immediately:

```powershell
.\.venv\Scripts\python.exe run_solver.py kmc `
  --indices 0 --beam 1000 --anneal-steps 1000000 `
  --tag portable_smoke
```

Competitive settings use much larger annealing budgets and beams and can take minutes
per puzzle. The exact settings are deliberately explicit rather than hidden in the
wrapper.

To rebuild KMC:

```powershell
.\build_kmc.ps1
```

or:

```bash
sh build_kmc.sh
```

The Rust build downloads crates from crates.io unless they are already cached.

## Merge and verify candidates

Merge a full candidate CSV against the included verified reference:

```powershell
.\.venv\Scripts\python.exe run_solver.py merge `
  --candidate-csv submissions\current_best_171019_verified.csv `
  --candidate-csv outputs\candidate.csv `
  --output outputs\merged.csv `
  --report outputs\merged.json
```

`07_merge_verify.py` also reads the sample submission as a final completeness fallback,
commute-reduces every candidate and independently replays every row.

## Integrity and reproducibility

`MANIFEST.json` records the byte size and SHA-256 digest of every distributed file except
the manifest itself. Run:

```text
python verify_manifest.py
```

Outputs, virtual environments, Rust build products and caches are not part of the
manifest and may be created locally.

Run the complete bundled regression suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

or `.venv/bin/python -m pytest -q` on Linux/macOS.

## Important limitations

- The neural solver is a capability baseline, not a competitive scoring solver.
- The included 171,019 CSV is a verified reference artifact, not regenerated by one
  short command; it is the min-merge of a large prior campaign.
- The KMC source is third-party research code with local modifications. See
  `THIRD_PARTY.md` before redistributing the bundle.
- No Kaggle API token or other credential is included.
