# Christopher's Jewel LLM evidence package

This archive contains the compact puzzle definition and the measured evidence
referenced by `LLM_TRAINING_PLAN.md`. It intentionally omits the 60 MB mixed
training dataset, the neural checkpoint, and the roughly 1.1 GB depth-8 ball.
Their SHA-256 identities or compact metadata are included where relevant.

## Direct answers

### Admissible lower bounds on the 1,000 competition states

`metrics/admissible_lower_bounds_1000.csv` contains one row per official state.
The component bound is

```text
max(ring bound, five-edge PDB for pieces 0..4,
    five-edge PDB for pieces 5..9)
```

It is then raised by one when required by solution-length parity. For the ten
states in the exact depth-8 ball, exact distance replaces the lower bound. The
result remains admissible and has:

- 1,000 states;
- mean 8.532;
- median 9;
- range 1--11;
- sum 8,532;
- 429 non-ball states tightened by parity;
- 10 states carrying exact distances.

The two complete PDB arrays and their metadata are included under `artifacts/`,
so the raw component bound is reproducible from the public puzzle definition.
The depth-8 arrays are too large for this compact archive; their layer metadata
is included and every exact value used in the CSV is identified explicitly.

### What is 93.92% measured on?

It is the set-valued top-1 policy accuracy on held-out exact rows:

```text
48,050 correct / 51,161 exact validation rows = 93.919196...%
```

For each row, the argmax among 12 policy logits is correct if it belongs to the
complete set of actions that reduce exact distance by one. The full validation
partition has 85,061 rows; 33,900 non-exact demonstration/public rows are not in
this metric. Validation membership is a stable hash of the 51-bit state rank,
so a state cannot cross between train and validation.

The exact sample was drawn with replacement. The 51,161 rows contain 36,080
unique states and 15,081 duplicate rows. Therefore 93.92% is row-weighted; it is
not unique-state-weighted accuracy, accuracy on the 1,000 competition states, or
an end-to-end solve rate. See `metrics/model_validation_metrics.json` for the
denominator, depth histogram, checkpoint metrics, and artifact hashes.

### Compact-solver budget and throughput for 17,012

The 17,012 score is the sum of 1,000 replay-verified raw candidate path lengths.
It was produced with:

- adaptive beam widths 1, 4, 16, and 64;
- at most 18 learned layers per width;
- one action per parent at width 1, up to four at wider widths;
- exact 51-bit deduplication and immediate-inverse suppression;
- learned policy/regret ranking plus the admissible PDB/ring bound;
- exact completion on entering the depth-8 ball;
- no incumbent-length bound (`raw-any-solution`);
- no explicit node cap.

The structural maximum is 1,449 expanded parent states per case if every width
exhausts all 18 layers. The recorded run expanded 571,138 parent states in
621.318 seconds on the local RTX 4090 Laptop GPU: 919.2 expanded parent states/s
end to end. Mean/median/p95/max expansions per state were
571.1/556/952.05/1,326. Ten states needed zero neural expansions because the
exact ball solved them directly.

An expanded parent is a frontier state presented to the neural scorer, not a
generated child edge. Generated-edge totals were not retained in this run. See
`metrics/compact_solver_benchmark.json` and
`evidence/competition_v4_public_full.json` for the compact and per-state records.

## Archive layout

```text
README.md                              this file
PROJECT_README.md                      current Jewel implementation README
source/jewel/official.py               official 48-position adapter
source/jewel/puzzle.py                 compact coordinates required by adapter
source/jewel/pdb.py                    PDB and lower-bound implementation
source/jewel/ball.py                   exact-ball loader
data/puzzle_info.json                  official public puzzle definition
data/test.csv                          the 1,000 official states
data/public_16490.csv                  verified solution upper bounds
artifacts/*.npy, *.json                the two complete PDBs and ball metadata
metrics/*.csv, *.json                  requested tables and compact reports
evidence/*.json                        underlying training/search records
MANIFEST.sha256                        hashes of every other archive member
```

`official.py` uses package-relative imports, so retain the `source/jewel/`
layout and add `source/` to `PYTHONPATH` when importing it outside this project.
