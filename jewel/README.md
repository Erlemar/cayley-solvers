# Christopher's Jewel: exact data + Transformer + interactive search

This directory contains an end-to-end solver for the Kaggle Christopher's
Jewel competition.  It combines a compact exact puzzle representation, a
depth-8 reverse ball, two complete edge pattern databases, mixed supervised
data, a structured Transformer policy/value model, and adaptive beam search
with exact endgame completion.

## Correctness boundary

`puzzle.py` represents a state as 12 permuted/oriented edges plus six C4 ring
orientations.  The reachable group has

`12! * 2^22 = 2,009,078,326,886,400`

states and fits in a 51-bit rank.

`official.py` infers and verifies the full isomorphism to the competition's
48-position permutations.  The official move directions are the inverse of
the TWS base directions, and the edge orientation reference gauge differs on
six slots.  Unit tests verify all 12 named actions and 200-action random paths
in both representations.

Pre-official-validation artifacts are unsafe; see `INVALIDATED.md`.  Current
artifacts and defaults use the `_official` suffix.

## 1. Exact reverse ball

`artifacts/ball_d8_official/` stores every state through exact distance 8:

| Depth | Layer size |
|---:|---:|
| 0 | 1 |
| 1 | 12 |
| 2 | 114 |
| 3 | 1,068 |
| 4 | 9,819 |
| 5 | 89,392 |
| 6 | 807,000 |
| 7 | 7,231,464 |
| 8 | 64,352,783 |
| **Total** | **72,491,653** |

Build it with:

```powershell
.venv\Scripts\python.exe -m jewel.build_ball
```

## 2. Pattern databases and exact search

The two edge PDBs each cover all `12P5 * 2^5 = 3,041,280` abstract states:

- `artifacts/pdb_edges_0_4_official.npy`
- `artifacts/pdb_edges_5_9_official.npy`

Both have maximum distance 10.  Their maximum is an admissible lower bound.
`exact_search.py` uses this bound in parity-aware IDA*, with direct completion
from the exact ball.

Across the 1,000 official competition states, the raw maximum of the two PDBs
and the ring bound has mean 8.102 (median 8, range 1--11).  Tightening that
bound to the known solution-length parity, and substituting exact distance for
the ten states inside the depth-8 ball, gives a certified admissible lower
bound with mean 8.532 and sum 8,532.  The complete per-state table is
`results/admissible_lower_bounds_1000.csv`; its construction and component
definitions are recorded in `results/admissible_lower_bounds_1000_summary.json`
(the evidence archive stores both under `metrics/`).

With a 1,000,000-node/10-second per-case budget, the corrected exact benchmark
solved 14/25 held-out non-backtracking scrambles: 5/5 at walk depth 8, 5/5 at
10, 3/5 at 12, 1/5 at 14, and 0/5 at 16.  Solved IDA* cases are optimal; the
remaining cases hit the time limit rather than returning a false result.

## 3. Data mixture

`artifacts/train_mixed_v4_public.npz` contains 1.5 million samples:

- 900,000 exact depth-1--8 states with the complete set of descending actions;
- 200,000 depth-8--20 non-backtracking demonstrations, weight 0.25;
- 400,000 samples from 8,529 distinct deep states along all 1,000
  replay-verified public competition trajectories, weight 1.0.

Public trajectory states already inside the exact ball are excluded, so exact
labels always take precedence.  Validation assignment is a stable function of
the 51-bit state rank, preventing duplicate states from crossing the split.

## Transformer

`models/transformer_v4_public/best.pt` is a 4,803,621-parameter model with six
Transformer blocks, width 256, eight attention heads, and FFN width 1024.  It
uses edge, orientation, ring, action-query, and CLS tokens.  Heads predict:

- set-valued next-action policy;
- geodesic-action logits and per-action regret;
- scalar distance and a distance CDF.

The best held-out exact metrics after public-trajectory fine-tuning are:

- top-1 descending action: **93.92%**;
- top-2 contains a descending action: **98.00%**;
- probability mass on all descending actions: **0.919**;
- distance MAE: **0.251**.

The 93.92% figure is specifically **48,050 / 51,161 exact validation rows**.
The full validation partition has 85,061 rows, but the descending-action
metric filters to rows whose complete optimal-action mask comes from the exact
depth-8 ball.  Validation membership is a stable hash of the 51-bit state rank,
so a state cannot occur in both training and validation.  Exact states were
sampled with replacement, however: the 51,161 rows contain 36,080 unique
states.  Consequently 93.92% is a row-weighted held-out exact-action metric,
not accuracy on the 1,000 competition states and not a unique-state-weighted
metric.

## GFlowNet backward policy

`gflownet.py` implements the reversed-graph construction from GFlowNet
pathfinding.  The solved Jewel is the source.  The learned forward policy has
12 scrambling actions plus `stop`; the learned backward policy has 12 solving
actions.  Training uses uniform terminal reward, fixed
`Z = 2,009,078,326,886,400`, masks forward moves that re-enter the source, and
applies regularized trajectory balance to every sampled trajectory prefix.
The backward probability is gathered at the inverse of the corresponding
forward action, which is essential for the official move convention.

The current network is a 2,775,065-parameter residual MLP over the official
48-position state.  It is trained without solution demonstrations:

```powershell
.venv\Scripts\python.exe -m jewel.train_gflownet `
  --steps 200000 --compile --batch-size 128 --trajectory-length 20 `
  --hidden 512 --res-blocks 3 --reg-coef 1e-5 `
  --out jewel\models\gflownet_v1
```

The run completed 200,000 updates.  Selection on the fixed 512-state rollout
set chose the step-175,000 checkpoint (460/512 solved); the final checkpoint
scored 450/512.  Independent evaluation of the selected checkpoint gives:

- exact validation top-1 descending action: **95.39%** row-weighted and
  **93.50%** after duplicate states are collapsed;
- exact validation top-2: **98.66%** row-weighted and **98.11%** unique-state;
- probability mass on the complete descending set: **0.893** row-weighted and
  **0.849** unique-state;
- greedy full-path solves: **763/1,000**, all replay-verified in both compact
  and official coordinates, with mean solved length 25.95 under a 64-move cap.

For comparison, the supervised Transformer's published row-weighted exact
top-1 is 93.92%.  The metrics are comparable because both use the same 51,161
complete exact validation rows; the unique-state GFlowNet metric uses 36,080
states.  Greedy coverage is not complete, so deployment should still use
search and an incumbent rather than raw argmax rollout.

Evaluate both exact next-action labels and greedy full solutions with:

```powershell
.venv\Scripts\python.exe -m jewel.evaluate_gflownet `
  --checkpoint jewel\models\gflownet_v1\best.pt `
  --out jewel\results\gflownet_v1_evaluation.json
```

## Policy-guided exact IDA* and PHS

`policy_search.py` provides three complementary modes:

- policy-guided IDA* uses the GFlowNet or Transformer only to order children;
  every prune uses the exact ball or an admissible PDB/ring bound.  Therefore a
  result marked `optimal=true` is a shortest-path certificate;
- PHSh and PHS* combine cumulative policy probability with the admissible
  heuristic.  They are usually much faster at finding a valid path, but PHS is
  not a shortest-path algorithm and does not claim optimality unless the path
  length equals the root lower bound;
- `phs_then_exact` first obtains a PHS incumbent, then passes it to IDA* for
  strict improvement or proof.

Neural frontier parents are scored in batches of 256.  Pre-scoring does not
change their heap priorities: all parents are reinserted before expansion, so
the best-first PHS expansion order remains identical to scalar scoring.

Query one state with independent compact and official replay verification:

```powershell
.venv\Scripts\python.exe -m jewel.policy_search_cli `
  --initial-state-id 20 --policy gflownet `
  --checkpoint jewel\models\gflownet_v1\best.pt `
  --algorithm phsh-exact --time-limit 30 --node-limit 500000
```

Run deterministic random-walk and competition benchmarks with:

```powershell
.venv\Scripts\python.exe -m jewel.benchmark_policy_search `
  --policy gflownet --gflownet jewel\models\gflownet_v1\best.pt

.venv\Scripts\python.exe -m jewel.evaluate_policy_search `
  --policy gflownet --checkpoint jewel\models\gflownet_v1\best.pt `
  --sample-size 100 --time-limit 1 --node-limit 25000 `
  --model-greedy-incumbent --incumbent-bound
```

Measured search results for the selected GFlowNet are:

- on ten fixed depth-12 random scrambles, both IDA* orderings certified all ten
  optimal paths.  Policy ordering reduced mean expansions from 6,203 to 4,198
  (32.3%), although scalar DFS inference increased total wall time from 15.6 to
  19.0 seconds;
- on nine fixed depth-10/12/14 scrambles, PHSh and PHS* both solved 9/9.
  PHSh returned shorter paths on average (11.78 versus 12.89), while PHS* used
  fewer expanded parents (27 versus 263).  Three cases were certified directly
  because the returned length met the exact root lower bound;
- on a deterministic random sample of 100 official states, raw PHSh solved and
  dual-verified 48/100 within three seconds per state.  A greedy policy
  incumbent plus one second of PHSh raised coverage to 83/100;
- adding the known 16,490-score public path as a hard incumbent yielded 100/100
  valid results and preserved the sample score exactly (1,636).  No strict
  improvements were found.  This safe incumbent mode is therefore correct but
  is not currently a stronger Kaggle optimizer.

The full reports are `results/gflownet_v1_evaluation.json`,
`results/policy_ida_gfn_depth12_10.json`,
`results/policy_search_gfn_final_random.json`, and
`results/gflownet_safe_phsh_competition_random100.json`.

## GFlowNet beam and best-first search

`gfn_beam.py` turns the backward GFlowNet into a complete practical solver for
the competition set.  It provides:

- layered policy beam search with adaptive widths, cumulative negative log
  backward probability, PDB/ring ranking, exact-ball completion, inverse-move
  suppression, and exact 51-bit state deduplication;
- batched global best-first search for a larger fallback budget;
- a hybrid entry point that tries the cheap adaptive beams before best-first.

The selected beam settings are widths `1,4,16,64`, four actions per parent,
18 learned layers, and heuristic weight 0.2.  On all 1,000 official states:

- solved and dual-replay verified: **1,000/1,000**;
- raw GFlowNet beam score: **18,492**;
- mean/median/max raw length: **18.492 / 19 / 23**;
- mean expanded parents: **858.7**;
- mean inference time: **0.722 seconds/state** on the local RTX 4090 Laptop;
- exact lower-bound certificates: **10**.

The raw paths tie the 16,490 incumbent on 245 states and are longer on 755;
there are no strict wins.  This establishes full standalone coverage, not a
better Kaggle score.  The supervised Transformer beam remains more efficient
at 17,012 moves.  A 20-state best-first probe also solved 20/20 with four
branches, but took about 3.0 seconds/state and produced longer paths, so beam is
the default and best-first is a fallback/research mode.

Query one state and receive the next move plus the complete verified path:

```powershell
.venv\Scripts\python.exe -m jewel.gfn_query `
  --initial-state-id 20 --mode hybrid
```

Run and preserve the full raw and safely merged solution files with:

```powershell
.venv\Scripts\python.exe -m jewel.evaluate_gfn_search `
  --checkpoint jewel\models\gflownet_v1\best.pt `
  --mode beam --raw-any-solution --widths 1,4,16,64 `
  --max-learned-steps 18 --beam-branch-actions 4 `
  --heuristic-weight 0.2 `
  --out jewel\results\gfn_beam_full_1000.json `
  --raw-submission jewel\results\gfn_beam_full_1000_raw.csv `
  --submission jewel\results\gfn_beam_full_1000_merged.csv
```

## Interactive inference

`interactive.py` tries beam widths 1, 4, 16, and 64, ranks children with the
learned policy/regret plus the PDB lower bound, forbids immediate inverses,
deduplicates exact 51-bit states, and completes from the depth-8 ball.

On 30 held-out random scrambles at walk depths 14, 16, and 18 it solved 30/30.
After merging with the known scramble inverse, mean lengths were 13.2, 14.8,
and 17.2 respectively, with 9/30 strict shortenings.

On all 1,000 official competition states:

- model paths found: **1,000/1,000**;
- compact-coordinate replays: **1,000/1,000**;
- official 48-position replays: **1,000/1,000**;
- raw model score: **17,012**;
- raw ties with the 16,490 public/leader path: **874/1,000**;
- safely merged score: **16,490**;
- mean inference time: **0.621 s/state** on the local RTX 4090 Laptop GPU.

The raw 17,012 run used `--raw-any-solution`, adaptive beam widths
`1,4,16,64`, at most 18 learned layers per width, one branch at width 1 and up
to four branches at wider beams.  It had no explicit node cap; the structural
maximum is 1,449 expanded parent states per case if every width exhausts every
layer.  In the recorded run it expanded **571,138 parent states total** (mean
571.1, median 556, p95 952.05, maximum 1,326), took 621.318 seconds, and
therefore processed **919.2 expanded parent states/s end to end**.  An expanded
parent state means one frontier state presented to the neural scorer; it is
not the number of generated child edges.  Ten states were completed directly
from the exact ball with zero neural expansions.

The public score 16,490 is already the current leading score, so the
incumbent-bounded shortcut search correctly found no strict improvements.

### Query one next move

Use the persistent `JewelQueryEngine` when an application repeatedly sends a
state and needs one official move in response:

```python
from jewel.query import JewelQueryEngine

engine = JewelQueryEngine()  # load the checkpoint and exact ball once
result = engine.query(state)  # raw one-step model query
print(result["next_move"])
```

The input is the competition's 48-integer `initial_state` representation. The
returned `next_move` is an official competition generator such as `DBLBBBR` or
`-DRFBR`. The Python `query` method is the raw one-step interface, optionally
guarded by the exact ball. Use `solve_query(state, pdbs)` when the first move
must belong to a complete replay-verified solution.

For a one-shot command, paste a state or query a row of the local test set:

```powershell
.venv\Scripts\python.exe -m jewel.query --state "<48 comma-separated integers>"
.venv\Scripts\python.exe -m jewel.query --initial-state-id 0 --move-only
.venv\Scripts\python.exe -m jewel.query --initial-state-id 999 --model-only
```

The command defaults to interactive beam search, PDB guidance, exact endgame
completion, and independent replay verification. `--model-only` explicitly
requests the raw one-step argmax instead. The normal JSON response includes the
verified full `solution_path`, its length and search statistics, along with
policy probability, predicted geodesic probability and regret, model-predicted
distance, exact distance when known, and the top ranked alternative moves.

Run the full evaluation and independent verifier with:

```powershell
.venv\Scripts\python.exe -m jewel.evaluate_competition `
  --checkpoint jewel\models\transformer_v4_public\best.pt `
  --widths 1,4,16,64 --raw-any-solution `
  --out jewel\results\competition_v4_public_full.json `
  --submission jewel\results\competition_v4_public_merged.csv

.venv\Scripts\python.exe -m jewel.verify_submission `
  jewel\results\competition_v4_public_merged.csv
```

The independent verifier reports 1,000 rows, score 16,490, 1,000 valid
official replays, and no problems.
