# Prefix-Anchored TPU Repair

Date: 2026-06-16

This note documents the pid 992 rescue that found a 69-move path from a 72-move
public floor, and turns it into a repeatable workflow for future hard pids.

## Finding

The 128M AZ-v5 V-only TPU run originally found pid 992 at 76 moves, while the
current floor from `submission_73731.csv` / `half_split_rank201_400_b16k.csv`
was already 72. Prefix-survival tracing showed a misleading early failure:
the known 72-path prefix dropped at step 4.

Root cause was not the value model. It was seed padding in the TPU V-only beam.
The huge seed frontier contained the real first-move children followed by
duplicates of the last real child. Before the beam is full, those duplicate
real states can pollute per-owner routing/topK. The kernel already treats the
all-zero state as padding, so the fix is to zero-pad unused seed slots.

After the fix:

- direct pid 992 AZ-v5 V-only 128M improved from 76 to 75;
- the 72-path prefix survived through step 10 and dropped at step 11;
- anchored repair from prefix 11 found a 59-move suffix: 11 + 59 = 70;
- anchored repair from prefix 10 found a 59-move suffix: 10 + 59 = 69.

Final verified merged CSV:

`megaminx/submissions/half_split_pid992_prefix10_azv5_69.csv`

Full verification:

```text
valid 1001/1001 total 73438
```

The pid 992 path improved 72 -> 69.

## Reusable Workflow

1. Pick the current best path for the target pid from the working floor CSV.
   Public/community paths are valid references for prefix anchoring; the final
   path still must verify from the original state.

2. Run a stop-on-drop prefix trace:

```bash
~/tpu-env/bin/python gcp_beam_v_only.py \
  --b-global 134217728 \
  --start-pid 992 --end-pid 993 \
  --num-steps 100 --alpha 2 \
  --internal-bs 131072 --parent-chunk 1048576 \
  --tree-dir /dev/shm/trees \
  --out /mnt/data/out/pid992_trace.json \
  --v-checkpoint m_az_v5_73614_orbit_v_only.pt \
  --hidden-dims 3072,1024 --num-res-blocks 2 \
  --trace-path-csv /mnt/data/v6e/half_split_rank201_400_b16k.csv \
  --trace-pid 992 \
  --progress-every 1 --stop-on-trace-drop
```

3. Let the trace guide the anchor, but convert the logged step to a path prefix
   length first. The packed V-only solver keeps first-move children as seed
   step 0, while progress logs start at loop step 1. So a trace log row
   `step=j` checks the path prefix of length `j + 1`. If an anchored suffix
   trace from prefix `p` drops at `step=1`, the beam kept the first suffix move
   implicitly as the seed, then dropped the full prefix `p + 2`.

   In practice, try the last kept prefix, the first dropped prefix, and one or
   two earlier anchors. Earlier anchors give the beam more freedom but increase
   suffix length.

4. Build a synthetic one-row test CSV from the prefix:

```bash
.venv/Scripts/python.exe megaminx/scripts/104_make_prefix_state_csv.py \
  --pid 992 \
  --prefix-len 10 \
  --path-csv megaminx/submissions/half_split_pid992_prefix11_azv5_70.csv \
  --out megaminx/data/pid992_prefix10_from70_test.csv \
  --suffix-out megaminx/data/pid992_prefix10_from70_suffix_trace.csv
```

5. Run the TPU beam on the synthetic state:

```bash
~/tpu-env/bin/python gcp_beam_v_only.py \
  --test-csv /mnt/data/v6e/pid992_prefix10_from70_test.csv \
  --b-global 134217728 \
  --start-pid 0 --end-pid 1 \
  --num-steps 80 --alpha 2 \
  --internal-bs 131072 --parent-chunk 1048576 \
  --tree-dir /dev/shm/trees \
  --out /mnt/data/out/pid992_prefix10.json \
  --v-checkpoint m_az_v5_73614_orbit_v_only.pt \
  --hidden-dims 3072,1024 --num-res-blocks 2 \
  --progress-every 5
```

6. Splice the prefix and suffix back into a full submission and verify:

```bash
.venv/Scripts/python.exe megaminx/scripts/105_splice_prefix_suffix_json.py \
  --pid 992 \
  --prefix-len 10 \
  --base-csv megaminx/submissions/half_split_pid992_prefix11_azv5_70.csv \
  --suffix-json megaminx/results/pid992_prefix10_from70_azv5_zeropad_128m_a2.json \
  --out megaminx/submissions/half_split_pid992_prefix10_azv5_69.csv
```

Then run `verify_submission` on the whole CSV. Do not trust a partial/suffix
solve unless the final spliced path verifies from the original state.

## Operational Notes

- For 128M on v6e-8, `b_global=134217728`, `B_local=16777216`, `alpha=2`,
  `parent_chunk=1048576`, and `internal_bs=131072` gave ~208 seconds/step.
- `alpha=3` fit but did not change pid 992's true drop step after zero-padding;
  it was slower and not useful for that case.
- The trace cutoff is `1e9` until the beam has enough real states to fill the
  global frontier. Ignore early cutoff values before the beam fills.
- If using widths above 128M on 8 chips, the V-only packed backpointer cannot
  stay 24/3/5 `uint32`; `B_local` exceeds 2^24 and needs a wider `uint64`
  packing.
- For 256M global on 8 chips, use 25/3/5 `uint64` backpointers. The tested
  V-only configuration was `b_global=268435456`, `B_local=33554432`,
  `alpha=1`, `parent_chunk=1048576`, and `internal_bs=131072`, at about
  422 seconds/step. `alpha=1` was chosen because `alpha=2` would double the
  receive pool at 256M and likely hit HBM pressure.

## Pid 994 256M Negative Result

On 2026-06-17, the same AZ-v5 V-only approach was run on pid 994 at 256M global
beam using `m_az_v5_73614_orbit_v_only.pt`. The current path from
`submission_73731.csv` / `half_split_pid992_prefix10_azv5_69.csv` is 71 moves.

Results:

- direct stop-on-drop trace: logged hits through step 5 and drop at step 6,
  which means the path prefix of length 7 dropped; no direct solve before the
  drop;
- prefix 5 anchored suffix search, 256M, 75-step horizon: no verified solve by
  the existing 66-move suffix threshold;
- prefix 6 anchored suffix search, 256M, 66-step horizon: no verified solve by
  the existing 65-move suffix threshold;
- prefix 6 anchored suffix trace dropped at logged step 1, meaning prefix 8
  fell out after prefix 7 was implicitly kept as the seed;
- prefix 20 anchored suffix search, 256M, 51-step horizon: no verified solve by
  suffix length 50, so no one-move improvement from that deeper anchor.

Local evidence:

`megaminx/reports/pid994_256m_2026-06-17/pid994_256m_logs/`

Verdict: this exact V-only 256M prefix-anchor recipe did not improve pid 994.
The useful follow-up is not "make the same run wider" immediately; it is to
trace successive anchored suffixes and/or try a different scoring stack
(inverse/NISS, qshort-aligned V+Q, or a small near-solved exact table) for the
suffix repair stage.

## Pid 994 Alpha/NISS Follow-up

On 2026-06-18, the follow-up used the same AZ-v5 orbit V-only checkpoint with
`b_global=201326592` (192M), `alpha=2`, `parent_chunk=1048576`, and
`internal_bs=131072`. This was deliberately narrower than 256M so the receive
pool could keep the extra alpha slack on v6e-8.

Results:

- direct stop-on-drop trace survived through logged step 6 and dropped at step
  7; this is one logged step deeper than the 256M `alpha=1` trace, so alpha
  slack helped retention, but still did not solve the original forward path;
- inverse-only NISS (`--invert`) found a verified 69-move path at inverse step
  68, wall 23,705 seconds;
- merging that path into `half_split_pid992_prefix10_azv5_69.csv` gives
  `megaminx/submissions/half_split_pid992_994_niss69_azv5_192m_a2.csv`;
- full verification passes: `1001/1001` valid, total `73,436`.

Local evidence:

`megaminx/reports/pid994_alpha_niss_2026-06-18/pid994_alpha_niss_logs/`

Verdict: for pid 994, alpha slack was useful diagnostically but NISS was the
decisive axis. The reusable follow-up pattern for stubborn single pids is:
try a cheaper alpha-retention gate first, then run inverse/NISS before spending
more wall on deeper forward prefix anchors.
