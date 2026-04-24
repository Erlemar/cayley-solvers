---
description: Combine candidate submissions, post-process, verify, and submit to Kaggle.
---

# /submit

Bundles the 4-command submission pipeline:

1. `combine_submissions.py` — take min-length per puzzle across all candidate CSVs + Kociemba fallback.
2. `post_process_submission.py` — pair-cancel + BFS-d5 window shortening.
3. `verify_submission` sanity check on the final CSV.
4. `kaggle competitions submit`.

## Usage

- `/submit <description>` — uses all `submissions/*.csv` as candidates EXCEPT the output of previous `/submit` runs (named `sub_*.csv`).
- `/submit --only a.csv b.csv ... "description"` — specify candidates explicitly.

## Execution

Do the following, in order:

1. Pick candidate CSVs. Unless the user passed `--only`, default to `submissions/fast_b8k_mitm6.csv submissions/s1_b4k.csv submissions/s2_b4k.csv submissions/s3_b4k.csv submissions/e3_khoruzhii_b65k.csv submissions/e5_khoruzhii_b65k.csv` (these are the landmark runs that have always helped in ensembles — verify they exist before using).

2. Generate a timestamped output path `submissions/sub_$(date +%Y%m%d_%H%M%S).csv` — this is the *raw* combine output before post-processing.

3. Run the combine step:

   ```bash
   .venv/Scripts/python.exe scripts/combine_submissions.py \
     --candidates <CSVs> \
     --fallback data/kociemba_fallback.csv \
     --out <raw_out>
   ```

4. Run the post-process step (appending `_pp` to the output name):

   ```bash
   .venv/Scripts/python.exe scripts/post_process_submission.py \
     --in <raw_out> --out <pp_out> \
     --bfs-table data/bfs_table_d5.pkl
   ```

5. Report the combined total move count and post-processed total.

6. Ask the user to confirm submission (the `-m` description is whatever was passed to `/submit`). Run:

   ```bash
   export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN
   .venv/Scripts/kaggle.exe competitions submit -c cayleypy-ihes-cube -f <pp_out> -m "<description>"
   ```

7. After submission, immediately check status with:

   ```bash
   .venv/Scripts/kaggle.exe competitions submissions cayleypy-ihes-cube | head -3
   ```

## Gotchas

- Never skip step 2 verification — a path with an unknown generator name will return `SubmissionStatus.COMPLETE` with score 999999 or similar on Kaggle (silent failure).
- If any candidate CSV doesn't exist, remove it from the list and proceed — don't abort the whole pipeline.
- Always reuse the same `data/kociemba_fallback.csv` as the fallback floor; never use `sample_fallback.csv` (sample-quality, 500K+ moves).
