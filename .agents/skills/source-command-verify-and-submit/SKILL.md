---
name: "source-command-verify-and-submit"
description: "Verify an already-built submission CSV then submit to Kaggle."
---

# source-command-verify-and-submit

Use this skill when the user asks to run the migrated source command `verify-and-submit`.

## Command Template

# /verify-and-submit

Thin submit path: skip the combine/post-process pipeline, just verify a pre-built CSV and submit it to Kaggle. Use this when you already have a finalized submission CSV (e.g., from combine_submissions + post_process ran earlier, or a single-shot solve output that's already PP'd).

## Usage

- `/verify-and-submit <csv_path> <description>` — verify then submit.
- `/verify-and-submit <csv_path> --force <description>` — submit even if verify reports any invalid paths (rarely needed; usually a verify failure = silent 999999 score).

## Execution

1. Confirm the CSV exists. Abort with a clear message if not.

2. Run verification:

   ```bash
   .venv/Scripts/python.exe -c "
   from pathlib import Path
   import sys
   sys.path.insert(0, 'src')
   from cayley.puzzle import PictureCube
   from cayley.verify import verify_submission
   puzzle = PictureCube.load('data/puzzle_info.json')
   r = verify_submission(puzzle, 'data/test.csv', '<csv_path>')
   print(f'verify: {r.n_valid}/{r.n_total} valid, total_moves={r.total_moves}, all_valid={r.all_valid}')
   sys.exit(0 if r.all_valid else 1)
   "
   ```

3. Report the total_moves count and verification status.

4. If verify failed and `--force` not set, abort. Otherwise, ask user to confirm submission.

5. Submit:

   ```bash
   export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN
   .venv/Scripts/kaggle.exe competitions submit -c cayleypy-ihes-cube -f <csv_path> -m "<description>"
   ```

6. Check status:

   ```bash
   .venv/Scripts/kaggle.exe competitions submissions cayleypy-ihes-cube | head -3
   ```

## Gotchas

- A typo in a generator name returns `SubmissionStatus.COMPLETE` with score 999999 (silent failure). Verification catches this before submission.
- The user policy (as of 2026-04-20): do NOT submit community-merged CSVs; only submit our own work. Check the CSV's provenance before running if unsure.
- This command is for pre-built CSVs only. If you need to combine candidates + post-process first, use `/submit` instead.
