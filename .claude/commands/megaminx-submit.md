---
description: Verify a megaminx submission CSV, submit to Kaggle, and update HANDOFF / SUBMISSION_JOURNEY / to_do_shortlist with the new score.
---

# /megaminx-submit

Bundles the post-merge submission ritual: verify → kaggle submit → update three
tracking docs that easily drift if updated by hand.

## Usage

- `/megaminx-submit <csv_path> "<description>"` — verify, submit, update docs.
- `/megaminx-submit <csv_path> --dry-run "<description>"` — verify + show planned updates, do NOT submit.

## Execution

### 1. Verify

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -c "
import sys
sys.path.insert(0, 'megaminx/src'); sys.path.insert(0, 'src')
from megaminx.puzzle import Megaminx
from cayley.verify import verify_submission
from pathlib import Path
puzzle = Megaminx.load(Path('megaminx/data/puzzle_info.json'))
report = verify_submission(puzzle, Path('megaminx/data/test.csv'), Path('<csv_path>'))
print(f'verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves:,}')
sys.exit(0 if report.all_valid else 1)
"
```

If `n_valid != 1001` or non-zero exit: ABORT. A failed verify is a silent
999999-score Kaggle submission. Show error and exit.

Capture the `total_moves` for the doc updates below.

### 2. Submit (skip on --dry-run)

```bash
export KAGGLE_API_TOKEN=KGAT_630ac26efca89d28c5b2d496b238b71c
.venv/Scripts/kaggle.exe competitions submit -c cayley-py-megaminx \
    -f <csv_path> -m "<description>"
```

Confirm "Successfully submitted" before proceeding to docs.

### 3. Update HANDOFF.md score progression

In section 2, prepend a new row to the score table. Mark the previous
"current best" row as superseded. Format:

```
| YYYY-MM-DD | <one-line description matching kaggle msg> | **<new_total>** | current submitted best |
```

Also update the "current best" reference in section 8 ("Where we ended") if it
exists, and the headline number anywhere it appears (currently sections 2, 8,
and 13).

### 4. Update SUBMISSION_JOURNEY.md

In the "Score progression at a glance" table, append a new row at the bottom:

```
| <next_n> | YYYY-MM-DD | **<new_total>** | — | -<delta_from_prev> | <one-line> |
```

If this is a major submission (not just a -100 follow-up), append a new
"Submission N" deep-dive section before the "Side findings" section.

Update the headline at the top: `# Megaminx — submission journey to <new_total>`
and the percentage / total moves saved.

### 5. Update to_do_shortlist.md

Edit the "State of play" section at the top:
- Update "Best submission: <new_total>"
- Update "Today's session saved" delta if today's running total

### 6. Final sanity check

```bash
grep "<new_total>" megaminx/HANDOFF.md megaminx/SUBMISSION_JOURNEY.md megaminx/to_do_shortlist.md | wc -l
```

Should be at least 5 (multiple mentions across docs). If 0 or 1, something didn't update.

### 7. Report

Print a summary:
```
Submitted <csv_path> = <new_total> (delta -X)
HANDOFF.md, SUBMISSION_JOURNEY.md, to_do_shortlist.md updated
Top of LB at last check: ... (#1 ...; #2 ...; us @ <new_total>)
```

## Gotchas

- **Don't run /megaminx-submit twice for the same CSV.** Each invocation costs a
  daily Kaggle slot (5/day). Check Kaggle leaderboard before re-submitting.
- **`KAGGLE_API_TOKEN` env var must be set inline** — Bash doesn't carry it
  across commands by default. Always `export KAGGLE_API_TOKEN=... && kaggle ...`
  in the same line.
- **`verify_submission` typo silent failure**: a malformed move name in the CSV
  silently returns SubmissionStatus.COMPLETE with score 999999. The local
  verify catches this — never skip step 1.
- **The 3 docs MUST stay synchronized**. Future-Claude reads HANDOFF.md as
  ground truth; if it disagrees with the leaderboard, the next session will
  burn time investigating "what's the actual current best?"
- **User policy**: do NOT submit community-merged CSVs. Only submit our own
  work or our work merged with ours.
