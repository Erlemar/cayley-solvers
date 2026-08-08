---
description: Merge every tetraminx source, independently replay-verify, submit to Kaggle, and confirm the score
---

Take the current best tetraminx result all the way to a scored Kaggle submission.
Arguments: `$ARGUMENTS` (optional, e.g. `--dry-run` to stop before submitting).

The standing submission rule lives in `tetraminx/HANDOFF.md` — as of 2026-08-03 it is
**beat 28,481 (Rokicki)**, and our submitted best is **28,467**. Do not submit a file
that does not beat the recorded bar unless the user says so explicitly.

## 1. Refresh every source FIRST (this is where merges silently lose moves)

```
gcloud compute tpus tpu-vm list --project=gen-lang-client-0977634337 --zone=us-east5-a --format="value(name,state)"
gcloud compute instances list --project=gen-lang-client-0977634337 --filter="status=RUNNING"
```

- **Re-pull ALL beam JSONs from every live box**, not just the one being watched —
  `scp <box>:~/out/*.json tetraminx/results/<box>/`. A stale pull caps the merge with no
  warning; on 2026-08-03 the final full re-pull took JSONs 78 → 121 and found moves.
- **Copy anything valuable into `tetraminx/submissions/`** rather than relying on
  `--extra` over session scratchpads — those are temp dirs whose set changes run to run.
- **Look for stray result files by CONTENT, not name or location.** Files worth −22 were
  found in `megaminx/` and in an unrelated `rogii-wellbore-geology-prediction/` folder.
  A candidate qualifies only if the header is `initial_state_id,path` AND its move
  alphabet is a subset of the tetraminx generators (a megaminx file passes the first
  test and fails the second).

## 2. Merge

```
.venv/Scripts/python.exe tetraminx/scripts/90_merge_all.py \
    --baseline tetraminx/submissions/community_28843.csv \
    --extra <sibling-scratchpads...> tetraminx/results/<box>... --write
```

Read the credited-source table: it tells you which files actually contributed, which is
the only honest basis for attributing a gain.

## 3. Verify INDEPENDENTLY of the merge

The merge re-reads and replays its own output, but do not let it vouch for itself. Replay
every path against the ORIGINAL `tetraminx/data/test.csv` state with a separate script:
load `puzzle_info.json`, apply `state = state[gen[move]]` per move, require the solved
state. Assert all four of:

- 1000 rows and 1000 distinct pids, none missing
- zero unknown move names
- zero paths that fail to reach solved
- the replayed total equals the filename total

Anything non-zero → **do not submit** (a bad generator name scores silently).

## 4. Submit and confirm

```powershell
$env:KAGGLE_API_TOKEN="<token>"; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
& "C:\Users\and-l\cayley\.venv\Scripts\kaggle.exe" competitions submit `
    -c cayley-py-professor-tetraminx-solve-optimally `
    -f tetraminx\submissions\FINAL_tetraminx_<N>.csv -m "<what produced it>"
```

Then **confirm the scored number matches the local total** — this is what catches a
silent-fail submission:

```powershell
& "...\kaggle.exe" competitions submissions -c cayley-py-professor-tetraminx-solve-optimally
```

PowerShell needs the token re-exported per invocation (CLAUDE.md 7d); a 401 is almost
always that, not a scope problem.

## 5. Report

Give total, `avg`, `avg>1`, the margin vs the bar, and the credited-source split. If
external files contributed, say so — but note that results from others running OUR
published notebook are downstream of our own solver, not a third-party source.

## Do NOT

- Submit on the merge's self-check alone.
- Quote a total without having re-pulled live machines in the same session.
- Claim a merged total is "final" — the source set is not stable; say "best verified
  merge over everything currently reachable".
