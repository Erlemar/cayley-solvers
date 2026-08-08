---
description: N-way per-pid min over every tetraminx result source, replay-verified, into a full submission
---

Run the Rule-26 merge for tetraminx: keep the shortest VERIFIED path per pid across
every source, then write and re-verify a complete 1000-row submission.

## Do this

1. **Pull anything still on a live machine first** — results on a VM are not in the
   merge. Check for running boxes and copy their output down:
   ```
   gcloud compute tpus tpu-vm list --project=gen-lang-client-0977634337 --zone=us-east5-a --format="value(name,state)"
   gcloud compute instances list --project=gen-lang-client-0977634337 --filter="status=RUNNING"
   ```
   Beam JSONs live in `~/out/*.json` on the TPU boxes; pull into
   `tetraminx/results/<box>/`.

2. **Check sibling Claude sessions.** Other sessions write to their own scratchpads
   and their results are invisible here:
   ```
   ls C:/Users/and-l/AppData/Local/Temp/claude/C--Users-and-l-cayley/*/scratchpad/
   ```
   Pass any result dirs via `--extra`.

3. **Run the merge:**
   ```
   .venv/Scripts/python.exe tetraminx/scripts/90_merge_all.py \
       --baseline tetraminx/submissions/community_28843.csv --write
   ```
   Add `--extra <dir>...` for sibling-session or freshly pulled output.

4. **Read the output critically.** The script prints pids covered, the total, a
   credited-source table, the length histogram, and the delta vs the baseline. It
   re-reads the written file and re-verifies every row; a non-zero `invalid` or a
   total mismatch prints `*** MISMATCH -- do not submit ***` and exits 1.

5. **Report the number against the bar (28,481), not against our previous best.**
   Do not submit unless it beats 28,481 — that is a standing instruction.

## Why this exists

Results scatter across `submissions/`, `results/**`, sibling session scratchpads
and whatever VM was up that day. On 2026-07-29 the recorded best was 28,821 while
the verified min over everything was 28,718 — 103 moves unclaimed because nothing
merged them. On 2026-08-03 this script's first run found 28,559 against a
hand-merged 28,561, because the hand-merge scanned `results/top100/` but not
`results/v6e8/`. Both misses were silent.

See memory `[[merge-all-sessions-before-quoting-score]]` and CLAUDE.md Rule 26.
