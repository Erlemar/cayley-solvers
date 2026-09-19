# IHES search campaign, 2026-09-06

## STOPPED BY USER — 2026-09-12, 15:15 UTC

No shorter length-24 paths were found. Best verified total remains **21,870**;
`submissions/ihes_20260912_1410_verified.csv` has 1003/1003 valid paths.
Local searches, queue, and collectors were stopped; CIM confirmed no matching
processes remained. Kaggle's Active Events UI confirmed shards 1/3/4/5 cancelled,
shard 2 successful, and zero active events. The two-hour automation is PAUSED.
Do not resume searches or monitoring without a new user request.

The interrupted local 810/root 21 journal had no hits when applied without search.
Interrupted cases remain unresolved. No global optimality proof for length 24.
An optional final API/record-update script was rejected by automatic approval
review and removed without execution; shutdown verification used the UI and CIM.
Historical RUNNING entries below and collector/queue status files are superseded
by this shutdown note. Existing submissions, journals, and logs are preserved.

User objective: shorten `submission_ihes.csv` to a total of at most 21,839.
Authorized compute: local machine and existing free Kaggle quota; no paid cloud.
User additionally authorized **up to five simultaneous CPU Kaggle notebooks**.

## Latest state, 2026-09-12

### USER PRIORITY CHANGE: focus on incumbent length 24

14:10 UTC check: **706/f0.-f1 (root 3) and 764/f0.r1 (root 8) fully
exhausted**, 262 suffixes each, zero timeouts/hits, native time 3694.06 s and
24719.50 s (764 combines wave-24/26). 936/-f0.r0 (root 22) first 128 suffixes
also exhausted in 13488.21 s; parent still partial. Prefix/native/wave audits
and 1003-path replay passed. Total **21,870**, zero saved; no whole-puzzle proof.
Audit `data/ihes_pdb/status_20260912_1410.json`; combined valid artifact
`submissions/ihes_20260912_1410_verified.csv`. Baseline preserved.

**Local queue 1210 continues**, launcher **48972**. 706 root 3 complete/audited;
706/f0.-r1 root **9** active, child **27884**, 242 suffixes exhausted at this
check; 810/-f0.-f2 root **21** queued. Queue/native errors empty. Existing
queue and candidate path conventions below remain current; do not duplicate.

**Remote five CPU notebooks**: wave-28 slots 1/2/3 continue unchanged;
wave-26/27 collectors FINISHED. New **wave 29** uses the freed slots:

- Slot 4: PID 936, -f0.r0 (root **22**), ONLY remaining IDs **128--261**,
  canonical shard-4 ref version **20**. Combine with audited wave-27 first half.
- Slot 5: PID 764, f0.-d1 (root **15**), IDs **0--127**, canonical shard-5
  ref version **18**. Later continue only IDs 128--261.

Both confirmed RUNNING. Same k=4/residual depth 18, 1200 s per suffix, 8 GiB,
nine-hour cap. New 764 selection excludes exhausted roots 2/8 and identity 1;
selection cost/source hashes recorded in manifest.

- Wave ID: `20260912_wave29_L24_next_parents`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave29_manifest.json`, `cpu_wave29_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave29_shard4`, `ihes_wave29_shard5`.
- Collector stem: `data/ihes_pdb/cpu_wave29_collector`, launcher **63944**.
- Collection: `submissions/ihes_20260912_wave29/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260912_wave29_verified.csv`.

Wave-28 collector **60940** remains active; all collector errors empty.
Exhausted roots: 810 **7,10,18,22,9,15**; 706 **16,13,0,7,10,5,3**;
936 **18,19**; 106 **19,22**; 592 **7,10**; 764 **2,8**; 680 **18**.
Next: audit local queue completions and remote continuation/half-parent results;
continue only missing suffixes, keeping all searches on the seven L24 PIDs.

### Previous checks

12:10 UTC check: **four parents fully exhausted**: 706/f0.-f2 (root 5,
4960.88 s), 810/f0.-d1 (root 15, 5413.82 s), 106/-f0.r0 (root 22,
26161.13 s across wave-23/26), 592/f0.r2 (root 10, 22391.77 s across
wave-24/26). Each has all 262 suffixes, no timeouts/hits. 680/f0.f2 root 4
first 128 suffixes also exhausted in 15654.93 s, still a partial parent.
Prefix/native/wave audits and 1003-path replay passed; **21,870**, zero saved.
No whole-puzzle optimality claim. Audit `data/ihes_pdb/status_20260912_1210.json`;
combined valid artifact `submissions/ihes_20260912_1210_verified.csv`.

**Current local: NEW queue 1210**, launcher **48972**; prior 0809 queue FINISHED.
Stem for script/plan/status/log/error/PID: `data/ihes_pdb/local_queue_20260912_1210`.
Three sequential jobs, all 262 suffixes per parent:

1. PID 706, f0.-f1, root **3**, active at launch.
2. PID 706, f0.-r1, root **9**.
3. PID 810, -f0.-f2, root **21**.

Same one-worker limits: k=4/residual depth 18, 300 s per suffix, 16 threads,
8 GiB. Startup cache/bounds verified, errors empty. Per-job stems:
`data/ihes_pdb/local_L24_split4_<pid>_p<root>_20260912`;
candidates `submissions/ihes_20260912_L24_split4_<pid>_p<root>.csv`.
Read queue `.status.json` before any local launch; do not duplicate queued jobs.

**Remote five CPU notebooks**: wave-27 slot 4 (936 root 22, IDs 0--127) and
wave-26 slot 5 (764 root 8, IDs 128--261) continue. Wave-25 collector FINISHED;
wave-26 already marked slots 1/2 finished before reuse. **New wave 28**, all
confirmed RUNNING, same k=4/residual depth 18, 1200 s, 8 GiB, nine-hour cap:

- Slot 1: PID 106, -f0.-r0, root **23**, IDs **0--127**, version **21**.
- Slot 2: PID 592, f0.f2, root **4**, IDs **0--127**, version **18**.
- Slot 3: PID 680, f0.f2, root **4**, ONLY remaining **128--261**, version **18**.

New-parent selection excludes identity root 1 and all exhausted roots; costs
and source hashes in manifest. Combine 680 continuation with wave-25 first half.

- Wave ID: `20260912_wave28_L24_next_parents`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave28_manifest.json`, `cpu_wave28_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave28_shard1`, `shard2`, `shard3`.
- Collector stem: `data/ihes_pdb/cpu_wave28_collector`, launcher **60940**.
- Collection: `submissions/ihes_20260912_wave28/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260912_wave28_verified.csv`.

Other collectors: wave 26 **55744**, wave 27 **40348**.
Exhausted roots: 810 **7,10,18,22,9,15**; 706 **16,13,0,7,10,5**;
936 **18,19**; 106 **19,22**; 592 **7,10**; 764 **2**; 680 **18**.
Next: audit completed queue and continuation results; continue new half-parent
slices only at missing suffix IDs. Baseline unchanged, all work remains L24.

10:09 UTC check: **706/f0.r2 (root 10) and 936/-f0.-f1 (root 19) fully
exhausted**, 262 suffixes each, zero timeouts/hits. Native times 3637.50 s and
21887.91 s (936 combines wave-23/25 slices). Prefix/native/wave audits and
1003-path replay passed; **21,870**, zero saved. These remain opening exclusions,
not whole-puzzle optimality proofs. Evidence: `data/ihes_pdb/status_20260912_1009.json`;
combined valid artifact: `submissions/ihes_20260912_1009_verified.csv`.

**Local queue 0809 remains active**, launcher **62396**. 706 root 10 is
complete/audited; 706/f0.-f2 root **5** active, child **63160**, 197 suffixes
exhausted at this check; 810/f0.-d1 root **15** queued. Existing queue and
per-job paths below remain current. Error logs empty. Do not duplicate work.

**Remote: five CPU notebooks**. Wave-25 slot 3 (680 root 4, first 128 suffixes)
and wave-26 slots 1/2/5 (106 root 22, 592 root 10, 764 root 8, remaining
134 suffixes) remain RUNNING. Wave-25 slot 4 output was collected and marked
finished before reuse, so its collector will not claim the replacement wave.

**New wave 27**, slot 4: PID **936**, -f0.r0 (root **22**), suffix IDs
**0--127**, canonical shard-4 ref version **19**, confirmed RUNNING. Same
k=4/residual depth 18, 1200 s per suffix, 8 GiB, nine-hour cap. Selected by
completed depth-19 cost after excluding roots 18/19 and identity root 1.

- Wave ID: `20260912_wave27_L24_next_parents`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave27_manifest.json`, `cpu_wave27_launch.json`.
- Package: `kaggle_notebooks/ihes_wave27_shard4`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave27_collector` (launcher **40348**).
- Collection: `submissions/ihes_20260912_wave27/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260912_wave27_verified.csv`.

Other collectors: wave 25 **57064**, wave 26 **55744**, errors empty.
Exhausted roots: 810 **7,10,18,22,9**; 706 **16,13,0,7,10**;
936 **18,19**; 106 **19**; 592 **7**; 764 **2**; 680 **18**.
Next: audit completed slices and queue jobs. Continue 680/root 4 and new
936/root 22 only at IDs 128--261 after first slices settle; combine other
continuations with their earlier halves before full-parent claims.

08:09 UTC check: **706/f0.-r0 (root 7) and 810/f0.-r1 (root 9) fully
exhausted**, 262 suffixes each, 3731.56 s and 4003.49 s native time. Remote
106 root 22, 592 root 10, 764 root 8 each exhausted their first 128 suffixes
in 15723.08 / 9378.52 / 9530.60 s. All zero timeouts/hits. Full-prefix/native/
wave audits and 1003-path replay passed; **21,870**, zero saved. The three
remote parents are partial, not yet excluded. No whole-puzzle optimality proof.
Evidence: `data/ihes_pdb/status_20260912_0809.json`; combined valid artifact:
`submissions/ihes_20260912_0809_verified.csv`. Baseline preserved.

**Current local: NEW sequential queue 0809**, launcher **62396**. Old 0407
queue FINISHED. Stem for script/plan/status/log/error/PID:
`data/ihes_pdb/local_queue_20260912_0809`. Three queued parents:

1. PID 706, f0.r2, root **10**, active at launch.
2. PID 706, f0.-f2, root **5**.
3. PID 810, f0.-d1, root **15**.

One native worker, 262 suffixes per parent, k=4/residual depth 18, 300 s,
16 threads, 8 GiB. Read `.status.json` for current child and candidate.
Per-job stems `data/ihes_pdb/local_L24_split4_<pid>_p<root>_20260912`;
candidates `submissions/ihes_20260912_L24_split4_<pid>_p<root>.csv`.
Startup bounds/cache verified; errors empty. Do not duplicate queued work.

**Remote: wave-25 slots 3/4 continue**, with 680 root 4 first 128 suffixes
and 936 root 19 remaining 134 suffixes. Wave-23/24 collectors FINISHED.
**New wave 26** resumes ONLY suffix IDs **128--261** for:

- Slot 1: 106/-f0.r0 (root 22), canonical shard-1 ref version **20**.
- Slot 2: 592/f0.r2 (root 10), canonical shard-2 ref version **17**.
- Slot 5: 764/f0.r1 (root 8), canonical shard-5 ref version **17**.

All three confirmed RUNNING; total five CPU notebooks. Same k=4, residual
depth 18, 1200 s per suffix, 8 GiB, nine-hour cap. Wave ID
`20260912_wave26_L24_remaining`; manifest/receipt
`data/ihes_pdb/cpu_wave26_manifest.json` / `cpu_wave26_launch.json`;
packages `kaggle_notebooks/ihes_wave26_shard1`, `shard2`, `shard5`.
Collector stem `data/ihes_pdb/cpu_wave26_collector`, launcher **55744**;
collection `submissions/ihes_20260912_wave26/collection_status.json`;
eventual CSV `submissions/ihes_20260912_wave26_verified.csv`.
Wave-25 collector **57064** remains active.

Exhausted parents: 810 **7,10,18,22,9**; 706 **16,13,0,7**;
936 **18**; 106 **19**; 592 **7**; 764 **2**; 680 **18**.
Next: combine continuations with earlier slices before full-parent claims.
Exclude identity root 1 and all exhausted/active parents from future selections.

06:07 UTC check, continued at 06:11: **706/f0.f0 (root 0) and 680/-f0.f1
(root 18) fully exhausted**, each 262 suffixes, zero timeouts/hits. Recorded
native times: 3660.55 s and 33980.78 s (680 combines wave-20/22 slices).
936/-f0.-f1 (root 19) first 128 suffixes also exhausted in 9103.24 s; its parent
is still partial. Prefix/native/wave checks and 1003-path replay passed.
Total **21,870**, zero saved; no whole-puzzle optimality claim.
Audit: `data/ihes_pdb/status_20260912_0607.json` (explicit full vs partial
coverage); combined valid artifact: `submissions/ihes_20260912_0607_verified.csv`.

**Local queue 0407 continues**, launcher **54820**: 706 root 0 complete/audited;
706/f0.-r0 root 7 active (child **58648**, suffix IDs 0--250 exhausted at
06:11); 810/f0.-r1 root 9 queued. Check queue `.status.json` for transitions,
do not duplicate jobs. Queue and active native error logs empty.

**Five remote CPU jobs**: wave-23 slot 1 (106 root 22, suffixes 0--127),
wave-24 slots 2/5 (592 root 10 / 764 root 8, suffixes 0--127), and new
**wave 25 slots 3/4**, all confirmed RUNNING. Wave-22 collection is FINISHED;
wave-23 collector has marked slot 4 finished before reuse.

Wave-25 assignments, k=4/residual depth 18, 1200 s per suffix, 8 GiB, nine-hour cap:

- Slot 3: PID 680, f0.f2 (root **4**), suffix IDs **0--127**, version **17**.
- Slot 4: PID 936, -f0.-f1 (root **19**), ONLY remaining IDs **128--261**,
  version **18**. Combine with wave-23's audited first 128 later.

680 selection excludes exhausted root 18 and cancelling identity root 1;
root 1 cannot begin a shortest 22-move path. Final uploaded package is the
`wave25b` slot-3 revision, with root 4; the earlier prepared root-1 package
was NEVER uploaded. Manifest records actual notebook hash and selection.

- Wave ID: `20260912_wave25_L24_next_parents`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave25_manifest.json`, `cpu_wave25_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave25b_shard3`, `ihes_wave25_shard4`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave25_collector` (launcher **57064**).
- Collection: `submissions/ihes_20260912_wave25/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260912_wave25_verified.csv`.

Other collectors: wave 23 **61532**, wave 24 **39888**; error logs empty.
Exhausted roots: 810 **7,10,18,22**; 706 **16,13,0**; 936 **18**;
106 **19**; 592 **7**; 764 **2**; 680 **18**. Root IDs remain canonical
TWO-move parent IDs; journal prefix IDs are the local suffix IDs.
Next: audit local queue completions and remote slices; continue only missing
suffixes. For future new-parent selection, exclude identity root 1 as well as
exhausted/active roots. All active and new work remains on the seven L24 PIDs.

04:07 UTC check, continued at 04:09: **three more parents fully exhausted**:
810/-f0.r0 (root 22, 3723.99 s), 592/f0.-r0 (root 7, 28940.57 s across
wave-20/21 slices), 764/f0.f1 (root 2, 27466.64 s across wave-20/22 slices).
Each covers exactly 262 suffixes with no duplicates, timeouts or hits; prefix,
native completion/exhaustion, wave identity and 1003-path replay checks passed.
Still only opening exclusions, not whole-puzzle proofs. Total **21,870**.
Audit: `data/ihes_pdb/status_20260912_0407.json`; combined valid artifact:
`submissions/ihes_20260912_0407_verified.csv`. Baseline preserved.

**New current local queue**, launcher **54820**, stem (script/plan/status/log/
error/PID): `data/ihes_pdb/local_queue_20260912_0407`. Old 0005 queue FINISHED.
Three sequential parents, one native worker, 262 suffixes each:

1. PID 706, f0.f0 (root **0**), active at launch, child **65060**.
2. PID 706, f0.-r0 (root **7**), queued.
3. PID 810, f0.-r1 (root **9**), queued.

Same k=4/residual depth 18, 300 s per suffix, 16 threads, 8 GiB. First job
cache/additive bounds loaded; errors empty. Per-job stems:
`data/ihes_pdb/local_L24_split4_<pid>_p<root>_20260912`;
candidates: `submissions/ihes_20260912_L24_split4_<pid>_p<root>.csv`.
Inspect `.status.json` for current job before launching any local work.
Plans exclude all exhausted roots; the queue stops on failure or improvement.

**Remote five CPU jobs**: wave-23 slots 1/4 (106 root 22 / 936 root 19,
suffix IDs 0--127), wave-22 slot 3 (680 root 18, IDs 64--261), plus new
**wave 24** on completed slots 2/5. All provider states confirmed RUNNING.
Wave-21 fully collected and finished; wave-22 marked slot 5 finished before reuse.

Wave 24 searches ONLY suffix IDs **0--127** under new parents:

- Slot 2: PID 592, f0.r2 (root **10**), canonical shard-2 ref, version **16**.
- Slot 5: PID 764, f0.r1 (root **8**), canonical shard-5 ref, version **16**.

Same tested runner, k=4, residual depth 18, 1200 s per suffix, 8 GiB, nine-hour
cap. Parent selection uses completed depth-19 cost, excluding exhausted roots;
source hashes and costs are in the manifest. Remaining suffixes 128--261 are
NOT included; continue them later without repeating settled work.

- Wave ID: `20260912_wave24_L24_next_parents`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave24_manifest.json`, `cpu_wave24_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave24_shard2`, `ihes_wave24_shard5`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave24_collector` (launcher **39888**).
- Collection: `submissions/ihes_20260912_wave24/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260912_wave24_verified.csv`.

Other active collectors: wave 22 **40984**, wave 23 **61532**.
Exhausted parent inventory: 810 **7,10,18,22**; 706 **16,13**;
936 **18**; 106 **19**; 592 **7**; 764 **2**. 680 parent 18 still pending.
Next: audit 680 continuation with its earlier slice, plus completed queue jobs;
continue wave-23/24 parents only at unsearched suffixes after their first slices.

02:07 UTC check, continued at 02:12: **four more complete parents exhausted**:
706/f0.-d0 (root 13, 3300.82 s), 810/-f0.f1 (root 18, 3649.05 s),
936/-f0.f1 (root 18, wave-19 pilot plus wave-20 continuation, 24443.51 s),
106/-f0.-f1 (root 19, wave-20 slice plus wave-22 continuation, 20875.69 s).
Each union covers exactly suffix IDs 0--261, no duplicates, timeouts or hits.
Prefix identity, depth-18 native exhaustion, normal finish, wave identity and
1003-path candidate replays verified. These are ONLY opening exclusions,
not whole-puzzle optimality proofs. Total remains **21,870**, zero saved.
Audit and source hashes: `data/ihes_pdb/status_20260912_0207.json`.
Combined artifact: `submissions/ihes_20260912_0207_verified.csv`.

**Current local: third/final queued parent 810/-f0.r0 (root 22)**, child
**58028**, under queue launcher **19136**. First two queue jobs complete and
audited. At 02:12 suffix IDs 0--29 exhausted; current work continues normally,
error log empty. Queue status/paths remain those in the 00:05 entry below.
Do not start a competing local worker. Completed parent inventory:
810 roots **7,10,18**; 706 roots **16,13**; 936 root **18**; 106 root **19**.

**Remote: five CPU notebooks across waves 21/22/23**. Existing wave-21 slot 2
(592/root 7 suffixes 64--261) and wave-22 slots 3/5 (680/root 18 and 764/root 2,
suffixes 64--261) remain RUNNING. Wave-20 is fully collected and its collector
finished. Wave-22 already marked slot 1 finished before that ref was reused.

**New wave 23**, confirmed RUNNING, begins suffix IDs **0--127** for:

- Slot 1: PID 106, -f0.r0 (root **22**), canonical shard-1 ref, version **19**.
- Slot 4: PID 936, -f0.-f1 (root **19**), canonical shard-4 ref, version **17**.

Both selected by cheapest recorded completed depth-19 cost excluding exhausted
parents; costs/source hashes are in the manifest. Same k=4/residual depth 18,
1200 s per suffix, 8 GiB, nine-hour cap. This 128-case slice remains partial;
continue ONLY suffix IDs 128--261 if all initial cases settle without a hit.

- Wave ID: `20260912_wave23_L24_next_parents`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave23_manifest.json`, `cpu_wave23_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave23_shard1`, `ihes_wave23_shard4`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave23_collector` (launcher **61532**).
- Collection: `submissions/ihes_20260912_wave23/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260912_wave23_verified.csv`.

Other active collectors: wave 21 **52184**, wave 22 **40984**; error logs empty.
Next: audit 592/680/764 continuation results with their earlier 64-case slices,
and the local queue's final parent when finished. Keep all new work on L24 PIDs.

00:05 UTC check, continued at 00:08: **PID 810 parent f0.r2 (root 10)
fully exhausted**, 262 suffix cases, zero timeouts/hits, 3560.8 s native time.
Full prefix identity, native exhaustion and completion, and 1003-path replay
verified. This is PID 810's second excluded opening (roots 7 and 10), still
NOT a proof that its 24-move path is optimal. Total remains **21,870**.
Audit: `data/ihes_pdb/status_20260912_0005.json`; combined valid artifact:
`submissions/ihes_20260912_0005_verified.csv`. Original baseline untouched.

**Current local: bounded sequential queue of THREE new parents**, one native
worker at a time. Prevents idle time between two-hour checks. Each uses all
262 suffixes, k=4/residual depth 18, 300 s per case, 16 threads, 8 GiB:

1. PID 706, f0.-d0 (root **13**), active at launch; child PID 60036.
2. PID 810, -f0.f1 (root **18**), queued.
3. PID 810, -f0.r0 (root **22**), queued.

Queue launcher **19136**; script/log/error/PID/plan/status stem:
`data/ihes_pdb/local_queue_20260912_0005` (status suffix `.status.json`).
The status file identifies the CURRENT child and candidate; inspect it before
starting any local work. Per-job journal/native/log/error stems are
`data/ihes_pdb/local_L24_split4_<pid>_p<root>_20260912`.
Candidates are `submissions/ihes_20260912_L24_split4_<pid>_p<root>.csv`.
The queue replay-verifies each candidate and stops on failure or any improvement.
It does NOT itself certify parent exhaustion; audit journals/native logs later.
Plans preserve selection costs and source hashes, excluding known exhausted
parents. Costs rank work only, not solution likelihood. First job's cache and
additive bounds loaded; queue and child error logs empty.

**Remote unchanged: all five actual provider statuses RUNNING** at this check:
wave-22 slots 1/3/5, wave-21 slot 2, wave-20 slot 4. Existing collectors
40984/52184/62180 active with empty error logs; no uploads/restarts this check.
Assignments, versions and remote collection paths are in the 22:05 entry below.
Next: audit completed local jobs and remote continuations, combining earlier
slices before claiming full-parent coverage. Do not duplicate queued local jobs.

22:05 UTC check, continued at 22:10: **PID 706 parent f0.d2 fully exhausted**,
262 suffixes, zero timeouts/hits, 3462.07 s native time. Wave-20 slots 1/3/5
also completed their 64-suffix slices: PID 106 in 7459.78 s, 680 in 7874.65 s,
764 in 7119.85 s, all zero timeouts/hits. Prefix identity, exact native depth-18
exhaustion, normal completion, wave identity and 1003-path replay verified.
Only the selected opening of 706 is ruled out; no 24-move puzzle is proved
optimal. Combined valid artifact: `submissions/ihes_20260911_2205_verified.csv`,
**21,870**, zero saved. Audit: `data/ihes_pdb/status_20260911_2205.json`;
reproducible check: `data/ihes_pdb/check_20260911_2205.py`.

**Current local: PID 810 parent f0.r2 (root ID 10), suffix IDs 0--261**.
Previous 810 parent 7 and 706 parent 16 are complete; do not repeat them.
This parent was selected by cheapest recorded completed depth-19 cost among
remaining original 810 roots (50.546 s); selection provenance:
`data/ihes_pdb/local_L24_next_20260911_2205.plan.json`.
k=4/residual depth 18, 300 s each, 16 threads, 8 GiB. Stem for log/journal/
native/error/PID: `data/ihes_pdb/local_L24_split4_810_p10_20260911`
(launcher **26820**). Candidate:
`submissions/ihes_20260911_local_L24_split4_810_p10.csv`.
Startup confirmed table load and both additive bounds; error log empty.

**Current remote: five CPU jobs across waves 20/21/22**. Wave-20 slot 4
(936/-f0.f1 suffix IDs 24--261) and wave-21 slot 2 (592/f0.-r0 IDs 64--261)
remain RUNNING. Their collectors 62180 and 52184 remain active.

Wave 22 continues ONLY suffix IDs **64--261** under these existing parents:

- Slot 1: PID 106, -f0.-f1; canonical shard-1 ref, version **18**.
- Slot 3: PID 680, -f0.f1; canonical shard-3 ref, version **16**.
- Slot 5: PID 764, f0.f1; canonical shard-5 ref, version **15**.

All three confirmed RUNNING. k=4, residual depth 18, 1200 s per suffix,
8 GiB, nine-hour cap. The wave-20 collector already marked these refs finished
and will not download the replacement versions.

- Wave ID: `20260911_wave22_L24_remaining`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave22_manifest.json`, `cpu_wave22_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave22_shard1`, `shard3`, `shard5`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave22_collector` (launcher **40984**).
- Collection: `submissions/ihes_20260911_wave22/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260911_wave22_verified.csv`.

Next: combine each continuation with its earlier slice before assessing
full-parent coverage. All seven L24 puzzles remain the priority; no new L22 work.

20:05 UTC check, continued at 20:08: **PID 810 parent f0.-r0 is fully
exhausted**: all 262 canonical suffixes, zero timeouts/hits, 3717.5 s recorded
native search. This rules out only that two-move opening for a <=22 solution;
it does NOT prove the 24-move path optimal. Wave-20 slot 2 (PID 592, same
parent) exhausted suffix IDs 0--63 in 6202.18 s. Full-prefix identity, native
completion, wave identity and 1003-path candidate replay checks passed.
Both candidates remain **21,870**, zero saved. Audit and source hashes:
`data/ihes_pdb/status_20260911_2005.json`.

**Current local: PID 706 parent f0.d2 (root ID 16), suffix IDs 0--261**.
Selected by the cheapest completed depth-19 cost in the old PID 706 run
(528.383 s), as a cost heuristic. k=4/residual depth 18, 300 s each, 16 threads,
8 GiB. Log/journal/native/error/PID stem:
`data/ihes_pdb/local_L24_split4_706_p16_20260911` (launcher 63924).
Candidate: `submissions/ihes_20260911_local_L24_split4_706_p16.csv`.
Startup verified; native search is exhausting cases normally, error log empty.
Completed 810 candidate: `submissions/ihes_20260911_local_L24_split4_810_p7.csv`.

**Current remote: four wave-20 jobs plus wave-21 slot 2**, five CPU notebooks
total. Wave-20 slots 1/3/4/5 retain the assignments below and remain RUNNING;
collector 62180 continues. Slot 2's completed wave-20 output was collected
before reuse, and the old collector has marked that ref finished.

Wave 21 resumes ONLY PID 592/f0.-r0 suffix IDs **64--261** (198 cases), k=4,
residual depth 18, 1200 s each, 8 GiB, nine-hour cap. Confirmed RUNNING at
`artgor/ihes-exact-cpu-wave-2-shard-2`, version **15**.

- Wave ID: `20260911_wave21_L24_592_p7`.
- Manifest/receipt: `data/ihes_pdb/cpu_wave21_manifest.json`, `cpu_wave21_launch.json`.
- Package: `kaggle_notebooks/ihes_wave21_shard2`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave21_collector` (launcher 52184).
- Collection: `submissions/ihes_20260911_wave21/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260911_wave21_verified.csv`.

Next: collect and audit finished jobs; for wave-20 slots 1/3/5, continue only
unsearched suffix IDs under their existing parents. Combine 936's wave-19 pilot
with wave-20 continuation before assessing full-parent coverage. No new L22 work.

18:05 UTC check, continued at 18:09: **split4 pilots both finished without
timeouts**. Local 810 parent f0.-r0: suffix IDs 0--15 exhausted in 232.45 s.
Remote 936 parent -f0.f1: suffix IDs 0--23 exhausted in 2401.55 s. Independent
audit checked every full four-move prefix, suffix ID, native exhaustion marker,
normal completion, wave identity and candidate replay. Neither parent is yet
fully covered; neither puzzle is proved optimal. All candidates remain valid at
**21,870**, zero saved. Four old wave-18 jobs finished with 24 timeouts each and
valid unchanged candidates. Status/audit: `data/ihes_pdb/status_20260911_1805.json`.

**Current local: continue PID 810 parent f0.-r0**, suffix IDs **16--261** only,
300 s each, 16 threads, 8 GiB, k=4/residual depth 18. Uses the SAME journal and
native log `data/ihes_pdb/local_L24_split4_810_p7_20260911.jsonl` / `.solver.log`,
preserving the 16 pilot records. NEW launcher/log/error/PID stem:
`data/ihes_pdb/local_L24_split4_810_p7_resume1_20260911` (launcher 36504).
Candidate remains `submissions/ihes_20260911_local_L24_split4_810_p7.csv`.
Startup confirms 16 settled records and only 246 queued cases; no overlap.

**Current remote wave 20: five L24 fixed-parent jobs**, k=4, residual depth 18,
1200 s per suffix, 8 GiB, nine-hour cap. All five confirmed RUNNING. Wave ID:
`20260911_wave20_L24_parents`. Canonical refs remain shard 1 through 5;
versions **17/14/15/16/14**. Assignments:

- Slot 1: PID 106, parent -f0.-f1 (root ID 19), suffix IDs 0--63.
- Slot 2: PID 592, parent f0.-r0 (root ID 7), suffix IDs 0--63.
- Slot 3: PID 680, parent -f0.f1 (root ID 18), suffix IDs 0--63.
- Slot 4: PID 936, parent -f0.f1 (root ID 18), ONLY remaining suffix IDs 24--261.
- Slot 5: PID 764, parent f0.f1 (root ID 2), suffix IDs 0--63.

The four new parents were chosen by cheapest completed depth-19 cost in the
old timed-out jobs, not a claimed likelihood of improvement. Source hashes and
selection costs recorded by `scripts/73_prepare_ihes_l24_wave20.py`.

- Manifest/receipt: `data/ihes_pdb/cpu_wave20_manifest.json`, `cpu_wave20_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave20_shard1` through `shard5`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave20_collector` (launcher 62180).
- Collection: `submissions/ihes_20260911_wave20/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260911_wave20_verified.csv`.

All prior collectors finished; new errors empty and process start times verified.
PID 706's old depth-20 cases remain unresolved, not active this batch. No new
L22 search. Future audits must distinguish parent and suffix IDs; completing a
parent proves only that opening cannot lead to <=22, not that the 24 path is optimal.

16:05 UTC check, continued at 16:12: local four-case 1800-second escalation
finished with **4 timeouts**, no hits/exhaustions. Wave-17 PID 706 also completed
normally with **24 timeouts**. Every available candidate replayed 1003/1003 valid
at unchanged **21,870**. Evidence: `data/ihes_pdb/status_20260911_1605.json`.
The selected depth-20 tasks remain too large even with longer budgets; do not
repeat the same shallow partition and time limits indefinitely.

**New strategy under test: refine a timed-out two-move parent with two more
moves (k=4), leaving residual depth 18.** Added `--fixed-prefix=<dot-separated
moves>` to `scripts/27_prefix_split_ladder.py`. It enumerates all 262 distinct
two-move suffixes under that parent; `prefix_id` is now the SUFFIX ID, and the
record's `prefix` contains all four moves. Full coverage of one parent is still
only partial coverage of the original puzzle. Never add these suffix counts to
the old 262 root counts or use old L22 proof scripts unchanged.

Validation: `scripts/71_test_ihes_fixed_prefix.py` checked equality with all 324
two-move words for three parents and recovered a known native solution; all
1003 candidate paths replayed. Five existing parser failure regression tests
passed. Control evidence: `data/ihes_pdb/fixed_prefix_control_1789142953782227800`.
The known-solution control is NOT a submission improvement.

**Active local split4 pilot: PID 810, parent f0.-r0 (old parent ID 7)**,
suffix IDs 0--15, k=4, residual depth 18, 300 s each, 16 threads, 8 GiB.
Journal/log/error/PID stem: `data/ihes_pdb/local_L24_split4_810_p7_20260911`
(launcher 60044). Candidate:
`submissions/ihes_20260911_local_L24_split4_810_p7.csv`.
At 16:12, first eight suffix cases exhausted in roughly 9--20 s each, no hits.
This is tractable partial coverage, not a score gain or completed parent proof.
Let pilot finish; if evidence remains sound, resume only remaining suffix IDs
under the same parent using the same journal and matching fixed prefix.

**Active remote wave 19: PID 936, parent -f0.f1 (old parent ID 18)**,
slot 4 `artgor/ihes-exact-cpu-wave-2-shard-4` version **15**, confirmed RUNNING.
Suffix IDs 0--23, k=4, residual depth 18, 1200 s each, 8 GiB, nine-hour cap.
Wave ID `20260911_wave19_L24_936_p18`.

- Manifest/receipt: `data/ihes_pdb/cpu_wave19_manifest.json`, `cpu_wave19_launch.json`.
- Package: `kaggle_notebooks/ihes_wave19_shard4`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave19_collector` (launcher 4008).
- Collection: `submissions/ihes_20260911_wave19/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260911_wave19_verified.csv`.
- Builder `scripts/72_prepare_ihes_l24_split4.py` embeds the tested updated runner
  and records its SHA256; current notebook gates remain enabled. Use this builder
  for fixed-parent remote runs; the old template contains the older runner.

Old wave 18 (106/592/680/764, slots 1/2/3/5) is finishing, existing collector
61332 remains active. Slot 1 just reported COMPLETE at 16:12, collection pending;
do not reuse any of these refs until that job's outputs are preserved and verified.
Old wave-17 collector finished. Current errors empty. No score improvement.

14:05 UTC heartbeat: first three local 1800-second escalation cases timed out;
fourth case (936, prefix 18) remains running. Latest candidate independently
replayed 1003/1003 valid at unchanged **21,870**. No hits/exhaustions so far.
All five remote L24 kernels remain RUNNING through fresh provider checks;
no completed outputs. Existing solver/collector processes alive with matching
start times; error logs empty. Let current jobs finish without restarts or
duplicate work. Snapshot: `data/ihes_pdb/status_20260911_1405.json`.

12:05 UTC check: local L24 round 1 finished normally, **48/48 timeouts, no
hits or exhausted prefixes** (24 cases each for 810/936). Native logs confirm
all reached depth 20 before timing out; the five-minute cap is insufficient
for these sampled cases. Candidate independently replayed 1003/1003 valid at
unchanged **21,870**: `submissions/ihes_20260911_local_L24_810_936_round1.csv`.
All five remote wave-17/18 kernels remain RUNNING, error logs empty. Snapshot:
`data/ihes_pdb/status_20260911_1205.json`.

**Active local escalation 1**, launched 12:07 UTC: retry four previously timed
out cases at **1800 s each** (six times the prior budget), 16 threads, 8 GiB,
residual depth 20. Order: **(810, prefix 7), (936, prefix 19), (810, prefix 10),
(936, prefix 18)**. Selected as the two cheapest completed depth-19 searches
per PID; this is a cost heuristic, not evidence that solutions are more likely.
At most about two hours total; do not rerun the whole 48-case batch unchanged.

- Wrapper: `scripts/70_ihes_l24_local_escalation.py`.
- Journal/log/error/PID stem: `data/ihes_pdb/local_L24_escalation1_20260911`
  (launcher 30328). Plan/provenance/results: same stem plus `.plan.json`.
- Candidate, replayed after each successful native process:
  `submissions/ihes_20260911_local_L24_escalation1.csv`.
- Fresh unseeded journal contains only the longer-budget attempts. Source
  round-1 journal/native hashes are recorded in the plan. Any timeouts remain
  unresolved; successful exhaustion is only partial prefix coverage.
- Native bounds loaded and first selected case running; error log empty.

Remote waves 17 (706, slot 4) and 18 (106/592/680/764, slots 1/2/3/5) continue
unchanged with existing collectors. No additional remote launches or overlap.

10:05 heartbeat, checked 10:06 UTC: all five L24 kernels remain RUNNING through
fresh provider calls; no completed remote outputs. Local round 1 has 24 timeouts
for PID 810 and 22 for PID 936, zero hits/exhaustions. Last two PID 936 cases
still in progress; leave existing worker running. These are unresolved searches,
not optimality proofs. All three local processes (solver launcher and two
collectors) are alive with matching start times; error logs empty. No score
improvement; last verified total remains 21,870. Snapshot:
`data/ihes_pdb/status_20260911_1005.json`. Active manifests and paths below.

08:05 UTC check, continued at 08:07: **all active searches now target length 24**.
Old wave-16 PID 834 completed all 262 prefixes and was independently audited
optimal at 22: `data/ihes_pdb/pid834_proof_20260911.json`. All candidate replays
passed at unchanged **21,870**. Its collector finished; no further L22 work.

**Active wave 18: L24 PIDs 106, 592, 680, 764**, assigned to slots 1, 2, 3, 5
respectively. Each covers initial openings 0--23, residual depth 20, 1200 s each,
8 GiB, nine-hour cap. All four new provider statuses confirmed RUNNING. Wave ID
`20260911_wave18_L24_four`. Canonical refs remain
`artgor/ihes-exact-cpu-wave-2-shard-{1,2,3,5}`; versions **16/13/14/13**.

- Manifest/receipt: `data/ihes_pdb/cpu_wave18_manifest.json`, `cpu_wave18_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave18_shard1`, `shard2`, `shard3`, `shard5`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave18_collector` (launcher 61332).
- Collection: `submissions/ihes_20260911_wave18/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260911_wave18_verified.csv`.
- Journals: `prefix_L24.jsonl`; native logs: `prefix_L24.solver.log`.

**Wave 17 continues on L24 PID 706 in slot 4**; current ref/version, paths and
collector 40904 are below. Combined remote concurrency is exactly five CPU
notebooks. **Local 810 then 936 round 1 continues**, launcher 51040. At snapshot,
PID 810 had 22 timeout records, zero hits or exhausted prefixes; PID 936 had
not started. Timeouts leave these prefixes unresolved and are not proofs.
All active process start times match and error logs are empty. No score gain.
Status snapshot: `data/ihes_pdb/status_20260911_0805.json`.
The previously queued slot assignments below are now launched; do not duplicate.

06:05--06:11 UTC: user explicitly requested focusing on the seven 24-move PIDs
**106, 592, 680, 706, 764, 810, 936** to find shorter paths. **Do not launch
further length-22 targets.** All seven previous whole-path depth-22 trials in
`data/ihes_pdb/ordered_L24_trial.jsonl` timed out after about 600 s; none proved
optimal. New trials seek total length <=22 using k=2 prefixes and residual
depth **20**, not the old depth 18. Initial round covers prefixes 0--23 per PID;
partial coverage/timeouts are not proofs. Favor exploring the seven candidates
before sinking all compute into one. Goal and baseline remain <=21,839 / 21,870.

**Active local L24 round 1: PIDs 810 then 936**, 16 threads, 8 GiB cache,
300 s per prefix, opening IDs 0--23 for each (48 work units; at most ~4 h).
Journal/log/error/PID stem: `data/ihes_pdb/local_L24_810_936_round1_20260911`
(launcher 51040). Eventual CSV:
`submissions/ihes_20260911_local_L24_810_936_round1.csv`. Native log confirms
L=24, maxdepth=20 and both additive bounds loaded; error log empty.

**Active remote wave 17: L24 PID 706**, free slot 4 reassigned immediately,
`artgor/ihes-exact-cpu-wave-2-shard-4` version **14**, confirmed RUNNING.
24 prefixes (IDs 0--23), residual depth 20, 1200 s per prefix, 8 GiB, nine-hour
cap. Wave ID `20260911_wave17_L24_706`.

- Manifest/receipt: `data/ihes_pdb/cpu_wave17_manifest.json`, `cpu_wave17_launch.json`.
- Package: `kaggle_notebooks/ihes_wave17_shard4`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave17_collector` (launcher 40904).
- Collection: `submissions/ihes_20260911_wave17/collection_status.json`.
- Eventual CSV: `submissions/ihes_20260911_wave17_verified.csv`.
- Remote journal is now **prefix_L24.jsonl**, native log prefix_L24.solver.log.
  Collector download filter was extended to preserve L24 JSONL as well as L22.

**Next remote assignments as slots become free: slot 1 -> PID 106, slot 2 ->
592, slot 3 -> 680, slot 5 -> 764.** Prepare only free, collected slots with
`scripts/69_prepare_ihes_l24.py --template data/ihes_pdb/cpu_wave16_manifest.json
--manifest <new> --receipt <new> --wave-id <new> --folder-prefix <new>
--shards <free IDs> --pids <matching PIDs>`; default initial range 0--23.
Launch using `scripts/47_launch_ihes_prefix_wave3.py` and a separate collector.
This helper validates length 24, baseline hash, CPU metadata inherited from the
known template, code syntax and notebook hashes. Existing notebook preflight,
native parser regression tests, PDB audits and positive control remain intact.

**Finishing old remote wave 16 only:** slots 1, 2, 3, 5 still RUNNING on PID 834
at this check; keep until completion, preserve outputs, then replace with the
L24 assignments above. Slot 4's old 53 prefixes were collected and independently
audited before reuse: `data/ihes_pdb/status_20260911_0604.json`. Existing old
collector 19740 has already marked slot 4 finished and will ignore its new run.
At most five CPU notebooks remain active across both waves.

**Stopped old local L22 worker deliberately** to redirect compute. Identified
native PID 44376 (parent 44192 / launcher 38704) and stopped only that native
process; Python closed normally with failure code, preserved 334 exhausted
records plus one interrupted record labeled timeout. Original stem:
`data/ihes_pdb/local_prefix122_990_20260911`. PID 122 was already complete and
audited optimal at 22 (5544.37 s): `data/ihes_pdb/pid122_proof_20260911.json`.
PID 990 remains unresolved (72 settled prefixes plus interruption); do not resume
while L24 priority applies. No score improvement, baseline unchanged.

### Earlier work (superseded by L24 priority above)

04:03 UTC check, continued at 04:06: **PIDs 772 and 874 are proved optimal at
22 moves**, all 262 openings exhausted for each, no hits/timeouts. Independent
full-wave audit for 772 and local native/prefix audit for 874 passed:
`data/ihes_pdb/pid772_proof_20260911.json`, `pid874_proof_20260911.json`.
Local 874 took 6020.99 s; completed 170/874 batch replayed 1003/1003 valid.
All completed candidates min-merged at unchanged **21,870**, saved zero:
`submissions/ihes_20260910_wave15_verified.csv`. Baseline unchanged. Snapshot:
`data/ihes_pdb/status_20260911_0403.json`. Exclude 772 and 874 in future searches.

**Current remote wave 16: PID 834**, five disjoint shards covering 262 openings
(53/53/53/53/50), residual depth 18, 1200 s per prefix, 8 GiB cache, nine-hour
cap. All five fresh provider statuses confirmed RUNNING. Wave ID:
`20260911_wave16_split834`; canonical refs remain
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`, versions **15/12/13/13/12**.

- Manifest/receipts: `data/ihes_pdb/cpu_wave16_manifest.json`, `cpu_wave16_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave16_shard1` through `shard5`.
- Collection: `submissions/ihes_20260911_wave16/collection_status.json`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave16_collector` (launcher 19740).
- Eventual CSV: `submissions/ihes_20260911_wave16_verified.csv`.

**Current local targets: PIDs 122 then 990**, 16 threads, 8 GiB cache, 120 s per
prefix, rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix122_990_20260911` (launcher 38704). Eventual candidate:
`submissions/ihes_20260911_local_prefix122_990.csv`. Native bounds loaded;
local/collector error logs empty and start times verified. Previous workers
finished. No overlap. Next unassigned ranked targets: 682, 890, 572.

02:02 UTC check: **PID 170 is proved optimal at 22 moves**, all 262 openings
exhausted in 5615.62 s, no hits/timeouts. Independent native/prefix audit:
`data/ihes_pdb/pid170_proof_20260911.json`. Exclude PID 170 from future searches.
Same local worker continues on PID 874 (83/262 exhausted at snapshot, no hits
or timeouts). Local and collector processes remain alive with matching start
times, error logs empty.

Wave-15 PID 772 shards **1, 3, 4, 5 COMPLETE**, only shard 2 RUNNING through
fresh provider checks. All four completed outputs audited for wave identity,
exact assigned prefix/native exhaustion coverage and normal exit; every candidate
replayed 1003/1003 valid at unchanged **21,870**. **209/262 prefixes settled**,
not a full optimality proof. Snapshot and partial audit:
`data/ihes_pdb/status_20260911_0202.json`. Keep existing workers running without
overlap or restarts. Current paths remain below. No score improvement.

00:01 UTC check: **PID 644 is proved optimal at 22 moves**, all 262 openings
exhausted in 5726.67 s, no hits/timeouts. Independent native/prefix audit:
`data/ihes_pdb/pid644_proof_20260911.json`. Completed local 614/644 batch replayed
1003/1003 valid at unchanged **21,870**:
`submissions/ihes_20260910_local_prefix614_644.csv`. Exclude 614 and 644.

**Current local targets: PIDs 170 then 874**, launched 00:02 UTC, 16 threads,
8 GiB cache, 120 s per prefix, rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix170_874_20260911` (launcher 41456). Eventual candidate:
`submissions/ihes_20260911_local_prefix170_874.csv`. Native bounds loaded, error
log empty. Previous local worker finished. No overlap; next unassigned 834, 122.

**Remote wave 15 remains active on PID 772**. Shard 5 COMPLETE and audited:
correct wave identity, normal exit, all 50 assigned prefixes exhausted with
native evidence, candidate replayed 1003/1003 valid at 21,870. **50/262 settled**,
no full optimality proof. Shards 1--4 remain RUNNING through fresh provider
checks. Keep existing collector (launcher 58308) and remote jobs running.
Manifest/collection paths remain those in the 19:51 entry below. Snapshot with
partial audit: `data/ihes_pdb/status_20260911_0001.json`. No score improvement.

### Previous checks, 2026-09-10

22:00 UTC check: **PID 614 is proved optimal at 22 moves**, all 262 prefixes
exhausted in 7331.81 s, zero hits/timeouts. Independent native/prefix audit:
`data/ihes_pdb/pid614_proof_20260910.json`. Exclude PID 614 from future searches.
The same local worker continues on PID 644 (16/262 exhausted at snapshot,
zero hits/timeouts). All five wave-15 PID 772 kernels remain RUNNING through
fresh provider checks. Local worker and collector are alive with matching start
times, error logs empty. Keep existing workers running without overlap or
restarts. No new score improvement; last verified total remains **21,870**.
Snapshot: `data/ihes_pdb/status_20260910_2200.json`. Active paths below.

19:58 UTC heartbeat: no material change since 19:57. All five wave-15 kernels
still RUNNING; local PID 614 has 17 exhausted prefixes, no hits/timeouts, 644
queued. Processes alive, error logs empty. Continue existing workers. Snapshot:
`data/ihes_pdb/status_20260910_1958.json`. Verified total unchanged at 21,870.

19:57 UTC user-requested continuation: all five wave-15 PID 772 kernels confirmed
RUNNING through fresh API calls. Local PID 614 has exhausted 12/262 openings,
zero hits/timeouts; PID 644 is queued. Existing local worker and collector are
alive with matching start times and empty error logs. Keep the current searches
running without duplicate launches. No new completed outputs or score change;
verified total remains **21,870**. Snapshot:
`data/ihes_pdb/status_20260910_1957.json`. Active paths remain below.

19:51--19:55 UTC execution of the delayed 11:05 heartbeat: **PIDs 996 and 910
are proved optimal at 22 moves**. Local 996 exhausted all 262 openings in
6245.22 s; proof: `data/ihes_pdb/pid996_proof_20260910.json`. Completed local
558/996 candidate replayed 1003/1003 valid. Wave-14 collector had reached its
ten-hour collection deadline; fresh API calls found all five kernels COMPLETE.
Reran collection without restarting kernels, recovered all outputs and audited
wave identity, all 262 prefix/native exhaustion blocks and every candidate:
`data/ihes_pdb/pid910_proof_20260910.json`. Verified min-merge remains **21,870**,
saved zero: `submissions/ihes_20260910_wave14_verified.csv`. Baseline unchanged.
Snapshot `data/ihes_pdb/status_20260910_1105.json` uses the heartbeat label;
its stored Unix timestamp reflects actual execution. Cause of the delay was
not established. Do not infer continuous two-hour checks during this gap.

**Current remote wave 15: PID 772**, five disjoint shards covering 262 openings
(53/53/53/53/50), residual depth 18, 1200 s per prefix, 8 GiB cache, nine-hour
cap. All five provider statuses confirmed RUNNING at 19:55 UTC. Wave ID:
`20260910_wave15_split772`; canonical refs remain
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`, versions **14/11/12/12/11**.

- Manifest/receipts: `data/ihes_pdb/cpu_wave15_manifest.json`, `cpu_wave15_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave15_shard1` through `shard5`.
- Collection: `submissions/ihes_20260910_wave15/collection_status.json`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave15_collector` (launcher 58308).
- Eventual CSV: `submissions/ihes_20260910_wave15_verified.csv`.

**Current local targets: PIDs 614 then 644**, started 19:52 UTC, 16 threads,
8 GiB cache, 120 s per prefix, rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix614_644_20260910` (launcher 18936). Eventual candidate:
`submissions/ihes_20260910_local_prefix614_644.csv`. Native search has already
exhausted initial openings; local and collector error logs empty, process start
times match. Previous workers finished. No overlap. Next unassigned ranked
targets: 170, 874, 834. Add 996 and 910 to the proved exclusion list.

06:34 UTC check, continued at 06:37: **PIDs 46 and 558 are proved optimal at
22 moves**. Wave-13 PID 46 openings 51/52 exhausted in 520.52/308.54 s, no
timeouts. `scripts/68_audit_ihes_pid46.py` independently rechecked all six source
runs, including wave 12's interrupted tail, every prefix/native marker, all
candidate replays and baseline identity. Full 262-opening proof:
`data/ihes_pdb/pid46_proof_20260910.json`. Local 558 exhausted 262 openings in
3474.34 s; proof: `data/ihes_pdb/pid558_proof_20260910.json`. Exclude both targets.
Verified total remains **21,870**, zero saved. Latest merged artifact:
`submissions/ihes_20260910_wave13_verified.csv`.

**Current remote wave 14: PID 910**, five disjoint shards covering all 262
openings (53/53/53/53/50), residual depth 18, 1200 s per prefix, 8 GiB cache,
nine-hour cap. All five fresh provider statuses confirmed RUNNING. Wave ID:
`20260910_wave14_split910`; canonical refs remain
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`, versions **13/10/11/11/10**.

- Manifest/receipts: `data/ihes_pdb/cpu_wave14_manifest.json`, `cpu_wave14_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave14_shard1` through `shard5`.
- Collection: `submissions/ihes_20260910_wave14/collection_status.json`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave14_collector` (launcher 47644).
- Eventual CSV: `submissions/ihes_20260910_wave14_verified.csv`.

**Local worker continues on PID 996**, 156/262 openings exhausted at snapshot,
zero hits/timeouts. Current journal/log/error/PID stem remains
`data/ihes_pdb/local_prefix558_996_20260910` (launcher 59608). Eventual candidate:
`submissions/ihes_20260910_local_prefix558_996.csv`. Both active processes verified
with matching start times; error logs empty. Previous collector finished.
Snapshot before continuation: `data/ihes_pdb/status_20260910_0634.json`.
Next unassigned ranked targets: 614, 644. No overlap or local restart.

04:34 UTC check, continued at 04:37: **PIDs 358 and 604 are proved optimal at
22 moves**, 262 exhausted openings each, no hits/timeouts. Independent audits:
`data/ihes_pdb/pid358_proof_20260910.json` (8036.47 s) and
`data/ihes_pdb/pid604_proof_20260910.json` (4944.15 s). Completed local batch
replayed 1003/1003 valid. Final wave-12 min-merge includes all completed local
batches and remains **21,870**, saved zero:
`submissions/ihes_20260909_wave12_verified.csv`. Baseline unchanged.

**PID 46 is NOT yet proved optimal.** Wave-12 shard 1 reached the nine-hour cap
with native return -15, 51 recorded exhausted prefixes and an interrupted
unrecorded opening 51. Full-wave audit correctly rejected this as incomplete.
Independent partial audit verified all five source candidates and every recorded
exhaustion block: **260/262 settled; only opening IDs 51 and 52 unresolved**.
Artifact: `data/ihes_pdb/wave12_partial_audit_20260910_0434.json`, generated by
`scripts/67_prepare_ihes_wave13_close46.py`. Snapshot:
`data/ihes_pdb/status_20260910_0434.json`.

**Current remote wave 13: only PID 46 openings 51 and 52**, one private CPU
notebook, 1200 s each, 8 GiB cache. Fresh unseeded journal limited to those two
IDs; no settled work repeated. Wave ID `20260910_wave13_close46`.
`artgor/ihes-exact-cpu-wave-2-shard-1` version **12** confirmed RUNNING; other
four slots are COMPLETE and idle. Previous wave-12 collector finished normally.

- Manifest/receipt: `data/ihes_pdb/cpu_wave13_manifest.json`, `cpu_wave13_launch.json`.
- Package: `kaggle_notebooks/ihes_wave13_shard1`.
- Collection: `submissions/ihes_20260910_wave13/collection_status.json`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave13_collector` (launcher 59420).
- Eventual CSV: `submissions/ihes_20260910_wave13_verified.csv`.
- Final PID 46 audit must combine original wave-12 five shards (including its
  interrupted native tail) and these two new records; generic full-wave/chain
  audit scripts assume normal exit/full coverage and cannot be used unchanged.

**Current local targets: PIDs 558 then 996**, 16 threads, 8 GiB cache, 120 s per
prefix, rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix558_996_20260910` (launcher 59608). Eventual candidate:
`submissions/ihes_20260910_local_prefix558_996.csv`. Native bounds loaded, new
local/collector error logs empty. Next unassigned ranked targets: 910, 614, 644.
Add PIDs 358 and 604 to the proved exclusion list below. Earlier worker entries
are historical.

02:32 UTC check: wave-12 PID 46 shards **2--5 are COMPLETE**; only shard 1
remains RUNNING through fresh provider checks. All four completed outputs were
audited for wave identity, exact assigned ranges, native exhaustion markers and
normal exit, and replayed 1003/1003 valid at unchanged **21,870**. PID 46 has
**209/262 prefixes settled**, zero hits/timeouts in completed shards, no full
optimality proof yet. Audit: `data/ihes_pdb/wave12_partial_audit_20260910_0232.json`.

Local PID 358 continues (235/262 exhausted at snapshot, zero hits/timeouts);
PID 604 has not started. Local worker 58544 and collector 53092 remain alive
with matching start times; error logs empty. Keep existing workers running,
without overlap or restarts. Current paths are below. Snapshot:
`data/ihes_pdb/status_20260910_0232.json`. No score improvement.

00:31 UTC check: **PIDs 730 and 272 are proved optimal at 22 moves**, each with
262 exhausted openings, no hits/timeouts. Independent native/prefix audits:
`data/ihes_pdb/pid730_proof_20260910.json` (3031.72 s) and
`data/ihes_pdb/pid272_proof_20260910.json` (4203.24 s). Completed local batch
replayed 1003/1003 valid at unchanged **21,870**:
`submissions/ihes_20260909_local_prefix730_272.csv`. Exclude these targets.

**Current local targets: PIDs 358 then 604**, launched 00:33 UTC, 16 threads,
8 GiB cache, 120 s per prefix, rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix358_604_20260910` (launcher 58544). Eventual candidate:
`submissions/ihes_20260910_local_prefix358_604.csv`. Native log confirms both
additive bounds loaded; error log empty. Previous local worker finished.

**Remote wave 12: PID 46**, shards 3 and 5 COMPLETE; shards 1, 2 and 4 RUNNING
through fresh provider checks. Completed outputs have matching wave identity,
normal native exit, exact assigned coverage and all exhaustion markers; both
candidates replayed 1003/1003 at 21,870. **103/262 prefixes settled**, not a full
optimality proof. Audit and status: `data/ihes_pdb/status_20260910_0031.json`.
Keep the three running kernels and existing collector (launcher 53092) active.
No restarts or overlap. Wave-12 manifest/packages/collection paths are below.
Next unassigned ranked targets: 558, 996, 910. No score improvement.

Known proved length-22 PIDs to exclude: **26, 28, 40, 94, 244, 268, 272, 296,
300, 336, 468, 510, 534, 548, 554, 634, 638, 642, 684, 690, 730, 816, 850,
868, 916, 918, 952**. Earlier active-worker descriptions are historical.

### Previous checks, 2026-09-09

22:30 UTC check: **PID 952 is proved optimal at 22 moves**, all 262 openings
exhausted in 5605.77 s, no hits/timeouts. Independent native/prefix audit:
`data/ihes_pdb/pid952_proof_20260909.json`. Completed local 548/952 batch replayed
1003/1003 valid at unchanged **21,870**:
`submissions/ihes_20260909_local_prefix548_952.csv`. Exclude both 548 and 952
from future searches. Snapshot: `data/ihes_pdb/status_20260909_2230.json`.

**Current local targets: PIDs 730 then 272**, launched at 22:31 UTC, 16 threads,
8 GiB cache, 120 s per prefix, rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix730_272_20260909` (launcher 59056). Eventual candidate:
`submissions/ihes_20260909_local_prefix730_272.csv`. Native log confirms both
additive bounds loaded; error log empty. Previous local worker finished.

**Remote wave 12 remains active on PID 46**, all five fresh provider statuses
RUNNING at this check, no finished outputs or reported failures. Keep those
kernels and collector (launcher 53092) running. Manifest, packages and collection
paths remain those listed in the 18:29 entry below. No overlapping work; next
unassigned ranked targets are 358, 604, 558. No score improvement.

20:29 UTC check: **PID 548 is proved optimal at 22 moves**, all 262 prefixes
exhausted in 4391.85 s, zero hits/timeouts. Independent native/prefix audit:
`data/ihes_pdb/pid548_proof_20260909.json`. Exclude PID 548 from future searches.
Same local worker continues on PID 952 (127/262 exhausted at snapshot, zero
hits/timeouts). All five wave-12 PID 46 kernels remain RUNNING through fresh
provider checks. Local worker and collector processes are alive with matching
start times; both error logs empty. No new remote outputs or score improvement;
last verified total remains **21,870**. Keep existing workers running without
overlap. Snapshot: `data/ihes_pdb/status_20260909_2029.json`. Active paths below.

18:29 UTC check, continued at 18:32: **PIDs 510 and 638 are proved optimal at
22 moves**. Wave 11 completed all five shards; independent audit checked all
262 prefix identities, native exhaustion markers, wave identity and each candidate
replay. Proof: `data/ihes_pdb/pid638_proof_20260909.json`. Local PID 510 exhausted
all 262 in 5216.56 s; proof: `data/ihes_pdb/pid510_proof_20260909.json`. Completed
916/510 batch replayed 1003/1003 valid. Min-merge remains **21,870**, zero saved:
`submissions/ihes_20260909_wave11_verified.csv`. Baseline unchanged. Snapshot:
`data/ihes_pdb/status_20260909_1829.json`.

**Current remote wave 12: PID 46**, five disjoint shards covering 262 openings
(53/53/53/53/50), residual depth 18, 1200 s per prefix, 8 GiB cache, nine-hour
cap. All five fresh provider statuses confirmed RUNNING. Wave ID:
`20260909_wave12_split46`; canonical refs remain
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`, versions **11/9/10/10/9**.

- Manifest/receipts: `data/ihes_pdb/cpu_wave12_manifest.json`, `cpu_wave12_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave12_shard1` through `shard5`.
- Collection: `submissions/ihes_20260909_wave12/collection_status.json`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave12_collector` (launcher 53092).
- Eventual CSV: `submissions/ihes_20260909_wave12_verified.csv`.

**Current local targets: PIDs 548 then 952**, 16 threads, 8 GiB cache, 120 s
per prefix, rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix548_952_20260909` (launcher 56364). Eventual candidate:
`submissions/ihes_20260909_local_prefix548_952.csv`. Native log confirms both
additive bounds loaded; new local and collector error logs empty. Previous
workers finished; old collector PID 32840 was reused by an unrelated Lenovo
process, so never identify a worker by PID alone. Next unassigned: 730, 272, 358.

Known proved length-22 PIDs to exclude: **26, 28, 40, 94, 244, 268, 296, 300,
336, 468, 510, 534, 554, 634, 638, 642, 684, 690, 816, 850, 868, 916, 918**.
All active-worker descriptions below are historical.

16:29 UTC check: **PID 916 is proved optimal at 22 moves**, all 262 prefixes
exhausted in 7092.97 s, zero hits/timeouts. Independent audit:
`data/ihes_pdb/pid916_proof_20260909.json`. Exclude PID 916 from future searches.
The same local worker continues on PID 510 (5 prefixes exhausted at snapshot).
All five wave-11 PID 638 kernels remain RUNNING through fresh provider checks;
collector and local processes are alive, error logs empty. No finished remote
outputs or new candidates. Keep current workers running without overlapping
work. Last verified total remains **21,870**. Snapshot:
`data/ihes_pdb/status_20260909_1629.json`. Current paths are listed below.

14:27 UTC check, continued at 14:30: **PIDs 244 and 642 are proved optimal at
22 moves**. Wave 10 completed all five shards; independent audit checked all
262 distinct openings, native exhaustion markers, wave identity and each candidate
replay. Proof: `data/ihes_pdb/pid244_proof_20260909.json`. Local PID 642 exhausted
all 262 openings in 11888.86 s; proof: `data/ihes_pdb/pid642_proof_20260909.json`.
The completed 690/642 batch replayed 1003/1003 valid. Final min-merge remains
**21,870**, saved zero: `submissions/ihes_20260909_wave10_verified.csv`.
Original `submission_ihes.csv` is unchanged. Pre-launch snapshot:
`data/ihes_pdb/status_20260909_1427.json`.

**Current remote wave 11: PID 638**, five disjoint shards covering all 262
openings (53/53/53/53/50), residual depth 18, 1200 s per prefix, 8 GiB cache,
nine-hour cap. All five fresh provider statuses confirmed RUNNING. Wave ID:
`20260909_wave11_split638`; canonical refs remain
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`, versions **10/8/9/9/8**.

- Manifest/receipts: `data/ihes_pdb/cpu_wave11_manifest.json`, `cpu_wave11_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave11_shard1` through `shard5`.
- Collection: `submissions/ihes_20260909_wave11/collection_status.json`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave11_collector` (launcher 32840).
- Eventual CSV: `submissions/ihes_20260909_wave11_verified.csv`.

**Current local targets: PIDs 916 then 510**, 16 threads, 8 GiB cache, 120 s
per prefix, existing rank order. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix916_510_20260909` (launcher 5904). Eventual candidate:
`submissions/ihes_20260909_local_prefix916_510.csv`. Native log confirms additive
PDB and center-sum bound loaded. Both new error logs empty. Previous local worker
and collector finished; no overlap. Next unassigned ranked targets: 46, 548, 952.

Known proved length-22 PIDs to exclude: **26, 28, 40, 94, 244, 268, 296, 300,
336, 468, 534, 554, 634, 642, 684, 690, 816, 850, 868, 918**.

Recovery recorded from the 12:27 check: wave-10 collector had stopped on Windows
`PermissionError` replacing `collection_status.json`. Added bounded rename
retries and preserved snapshots instead of terminating on persistent status-file
locks (`scripts/37_collect_ihes_kaggle.py`); three regression tests passed in
`scripts/66_test_ihes_collector_lock.py`. Restart launcher 11112 used
`data/ihes_pdb/cpu_wave10_collector_restart1.log` / `.err.log`, collected the
remaining results and has now finished normally. No kernels were restarted.

### Previous checks (superseded active-worker descriptions below)

10:27 UTC check: **PID 690 is proved optimal at 22 moves**. All 262 prefixes
exhausted, no hits/timeouts, 4332.85 s; independent proof:
`data/ihes_pdb/pid690_proof_20260909.json`. Skip PID 690 in future searches.
Local worker continues on PID 642: 57 prefixes exhausted at the check, zero hits
or timeouts. All five wave-10 Kaggle kernels still report RUNNING through fresh
API calls; no provider failures, no finished output, collector and local error
logs empty. Keep current workers running without overlapping work. Last verified
total remains **21,870**. Snapshot: `data/ihes_pdb/status_20260909_1027.json`.

08:27 UTC check: **PIDs 684 and 816 are proved optimal at 22 moves**. All five
wave-9 shards finished normally, all 262 prefixes exhausted without hits/timeouts.
Fresh split-wave audit checked every assignment/native block and all candidates:
`data/ihes_pdb/pid684_proof_20260909.json` via `scripts/64_audit_ihes_split.py`.
Local PID 816 exhausted all 262 prefixes in 5703.99 s; proof:
`data/ihes_pdb/pid816_proof_20260909.json`. Completed 534/816 batch took 11410 s;
its final CSV independently replayed all 1003 paths at **21,870**, unchanged.

**Active remote wave 10:** unresolved PID **244**, all 262 two-move prefixes
split 53/53/53/53/50, residual depth 18, 1200 s per prefix, 8 GiB cache, nine-hour
cap. All five provider statuses confirmed RUNNING. Wave ID:
`20260909_wave10_split244`; canonical refs stay
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`, versions **9/7/8/8/7**.

- Manifest/receipts: `data/ihes_pdb/cpu_wave10_manifest.json`, `cpu_wave10_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave10_shard1` through `shard5`.
- Collection: `submissions/ihes_20260909_wave10/collection_status.json`.
- Collector log/error/PID stem: `data/ihes_pdb/cpu_wave10_collector`.
- Eventual CSV: `submissions/ihes_20260909_wave10_verified.csv`.

**Active local targets: PIDs 690 then 642**, 16 threads, 8 GiB cache, 120 s per
prefix, 262 cases each. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix690_642_20260909` (launcher 13352). Eventual candidate:
`submissions/ihes_20260909_local_prefix690_642.csv`. All older workers/collectors
finished. No local/remote target overlap. Snapshot before continuation:
`data/ihes_pdb/status_20260909_0827.json`.

Known proved length-22 PIDs: **26, 28, 40, 94, 268, 296, 300, 336, 468, 534,
554, 634, 684, 690, 816, 850, 868, 918**. Exclude them from future shortening runs.
Reusable fresh-wave preparation: `scripts/65_prepare_ihes_fresh_split.py`;
`scripts/47_launch_ihes_prefix_wave3.py` now accepts explicit manifest/receipt
paths while retaining its old guarded defaults. No native solver changes.

06:26 UTC check: **PID 534 is proved optimal at 22 moves**. All 262 prefixes
exhausted in 5702.28 s, zero hits/timeouts; independent proof:
`data/ihes_pdb/pid534_proof_20260909.json`. Skip it in future searches. The local
worker continues on PID 816 (55 exhausted prefixes at check, no hits/timeouts).

Wave-9 shards 1, 2, 3 and 4 are COMPLETE; all **212 assigned prefixes exhausted**,
no hits/timeouts, and their submissions independently replayed 1003/1003 valid
at unchanged **21,870**. Audit: `data/ihes_pdb/wave9_partial_audit_20260909_0626.json`.
Shard 3 completed just after the collector's poll, so it was collected and audited
separately under `submissions/ihes_20260909_wave9/manual/shard3`; the active
collector will also accept it into `remote/shard3` normally. Avoid double counting.
Only shard 5 remains RUNNING on the final 50 prefixes; no full proof for PID 684
yet. Collector/local error logs empty. Keep existing workers running without
overlap; choose the next remote batch after shard 5's evidence arrives.
Snapshot: `data/ihes_pdb/status_20260909_0626.json`.

04:25 UTC check: **PIDs 336 and 634 are proved optimal at 22 moves**. Local 634
exhausted all 262 prefixes in 7412.82 s; proof:
`data/ihes_pdb/pid634_proof_20260909.json`. Its completed two-PID batch (850/634)
took 19409 s and final CSV replayed 1003/1003 valid at **21,870**.
Wave 8 exhausted both final PID 336 cases in 1096.20 and 1095.72 s. The complete
audit `scripts/62_audit_ihes_pid336.py` rechecked all **13 source runs** across
waves 4--8, exact seed links, assigned prefixes, native exhaustion blocks, full
262-prefix coverage, baseline identity and every candidate replay. Proof:
`data/ihes_pdb/pid336_proof_20260909.json`. No score improvement.

**Known proved length-22 PIDs to exclude:** 26, 28, 40, 94, 268, 296, 300, 336,
468, 534, 554, 634, 850, 868, 918. Older <=21 proofs remain documented separately.

**Current remote wave 9:** all five CPU kernels confirmed RUNNING on unresolved
PID **684**, 262 disjoint two-move prefixes split 53/53/53/53/50, residual depth
18, **1200 s per prefix**, 8 GiB cache, nine-hour cap. If the wall cap interrupts
a shard, retain settled prefixes and resume only unfinished work later. Output
wave ID: `20260909_wave9_split684`. Canonical refs remain
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`; versions are **8/6/7/7/6**.

- Manifest/receipts: `data/ihes_pdb/cpu_wave9_manifest.json`, `cpu_wave9_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave9_shard1` through `shard5`.
- Collection: `submissions/ihes_20260909_wave9/collection_status.json`.
- Collector logs/error/PID: stem `data/ihes_pdb/cpu_wave9_collector` (launcher 36416).
- Eventual CSV: `submissions/ihes_20260909_wave9_verified.csv`.
- All previous kernels/collectors completed; do not relaunch their old targets.

**Current local targets: PIDs 534 then 816**, separately from remote PID 684.
16 threads, 8 GiB cache, 120 s per prefix, existing rank order. Journal/log/error/PID
stem `data/ihes_pdb/local_prefix534_816_20260909` (launcher 10624). Eventual CSV:
`submissions/ihes_20260909_local_prefix534_816.csv`. Prior local workers finished.
Status snapshot before continuation: `data/ihes_pdb/status_20260909_0425.json`.
Two-hour monitor active; original submission remains untouched at 21,870.

## Earlier checks

19:14 UTC check: **PID 850 is proved optimal at 22 moves**, all 262 prefixes
exhausted, no hits/timeouts, 11993.12 s. Independent proof audit:
`data/ihes_pdb/pid850_proof_20260908.json`. The same local worker has advanced to
PID 634 (107 exhausted prefixes at this check, no hits/timeouts); let it continue.
No new score improvement: replay-verified total remains **21,870**.

Wave 6 completed normally: 24 newly exhausted prefixes, **two still unresolved:
13 and 16**. Seed equality, exact assigned IDs, native blocks and candidate replay
were audited in `data/ihes_pdb/remote_partial_audit_20260908_1914.json`.
**PID 336 now has 260/262 prefixes settled**, not a full optimality proof yet.

**Current remote work: one wave-8 closing retry** (`20260908_wave8_close336`),
version 7 of `artgor/ihes-exact-cpu-wave-2-shard-1`. It carries the complete wave-6
journal and retries ONLY prefixes 13 and 16, at 3600 s each, 8 GiB cache, two hours
maximum scheduled search. IDs 14/15 in the selected 13--16 range are already
settled and skipped. Other four kernels are COMPLETE and idle. A second large
local worker was considered but not launched: a direct Windows memory query
showed only 3.05 GiB available of 31.75 GiB, so sharing the local machine would
add paging pressure. Existing local PID 634 remains undisturbed.

- Manifest/receipt: `data/ihes_pdb/cpu_wave8_manifest.json`, `cpu_wave8_launch.json`.
- Package: `kaggle_notebooks/ihes_wave8_shard1`.
- Collector logs/error/PID: stem `data/ihes_pdb/cpu_wave8_collector` (launcher 48664).
- Collection: `submissions/ihes_20260908_wave8`; expected CSV:
  `submissions/ihes_20260908_wave8_verified.csv`.
- Full PID 336 proof will need original wave-4 evidence plus wave-5 scoped retries,
  wave-6 head retries, wave-7 tail retries, and these final two wave-8 results.
  Verify every prefix identity and exhaustion marker; never overwrite settled
  coverage with stale timeout rows carried in a seed journal.

All earlier remote collectors finished; only the wave-8 collector is active.
Status snapshot before upload: `data/ihes_pdb/status_20260908_1914.json`.

17:13 UTC check: wave-7 shard 3 completed normally and all 11 assigned retries
exhausted. Seed hash/equality, exact assigned IDs, native exhaustion blocks and
submission replay independently verified. Audit:
`data/ihes_pdb/remote_partial_audit_20260908_1713.json`. Wave 7 is now fully
COMPLETE with unchanged 21,870 moves. **PID 336 coverage: 236/262 settled**.
Only wave-6 shard 1 remains RUNNING, covering the final 26 unresolved cases in
IDs 0--27. Its collector is active and error log empty. Other four CPU slots
are idle pending that result and selection of the next remote batch. No proof
of optimality for PID 336 yet and no score improvement.

Local PID 850 continues: 152/262 exhausted prefixes, no hits or timeouts; PID 634
has not started. Local error log is empty. Keep current remote and local workers
running without overlapping work. Status snapshot:
`data/ihes_pdb/status_20260908_1713.json`.

15:13 UTC check: **PID 554 is now proved optimal at 22 moves**. Its four remaining
prefixes all exhausted in a 474 s local retry, zero hits/timeouts. The independent
chain audit checked original 258 settled cases plus the four new native blocks,
seed equality, full 262-prefix coverage and final CSV replay. Proof:
`data/ihes_pdb/pid554_proof_20260908.json`; reusable checker:
`scripts/60_audit_ihes_prefix_chain.py`. Candidate
`submissions/ihes_20260908_local_prefix554_retry.csv` is 1003/1003 valid at
**21,870**. Skip PID 554 in future work. No score improvement.

Newly completed remote jobs: wave-5 shards 2 and 5 settled all 28+21 retries;
wave-7 shard 4 settled all eight assigned retries. Native blocks, scoped prefix
coverage, seed equality, wave identities and full candidate replays checked:
`data/ihes_pdb/remote_partial_audit_20260908_1513.json`.
**Known PID 336 coverage is now 225/262**. Its remaining 37 cases are actively
assigned: wave 6 shard 1 (26 in IDs 0--27) and wave 7 shard 3 (11 in IDs 28--40).
Both report RUNNING without failure. Shards 2/4/5 are COMPLETE and idle; wave-5
collector finished. Keep the two remaining workers and their collectors running;
choose the next remote batch after the current PID's remaining evidence arrives.
Do not interpret carried seed timeout rows in completed wave-7 output as new work.

**Active local continuation: PIDs 850 then 634**, next unresolved targets from
the existing ranking. Same 16 threads, 8 GiB cache, 120 s per prefix, 262 prefixes
per PID. Journal/log/error/PID stem:
`data/ihes_pdb/local_prefix850_634_20260908` (launcher 23300 at start).
Eventual candidate: `submissions/ihes_20260908_local_prefix850_634.csv`.
All earlier local workers have finished. Status snapshot:
`data/ihes_pdb/status_20260908_1513.json`. Two-hour monitor remains active.

13:12 UTC check: **PID 40 is proved optimal at 22 moves**, all 262 prefixes
exhausted, zero timeouts/hits (6442.99 s). Proof:
`data/ihes_pdb/pid40_proof_20260908.json`. The 554/40 local batch completed in
19692 s; final candidate independently replayed 1003/1003 at **21,870**.
PID 554 remains unresolved at exactly four prefixes: **89, 92, 170, 173**.

**Active local work:** retry ONLY those four PID 554 prefixes with **600 s** each,
16 threads, 8 GiB, seeded from the completed 554/40 journal. All 520 settled
records are skipped. Logs/journal/error/PID share stem
`data/ihes_pdb/local_prefix554_retry_20260908` (launcher 25932 at start).
Eventual CSV: `submissions/ihes_20260908_local_prefix554_retry.csv`.
Proof audit must combine the original native logs with the four retry blocks;
`53_audit_ihes_live_prefix.py` alone rejects the duplicate seeded PID records.
The prior 554/40 worker has finished.

Wave-5 shards 3 and 4 completed normally, settling all **16+17** retried prefixes.
Each now has all 53 of its original range exhausted. Original seed equality,
new native exhaustion blocks and candidate replays were independently checked:
`data/ihes_pdb/wave5_partial_audit_20260908_1312.json`. Score remains 21,870.
Known settled PID 336 prefixes now total **168/262**; no full optimality proof yet.

**Current five CPU jobs:** canonical names are unchanged.
- Shard 1, version 6: wave 6, retrying IDs 0--27.
- Shard 2, version 5: wave 5, retrying IDs 53--105.
- Shard 5, version 5: wave 5, retrying IDs 212--261.
- Freed shards 3 and 4, now version 6: new wave 7 (`20260908_wave7_resume336_tail`),
  retrying the last **19 previously unassigned** cases: 11 within IDs 28--40 and
  8 within IDs 41--52. Same 1200 s per prefix / 8 GiB / nine-hour cap. No overlap.

Wave-7 manifest/receipts: `data/ihes_pdb/cpu_wave7_manifest.json` and
`cpu_wave7_launch.json`; packages `kaggle_notebooks/ihes_wave7_shard3`, `shard4`.
Collector log/error/PID stem: `data/ihes_pdb/cpu_wave7_collector` (launcher 21268).
Collection directory: `submissions/ihes_20260908_wave7`; eventual CSV:
`submissions/ihes_20260908_wave7_verified.csv`. The wave-5 collector continues
only shards 2 and 5; it already preserved completed shards 3 and 4.

For the final PID 336 coverage audit, use wave-6 records ONLY for IDs 0--27,
wave-7 records ONLY for 28--52, and wave-5 records for 53--261. Seed journals
include records outside their new job's scope; never let a carried seed timeout
overwrite an independently settled result. All 262 IDs are now either settled
or assigned to an active retry. Status before new uploads:
`data/ihes_pdb/wave6_status_20260908_1312.json`.

11:12 UTC check: **wave 4 is fully COMPLETE**, no improvements. Shard 1 finished
normally with 8 exhausted and 45 timed-out prefixes. Full original wave totals:
**135 exhausted / 127 timeout / zero hits**, covering all 262 PID 336 prefixes.
Final merged CSV `submissions/ihes_20260908_wave4_verified.csv` was independently
replayed again: 1003/1003 valid, **21,870**. Shard-1 seed identities and native
exhaustion blocks were checked by `scripts/56_prepare_ihes_wave6_head.py`.
There is still no optimality proof for PID 336.

**Current CPU allocation:** wave-5 shards 2--5 remain RUNNING; the freed shard-1
slot now has **wave 6**, output ID `20260908_wave6_resume336_head`, kernel version
6 on `artgor/ihes-exact-cpu-wave-2-shard-1`. It preserves the complete shard-1
journal and retries ONLY 26 unresolved prefixes within IDs 0--27, 1200 s each,
8 GiB cache, nine-hour overall cap. Maximum scheduled search is 31200 s, leaving
startup margin. **19 unresolved IDs in 28--52 remain unassigned**; they are listed
in the wave-6 manifest and must be scheduled later, not counted as covered.
No overlap with active wave-5 IDs 53--261, and no settled prefix is repeated.

- Wave-6 manifest/receipts: `data/ihes_pdb/cpu_wave6_manifest.json`, `cpu_wave6_launch.json`.
- Package: `kaggle_notebooks/ihes_wave6_shard1`.
- Collection: `submissions/ihes_20260908_wave6/collection_status.json`.
- Collector logs/PID: stem `data/ihes_pdb/cpu_wave6_collector`.
- Eventual merged CSV: `submissions/ihes_20260908_wave6_verified.csv`.
- Use wave-5 latest deduplicated records for IDs 53--261 and wave-6 for IDs 0--52;
  remember wave-6 carries unresolved seed rows for its 19 unscheduled cases.

Local PID 554 has completed its first pass: 258 exhausted / 4 timed out / no hits.
It is NOT proved optimal. PID 40 is active with 54 exhausted prefixes and no
timeouts at this check. Let that worker continue; afterward retry only unresolved
local cases with more time. Logs remain empty of errors. Snapshot before new
upload: `data/ihes_pdb/wave5_status_20260908_1112.json`.

09:11 UTC check: wave-4 shards 2--5 **COMPLETE**, shard 1 still RUNNING. The
completed shards contain 209 prefixes: **127 exhausted, 82 timed out, zero hits**.
All four native return codes 0; their wave IDs, prefix IDs/transformations,
exhaustion markers and complete submission replays were independently audited.
Partial audit: `data/ihes_pdb/wave4_partial_audit_20260908_0911.json`.
The partial merged submission is 1003/1003 valid at **21,870**. This is partial
coverage of PID 336, not an optimality proof. Retain all settled records.

**Current Kaggle allocation: five running CPU kernels total.** Canonical refs
remain `artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`, all now version 5:

- Shard 1: original **wave 4**, IDs 0--52, 600 s per opening, still active.
- Shards 2--5: new **wave 5** (`20260908_wave5_resume336`), preserving each shard's
  completed wave-4 journal and retrying ONLY its unresolved prefixes. Their retry
  counts are 28/16/17/21, totaling 82. Limit increased to **1200 s per prefix**,
  same 8 GiB cache and nine-hour wall cap. All four confirmed RUNNING after upload.
  Their ID ranges stay disjoint from shard 1 and each other. No settled work repeats.

Wave-5 manifest/receipts: `data/ihes_pdb/cpu_wave5_manifest.json` and
`cpu_wave5_launch.json`. Packages: `kaggle_notebooks/ihes_wave5_shard2` through
`shard5`. Collection: `submissions/ihes_20260908_wave5/collection_status.json`;
journals under `remote/shardN/prefix_L22.jsonl`. Logs/PID share stem
`data/ihes_pdb/cpu_wave5_collector` (launcher 11100 at launch). Eventual merge:
`submissions/ihes_20260908_wave5_verified.csv`. The original wave-4 collector stays
alive to collect shard 1; it already marked shards 2--5 finished and will not
confuse their new versions. Both collectors enforce distinct wave IDs.
Next audit must combine wave-4 shard 1 with latest wave-5 results, deduplicating
seeded records by prefix ID and accepting only sound settled evidence.

Local search remains on PID 554: 137 recorded prefixes at this check, 135
exhausted and 2 timeouts, zero hits. PID 40 has not started. Worker 52248 and
launcher 38928/Python 31524 are active; error log empty. Preserve those two
unresolved prefixes for a longer retry after this batch, rather than counting
them as proofs. No duplicate local work launched. Status snapshot:
`data/ihes_pdb/wave4_status_20260908_0911.json`.

07:10 UTC scheduled check: **PID 94 is now proved optimal at 22 moves**. Its
262 prefixes all exhausted at residual depth 18, zero hits/timeouts, 10750.96 s.
Independent audit: `data/ihes_pdb/pid94_proof_20260908.json`. The full local batch
(918 and 94) completed normally in 21205 s wall; final CSV independently replayed
1003/1003 valid at **21,870**. Candidate:
`submissions/ihes_20260908_local_prefix918_94.csv`; verification summary:
`data/ihes_pdb/local_prefix918_94_20260908_verified.json`. Skip 94 in future runs.

**Active local continuation now: PIDs 554 then 40**, next unresolved cases in
the existing rank file, excluding proved cases and remote target 336. Same
16-thread/8 GiB/120 s-per-prefix configuration; 262 openings per PID. Journal,
log, error log and PID file share stem `data/ihes_pdb/local_prefix554_40_20260908`.
Eventual candidate: `submissions/ihes_20260908_local_prefix554_40.csv`.
The previous 918/94 worker has finished; do not restart it.

All five wave-4 Kaggle shards for PID 336 still report RUNNING via fresh API;
no provider failures, collector error log empty, 0/5 completed. Keep remote runs
unchanged; they are about six hours into their nine-hour caps. No new remote
artifact to merge. Snapshot: `data/ihes_pdb/wave4_status_20260908_0710.json`.

05:09 UTC scheduled check: **PID 918 is now proved optimal at 22 moves**. Its
262/262 prefix searches exhausted at residual depth 18 with zero timeouts/hits
(10450.82 s accumulated search time). Independent audit matched every prefix and
native exhaustion block to the unchanged replay-valid baseline. Proof artifact:
`data/ihes_pdb/pid918_proof_20260908.json`; checker:
`scripts/53_audit_ihes_live_prefix.py`. This audit handles a completed PID while
the same worker continues a later PID; it hashes only the settled evidence.
Skip PID 918 in future shortening runs. Score remains **21,870**, 1003/1003 valid.

The local worker has advanced to PID 94: 111/262 prefixes exhausted at this
check, zero hits/timeouts. All five remote wave-4 kernels remain RUNNING by fresh
API status, no provider failures; collector active and error log empty, 0/5
finished shards. Both current searches continue without duplicate launches.
Snapshot: `data/ihes_pdb/wave4_status_20260908_0509.json`.

03:09 UTC scheduled check: all five wave-4 kernels remain RUNNING by fresh API
calls, with no reported failures. Collector alive and error log empty; no finished
remote shards. Local PID 918 has **191/262 prefixes exhausted**, zero hits and
zero timeouts; PID 94 has not started. Native worker 50612 and local launcher
38216/Python 7020 are active. Leave both searches running without duplicates.
No new candidate or optimality proof yet; last verified total remains 21,870.
Snapshot: `data/ihes_pdb/wave4_status_20260908_0309.json`.

User-requested mathematical investigation is documented separately in
`IHES_PID296_POTENTIAL.md`. Scripts 51/52 synthesized and independently checked a
global piece-coefficient potential using exact integers; it evaluates to ~7.545
on PID 296, while the existing corner/center bound gives 12. Neither reaches 22,
so neither replaces the exhaustive optimality proof. No score improvement or
new search bound was claimed or deployed from that investigation.

01:06 UTC heartbeat: local PID 296 completed all 259 remaining prefixes in
6349 seconds, no hits or timeouts. Together with its three seeded local proofs,
**all 262 prefix identities exhausted through residual depth 18**. Script
`48_audit_ihes_prefix296.py` independently checked prefix identities, explicit
native exhaustion blocks, normal native completion, baseline hash and full CSV
replay. **PID 296 is now proved optimal at 22** by full coverage through 20 plus
move-count parity. Proof: `data/ihes_pdb/pid296_proof_20260908.json`. Skip PID 296
along with 468 and older proved PIDs. Score unchanged at **21,870**, 1003/1003 valid
in `submissions/ihes_20260908_local_prefix296.csv`. The native wrapper's final
0/1 proof count excludes resumed seeds; the independent combined audit resolves it.

The CPU timing gate completed successfully: known opening 0 exhausted in
**232.95 seconds**, comfortably within the new 600 s cap. This is a timing result,
not a new proof or score improvement. Manifest and pass marker bind the result.

**Current remote wave:** all five kernels confirmed RUNNING after upload, wave ID
`20260908_wave4_split336`. Each searches a disjoint portion of PID **336**:
opening IDs 0--52, 53--105, 106--158, 159--211, 212--261. Native residual depth 18,
600 s per opening, 8 GiB cache, available CPU threads, nine-hour overall cap. Union
is exactly 262; only complete, verified union coverage could establish optimality.
Canonical refs remain `artgor/ihes-exact-cpu-wave-2-shard-1` through `-5`.
Shard 1 is version 5; shards 2--5 are version 4. No overlap with local targets.

- Manifest: `data/ihes_pdb/cpu_wave4_manifest.json`.
- Receipts: `data/ihes_pdb/cpu_wave4_launch.json`.
- Packages: `kaggle_notebooks/ihes_wave4_shard1` through `shard5`.
- Collection: `submissions/ihes_20260908_wave4/collection_status.json`.
- Journals: `remote/shardN/prefix_L22.jsonl` under that collection directory.
- Collector logs/PID: `data/ihes_pdb/cpu_wave4_collector.log`, `.err.log`, `.pid`.
- Future verified merge: `submissions/ihes_20260908_wave4_verified.csv`.

**Current local search:** PIDs **918 then 94**, ordered by the existing rank file,
262 prefixes each, 16 threads, 8 GiB cache, 120 s per opening. Launcher PID 38216
at launch. Journal/log/error/PID share stem
`data/ihes_pdb/local_prefix918_94_20260908`. Eventual candidate:
`submissions/ihes_20260908_local_prefix918_94.csv`. Replay-verify before merging;
require full prefix identity/native-marker coverage before claiming a proof.
The old gate collector and PID 296 worker have finished. Two-hour monitor active.

## Previous wave and timing gate

23:05 UTC check (Sep 8 local date): **wave 3 COMPLETE, zero improvements**.
All five native return codes 0; all 1,307 newly run remote openings timed out.
The only three `none` records are the previously proved local seeds in shard 1.
All 1,310 records recovered and audited: `data/ihes_pdb/wave3_audit.json`.
Independent replay again confirms 1003/1003 at **21,870** in
`submissions/ihes_20260907_wave3_verified.csv`. No new optimality proof.

The collector's filename filter omitted `prefix_L22.jsonl`; it has been fixed
and all five journals re-downloaded. Their native logs show the additive bound
enabled, but only ~4.2 million nodes/s on 4 Kaggle threads versus ~42--53 million
locally on 16 threads. Thus the 120 s remote cutoff was too short to close these
~1-billion-node subproblems. Do not repeat that remote configuration.

**Active continuation:** local PID 296 prefix search, 16 threads/8 GiB/120 s per
opening, resumes the wave-3 journal and skips the three settled seeds. Launcher
PID is in `data/ihes_pdb/local_prefix296_20260908.pid` (32524 at launch).
Journal/log/error log share stem `data/ihes_pdb/local_prefix296_20260908`.
Output will be `submissions/ihes_20260908_local_prefix296.csv`. First resumed
opening (ID 3) already exhausted in 14.38 s. Deduplicate by prefix ID when counting
coverage because copied timeout records precede new results. Check all 262
identities and matching baseline before claiming an optimality proof.

One private **600-second CPU timing gate** was uploaded and confirmed RUNNING as version 4 of
`artgor/ihes-exact-cpu-wave-2-shard-1`. It repeats known PID 296 opening 0 at
residual depth 18 to validate a realistic cutoff, and must produce
`prefix_timing_gate_passed.json` with verdict `none` before scaling.
This is deliberately a timing control, not an improvement search. Other four
Kaggle slots are idle pending its result. Use `20260908_prefix_timing_gate` as
output wave ID. Manifest/receipt: `data/ihes_pdb/prefix_timing_gate_manifest.json`
and `prefix_timing_gate_launch.json`. Hidden collector logs/PID share stem
`data/ihes_pdb/prefix_timing_gate_collector`; collection directory is
`submissions/ihes_20260908_prefix_timing_gate`, eventual verified output
`submissions/ihes_20260908_prefix_timing_gate_verified.csv`.
Next heartbeat: collect gate, compare measured runtime, and only if passed resume
other PIDs with an adequate cutoff. Avoid overlapping local PID 296 actual search.
The two-hour heartbeat remains active; original submission unchanged.

## Prior wave-3 monitoring

21:05 UTC scheduled check: fresh API calls show all five version-3 kernels still
RUNNING, with no provider failure reported. Collector processes 39604 and 28524
remain active, error log empty, 0/5 completed shards. No completed results to
verify or merge; last verified total remains 21,870. The searches have run about
7 h 47 min since upload and remain within their nine-hour caps. Keep them running;
the active collector will collect finished outputs before the next heartbeat.
Snapshot: `data/ihes_pdb/wave3_status_20260907_2105.json`.

19:05 UTC scheduled check: all five version-3 kernels remain RUNNING by fresh
Kaggle API status; no provider failures reported. Collector processes 39604 and
28524 are active, the error log is empty, and 0/5 shards have finished. No new
completed output is available to verify or merge; last verified total is 21,870.
Searches continue within their nine-hour caps; no duplicate work launched.
Snapshot: `data/ihes_pdb/wave3_status_20260907_1905.json`.

17:03 UTC scheduled check: fresh API status remains RUNNING for all five
version-3 kernels, with no reported provider failures. Collector processes 39604
and 28524 are alive, the error log remains empty, and 0/5 shards have finished.
No newly completed artifacts to replay or merge; verified total remains 21,870.
Current workers were left running without duplicate work. Snapshot:
`data/ihes_pdb/wave3_status_20260907_1703.json`.

15:03 UTC scheduled check: all five version-3 kernels still report RUNNING via
fresh Kaggle API calls; no provider failure reported. The local collector is
alive (launcher PID 39604, Python worker 28524), its error log is empty, and its
latest status has 0/5 finished shards. No new output is available to verify or
merge; last verified total remains 21,870. Leave the current searches running
without duplicate launches. Snapshot: `data/ihes_pdb/wave3_status_20260907_1503.json`.

13:18 UTC: **five version-3 CPU kernels confirmed RUNNING** on a new resumable
prefix-search wave. Canonical refs deliberately stay
`artgor/ihes-exact-cpu-wave-2-shard-1` through `-5` (titles/slugs retained to avoid
renaming); the output wave ID is **`20260907_wave3_prefix`**. Assigned PIDs in shard
order: **296, 918, 336, 94, 554**. These are previously unresolved targets; PID 468
is excluded as proved optimal. Each PID has 262 distinct two-move opening products,
searched through residual depth 18, 120 seconds per opening, 8 GiB cache, available
CPU threads (normally 4), nine-hour overall cap. Completed openings persist in
`prefix_L22.jsonl`. Use those records to resume only missing/timeout openings in a
later wave; preserve baseline hash, prefix identity, metric and max_depth=18.
Timeouts and partial coverage are never proofs. No competition submission made.

- Configuration: `data/ihes_pdb/cpu_wave3_manifest.json`.
- Upload receipts: `data/ihes_pdb/cpu_wave3_launch.json` (all five version 3).
- Collector: `submissions/ihes_20260907_wave3/collection_status.json`.
- Logs/PID: `data/ihes_pdb/cpu_wave3_collector.log`, `.err.log`, `.pid`.
- Future verified merge: `submissions/ihes_20260907_wave3_verified.csv`.
- Notebook folders: `kaggle_notebooks/ihes_wave3_shard1` through `shard5`.
- Journal outputs: `remote/shardN/prefix_L22.jsonl` under the collection directory.
- The collector rejects previous-version output until the wave ID matches.

Wave 2 is **COMPLETE**, all 120 searches timed out, zero hits, zero exhaustion,
all five native return codes 0. Final independent replay: 1003/1003 valid at
**21,870**, in `submissions/ihes_20260907_wave2_verified.csv`;
audit `data/ihes_pdb/wave2_audit.json`. No improvement toward the <=21,839 target.

Three new cost-partition probes (scripts 43--45) were admissible and passed every
abstract-edge check and all 22,873 incumbent trajectory states. The strongest,
corner permutation plus free-middle center components, adds a stronger bound at
only 15 states with >=11 moves remaining (253 overall, maximum gain 2).
These are **not promoted** to native search; no speedup or score gain claimed.
The new campaign instead preserves completed exact subproblems across runs.

`27_prefix_split_ladder.py` now requires an explicit exhaustion marker at the
requested depth, propagates native failures, records raw solver logs and avoids
claiming full coverage from a partial count. Five parser regression tests cover
abort, silent EOF, timeout, correct exhaustion and wrong-depth exhaustion. Local
prefix encoding positive control passed on PID 1. Three actual PID 296 openings
(IDs 0,1,2) exhausted in 19.81, 24.80 and 23.87 seconds with 16 threads; they are
seeded into shard 1 and skipped by resume. Their candidate independently replayed
all 1003 paths at 21,870. Every notebook repeats the cache/hash audit, prior
positive control, parser tests and prefix positive control before searching.
No local search remains active; only the wave-3 collector is running locally.
The existing two-hour heartbeat remains ACTIVE.

## Previous checks and completed waves

Scheduled check around 11:03 UTC: all five wave-2 kernels remain RUNNING without
reported failures; the collector is active and its error log is empty. The long
local batch completed cleanly in 20086 s: zero hits, one exhaustive result, and
three timeouts. Its final CSV independently replay-verified 1003/1003 paths at
21,870; verification summary is `data/ihes_pdb/local_long_L22_20260907_verified.json`.
PIDs 296, 336 and 918 reached 5400-second deadlines and remain unresolved. **PID 468
exhausted the search through depth 20 in 3880.42 s**, with an explicit native
`No solution found in 20` marker. Its recorded window matches the valid 22-move
incumbent, so parity makes that incumbent optimal. Proof provenance is recorded
in `data/ihes_pdb/pid468_proof_20260907.json`; skip PID 468 in future shortening
campaigns. No score improvement: verified total 21,870. The five healthy Kaggle
workers were left running. Local follow-up selection awaits their results, avoiding
an immediate repeat of the completed batch. The existing wave-2 collector already
includes the completed local CSV in its eventual merge.

The 07:00 UTC check had one timeout (PID 296). The second active local PID then
was 336, not 468: without a rank-order file the ladder sorts selected PIDs.

All five first-wave search kernels and the cache preflight completed. The private
cache dataset is ready. The first-wave collector independently replay-verified
1003/1003 paths at **21,870**, with zero improvements. Actual search outcomes:
119 recorded 1200-second timeouts, no hits and no exhaustion proofs; the original
16 GiB worker was stopped by the nine-hour watchdog during its last case. Its
23 records persist. The four cache-reusing workers each finished all 24 cases.
Both earlier local batches also completed: 12 timeouts, zero improvements.

The user asked to check status and continue. **Five second-wave kernels are now
RUNNING**, verified through the Kaggle API, on 120 fresh, disjoint length-22 PIDs.
They reuse the checked 8 GiB cache, shuffle native opening subtrees, search all
depths through 20 (no fixed minimum), allow 1200 s per PID, and retain a nine-hour
overall cap. Every notebook repeats the independent audit and positive control.

Current canonical refs (Kaggle changed the slugs when the titles changed; kernel
IDs are unchanged and these are version 2 of the existing five notebooks):
`artgor/ihes-exact-cpu-wave-2-shard-1` through `artgor/ihes-exact-cpu-wave-2-shard-5`.
Always use these returned refs for status/output calls; the old API slugs now 404.

- Immutable configuration: `data/ihes_pdb/cpu_wave2_manifest.json`.
- Upload receipts: `data/ihes_pdb/cpu_wave2_launch.json`.
- Live collection: `submissions/ihes_20260907_wave2/collection_status.json`.
- Collector logs: `data/ihes_pdb/cpu_wave2_collector.log` and `.err.log`.
- Eventual verified merge: `submissions/ihes_20260907_wave2_verified.csv`.
- Collector checks the output wave ID before accepting a completed version;
  downloads are staged outside the merge folder until that check passes.

A separate local batch searched PIDs 296, 468, 918 and 336 for up to **5400 s each**
(six hours maximum search time), 16 threads, shuffled opening order and all depths
through 20. Startup read the validated cache and enabled both additive bounds;
PID 296 finished depth 18 in 19.047 s before continuing. This is progress, not a hit.
Logs/journal: `data/ihes_pdb/local_long_L22_20260907.*`.
Verified output: `submissions/ihes_20260907_local_long.csv`, included by the wave-2
collector if available when merging. No paid compute or competition submission.

Historical details below describe the first-wave preparation and runs.

## Verified baseline

All 1,003 paths independently replay correctly, totaling **21,870** moves.
SHA-256: `0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab`.
Every legal move changes edge permutation parity, so total improvements are even.
The first total satisfying the requested threshold is therefore **21,838** (32 saved).
The input file has not been overwritten.

The content-based local merge checked 1,001 submission-shaped files: zero improvements.
Nine newly downloaded candidate CSVs from five recent public IHES Kaggle kernels also
contributed zero improvements. Verified copies:

- `submissions/ihes_20260906_all_sources_verified.csv`
- `submissions/ihes_20260906_public_merge_verified.csv`

The old 24,618 / 24,068 headlines in README, skill and IDEAS are obsolete. The August
2026 section of EXPERIMENTS is the relevant prior work. Do not repeat its local rewrites.

## New exact lower bound

An outer slice affects corners and center twists; a middle slice affects center
permutation but no corners. This partitions move costs exactly:

`minimum outer moves for (corners, center orientation sum mod 4)`
`+ minimum middle moves for center permutation`.

Middle turns preserve the sum of the six center orientations modulo four under the
verified v2 piece convention. Corner permutation parity fixes the low bit of that sum,
so the stronger table stores two entries per corner coordinate.

Tables in `data/ihes_pdb/`:

- `corners_v1.bin`: 88,179,840 exact outer-turn distances.
- `corners_center_sum_v1.bin`: 176,359,680 exact outer-turn distances.
- The 24-element middle-only center-permutation distance table is built at startup.
- Experimental corner-orientation / full-center and edge-orientation / full-center
  tables were also built, but their bounds are weaker and are not used by the native hook.

`scripts/32_audit_ihes_pdb.py` independently checks all 22,873 states on incumbent
trajectories, 12,000 random-walk states, and 7,200 consistency edges. It also checks that
minimizing the stronger table over the two sum values exactly recovers all 88,179,840
entries of the corner table. All passed. `audit_verified.json` records the result.

`scripts/ihes_twsearch_pdb_hook.h` checks all 18 official move transformations against
twsearch before enabling its optional lower bound. It is injected into an isolated
copy of solve.cpp; the original native solver is not overwritten.

Matched CPU benchmark, PID 30, proving no path of length <=18, 16 threads, same 8 GiB
quarter-turn hash table and same compiled native code with optional bounds toggled:

| Bound | Search seconds | Verdict |
|---|---:|---|
| Existing hash table only | 170.656 | none |
| Hash + corners + middle count | 53.03 | none |
| Hash + corners/center-sum + middle count | 43.10 | none |

This is a **3.96x measured speedup on one proof**, not a submission improvement or a
generalized speed guarantee. Both new variants passed an end-to-end positive control:
PID 1 inflated from 8 to 10 moves returned to 8; resulting full CSV replayed 1,003/1,003.

## Correctness fix in result collection

The existing `24_twsearch_ladder.py` classified an aborted native process after `Solving`
as `none` and returned exit zero. That was observed during a deliberately strict new
metric guard. It now requires an explicit `No solution found in ...` line for a no-hit
proof, classifies missing completion / crashes as unresolved, propagates a failed native
exit, and preserves raw native logs beside the journal.

Four regression tests in `scripts/test_ihes_ladder_failures.py` cover crash, silent EOF,
explicit timeout and genuine exhaustion. All passed. The failed pilot record
`data/ihes_pdb/hook_pid30.jsonl` is explicitly marked `error`, not a proof.

## Active / prepared work

The bounded local exact trial searched PIDs 296, 468, 918, 336, 94 for length <=20,
with 600 seconds per PID. It uses the stronger bound and the existing 8 GiB hash table.
All five reached their deadlines without a hit and remain undecided. It completed
at approximately 17:17 local time: 3008 s wall, zero hits, five timeouts, exit 0.
The resulting CSV replay-verified 1003/1003 at 21,870. Journals and raw logs:

- `data/ihes_pdb/ranked_L22_trial.jsonl`
- `data/ihes_pdb/ranked_L22_trial.solver.log`
- `data/ihes_pdb/ranked_L22_trial.log`
- eventual replay-verified `submissions/ihes_pdb_ranked_L22_trial.csv`

The local GPU compared ordinary E6 beam 65,536 against the same beam with the new
PDB floor, using the existing solver's `pdb_lookup` / `max` hook. No changes were made to
the user's already-modified `src/cayley/khoruzhii_search.py`.
Results: `submissions/ihes_pdb_beam_probe.json` and corresponding verified CSV.
Completed result on PIDs 30, 296, 468, 918, 336, 94: ordinary E6 totals 140 moves;
PDB-clamped E6 totals 144. No incumbent improvement. Paths actually changed on four
PIDs, confirming the hook was exercised. Do not scale this beam variant as-is.

Opening-order experiment: E6 ranks the incumbent's first-three-move state at median
1,042.5 among 5,832 raw opening words on 42 PIDs, versus 1,622.5 for AZ v2 and 1,432.5
for their standardized mean. E6 is used for ordering only; no subtree is discarded.
The new native ranked variant matched and retained all 3,732 canonical subtrees on its
PID-30 control. This is an ordering diagnostic, not evidence of a new shorter solution.
The subsequent known-length-20 rediscovery probe was inconclusive: ordinary order
timed out after 90 s; ranked order exceeded the wrapper's 150 s deadline after a
64.7 s cache load. Neither found an improvement. The wrapper now allows 300 s and
cleans up its process tree on timeout. Do not claim a speed advantage from this probe.

A sequential follow-up started at approximately 17:17 local time through
`scripts/38_ihes_ordered_L24.ps1`. The ranked binary searches all seven length-24 PIDs
(106, 592, 680, 706, 764, 810, 936) at threshold 22, 600 s each, 16 threads and
the existing 8 GiB table. Maximum search time is 70 minutes plus table loading.
Journal/log stem: `data/ihes_pdb/ordered_L24_trial`.
Eventual verified CSV: `submissions/ihes_pdb_ordered_L24_trial.csv`.
Startup confirms the bound is enabled and all 3732 opening subtrees were matched
and retained for the first PID. The 8 GiB table loaded in 5.344 s.

The private CPU-only Kaggle notebook is prepared in
`kaggle_notebooks/ihes_exact_additive_cpu/`, with bundled source/puzzle inputs, source
hashes, positive controls, memory sizing, per-PID deadlines, and a nine-hour wall cap.
The user explicitly approved all Kaggle uploads on 2026-09-06. Version 1 was uploaded
and the Kaggle API confirmed **RUNNING** at approximately 17:00 local time:
https://www.kaggle.com/code/artgor/ihes-exact-additive-pdb-search
The initial automatic-review block is resolved by that explicit approval.
`launch_response.txt` in the notebook folder records the successful upload.
The API does not yet expose a downloadable execution log; RUNNING confirms dispatch,
not successful completion of the build or positive control. Check these before treating
any remote run as a valid search result.

`scripts/37_collect_ihes_kaggle.py` is running as a hidden local collector. It polls
every three minutes, with a ten-hour deadline, downloads all pages of small result
artifacts and the execution log, and replay-verifies a deterministic min merge.
It preserves the original baseline, rejects duplicate/extra/missing IDs, and refuses
to merge if the baseline SHA changes. Valid-baseline, duplicate-ID and invalid-path
checks passed. It uses existing Kaggle credentials without exporting them.
The local machine must remain on for automatic collection.

- Collector log: `data/ihes_pdb/kaggle_collector.log` (and `.err.log`).
- Durable status: `submissions/ihes_20260906_kaggle/collection_status.json`.
- Downloads: `submissions/ihes_20260906_kaggle/remote/`.
- Eventual verified merge: `submissions/ihes_20260906_search_verified.csv`.
- The collector does not submit to the competition or claim success on a timeout.

## Five-notebook CPU expansion

The four additional workers are prepared with disjoint sets of 24 length-22 PIDs;
including the original worker this covers 120 distinct PIDs. Assignments and references
are in `data/ihes_pdb/cpu_fleet_manifest.json`. All use threshold 20, 1200 seconds
per PID, CPU only, private notebooks, and a nine-hour wall cap.

To avoid repeating a large cache build four times, a private dataset upload is in
progress: `artgor/ihes-additive-exact-search-cache`. It contains the existing verified
8 GiB hash table (4,960,490,565-byte compressed file), plus the two additive PDBs.
Total large-file bytes: 5,225,030,085. SHA-256 values are recorded in
`kaggle_datasets/ihes_exact_search_cache/cache_manifest.json`; source files are linked,
not overwritten. The first upload invocation failed before transferring data because
the SDK mishandled slash-separated relative paths; native absolute Windows paths fixed it.

`scripts/41_launch_ihes_cpu_fleet.py` is running as a hidden controller. After the
upload commits and the dataset becomes ready, it launches a separate short private
CPU preflight. That notebook checks every cache hash, compiles the native solver on
Kaggle, reruns the independent additive-bound audit, and solves the known control
using the uploaded hash table. Only after the preflight completes successfully will
it upload the four extra search notebooks. The preflight finishes before the four
workers start, so this sequence never intentionally exceeds five active CPU notebooks.
The new workers reuse the 8 GiB table and remove temporary input links before export.

- Preparation: `scripts/39_prepare_ihes_cpu_fleet.py`.
- Launch status: `data/ihes_pdb/cpu_fleet_launch_state.json`.
- Launch logs: `data/ihes_pdb/cpu_fleet_launch.log` and `.err.log`.
- Preflight results: `data/ihes_pdb/cpu_gate_output/`.
- Shard notebooks: `kaggle_notebooks/ihes_exact_cpu_shard2/` through `shard5/`.
- The first controller attempt encountered Kaggle's expected 403 for the not-yet-created
  private dataset. It launched nothing. The corrected controller waits for the upload
  completion record first; its state preserves the failed attempt.

After dispatch, `scripts/40_collect_ihes_cpu_fleet.py` will collect all five notebooks,
using a separate output folder to avoid racing the original single-run collector:
`submissions/ihes_20260906_cpu_fleet/collection_status.json`. The eventual deterministic,
replay-verified fleet merge is `submissions/ihes_20260906_fleet_verified.csv`.
These are queued actions until the launch-state file confirms dispatch; do not describe
all five workers as running merely because the packages exist.

## Current score status

**No verified improvement yet; best remains 21,870. The requested target is not met.**
Never count a speedup, a positive-control repair, a timeout, or an alternative longer
neural solution as a score improvement.
