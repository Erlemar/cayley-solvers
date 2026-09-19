Cube666 neural beam: findings, failures, and first-10 diagnostic
Date: 2026-09-04
Status timestamp: 2026-09-04 00:35 PDT
Document status: The completed whole-PID campaign findings are frozen from the authoritative 2026-09-03 report. Two width-8,192/depth-400 first-10 shards are active; their results are intentionally left pending below until both shards complete and the merged output is replay-validated.

The detailed artifact audit and full experiment log remain in docs/whole_pid_scalar_v_campaign_status_2026_09_03.md, SHA-256 1f771a294284a8e8835deacd610f4bd65e78adf5ff71f242ba15c440c86c76b4. This document is a shorter synthesis organized around what was tried, what failed, and what the evidence now says.
Executive conclusion
There is no working or promoted neural whole-PID Cube666 solver yet.

Classical KMC solved PID 502 in a replay-verified 195 moves. No tested direct neural beam solved PID 502.
The direct scalar-V failures are not explained by a depth-4 cap. Four candidates were run continuously to depth 400 at width 8,192 and all failed.
Increasing hybrid-search width from 20,480 to 524,288 did not change the selected endpoints on the ten selected regressions. In a separate, not causally matched run, the depth-8 arm totaled 2,166 moves versus 2,160 for depth 4; both lost to KMC.
The strongest scalar checkpoint, C2500, improves the fixed high-cost validation slice but still loses the known PID 502 solution path at depth 4 and fails direct width-8,192/depth-400 search.
More imitation, symmetry, full-trajectory labels, explicit pair losses, early-prefix sampling, and substantially longer training each improve some offline metric. None satisfies the complete gate or produces a full path.
The q96 campaign produced useful, clean training data, but matched post-q96 training regressed the primary high-cost ordering guard. The pair loss fit four additional training-pair directions without improving general action ordering.
A 36-action direct-Q parent forward uses about 36 times fewer neural input rows than scalar scoring of all 36 children, but its end-to-end beam speedup was not measured. The tested C2500-distillation heads do not rank exact children faithfully enough to justify a wide beam.

The central failure is systematic ranking error under the model's own deep search distribution. More width preserves more states, but it does not fix a score that repeatedly prefers false low-value basins over the path that must survive for hundreds of layers.
Evidence classes
Three experiment families must not be conflated.
Hybrid V plus KMC
The learned model searches a short prefix, then classical KMC completes from selected endpoints. Reported path lengths include KMC. These experiments test whether learned search can find a better handoff state; they do not show that the neural model can solve the full PID.
Direct scalar-V beam
Every retained state is scored with the state-only DenseValueBN scalar value model. Search is continuous, uses exact global transpositions and the exact d5 plus terminal d6 goal checks, and has no KMC completion, classical path, restoration handoff, or corner repair. A replay-verified solution from this family would be a genuine neural-beam whole-PID solve.
Direct-Q experiments
One parent forward emits scores for all 36 Cube666 actions. The new surrogate experiments trained against C2500 child values and measured fidelity before authorizing search. They were rejected at the fidelity gate, so no wide C2500-surrogate beam was run. This does not test direct Q trained on exact distances or a stronger policy target. The older random-walk direct-Q lineage is separate and is summarized below.
Starting data and target semantics
The base corpus contains replayed states from 8,874 successful full paths over 493 roots:

Split
Roots
Unique states
Train
450
963,970
Validation
43
92,834
Total
493
1,056,804


Its label U(s) is the shortest replay-verified suffix observed for that exact state. It is an upper bound on optimal distance, not the true distance. A path action is a demonstrated positive, but an unobserved action is not a certified negative. This distinction explains why value MAE can improve while the one-step ordering required by beam search gets worse.

The corpus manifest self-hash is 48d00ed36e94306ca94d2008ab728228e010c9d1ac8add59d3140b67581f19fa.

The same corpus supplies 94,700 train and 9,222 validation non-tied same-root/same-depth preference pairs. For cross-policy step 350, strict pair accuracy is 55.345% train and 60.421% validation, but only 39.763%/41.129% at depths 0-15 and 45.950%/46.977% when the larger observed U is at least 160. Aggregate pair accuracy therefore hides the region that determines early whole-PID survival.
Hybrid width and depth controls
These controls used the earlier 6,000-update KMC-on-policy scalar checkpoint, SHA-256 ac195ec763906d955016d282106fb94eed7eea321b55b1806a97b220673279cf, not C2500. Their completion budgets were asymmetric: the classical reference received three fresh KMC calls per root, 30 total, while each learned-endpoint arm received 12 KMC calls per root, 120 total.

The ten largest raw confirmation regressions were first rerun with depth 4 and beam width 524,288. Classical KMC totaled 2,096 moves. Learned width 20,480 and learned width 524,288 both totaled 2,160. All 30 root/residual-band searches selected the same four guarded endpoints in the same order at both widths, even though the wider arm evaluated 896,805 unique depth-4 states per band.

Depth 8 at width 524,288 did not rescue the same cases:

Method
Total moves
Versus KMC
Classical KMC
2,096
reference
Learned depth 4, width 524,288
2,160
+64
Learned depth 8, width 524,288
2,166
+70


Depth 8 beat KMC on zero roots, tied one, and lost nine. It found zero exact d5-ball hits. The 30 searches scored 2,103,463,506 states in 2,704.863 seconds and peaked at 11,529,187,328 allocated CUDA bytes. The exact-restoration guard accepted 283 of 92,160 inspected candidates. The 120 handoffs selected for KMC used 16-move prefix/repair sequences that all primitive-reduced to empty. Those selected handoffs were identity-equivalent detours, not genuine progress before KMC; that statement does not apply to every inspected candidate.

These are post-hoc hybrid diagnostics. They reject more width and depth under that exact-restoring handoff; they do not reject all possible neural search or handoff designs.

The source report hashes are:

width-524288 report self/file: 4f74698f8fb86db3d7b60bf09f1fea0a83ff4f2419db7bade7f8a650b0884346 / 06385a217433fb85204b16a8cbf7566fcf989e6b92cae6b8be0e89a90037c956;
depth-8,width-524288 report self/file: 5f3b930f622e1d42440af19b41f29f4560424600add89e744c150bdfb68b2bc6 / ce6a2bdf2e0ba1570506ba6a3d1c4b59c8f6f726e49efd5f626bfeb0f81dbc99;
archived GDrive findings document cube666_onpolicy_final_2026_09_02/2026_09_01_cube666_depth4_guard_and_onpolicy.md: 3b7182501d4a93e675c47a7a6e3b03d65ea17d7bd99f0aaa5091a215ef407c3b.
Direct scalar-V baseline and gates
All offline values below use the same fixed 1,376-row sample from 43 validation roots. Lower rank/MAE and higher top-8 are better. None is a solve claim.

Checkpoint
Overall rank
Overall top-8
160+ rank
160+ top-8
160+ MAE
K108 base
10.650
54.14%
15.285
33.03%
122.93
Clean-from-K108 step 1875
10.854
54.72%
12.084
47.75%
55.11
Cross-policy DAgger step 350
10.444
55.38%
11.237
50.75%
58.00
Continuation C2500
9.127
61.48%
10.925
51.05%
46.43


Direct search still failed:

Model/search
PIDs
Width
Depth cap
Result
Candidate edges per PID
v1 step 1500
502
65,536
220
fail
510,759,900
Cross-policy step 350
502
8,192
400
fail
117,119,628
Round-2 symmetry D750
502
8,192
400
fail
117,119,628
Early-prefix E600
502
8,192
400
fail
117,119,628
Continuation C2500
502
8,192
400
fail
117,119,628


Each depth-400 run retained 3,261,515 globally unique states. The wider depth-220 run scored 484,787,069 unique candidate states and retained 14,253,311 global states. A shallow stopping cap is not the blocker in these runs.

The depth-400 report self-hashes are 4b3da2b0134d049088337543ce6af1550dbbecbf2bcfbff7a88ca0129cf0bb8c for cross-policy step 350, 8a02b5785ddb14bc5bcb3c4db0846552a2f628c75643ec923f66bc2452b63cdc for D750, dadefc52a911490dae194a1c03a4dea037c6627c709f0f4e23e9dcbff650abe3 for E600, and 6222482629162a6f3176644fb59d7c0ad54443d0df69a8670b09ee3f7e182904 for C2500.
Known-path survival
The replay-verified 195-move KMC path for PID 502 was traced through a fixed width-8,192 beam. Ranks are zero-based and depend on the preceding frontier.

Model
d1 rank
d2 rank
First eviction
Width lower bound on that frontier
Whole-PID step 1500
30
990
d3 rank 28,659
28,660
Cross-policy step 350
22
798
d3 rank 23,828
23,829
Round-2 symmetry D750
29
720
d3 rank 19,728
19,729
Trajectory-complete T500
33
804
d3 rank 21,521
21,522
Early-prefix E600
31
627
d3 rank 10,217
10,218
Continuation C2500
16
242
d4 rank 214,975
214,976


C2500 retains depth 3, where the path state ranks 6,246, then loses depth 4. At width 16,384, E600 retains depth 3 but ranks the next path state 431,700. The evidence is consistent with systematic early misranking, not a single slightly-too-small beam.

C2500 checkpoint SHA-256 is 128f0ea1ab7b0df7f9b72ca054d3fcef1b81bd3feabca51a851fdad0a2feeb95. Its survival report self-hash is c4656ba13f2838f2f3ec6cdf98fa1512a7e1f85163db288d3317bd7b2a8b9d77.
Scalar-model experiment inventory
Experiment
Intended change
Measured result
Verdict
First-round DAgger
Put KMC labels on model frontier states
Cross-policy s350 reached fixed 160+ rank 11.237/top-8 50.75%, but PID 502 direct search failed
Useful parent, not solver
Round-2 scalar/imitation/symmetry
More q16 replay, action imitation, verified cube frames
D750 improved q4 rank to 12.75 and q2/q2b rank to 16.00, but hard rank regressed to 11.784 and d3 path rank remained 19,728
Reject
Full stitched trajectories
Train on behavior prefixes plus KMC suffixes
T500 improved q4/q2 ranks but regressed hard rank/top-8 to 11.778/45.95%; d3 path rank 21,521
Reject
Explicit q16 pair loss
Train top/cutoff ordering
Training-pair agreement rose 5/8 to 7/8, but held-out q4 rank only tied 13.50
Reject
Early-prefix curriculum
Oversample verified depths 0-8
E600 moved d3 path rank to 10,217 and q4 rank to 11.75, but hard rank/top-8 regressed to 13.333/39.64%; direct search failed
Reject
Long coverage controls
5.12M parent draws per arm
Plain/symmetry/action40 moved different diagnostics; no 5,000-step checkpoint passed the parent hard guard
Reject
Action40 continuation
Continue strongest long arm 3,000 updates
C2500 reached hard rank/top-8 10.925/51.05%, but q4/q2 regressed, d4 path rank was 214,975, and direct search failed
Data-collection behavior only
Base counterfactual pairs
640,000 same-root/depth pair samples
Validation pair accuracy fell 60.421% to 59.022%; S5000 hard rank/top-8 12.880/40.24%
Reject
Persistent prefix cohorts
Preserve alternative depth-2 prefixes
Reference reached d5 rank 23,400; implied uniform frontier 24.85M states and 894.67M edges/layer
Reject at measured resource bound


The full-trajectory and pair-label experiments are especially important: more supervised data did not make the labels more expert. A behavior prefix followed by a KMC suffix is a valid replayable path, but it does not prove that each prefix action is locally optimal.
C2500: best scalar checkpoint, not a solver
C2500 is the local step-2500 checkpoint from the continued action40 arm.

Diagnostic
Parent action40 s5000
C2500
Fixed overall rank / top-8 / MAE
9.426 / 60.10% / 46.17
9.127 / 61.48% / 40.65
Fixed 160+ rank / top-8 / MAE
11.405 / 48.65% / 57.02
10.925 / 51.05% / 46.43
q4 teacher-action rank
13.50
14.50
q2/q2b teacher-action rank
18.00
20.50


It is the first candidate to beat the original cross-policy s350 parent on both fixed high-cost rank and top-8. It simultaneously regresses the small oracle action controls, evicts the known path at depth 4, and fails direct depth 400. It was therefore frozen for q96 data collection, not promoted.

Its fixed, q4, q2/q2b, base-counterfactual, survival, and direct report self-hashes are 20f5e5d395380be2dc1e32aee2de6bafa65b24117af54808b50ba081e52b27c0, 192df02abfd502f753dc935c4dc5bdb3dc146edb9b807f10cc63f00d0fa38844, 2f18866ba27accac63d2f95cd16f8dc26642d16abed435745f30b1c1b5ac2119, da1d8bde9e8500fca9cff0ca5fb89e6d9c03bfcd6f608c1570557a653be104b7, c4656ba13f2838f2f3ec6cdf98fa1512a7e1f85163db288d3317bd7b2a8b9d77, and 6222482629162a6f3176644fb59d7c0ad54443d0df69a8670b09ee3f7e182904.
Direct-Q correction and experiments
The successful 555 production approach uses one 30-output Q(s,a) forward per parent, not scalar V on every child. This is an important architectural correction: a faithful Cube666 analogue would emit 36 actions per parent.

Direct Q is not an untested cure. The earlier Cube666 random-walk Q lineage used 12,288,262,144 fresh-state draws and still failed the hard full-PID gate. The 555 and archived Cube666 evidence SHA-256 values are 91005696b3dfae3269383958f333271a11d9faf9123537d3c55d807ea909778d and 15218b511114e19a5f00d1c066564357a5b40a91f7bb2eb4fefc716ba8279f37; the old q666_a.yaml SHA-256 is fb9838003cfb5543675d6e417acac8e4b2b497b329fade05439746e2292ad815.

The bounded new experiments asked whether one-pass direct Q could reproduce C2500's exact scalar child ranking cheaply enough to widen search:

Model
Teacher-best top-1 / top-4 / top-8
Mean top-8 set overlap
Verdict
Frozen C2500 trunk plus 75,301-parameter head
9.45% / 27.83% / 43.68%
33.36%
Reject, no wide beam
Fully trainable C2500-derived student
13.74% / 38.52% / 55.74%
36.57%
All 8 gates fail
Same student, backbone LR 1e-4
17.30% / 44.84% / 61.63%
39.60%
All 8 gates fail


For reference, uniform top-1/top-4/top-8 floors are 2.78%/11.11%/22.22%. The models are above random but still discard the teacher-best child too often for hundreds of consecutive beam layers. High global retained-set overlap did not repair weak within-parent sibling ordering. Even the exact C2500 teacher does not solve PID 502, so an imperfect C2500 surrogate cannot establish a better policy by fidelity alone.

The frozen-head, trainable-student, and higher-backbone-LR report file hashes are 1384e6f2fc359ec0ff5914f6118d671a1d02f0fcbe83398ed44c2c81e858726a, 1106f4e8d979140008d4a512ce61f769892f1629ccb230118b43d5da4fbd3706, and a04f95624d889c6b48a2168189f14fbc21df890893dc01ab2e79d044ede2084f.
q96 better-label campaign
C2500 exported top and cutoff states from 16 fresh base-training roots at depths 3, 32, and 64. KMC completed all 96 scheduled queries with no censoring.

Pair result across 48 same-root/depth pairs
Count
Beam-top has shorter observed completion
29
Cutoff has shorter observed completion
17
Tie
2


Mean U(cutoff)-U(top) is exactly +3.0 moves. Beam-top is better on average, but cutoff wins often enough that rank 0 is not an expert label.

After exact-state minimum-U aggregation and the full-base plus all-36-child leakage filter, the final corpus contains 22,664 train rows, zero validation rows, and U=3..209. The safe index contains 96 strict demonstrated transitions and 46 non-tied safe pairs.

Key q96 artifacts:

Artifact
File SHA-256
Self-hash
Finalization report
08ba72fd5c982c88484d99dfc1ddbd4ed4b22ed2827434f6f1ea60b6453c2a7e
ac623cf09b2947b62166c0e2996dbf8958cf06766e6cc7644d50c0c02566cb0d
Filtered corpus manifest
06fc8c26d4b9f80e4e6040b8e825afbcaf6478ac0c947467027d6d94b805e509
4269e531250e6d205d7fd8ebc877fdc046d158a0a165e5ee30238203fe676921
Safe pair index
19ae2bf6a49bd7d1d60b1bf59087ed9b3ab4011bcb5338e2d5d71c1ea9684611
b719250c305213dafae77eb2f0b8cec00a1cc1733c93880f62d1ef4223d80383


These 96 query labels/results are training data; suffix materialization from them contributes to the 22,664-row corpus. They cannot serve as held-out evidence for C2500 or for the post-q96 models trained from them.
Matched post-q96 result
Two 3,000-update arms started from C2500 with identical seed and source sampling. The control sampled no explicit pairs. The treatment sampled at most 32 safe pairs per update with loss weight 0.05, 96,000 pair samples total.

Candidate
Fixed 160+ rank / top-8
q4 rank / top-8
q2/q2b rank / top-8 / pair order
q96 rank / top-8 / pair order
C2500 parent
10.925 / 51.05%
14.50 / 50%
20.50 / 0% / 2/2
14.063 / 35.42% / 29/46
No-pair step 3000
11.237 / 48.35%
15.25 / 50%
21.00 / 0% / 2/2
14.490 / 36.46% / 29/46
Pair-0.05 step 500
11.288 / 49.25%
15.25 / 50%
19.75 / 0% / 2/2
14.427 / 37.50% / 29/46
Pair-0.05 step 3000
11.405 / 47.15%
15.50 / 50%
20.75 / 0% / 2/2
14.583 / 34.38% / 33/46


No checkpoint from either arm clears the parent fixed 160+ guard. The pair arm learns four additional q96 pair directions at step 3000, but q96 is its training set; q96 action rank/top-8 and held-out high-cost ordering worsen. Both arms are rejected. No post-q96 reference-survival or direct search was authorized.

Selected comparison checkpoint SHA-256 values are 6d87c4883bd8428a07bc273943e3a678caee398a83aa79547932b13835e601c0 for no-pair step 3000 and df2a9c9302c0021aa339fb2e55515ad0d3b46da04b160697b75ed83e5aac26a6 for pair step 500.

The no-pair and pair fixed-report file/self hashes are 63151184a0391cfc1e42462ccf38f86c904884530c23297824b58011119bbaaa / 39fa2f050d459046996abd283da57b29dbbbbad8ed23bfc0be941173202783d6 and c2e43f57441e77d37e2ccc81cadd41a7dd48451e0baa5551acf3a746b3d9a1dd / bbb673c92a144a3e2ba31e35bced732fedc704cf031950d1f4ad4b49e2a65bc6. The combined q4, q2/q2b, and q96 report file/self hashes are 6db9d00abbbee111e54a79707fc62d84ebdc972e6103cb63d6e207f85ec13e57 / 09c27af43d718594a72fab44e7cdac3137c66035ba7fa8347ea1f473f3f90c51, 324d23657e9cd4a2324e65107873f220263b7735636f6f75d7509df3e9865789 / 7f684ea0bc594456bd299b7f9897f20546ed32ad531c4d7c892eae0bd1d80c0d, and 137137eee15cf20d5e01f1ca924647c1d7d316a896eeb07c6714503a2a34a2ec / ba54767bdd3781b5a81f6573a776457450efb8e6f8c534f58663cdaaefc08c2f.

The final adjudication rejects both arms and binds 28 input identities. Its file/self hashes are cea73c13006e04dd4d6e1e3c27879c32004fefe044bc299a0d25ee0762d6fc61 and 94d2cefef5d701b11f907686d70428e121b8f993cc4329bd3c579ae4a7ae0862.
Measured failure mechanism
The accumulated evidence supports six linked conclusions.

The model leaves its label distribution immediately. Base training follows successful KMC trajectories; beam search visits its own states.
Deep scores show false progress. Mean model V falls from about 137 at depth 0 to 27 at depth 128, while KMC completion from sampled deep frontier states still costs roughly 182-200 moves.
Sibling ordering is weaker than aggregate calibration. MAE and cross-parent ordering can improve while the correct next action is evicted.
Observed suffixes are upper-bound demonstrations. They are valid paths, not proof of locally optimal actions or negative labels for alternatives.
Pair objectives fit their source pairs. q16 and q96 pair accuracy moves in training, but the fixed held-out high-cost ordering does not improve.
Search scale cannot repair a systematic scorer. More width, depth, ensembles, symmetry frames, and persistent cohorts preserve more wrong states without establishing the correct long-horizon path.
What the campaign does and does not establish
Established:

The tested scalar-V line does not solve the measured whole-PID cases.
C2500 is the strongest tested scalar behavior by the fixed high-cost guard, but it is not a solver.
The current upper-bound and on-policy pair labels are insufficient to repair the early/deep ranking failure under the tested objectives.
C2500-distilled direct Q is too inaccurate for a wide search at the declared gates.
Blind width/depth scaling is not supported by the measured survival and hybrid-search evidence.

Not established:

That all direct-Q architectures fail. Exact-distance, stronger-policy, or search-aware direct-Q targets were not trained and evaluated to a decision-eligible budget here; the path-action CE experiment was only a four-update plumbing smoke test.
That no alternative beam state representation can work.
That no non-restoring neural-to-classical handoff can help.
That a model trained against actual downstream completion cost cannot solve full PIDs.
First-10 direct scalar-V diagnostic: results pending
Two active shards are running C2500 with the direct scalar-V beam at width 8,192 and depth cap 400 on literal PIDs 0-9. This is a diagnostic requested on the literal first ten submission PIDs. It is not a held-out promotion gate.

Submission order was checked against gdrive/artgor_meta/cube666_submission_176881_2026-08-22.csv (SHA-256 47398c5139f96311aead6b32e6393346332719942288ce8f1dec7bc006b4af82): its first ten data rows are PIDs 0-9. Search loads original states only from data/test.csv (SHA-256 19a4a192fae217dfb693364512fb7d49cd598acd173c2f2434031cda992887fd); it does not load the submission's solution paths.

Both shards require the explicit --allow-seen-roots diagnostic override; the evaluator still rejects checkpoint-seen roots by default. Their contracts bind allow_checkpoint_seen_roots: true and runner SHA-256 4d299e75a943118899f3a7d1d4c0ee965d14f5ccd190a868411421303a333161.

Width 8,192/depth 400 is the campaign's bounded full-depth direct-search gate, not the earlier width-524,288 short-prefix hybrid setting. A width-524,288, depth-400 direct search was not attempted; this diagnostic only answers the configured width-8,192 question. Each fully exhausted root evaluates 117,119,628 candidate edges.

Without KMC here describes inference: neither KMC nor a stored classical path is loaded or called by the search. The model's historical training labels were produced from replay-verified solver trajectories, including KMC-derived data; this run cannot erase that training provenance.

C2500's actual training roots overlap PIDs 2, 4, 5, 6, and 7. Its broader source plan also contains PID 1, so the evaluator's conservative seen-root union is 1, 2, 4, 5, 6, and 7. There is no validation-root or frozen-eval overlap. PIDs 0, 3, 8, and 9 are unassigned rather than members of a predeclared evaluation cohort; 0, 3, and 8 also appeared in prior direct evaluation. Therefore this cohort must not be described as fresh, held out, or an unbiased solve-rate estimate. No substitute root set is used.

Model checkpoint:

data/whole_pid_value_dagger_round4_coverage_imitation40_continue3000_s5000_v1/checkpoint_00002500.pt

SHA-256:

128f0ea1ab7b0df7f9b72ca054d3fcef1b81bd3feabca51a851fdad0a2feeb95

Do not enter partial shard results in this table. Fill it only after both shards finish, the merged inventory covers every PID exactly once, and every claimed solution replay-verifies.

PID
Checkpoint seen-root provenance
Solved
Path length
Replay verified
Candidate edges
Retained global states
Wall seconds
0
none
pending
pending
pending
pending
pending
pending
1
source plan only
pending
pending
pending
pending
pending
pending
2
training + source plan
pending
pending
pending
pending
pending
pending
3
none
pending
pending
pending
pending
pending
pending
4
training + source plan
pending
pending
pending
pending
pending
pending
5
training + source plan
pending
pending
pending
pending
pending
pending
6
training + source plan
pending
pending
pending
pending
pending
pending
7
training + source plan
pending
pending
pending
pending
pending
pending
8
none
pending
pending
pending
pending
pending
pending
9
none
pending
pending
pending
pending
pending
pending


Reserved merged result: pending both shards.
Reserved solve count: pending.
Reserved report file/self hashes: pending.
Reserved interpretation: diagnostic only, regardless of outcome.
Recommended next direction
Do not spend another campaign merely continuing C2500 distillation or widening the same scalar scorer. A defensible next experiment should change the source of action information:

Obtain labels tied to actual downstream completion cost or a stronger policy, with root-disjoint validation selected before training.
Train a native 36-action model on those targets rather than only imitating C2500 child values.
Gate first on exact within-parent child ranking and reference survival, then on a fixed direct full-depth beam operating point.
Keep hybrid completion results separate from direct neural solve results.
Artifact and migration index
Artifact
SHA-256
Authoritative 2026-09-03 campaign report
1f771a294284a8e8835deacd610f4bd65e78adf5ff71f242ba15c440c86c76b4
Final post-q96 adjudication file
cea73c13006e04dd4d6e1e3c27879c32004fefe044bc299a0d25ee0762d6fc61
Final post-q96 adjudication self-hash
94d2cefef5d701b11f907686d70428e121b8f993cc4329bd3c579ae4a7ae0862
Migration top-level manifest
92a88bfc05681202bb0aa325cfe9530efd24bfb10c16dd90d5eaa1fc03695f0b
Migration delta archive
1d2bc5d31b989d053beea6409a3402cf730484cf77cb870e4731c062b13afa33


The migration bundle is gdrive/artgor_meta/cayley/cube666_whole_pid_scalar_v_campaign_2026_09_03. It contains the costly raw q96 evidence, final corpora and reports, C2500, and the selected post-q96 comparison checkpoints. Rejected direct-Q and non-selected periodic weights are intentionally excluded while their reports and frozen source are preserved.

