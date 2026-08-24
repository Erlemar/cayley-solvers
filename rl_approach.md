Beam-AVI: reinforcement learning for the 4x4x4 cube
Result: 2499 moves vs the 46,662 community floor's 2609 on the hardest 54 test scrambles at deployment width (B=2^20) -- -4.2 percent below the floor, beating the previous best model by 124 moves, 37W/4L/13T. First model in this project to go under the floor rather than approach it.

Code: cube444_tf/{beam_avi.py, gen_harvest.py, probe/instrumented.py}. Deployable checkpoint: runs/T/r020 (189k steps) -- but see s7, r010 at 90k may be better.


1. The problem
Find short paths to the identity in the Cayley graph of the 4x4x4 cube: 24 generators, 7.4e45 states, test scrambles at true distance 44-50. The competition metric is total moves over 1043 scrambles, not solve rate.

The deployed solver is a width-B beam search scored by a learned Q head, where Q(s,a) = d(apply(s,a)). One forward per parent prices all 24 children, which is what makes B=2^20 affordable. So the learning problem is: produce a scoring function whose induced ranking makes a bounded-width beam return short paths.

Solving was never the bottleneck -- the beam solves ~100 percent. Solving short is.
2. Why the incumbent objective cannot work at depth
The standard recipe (DeepCubeA's ancestors, CayleyPy's "diffusion distance", every public notebook) regresses the random-walk index: walk k steps from solved, label the pivot k.

That label is exact when shallow and worthless when deep. A non-backtracking walk of length 58 lands at true distance ~44, so past depth ~40 the label is ~14 moves adrift. Measured on the shipped model, same metric, two pivot sources:

depth band
random-walk pivots
geodesic-path pivots
38-42
pair 0.564
0.877
44-50
pair 0.526 (chance)
0.792


The deep failure is a LABEL defect, not a model defect. And ~1000 of the 1043 test scrambles live in exactly that band. Six independent levers (capacity, beam width, anchor ball size, dual head, training time, label fixes) were measured against it and none closed it, because all six were model-side or search-side. This is label-side.
3. The RL formulation
Only two mechanisms manufacture correct signal at depth, and both are RL: bootstrapping and learning from the searcher's own output. Beam-AVI is both.

Approximate value iteration on the Q head, with a perfect model of the dynamics:

Q(s,a)  <-  0                                if apply(s,a) is solved
            1 + min_a' Q_target(child, a')   otherwise

This is the deterministic, known-transition special case of Q-learning -- the same algorithm class as DeepCubeA (Nature MI 2019). The target contains no walk index, so it is immune to the defect in s2. Two things ground it, both load-bearing:

the terminal case above, and
exact d<=6 BFS anchors in every batch (67,041,677 states, all 24 columns exact). Q == 0 is also a fixed point of the recursion; the anchors pin the absolute scale.
Precondition, and why 444 and not 666
Bootstrapping cannot create signal that the target net does not already have. On 666, every model is at chance past depth 85 and a faithful 20k-step Bellman run took beam wins 13 -> 1. Before committing GPU we measured the precondition directly on 444: replay the reference solution through the beam and record the rank of the on-path candidate among all candidates.

At B=2^20, cross-parent steps: mean percentile 0.034 (z = 30.2) against a 0.5 no-signal null. There is ample deep signal to bootstrap from. Run this probe before any AVI run on a new puzzle (probe/p1_path_rank.py); it costs ~1 GPU-hour and is a hard go/no-go.
4. The idea that makes it work: the beam IS the sweep
AVI needs states to back up. Every prior implementation drew them from random walks -- including this project's own, whose sampler is k~U[2,45] with the pivot inside the walk: mean depth 11.75, 0.8 percent of mass past depth 40, against a test set that starts past 40. The code's own docstring flags this as an open follow-up and then uses the default.

The fix is free, and it comes from the search:

A width-B beam step forwards B parents to score all 24 children, keeps the top B, and at the next step forwards exactly those survivors. So min_a' Q(child, a') is already on the GPU. The beam is computing a Bellman backup on B states per step and discarding it.

Harvesting it gives a sparse-column masked target -- the exact shape the trainer already consumes -- on the deployment state distribution by construction. Measured:





wall overhead
+0.2 percent (interleaved control, reps 1+)
search perturbation
none -- probed and control paths byte-identical
target correctness
verified against an independent recompute, within one bf16 ULP
yield
~190k rows per scramble at B=65536, ~24 s


Because targets are precomputed by the generating pass, training also stops paying the 24-forwards-per-state backup and runs at ordinary supervised throughput.
The bias this has, and the fix
The free harvest only backs up surviving children, and a child survives because the beam ranked Q(parent, action) low. The target then reads that same child's min Q, so the stream is systematically optimistic and self-confirming -- it never learns that a discarded action was bad.

Corrected with a second stream: --full-expand 256 samples 256 parents per step without reference to Q and expands all 24 children explicitly. Unbiased, separately weightable, costs +9.5 percent (= 256*24/65536, matching the cost model exactly).
5. The loop
per round (x40, ~30 min each on one A100):
  1. freeze the target net = current online weights
  2. GENERATE  beam B=65536, 70 steps, 50 fresh scrambles     (~20 min)
               -> ~9.5M sparse rows + ~0.6M dense rows
  3. TRAIN     9,000 steps, batch 1024                        (~10 min)
  4. refresh the target net, re-generate on the new weights

Two correctness properties that are easy to get wrong:

The scramble seed is derived from the round number. A fixed seed re-draws the identical sample every round and the buffer silently stops being fresh.
The gate pids are passed as an explicit exclusion list, not a count. The incumbent Bellman code held out 100 pids by its own seeded permutation, which left 47 of the 54 gate pids inside the training set.
6. The loss
L = masked MSE on the harvested column            sparse, free, ~9.5M rows/round
  + 1.0 * MSE over all 24 columns                 dense, unbiased, 256 parents/step
  + 2.0 * MSE against exact d<=6 BFS anchors      every batch, pins the scale

Plus 24-frame symmetry augmentation (sym(s,R) = color_map_R[s[rotation_R]] with the action column relabelled -- the slot-permutation-only form is legal-looking and wrong), and a loose global clip of the target to [0, 60] that only bites if the bootstrap diverges upward.

Do not soften the absolute level term. Huber-capping it is measured on this puzzle to take the beam from 18/18 to 5/18 solved, while improving every offline metric. The beam takes a global top-B across (parent, action) pairs, so cross-parent comparability of the raw level is what it runs on.

lr 2e-5, wd 3e-3, grad clip 1.0, 200-step warmup, warm-started from the incumbent. Model: PieceTransformer, 3.38M params, d_model 256 / 4 layers / 8 heads / ff 1024.
7. Results
Deployment width, B=2^20, all 54 hardest pids, both arms solve 54/54, floor 2609:

arm
total
vs floor
beats floor on
moves saved
s3 (previous best)
2623
+0.5%
15 pids
34
Beam-AVI r020
2499
-4.2%
40 pids
114


Head-to-head -124 moves (-4.73%), 37W/4L/13T.

Attribution and saturation -- matched on the 50 pids all five arms solved (floor 2415), every arm warm-started from the same checkpoint, B=65536:

arm
Bellman states
steps
total
vs s3
W/L/T vs s3
s3
--
0
2741
--
--
r005
beam frontier
45k
2587
-154
30W/9L/11T
r010
beam frontier
90k
2527
-214
36W/5L/9T
r020
beam frontier
189k
2569
-172
38W/9L/3T
C20
random walk
189k
3269
+528
2W/46L/2T


Two things fall out:

The state distribution is the entire effect. C20 is not merely no better -- it is far worse than the checkpoint it started from, losing to it on 46 of 50 pids. Identical steps, anchors, lr, seed, init. The only difference is where Bellman states come from. So the finding is not "scale up Bellman"; it is bootstrapping on the deployment distribution works, and bootstrapping on the random-walk distribution at the same budget destroys the model. This also explains why "the training axis is closed" kept being the conclusion: the incumbent s3 came from random-walk Bellman, but only 6,000 steps. A little helps; 189k wrecks it.
The recipe is ~5 hours, not 20. The gain is essentially complete by 45k steps and r010 at 90k is the strongest arm on the matched set. Rounds 10-39 bought nothing.
8. Two readings we got wrong
E[target] was inverted. It rose 13.1 -> 32.5 over arm T and was repeatedly flagged as possible divergence (it is the [[cube666-bellman-555recipe-dead]] signature). It was the opposite: the model had been underestimating depth by ~15 moves and the climb was it learning the true scale. The control's flat E[target] was the pathological one -- flat meant the shallow states carried no new information. On this puzzle, flat E[target] under bootstrapping is a warning, not a health sign.

Gate the interior, not just the end. Arm T's solve rate fell in its final quartile while E[target] climbed. Gating only the last checkpoint would have read as a modest win; the interior checkpoints are where the model actually is. When E[target] has not plateaued, always gate a mid-run checkpoint.
9. Measurement traps hit while building this
Recorded because each produced a confident wrong number:

Warm-up masquerading as overhead. A probe whose body was return measured +15.8 percent because the probed arm always ran first. Interleave A/B/A/B and drop rep 0. True cost was +0.2 percent. The tell was that the delta did not move with the variable (cap=1 and cap=16384 both +15/16).
bench_beam's result DB path encodes seed/tail/history/mlp but NOT beam width, so a 65k run and a 2^20 run collide. Reading the live DB while a later gate was running produced a fabricated "-276 moves, 31W/0L". Read results from the gate's --out JSON, never the DBs.
A boolean-mask assignment in the harvest hot path synchronised the device every step. Use torch.where; keep the hot path free of .item(), .nonzero(), bool indexing, .cpu().
bf16 tolerance must be scale-aware. One ULP at a target of 34 is 0.25, so a fixed 0.05 threshold fails on rounding alone. bellman_targets itself differs from an fp32 recompute by more than the two bf16 paths differ from each other.
A stray cd persists across Bash calls and silently broke a monitoring check. Absolute paths everywhere.
10. Reproducing it
PY=/home/artgor/cube444_a100_handoff/.venv/bin/python
cd /home/artgor/cube444_tf

# 0. GO/NO-GO: is there deep signal to bootstrap from? (~1 h)
$PY probe/p1_path_rank.py --weights runs/s3/best/model.pth \
    --info runs/s3/best/model.json --n-pids 20 --beam 65536 --no-compile
#    cross-parent percentile must beat the 0.5 null. At chance => stop.

# 1. TRAIN (~5 h for the useful 10 rounds; we ran 40 and wasted 30)
$PY beam_avi.py --name T --init runs/s3/best/model.pth \
    --init-info runs/s3/best/model.json \
    --rounds 10 --train-steps 9000 --batch 1024 \
    --gen-pids 50 --gen-beam 65536 --full-expand 256 \
    --anchors data/q_anchors_d6.pt --lr 2e-5 --seed 0

# 2. GATE -- control in the SAME invocation, judged on merge gain not the mean
$PY bench_beam.py --arms "s3=runs/s3/best,r010=runs/T/r010" \
    --pids <the 54> --beam 1048576 --steps 110 --mlp-weight 0.0

Never promote on training loss. This project has four retractions from probe-to-beam dissociation; only a replay-verified beam total on >=54 pids counts, and deltas under ~2 percent are noise.
11. Open
r010 vs r020 at B=2^20 (~9 h). s7 suggests r010 is the better checkpoint and it would be expensive to run the full set on the wrong one.
Full 1043-pid run at 2^20 (~72 h) -- the only path to an actual submission. Consider targeting the pids whose current merged path is longest; on tetraminx yield was strongly graded by path length.
Track B: a policy head + GRPO. There is no policy head in either codebase (az_head is a scalar value head). A 24-logit head on the same trunk, behaviour-cloned on the beam's own action choices from the harvest, then improved against the exact verifier, would give a structurally different solver for the merge -- which is where the published CayleyPy work finds most of its remaining moves (29-agent ensemble 48.98 -> 46.51).

