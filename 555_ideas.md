1. Switch to orbit-factored one-hot
This should be tested independently because it has almost no inference-cost downside.
The six moving 24-sticker orbits can be represented as six 24×24 permutation matrices:
\[
6 \times 24 \times 24=3456
\]features, plus any required fixed-centre metadata. The project’s proposed slot-wise form is 150×24 = 3,600 features—exactly the width of the current flattened 24-dimensional embeddings.
Benefits:
lossless within each legal slot orbit;
no learned global sticker-embedding bottleneck;
easier symmetry relabelling;
same first-layer size and approximately the same throughput;
exposes the physical orbit structure directly.
First test only the encoding change with the same 1024×10 trunk, objective, seed family, updates, and data. If it helps, a later model can use six shared orbit encoders followed by a small six-token mixer. Attention over six orbit tokens is cheap; sticker-level attention is unnecessary.

2. Limited-horizon Bellman on real beam regions
One-step warm Bellman was the strongest existing training lever. The next version should supply denser, more search-relevant targets.
For every parent/action child:
Search 4–8 steps from the child.
Use an exact endgame distance on ball hits.
Otherwise back up the best frontier target.
Train all 30 action columns, not only undo/next.
Sample heavily from actual failed and near-successful beam frontiers.
This attacks the unlabelled-column problem directly.
Use double targets:
network A selects the frontier/action;
network B evaluates it.
Otherwise, taking a minimum over many noisy leaves will create optimistic bias and can make the bootstrap eat itself. Retain exact anchors in every batch and monitor E[target], deep action margins, and single-frame beam coverage.
Compare:
another 20k ordinary one-step Bellman continuation;
20k limited-horizon Bellman updates;
from the same starting checkpoint and at the same learning rate/update budget.

3. The all-column argmin hinge — my main suggestion. You compute all 30 columns in one forward pass and supervise 2 of them. The other 28 are free supervision you're discarding, and there's an exact 
constraint available for them:                                                                                                                                                                         
																																																	 
▎ the undo child achieves d−1, which is the minimum any child can achieve, so min_a Q(s,a) == Q(s,undo) identically.                                                                                   
																																																	 
Add sum_a max(0, sg[Q(s,undo)] − Q(s,a)) — one-sided, stop-gradient on the undo term so it can't cheat by pushing Q(undo) down. Why this is the right shape:                                           
																																																	 
- It's the metric you measure. top-1 at depth (0.246) is exactly "is undo the argmin". Currently nothing in the loss optimizes it directly.                                                            
- It's saturation-immune. Purely relative to the state's own level, so the ceiling at 42 vs true 72 is irrelevant to it.                                                                               
- It attacks the actual beam failure. The beam takes a global top-B over (parent, action); it dies when bad children get spuriously low scores and crowd out good ones. This term suppresses exactly   
that.                                                                                                                                                                                                
- It's correct in every case, including alternative geodesics: a second geodesic child is equal to undo, and a one-sided hinge pushes it up to equal, not down.                                        
- It's free — 15× the label density at zero extra compute.                                                                                                                                             
																																																	 
Failure rate is the same ~7% as your existing sparse-Q labels (tetraminx: undo/next ordering right 92.75%, tie 6.14%, inverted 1.11%), and the same exact-ball correction fixes both.                  
					===
					
					
Comment to this idea:
The intuition is good—unlabelled columns probably produce spurious low Q values—but the claimed identity is false for random-walk training.
I would test a corrected version, not the zero-margin argmin hinge as written.
Why the identity is not exact
The training walk gives:
\[
s_p \xrightarrow{\text{undo}} s_{p-1}
\]so undo has a known solution of length at most \(p-1\). But the walk prefix is not necessarily geodesic:
\[
d^*(s_{p-1}) \le p-1,
\]and another child may have lower true distance than the undo child.
On 555 this matters particularly near k=80, while the random-state distance scale is around 72: the walk already contains slack. Non-backtracking only prevents immediate cancellation; it does not make the word shortest.
Because all 30 generators are odd, child distances have opposite parity to the parent. If undo is descending, it has distance \(d-1\); if it is ascending, it has distance \(d+1\), while another child can be at \(d-1\)—exactly two moves better. The proposed zero-margin hinge would incorrectly push that genuinely better child upward.
It also treats alternative optimal actions as negatives. Ties are allowed by the hinge, but any truly better alternative is suppressed.
The corrected exact constraint
Any two children of the same parent are at most two primitive moves apart:
\[
s_a \rightarrow s \rightarrow s_u.
\]Therefore the triangle inequality gives:
\[
|d^*(s_a)-d^*(s_u)| \le 2.
\]That suggests this one-sided lower-tail hinge:
\[
L_{\mathrm{lower}}
=
\frac{1}{30}\sum_a
\max\left(0,\operatorname{sg}[Q(s,u)]-2-Q(s,a)\right).
\]In code:
q_undo = q.gather(1, undo[:, None]).detach()
loss_lower = torch.relu(q_undo - 2.0 - q).mean()
This says:
No other child should be predicted more than two moves closer than the undo child.

Unlike “undo must be argmin,” this is compatible with:
undo being the wrong direction;
another action being genuinely two moves better;
multiple optimal actions;
non-geodesic random-walk prefixes.
It still attacks the suspected failure: arbitrary unlabelled columns falling far below the supervised undo value and stealing global beam slots.
Why it could help
Sparse-Q currently gives each row:
one supervised low column;
one supervised high column;
28 largely unconstrained columns.
A spurious action predicted 5–10 points below undo can dominate beam selection. Because beam trajectories are extremely sensitive to these errors, different symmetry frames can generate independent paths even if their overall Q predictions are similar.
The slack-2 hinge provides dense lower-bound supervision at negligible forward cost. It may:
reduce false-low actions;
improve global parent/action ranking;
reduce frame disagreement;
increase single-frame coverage.
The stop-gradient is correct: violations should push suspicious columns upward, not allow the loss to cheat by lowering the undo reference.
Possible symmetric version
True child distances also satisfy the upper side:
\[
Q(s,a) \le Q(s,u)+2.
\]So a full Lipschitz loss would be:
lower = torch.relu(q_undo - 2.0 - q)
upper = torch.relu(q - q_undo - 2.0)
loss_lip = (lower + upper).mean()
I would start with the lower term only. Excessively high Q columns do not enter the beam; false-low columns are the dangerous ones. The upper term could compress useful action margins without improving search.



					
4. Margin hinge on the labelled pair. max(0, 2 − (Q(next) − Q(undo))). This is top1_margin_weight in Vlad's framework, shipped at 0.0 in every config — untested everywhere. MSE does push the gap     
apart, but it also rewards absolute closeness to p, and when the state is ambiguous the absolute term wins and the gap collapses. Honest counter-evidence: three ranking probes tied on megaminx — but 
that was a scorer whose child ordering was already near-optimal. Yours is at 0.246. Different regime, worth the arm.                                                                                   
			   
			   
			   