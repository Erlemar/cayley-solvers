# Review of "Beyond V + beam: new solver paradigms for Cayley 444, 555 and 666" (brief dated 2026-09-04)

Reviewed 2026-09-05 against the repo's measured record (`RESEARCH_SYNTHESIS_2026-09-04.md`,
`IDEAS_DETAILED_2026-09-04.md`, the HANDOFF / EXPERIMENTS / VERDICTS files they cite) plus one
new check (section 2). The brief proposes seven directions organised into two products:
**A**, a learned solution compressor over complete words, and **B**, a recurrent program
constructor with exact execution as the checker.

## 0. Verdict in one table

| # | Direction | Brief's own rating | My rating | Reason (measured) |
|---|---|---|---|---|
| 1 | Learned superoptimizer over equivalent words (Product A) | strongest fresh shortening bet | **low on 444/555/IHES/tetraminx; bounded pilot on 666** | local slack is exhaustively closed to radius 12-13 on beam paths; cross-region slack measured 0; on 666 the slack is decomposition-level and the learned macro-context line already harvested it (-5.9k) |
| 2 | Shared-setup program synthesis | high upside, substantial risk | **medium as score insurance, not a paradigm** | a learned version of KMC's SA + insertion; ceiling set by the macro grammar's bits/move (~2.0-2.7 vs 4.86) |
| 3 | Recurrent constraint planner (Product B core) | strongest architectural departure | **lowest** | re-encodes pathfinding as CSP with the same information gap; boundary states reintroduce the aiming problem that killed bidirectional MITM and subgoal search; no supervision beyond corruption recovery |
| 4 | Masked action diffusion with block repair | component for 3 | **component only** | coordinated multi-region repair is subsumed by exact/beam rewriting of the union span below their radii; above them it has only random-word supervision |
| 5 | Orbit-consensus decoder | original, speculative | **keep the encoder, drop the decoder** | encoder = the cluster-factorised transformer already planned; direct word emission at H~100 has no foothold anywhere |
| 6 | Learned backtracking controller | bounded lane | **no as a solver; maybe as a non-neural budget allocator** | a sequential controller against a 2^21-wide parallel beam is an orders-of-magnitude throughput mismatch |
| 7 | Coupled stochastic bridges | postpone | **agree, postpone** | bidirectional MITM measured midpoint overlap ~0 (closest approach V(z) 14.5 vs true residual 49) |

Net: the brief is well argued, careful about exactness and controls, and correct that a bigger
encoder on the same scalar target is not a new idea. But its central premise (that the
representation, target and search must all change) does not follow from the record. The
binding constraint on the big cubes is **distance information at depth 80-100**
(`V_vis / M` 0.78 on 666 vs 0.9 on 555 and ~1.0 on 444), and none of the seven designs creates
that information: Product A needs an incumbent, and Product B hides the same gap inside a
recurrent network with less supervision and worse throughput.

## 1. What the brief gets right

- **Exactness as a final constraint, not a confidence.** Every output replays from the
  original state; fractional move mixtures and "almost solved" do not count. This matches our
  rules (replay-verify every splice; a timeout is not a proof).
- **Shortening is not solving.** Designs that need an initial solution are relabelled
  honestly; a classical incumbent shortened by a model is not a neural-only solve.
- **Random and monotone controls.** The NeuroCore citation (random refocusing matched neural
  guidance on SATCOMP 2018) is the right warning and is our rule 28 in another domain.
- **No numbers before pilots.** It refuses to attach percentages, which is more honest than
  most of our own planning docs.
- **The "not novel" list** (bigger transformer on the scalar target; fixed macro
  dictionaries; a plain full-666 GFlowNet port; diffusion with classical repair; generic MCTS)
  is accurate and saves us from re-running things.
- **The forward/backward Hamming invariance.** The brief derived that
  `Hamming(F_t, B_t)` is constant at every cut of a fixed word and checked it on the real
  rules. That is correct, and it is stronger than stated (section 2).
- **The e-graph deduplication warning** (all complete solutions share an endpoint, so
  endpoint dedup collapses the population) is a real implementation trap.

## 2. The invariance lemma, strengthened, and what it rules out

Let `F_t` be the state after the first `t` moves of a candidate word from the start and `B_t`
the state obtained by undoing the suffix from the goal. Write the residual as the position
permutation `r_t` with `B_t = F_t[r_t]`. Advancing the cut right-multiplies both `F` and `B`
by the same generator, so `r_{t+1} = g^-1 r_t g`: **the residual moves within one conjugacy
class along the entire word.** Checked today on the actual 666 generators (random reachable
start, solved goal, random 40-move non-solving words, three trials): the Hamming count is
constant (204 / 213 / 210) and the full cycle type of the residual is identical at all 41 cuts.

Consequence: any class function of the residual (Hamming distance, number of cycles, cycle
lengths, the classical "3-cycle unit" count, parity, any PDB-style max/sum over orbits of a
class quantity) is **constant along a fixed word** and carries zero information about where
the word is wrong. Only position-specific structure varies. This is broader than the brief's
remark and it touches three of its directions:

- Direction 3's chunk-boundary repair cannot be steered by any scalar residual; the boundary
  states have to be free variables, and then the constraint "chunk `j` connects `x_j` to
  `x_{j+1}` within `h` moves" is a distance oracle at radius `h`.
- Direction 5's "which component obligations remain inconsistent" is informative only as a
  set of positions, never as a count per component.
- Any blocked-Gibbs style "trajectory energy" built from residual statistics is flat.

The lemma also says something useful about our own practice: progress along a *path* must be
measured by re-scoring states, never by residual statistics of a fixed word. We already do
this, but it is worth writing down as a rule.

## 3. Direction by direction

### 3.1 Learned superoptimizer (Product A)

The brief's premise is that "greedy shrinking windows cannot cross a barrier" where a word
must first get longer or be rearranged before a distant cancellation becomes available. Two
things in the record cut against this.

First, our exact rewriter is not greedy shrinking. The MITM ladder replaces a window of
length `L` with **any** word of length `< L` that has the same effect, whatever intermediate
forms lie between; within its radius it is complete and non-monotone by construction. It is
complete to radius 12 on tetraminx (166,703 queries, **0** improvements), found the only
exact win at radius 13 (-8 over 6 pids), is complete to reach 11 on cube444 (every window
`<= 11` proven geodesic), reached its practical ceiling at reach 9 on cube555 (-224 on the
109,512 file, then 0), and on IHES 1,816 of 1,863 exact searches proved optimality and all
51 pids of length `<= 20` are proven optimal. The barrier the brief describes is therefore
not "windows cannot lengthen"; it is "windows longer than 26 cost ~11x per extra move of
depth".

Second, cross-region slack, the thing a learned non-monotone rewriter would exploit, was
measured directly and is zero on beam paths: cross-trajectory splicing found 0 of 6,979
exactly-known waypoints with slack (tetraminx), Knuth-Bendix completion over 1,844,967
derived rules found 0 (longest left-hand side 9, subsumed by radius 12), relation mining and
commutator libraries found 0 on IHES and megaminx, commutation reduction found 0 on 555 beam
output, the megaminx union-graph recombination headroom was 2 moves over 998 pids, and the SA
`commuting_swap` operator accepted 0.2% of proposals. A learned policy over the same relation
set has nothing left to find on those files, and the brief's own novelty test ("substantial
only if it learns sequences of non-improving, context-dependent transformations across
distant regions") is a test we have effectively run with exact and stochastic controls.

On 666 the situation differs and the brief is partly right: KMC paths are ~165 moves against
a counting bound of 102, so there is 50-60 moves per pid of slack. But that slack is not
local (primitive re-solves of KMC windows at spans 34 and 58 returned **0**) and it is not
rearrangement (`commuting_macro_reordered` returned 0). It is the choice of decomposition into
cycle units, which KMC's simulated annealing already optimises over macro sequences with a
cost model, and which our learned macro-context line already attacked for **-5,870**
(176,889 -> 171,019). That line *is* Product A on 666, and its ceiling is the finisher's
2.0-2.3 bits/move.

What survives: **Experiment A** as the brief specifies it, one day, on two inputs where slack
is known to exist: the 666 KMC paths and the 705-move factorised-finisher paths, with the
random-rewrite and monotone-learned controls. If the learned non-monotone arm beats both
controls by more than noise, that is a real finding; my prior is that it will not on KMC
paths and might on finisher paths (whose per-macro structure is loose). Do not run it on
444/555/IHES/tetraminx files; the ceiling there is measured.

### 3.2 Shared-setup program synthesis

The identity `(X A X^-1)(X B X^-1) = X (A B) X^-1` and the observation that independently
optimised cluster operations hide shared setups are both correct, and this is the most
practically useful of the seven. It is also not new in kind: KMC's bulk phase is exactly
shared work (an inner-inner commutator executes two parallel centre 3-cycles in four turns, an
inner-outer commutator six units in four), the insertion finisher places each macro "at the
best point in the existing path", and the SA energy `|P| + 3 M(P)` is the fully expanded
length the brief asks for. A typed-program synthesizer with a learned controller over holes
is a learned SA plus insertion beam.

The ceiling is structural. A pure-effect macro is a commutator or conjugate, so its length is
at least `2(|X| + |Y|)` for an effect worth a handful of bits; that is why the finisher runs
at 2.0-2.3 bits/move and the whole KMC path at 2.66 (Diener 2.75) against a 4.86 ceiling.
Reaching 140 moves needs 3.1 bits/move over the *whole* path, 116 needs 4.3; no grammar of
pure-effect macros gets there. The brief's remedy, "retain primitive leaves so the
representation does not exclude ordinary words", is right but returns the search to the
primitive-move problem, where the scorer's visibility is the constraint again.

Verdict: a legitimate way to take a few thousand moves off 666 before 2026-11-20 inside the
hybrid lane (the docs' P3'), with a well-designed decisive experiment (unseen pairs/triples of
cluster effects vs concatenation + reducer vs the fixed library at equal cost). Budget it as
score insurance, not as the route to a model-only competitive solver.

### 3.3 Recurrent constraint planner (Product B)

This is the brief's "strongest architectural departure" and, on the record, the least
grounded. The reformulation is: find `a_1..a_H` with `T(start, a) = goal`, introduce chunk
boundaries `x_j`, and let a recurrent network revise actions and boundaries until the plan is
consistent.

Three problems:

1. **The boundary states are the whole problem.** Each `x_j` is an element of a group with
   `3.1e149` elements; a random one is ~102 moves from everything. The chunk constraint
   "connect `x_j` to `x_{j+1}` within `h`" is precisely what our exact ladder solves for
   `h <= 13` and the V-beam bridge for `h` up to V's saturation (~30 on megaminx), and the
   outer problem of *choosing* the `x_j` is the aiming problem that killed bidirectional MITM
   (best learned midpoint: predicted residual 14.5, true 49; corridor at `d = 40` ~1e55
   states) and subgoal search (k=2: 2/6 solved and +80 moves; k=3: 0/6). A recurrent network
   that can place boundaries well has learned a distance function; if it can do that, we
   should use it as a V.
2. **No gradient, no residual.** Multiple shooting works in continuous control because the
   defect between chunks is a differentiable quantity. On a Cayley graph the constraint gives
   no gradient, and by section 2 every scalar residual along a word is constant. The network
   must learn a proxy from data, and the only data the brief proposes are corrupted valid
   words, which teach the corruption process (the brief concedes this in section 11).
3. **The cited evidence is small and local.** TRM/HRM solve Sudoku and mazes with local
   constraints and tiny state spaces; NeuroSAT solves small random SAT; Ren and Liu document
   incorrect fixed points. None has a horizon of 100+ discrete decisions under a single global
   group equality.

The proposed pilot, root-disjoint 16-32-move windows, is dominated by exact search: radius
13 solves any 26-window exactly on tetraminx, reach 9 does the same on 555. A positive
result at that scale would establish nothing beyond MITM, and the relevant scale (H >= 60,
where nothing works) has no supervision. I would not fund this beyond a 2-3 day check that it
even matches exact search at H = 20-30.

### 3.4 Masked action diffusion with block repair

The distinguishing claim is "coordinated compensation": changing two or three regions jointly
yields a valid shorter word where changing one does not. Any such joint edit is a rewrite of
the union span with the intermediate segment pinned, and the exact ladder or beam bridge over
that span already covers it below their radii (and found nothing on beam paths). Above their
radii the model has random-word supervision only, and the brief lists the resulting failure
modes itself (restores the long word, sparse endpoint reward, relaxed samples do not
discretise). The Blocked-Gibbs paper is on Sudoku and graphs with known constraint energies;
by section 2 no such energy exists along a cube word. Component for direction 3 at most.

### 3.5 Orbit-consensus decoder

Split this in two. The **encoder** (component encoders for corners, wings and centres with
explicit cross-component messages and shared weights for compatible orbit structures) is the
cluster-factorised transformer in `IDEAS_DETAILED_2026-09-04.md` section 1; the brief and I
converge on it independently, and its "do not multiply separately learned component
posteriors" is the stabiliser trap (`kept 24/24/0`) in Bayesian language. Adopt it.

The **decoder** (shared categorical action variables; emit or edit a word) is direct
sequence emission at H ~ 100. Every strong Cayley solver, including the GFlowNet paper the
brief cites and AlphaCube-style policies, still beam-searches over the policy; our own
measurement that policy-only stepping drifts (subgoal search) is the same fact. The
identifiability point the brief makes (many words per observation) also means "decode the
scramble" is the wrong target and "emit any short word" is the pathfinding problem restated.

### 3.6 Learned backtracking controller

A recurrent controller with an explicit stack that decides when to try, restore or stop is a
sequential process. Our production search is a global top-B over `2^21 x 36` candidates per
step, sharded over 8 TPU cores, with the step selection-bound rather than decision-bound.
The throughput gap is several orders of magnitude, and Stream-of-Search's Countdown is a
thousand-state toy. The brief's own main failure mode ("a slow approximation to an ordinary
search routine") is the likely outcome. The one real gap it points at, allocation of width,
frames and restarts per pid, is a bandit problem we currently solve by hand with the
"yield graded by path length" rule; a non-neural allocator is worth an afternoon, a learned
one is not.

### 3.7 Coupled stochastic bridges

The brief postpones it; agree. The measured reason is the bidirectional MITM result above:
independent forward/backward distributions at half depth have no overlap in a graph that
mixes at 102, and a learned coupling that "reduces a learned discrepancy" without raising
exact connection rates is the failure the brief itself names.

## 4. The central disagreement

The brief argues from analogy (compilers, quantum circuits, program synthesis, QEC decoding,
constraint solving) that the paradigm must change. The record argues from measurement that
the paradigm is fine and one number is too low: the scorer's visibility relative to the
target set's mixing depth. On 444 (`V_vis ~ M`) the V-beam is within a few percent of
optimal; on 555 (`0.9`) it solves at 2.35x inflation where the paper's recipe reaches
1.28-1.39x on the same states; on 666 (`0.78`) it does not solve at all. Every proposal in
the brief either consumes an incumbent (A, 4) or moves the missing distance information into
a different object (B, 3, 5, 7) without adding a source for it. The brief says as much in its
last section: "shorter verified alternative constructions are the crucial source of
improvement supervision", and those come only from a search that already works.

The brief also frames prior negative results as "interpretations of measured configurations,
not theorems". That is fair for beam-width or training-budget nulls. It is not fair for the
exact-rewriting results, which are exhaustive proofs at their radius, nor for the matched
controls behind the MITM and subgoal rejections.

## 5. What to take from it

1. **The conjugacy lemma** (section 2): add to the rules as "no class function of a fixed
   word's residual measures progress along that word".
2. **Experiment A on 666**, one day, on KMC paths and finisher paths, with the brief's random
   and monotone controls. Not on the other puzzles.
3. **Direction 2 as a macro-phase optimiser** inside the hybrid P3' lane, if score insurance
   before 2026-11-20 is wanted; its decisive experiment is the right one.
4. **The e-graph idea as a data structure**, not a solver: organise the 24,288 three-cycles,
   768 bulk effects and commutation relations as an e-graph over macro words, which is a
   cleaner substrate for the existing SA and for experiment A than the current libraries.
5. **The evaluation discipline** (root-disjoint splits, count every retry, matched random
   controls, exact replay as the only success) is already ours; the brief restates it well.

## 6. What not to do

Fund directions 3, 4, 5's decoder, 6 or 7 as solvers. They compete with an exact method
where the exact method is complete, and with the V-beam where the V-beam fails for a reason
they do not address.
