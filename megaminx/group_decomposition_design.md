# Group-theoretic decomposition — design notes

Goal: Solve megaminx in stages, each stage with smaller effective state space.

This is a multi-week structural project. The biggest potential ROI mentioned
(maybe -5K to -10K moves vs current 76,304 best), but uncertain feasibility on
megaminx given the larger group than Rubik's cube (20 corners + 30 edges
vs 8+12).

## Background: human megaminx solving

Skilled cubers use a stage-based decomposition (similar to Rubik's CFOP):

1. **Top star (5 edges)** — orient and place 5 edges of top face
2. **Top corners (5)** — place + orient 5 corners of top face
3. **Second layer (10 edges)** — between top and last layer
4. **Last layer cross (5 edges orientation, then permutation)**
5. **Last layer corners (orientation, then permutation)**

This gives ~60-100 moves for an experienced human, vs theoretical ~50 optimal.

Each stage's state space is much smaller than the full puzzle:

| Stage | Free dimensions | State space |
|---|---|---|
| Full megaminx | 20!/2 × 3^19 × 30!/2 × 2^29 | ~1.0 × 10^68 |
| Top star (5 edges) | 30P5 × 2^5 | ~5.5 × 10^7 |
| F2L corners + edges | varies | < 10^15 |
| Last layer | 5! × 5! × 3^4 × 2^4 | ~5.5 × 10^7 |

Top star and last layer are each ~5×10^7 states — small enough for exact BFS!

## Approach options

### Option A: Two-stage (F2L + LL)

1. **Stage 1: Solve "first 2 layers"** = top star + top corners + 2nd layer.
   Treat the full state but only require top 2 layers correct. Use neural V
   trained on this objective.
2. **Stage 2: Solve last layer** from end-of-stage-1 state. State space small
   enough for exact PDB.

Pros: simpler than 5-stage; PDB on last layer is feasible.
Cons: Stage 1 is still hard (state space ~10^40-50).

### Option B: Five-stage CFOP-style

Each stage: small state space. Most can use exact BFS / PDB.

Pros: each stage truly small, exact heuristic possible.
Cons: chain of 5 searches, error compounds; coordinator tricky.

### Option C: Hybrid (top + bottom symmetric)

Use top-face = bottom-face symmetry. Treat top half + bottom half as parallel
sub-problems. Solve top half, mirror, apply to bottom.

Pros: parallelism.
Cons: megaminx isn't quite top-bottom symmetric (12 faces, 5 fold per face).

## Recommended approach: A (two-stage F2L + LL)

Cleanest and biggest leverage.

### Phase 1: Define + label "F2L-correct"

A state is "F2L-correct" when:
- Top star (5 edges) in their home positions with correct orientation
- Top corners (5) in home positions with correct orientation
- Second-layer edges (10 edges between top and middle layer) in home positions

Total stickers F2L-correct: 5×2 (top edges) + 5×3 (top corners) + 10×2 (mid edges) = 35 stickers
of 120. Bottom half (60 stickers) and last-layer corners-of-top-face (15 stickers... wait)

Hmm, megaminx has 12 faces. F2L = top face fully solved + 5 adjacent faces' top halves
solved. Need to look up the exact sticker count.

**ACTION ITEM**: enumerate the F2L sticker set for our `puzzle_info.json` definition.

### Phase 2: Train V_F2L

Train a V model to predict "moves to F2L-correct" from any state.
- Random walks: start from F2L-correct state (any of many — 12 face orientations), walk
  k_max=50 steps.
- Label: walk depth.
- Loss: MSE.
- Architecture: same as m_curr_v3 (6M params). Or smaller.

### Phase 3: Build LL PDB

Last layer state space: 5 corners × 5! permutations × 3^4 orientations + 5 edges × 5! × 2^4 = 5.5×10^7.

Build BFS table from solved state, exhaustive expansion until all states reached.
Memory: 5.5×10^7 × 1 byte (depth) = 55 MB. Easily fits.

### Phase 4: Solver pipeline

Given pid initial state:
1. Run F2L beam search using V_F2L. Output: state X (F2L-correct), path π1.
2. Run LL solver (PDB lookup → IDA* or just direct optimal path) from X. Output: π2.
3. Concatenate: full_path = π1 + π2.

### Memory + compute estimates

- V_F2L training: ~6h on 4090 (similar to m_curr_v3).
- LL PDB build: ~10-30 min on 1 thread (depth ≤ 20 on 5.5e7 states).
- Inference: per-pid wall ~similar to current beam (since V_F2L is same arch).

### Risks

1. **F2L definition**: 12 face symmetries → which one is "the" F2L? Must canonicalize.
2. **Stage 1 may not be much easier**: even with V_F2L, exploring "F2L-correct" state space from random initial might be hard. Not all states are equidistant from F2L-correct.
3. **Stage 2 may not be tight**: PDB gives exact distance, but the F2L-correct state we reach might not be optimal-from-here.

Honest expected outcome: 50% chance of meaningful improvement (-1K to -5K moves), 50% chance
of cluster-class result (no improvement). The CFOP recipe at HUMAN scale gives -10% over
greedy (~7 moves saved per pid × 1001 = ~7K moves) but ML execution may not match that.

## Implementation plan (estimated 2-3 weeks)

### Week 1: F2L definition + training data

- [ ] Enumerate F2L sticker set in `puzzle_info.json`
- [ ] Write F2L-correct check function
- [ ] Generate F2L-walks training data (BFS from F2L-correct goal states)
- [ ] Validate: can we identify a state's distance to F2L-correct?

### Week 2: V_F2L training + LL PDB

- [ ] Train V_F2L (6h)
- [ ] Build LL PDB (30 min)
- [ ] Validate V_F2L on stratified hard pids (compare to V_full)

### Week 3: Solver pipeline + integration

- [ ] Implement two-stage solver: F2L beam + LL PDB lookup
- [ ] Test on subset of pids
- [ ] Compare path lengths to current best beam

### Stretch: 5-stage CFOP

If 2-stage works, try full 5-stage with PDB at each stage for tighter heuristics.

## Status

**2026-05-09**: Design drafted. Awaiting m_curr_v4 hardmine result before
committing engineering time. If hardmine breaks cluster, focus on training-side
gains; if not, commit to this multi-week project.
