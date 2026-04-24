 # Data generation & feature engineering ideas

Focused on changing **what the model sees during training** rather than the model/search.
All of these are orthogonal to Bellman refinement — they can stack.

**Current pipeline**: random walk from solved → emit `(state_at_step_k, k)` as
`(input, target)`; target is walk depth (upper bound on true distance); input is raw
72-element permutation, one-hot or embedded.

---

## 1. Mix exact BFS labels for near-solved states ★★★

**The idea**: we already have `data/bfs_table_d5.pkl` — 790K states with **exact**
shortest distances (0 to 5). For these states, walk depth overestimates worst. Drop in
exact labels.

**Implementation**:
```python
# src/cayley/data.py: new function
def sample_bfs_labeled(bfs_table, n_samples, device):
    # Randomly pick n_samples entries from the table.
    # Return (states, distances) where distances = true shortest from solved.
```
Then in `train_one_epoch`: 80% from random walks (existing), 20% from BFS table.

**Effort**: ~1 hour. **Expected effect**: measurable improvement on near-solved
predictions, which should in turn help beam search find end-of-path faster.

**Risk**: walks already oversample near-solved states slightly; the real gain is
*unbiased* labels near the goal, not just more data there.

---

## 2. Piece decomposition features ★★★

**The idea**: picture cube's 72 facelets decompose into 8 corners × 3 facelets, 12 edges
× 2 facelets, 6 centers × 4 orientations. This is already known — `experiment_log.md`
in `kaggle_research/cayleypy-ihes-cube/` records:

```python
CORNERS = [(0,38,48), (2,26,36), (9,12,50), (11,14,24),
           (21,59,60), (23,33,62), (35,45,71), (47,57,69)]
EDGES   = [(1,37), (3,49), (8,25), (10,13), (15,56),
           (20,27), (22,61), (32,39), (34,68), (44,51), (46,70), (58,63)]
CENTERS = {face*12 + offset: orientation
           for face in range(6) for offset, orientation in zip([4,5,7,6], [0,1,2,3])}
```

**The model currently has to learn cube geometry from scratch** — it must discover that
facelets 0 and 38 always move together. Piece decomposition bakes this in.

**Implementation sketch** (alternative input encoder):
```python
# For each state s:
#   corner_id[8]      — which of 8 corner pieces is at each corner slot (int 0-7)
#   corner_ori[8]     — orientation of that corner (int 0-2, via sticker order)
#   edge_id[12]       — which of 12 edge pieces is at each edge slot (int 0-11)
#   edge_ori[12]      — edge flip (int 0-1)
#   center_ori[6]     — center orientation (int 0-3)
# Total: 8+8+12+12+6 = 46 small-vocabulary int features vs 72 large-vocabulary features.
#
# Encode each with its own small embedding (e.g. 8 embeddings of (8,16), (3,16), etc.).
# Concat and feed through the trunk. ~ same parameter count, much stronger structural prior.
```

**Effort**: ~3-4 hours. Need to derive the state-to-pieces mapping and test carefully.

**Expected effect**: could be the change that breaks our plateau. `twsearch` uses this
decomposition to reduce state space 10^103 → 10^22; a NN with the same prior should
generalize much faster. Unknown if it helps or hurts loss floor, but likely helps solve
rate at same loss.

**Risk**: derivation is error-prone; need tests against random scrambles to verify each
piece's position and orientation are computed correctly. If a piece gets confused the
model will train on garbage labels.

---

## 3. Kociemba-path reversal for training data ★★

**The idea**: `data/kociemba_fallback.csv` has 1003 puzzles × ~38 avg moves = ~40K
real-distribution states. For each puzzle, replay the solution from solved to generate
intermediate states `s_0=solved, s_1, s_2, …, s_{k-1}` where `s_{k-1} = scrambled`.
Label `s_i` with `k - i` (distance along this path to solved — an upper bound).

**Why it matters**: random-walk training states are uniformly sampled through generator
space. Competition states aren't — they're sampled from graded difficulty buckets. This
gives the model the actual test distribution.

**Implementation**:
```python
# scripts/build_kociemba_walks.py
# For each (pid, path) in data/kociemba_fallback.csv:
#   state = scrambled; emit (state, len(path))
#   reversed_path = [invert(m) for m in path[::-1]]
#   for step, move in enumerate(reversed_path):  # 0..len(path)-1
#       state = apply(state, move)
#       emit (state, len(path) - 1 - step)
# Shuffle, save as data/kociemba_walks.pkl
```

Then mix into training like #1 (say 10% of each epoch).

**Effort**: ~1 hour. **Expected effect**: moderate; these labels are upper bounds too,
but at least they're from realistic states.

---

## 4. 1/k-weighted sampling (DeepCubeA curriculum) ★

**The idea**: currently walks emit one state per step, uniformly mixed across k=1..K_max.
Weight each sample by 1/k when computing the loss, so near-goal states dominate early
epochs:
```python
loss = (weights * elementwise_loss).mean()   # weights = 1 / k
```
Already implemented as `training.cfg.curriculum = True` (enabled in big_v1 which
regressed — but big_v1 had other changes too; never tested in isolation on small arch).

**Effort**: 2 minutes (already coded). **Expected effect**: small; mostly a schedule
change. Probably +1-2% at best.

**Risk**: can make the model undertrained at high k. Test on fast recipe first before
committing to a full run.

---

## 5. 24× rotational symmetry augmentation ★★

**Covered in `IDEAS.md` #3.** Listed here for completeness because it's a data-side
change. The cube's 24 rotational symmetries are orbit-preserving permutations of
facelets; each training state has 24 equivalents with the SAME distance label. Stacks
cleanly with every other item here.

Derivation complexity is the blocker: whole-cube rotations are NOT in the generator set
and center orientations rotate non-trivially. ~1 day of work to derive correctly.

---

## What NOT to do

- **Parity / invariant features** (corner orientation sum mod 3, edge flip sum mod 2):
  preserved by all generators, so constant for all reachable states. Zero information.
- **Hamming distance as an input feature**: the model with 72 facelet inputs trivially
  learns this internally. Adding it explicitly is redundant and takes a slot in the
  input representation.
- **Reversing random walks** (instead of forward walks): same sampling distribution,
  zero gain.
- **Using our own previous submissions as pseudo-labels**: circular, will reinforce the
  model's biases.

---

## Implementation order

Recommended sequence if we tackle multiple:

1. **#1 (BFS exact labels)** — 1h, gives immediate signal improvement; low risk.
2. **#4 (1/k curriculum)** — 5m, run an ablation to see if it helps on small arch.
3. **#2 (piece decomposition)** — 3-4h, biggest structural change; schedule when we have
   time to do careful unit tests against Kociemba's piece tables.
4. **#3 (Kociemba-path reversal)** — 1h, small but complementary.
5. **#5 (24× symmetry)** — full day; do only if #1-#4 still leave us >1K short of target.

Each of #1, #2, #3 could be combined with **Bellman refinement** (already coded in
`src/cayley/bellman.py`) for compound effect.
