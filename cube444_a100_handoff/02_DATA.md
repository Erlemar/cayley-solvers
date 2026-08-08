# 02 — Data, and the label-conflict fix

Everything is **self-generated + exact**. Zero community or human solutions ever enter
training. (The community CSV touched our pipeline exactly once, at the final min-merge,
never as a label.) Keep it that way — it is what makes results interpretable.

Two label streams feed the trainer, and **they contradict each other where they overlap**.
That conflict is the subject of the second half of this document and is the one novel
change you are being asked to make.

## 1. The exact BFS ball (anchors + the oracle)

```bash
python cube444/scripts/01_build_bfs.py \
  --puzzle-info cube444/data/puzzle_info.json \
  --out cube444/data/bfs_anchors.pt \
  --max-exact 5 --d6-sample 20000000 --chunk 400000
```

Writes `{"states": uint8 (N,96), "distances": int8 (N,)}`. Levels are exact to `d=5` and
sampled at `d=6` (the full d=6 shell is ~25.3M states; sampling is fine for anchors, but
see below — for the *oracle* you want it complete or hash-keyed).

Level sizes to assert against (branching 19.18): `1, 24, 552, 12144, ...`. If your d1 is
not exactly 24 or d2 not 552, the generator convention is wrong — stop and re-check
`test_symmetry.py`.

**Build the hash-keyed ball too** if you intend to use the oracle fix below at scale.
Store sorted Zobrist hashes + depths rather than raw states; a d<=6 table is ~25.3M
entries, 8 bytes each. Reuse the pattern in `src/cayley/bfs_table.py`.

## 2. Random walks (the bulk of the data)

Non-backtracking random walks from solved, `k_max = 60`, label = walk index. This is the
standard CayleyPy generator and is already wired into `src/cayley/training.py`.

`k_max=60` (not 80 as on megaminx) because the diameter here is ~37-40.

## 3. THE PROBLEM: walk labels are wrong, and on a colour cube they are wronger

### 3a. The general defect (measured on tetraminx, 200k pivots)

A non-backtracking walk forbids only the immediate inverse, so **a walk is not geodesic**.
With an order-3 generator, `g,g` has walk index 2 at true distance 1. Measured where walk
data overlaps the exact table:

* **37.1%** of walk pivots are inside the exact ball (so directly checkable)
* of those, **25.9%** have `walk index != true distance` — always `p >= d`, never below
* the labelled undo/next children are wrong in **21.9% / 31.0%** of checkable cases
* but the **ordering** is right **92.75%** of the time (tie 6.14%, inverted 1.11%)
* mean true gap `d(next) - d(undo)` = **1.717**, while the label always asserts **2**

So ~7.25% of labelled pairs assert a gap that does not exist.

### 3b. Why it is WORSE on this cube — the part specific to 444

On a permutation puzzle a walk of length `k` produces a group element `g` with
`d(g) <= k`, and the label error is just non-geodesic slack.

**Here the state is a colouring, and the colouring only pins `g` up to the `4!^6 =
191,102,976` stabiliser.** The true distance of the colouring is

```
d(colouring) = min over all g' in the coset g.Stab  of  |g'|
```

A walk hands you one specific `g` and labels the state `k`. The colouring's actual
distance is a minimum over ~1.9e8 group elements, and **can be far below `|g|`**. So on
this puzzle walk labels over-estimate for two independent reasons stacked on top of each
other — non-geodesic slack *and* coset collapse — and the second one has no analogue in
any puzzle we have measured this on.

This also means **duplicate states carrying different labels are common**: two walks of
different lengths can reach the same colouring, and both labels get trained on.

### 3c. Why sparse-Q survives it and MSE-on-depth does not

The sparse-Q objective is **relative** — it labels only `Q(s,undo) = p-1` and
`Q(s,next) = p+1`, so a uniform offset shifts both equally and the ranking is preserved.
That is the deeper reason sparse-Q beats walk-depth MSE, and it is why the fix below
matters *most* for the V/MSE path and *least* for the Q path. Do not skip it for the Q
path though: the ~7.25% of pairs asserting a nonexistent gap are inside the primary
objective there.

## 4. THE FIX — dedup + exact-label override

Two passes. Both are cheap because the exact ball is already resident for the anchors.

### Pass A — collapse duplicates, keep the minimum label

A colouring reached at walk index 12 and again at 30 is one training example, not two, and
its best available label is 12. Hash-dedup each generated batch (or the whole epoch's
buffer) and keep `min(label)`.

```python
def dedup_min_label(states: np.ndarray, labels: np.ndarray, ztab: np.ndarray):
    """states (N,96) uint8, labels (N,) int -> unique states with the MIN label each.

    Two walks reaching the same colouring at different lengths are one example. Keeping
    both trains the net to output two different numbers for one input; keeping the max
    trains it to the worse of two upper bounds.
    """
    h = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[1]):
        h ^= ztab[i][states[:, i]]
    order = np.lexsort((labels, h))          # group by hash, min label first
    h_s, s_s, l_s = h[order], states[order], labels[order]
    keep = np.empty(h_s.size, dtype=bool)
    keep[0] = True
    np.not_equal(h_s[1:], h_s[:-1], out=keep[1:])
    return s_s[keep], l_s[keep]
```

### Pass B — replace the label with the exact distance wherever the table knows it

```python
def exact_override(states, labels, table_hashes, table_depths, ztab):
    """Where a state is inside the exact ball, its walk label is a strict upper bound and
    the table has the truth. Returns (labels, n_fixed, n_in_table).

    Direction is one-way by construction: p >= d always, so this can only lower a label.
    Assert that -- if you ever see the table claim a LARGER distance than the walk, your
    hashing or your generator convention is wrong, not the table.
    """
    h = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[1]):
        h ^= ztab[i][states[:, i]]
    pos = np.searchsorted(table_hashes, h)
    np.clip(pos, 0, table_hashes.size - 1, out=pos)
    hit = table_hashes[pos] == h
    exact = table_depths[pos].astype(labels.dtype)
    assert not np.any(exact[hit] > labels[hit]), \
        "table claims a LONGER distance than the walk -- hashing/convention bug"
    n_fixed = int(np.count_nonzero(hit & (exact < labels)))
    labels = np.where(hit, exact, labels)
    return labels, n_fixed, int(hit.sum())
```

For the **sparse-Q** path the same override applies to the pivot and both labelled
children (3 lookups per row):

```python
# pivot p, undo child u, next child n
# label asserts  Q(s,undo) = p-1,  Q(s,next) = p+1
# if all three are in the table, use  d(u), d(n)  directly and DROP the row if
# d(n) - d(u) <= 0  (the label asserts an ordering the graph does not have)
```

**Alternative, simpler, nearly as good**: just *drop* in-table pivots from the walk
stream entirely. The anchors already cover that depth densely, so you lose no coverage and
you remove the conflict at the source. Prefer this if you want a one-line change.

### What to log

Print these every epoch — they are how you know the fix is live and how you report it:

```
walk rows generated      N
after dedup              N'        (dup rate = 1 - N'/N)
in exact table           M         (expect ~30-40% early, less as k_max is approached)
labels lowered           F         (expect ~25% of M)
rows dropped (bad order) D         (sparse-Q only)
```

**This is an objective change, so it only takes effect on a fresh run.** Do not warm-start
a fixed-label run from an unfixed checkpoint and attribute the difference to the fix.

## 5. The control you must run

Per the matched-control rule: **train one model with the fix and one without, changing
nothing else** — same seed, same epochs, same everything. Quote the delta only against
that control. If you cannot afford both, run the control at reduced epochs rather than
skipping it; an unmatched comparison here is worse than no comparison, because the fix is
plausible enough that you will believe a confounded result.
