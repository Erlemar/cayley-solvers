# The approach — what the method is, and why each piece is there

Companion to `README.md` (how to run) and `RESULTS.md` (the numbers).

---

## 1. The puzzle, measured

| quantity | value |
|---|---|
| stickers / `num_classes` | **150**, identity central state -> **PICTURE cube** |
| generators | 30 = {f,r,d} x {0..4} x {+,-}, all order 4, **all ODD permutations** |
| \|G\| | **6.198e91** (Schreier-Sims, exact) |
| BFS levels 1..5 | 30 / 735 / 17,760 / 427,877 / 10,292,614 -> branching **24.03** |
| counting bound | **66.4** -> diameter ~70-76 |
| slot orbits | **7**: [6, 24, 24, 24, 24, 24, 24] |
| symmetry frames | **48**, all verified on the transport identity |
| exact anchors | d<=4 = 446,403 · d<=5 = 10,739,017, built in <10 s |
| convention | `new[i] = state[gen[i]]` |

**It is a PICTURE cube, not a colour cube.** A state pins the group element exactly, so
`invert_state` is legal and bidirectional tricks are available. This is the opposite of the
4x4x4 in this project family, where inversion is forbidden — do not carry rules across.

**Where the score lives.** 1000 of the 1035 pids are random walks whose shipped baseline is
the inverse scramble of length `pid-34`; only 35 are the Santa benchmark. Solving a deep
random-walk pid saves hundreds of moves. Solving a Santa pid saves **zero** — our ~177 loses
to its 93.6 baseline. Half a day was spent optimising the wrong 35 pids before this was
noticed. Run the queue **deepest-first**: saving per pid is `base_p - L`, so ordering
front-loads the gain and any stopping point is near-optimal.

### Derivation traps

1. The mirror family needs **per-axis** direction flips. A global flip finds only 12 frames
   — and every one verifies, so the count is the only tell.
2. Dedupe frames by **action**, not slot map. The slot centraliser has order 24, so 288
   distinct maps share 48 relabels and act identically.
3. `state_dtype=torch.int8` wraps classes 128..149 negative into `nn.Embedding`.
4. Hamming mixing does **not** give the diameter — it saturates by scramble length ~35,
   implying ~49, well below the rigorous 66.4.

## 2. The model

`ResMLPQ`, **24,757,807 params**:

    embedding        3,600     150 sticker classes -> 24 dims
    input_stack  3,689,472     3,600 -> 1,024
    res_blocks  21,032,960     10 blocks x 2 x (1024x1024)
    q_head          30,750     1,024 -> 30, one score per generator
    v_head           1,025     auxiliary

**Why a Q head and not a value head.** The Q head scores all 30 children from a single
forward pass on the parent, so a beam step costs `B` evaluations instead of `B x 30`. That
30x is why a 2^21 beam runs on one A100 where the reference paper needed 69 agents at 2^24.
Q also beats V(child) on top-1 at depth (0.246/0.182 vs 0.145/0.153) — **the cheap head is
also the better one.**

**Encoding.** One-hot at 150 classes is a 22,500-dim input: 44.1M params, 27.4 ms,
0.69 Mstate/s. The learned `embed24` is 24.8M, 20.3 ms, **1.34 Mstate/s** — roughly 2x
faster for no measured loss.

## 3. Training

**Pretrain.** 11,719 epochs x 256 steps x 2,048 fresh states = **3,000,064 updates /
6.14B fresh states**, lr 3e-4 cosine to 0, 23.1 h on one A100.

**Exactly three label sources. The restriction IS the recipe:**

1. **Sparse-Q random-walk middles.** `k ~ U[2,80]`, non-backtracking, one pivot `p` inside
   the walk; label two of the 30 columns, `Q(s,undo)=p-1` and `Q(s,next)=p+1`. `k_max=80`
   comes from the counting bound, not from an oracle.
2. **Exact BFS anchors**, d<=4, all 30 columns exact, **256 rows/step at weight 1.0**.
   Load-bearing: without exact anchors in every batch the Bellman bootstrap settles at
   `V(solved) ~ 2`. Do **not** scale the dose — bigger and deeper anchor batches are 0-for-3
   on the sibling puzzle.
3. **Symmetry expansion**, 4 random frames of the 48, via
   `Q(sym(s,k), relabel[k,a]) == Q(s,a)`. Frames must be drawn **randomly per sample**; a
   greedy coverage table hands the same frames to a given action pair every step, buying
   column coverage with zero state diversity.

Adding a fourth label source is the pattern that keeps losing: path labels and the
sorted-profile / permutation-invariant family are both measured REJECTED on the sibling
puzzle, and their pre-flight check passes for rejected variants, so passing it means nothing.

**Bellman refinement, warm-started.** Expand all 30 children, regress on
`1 + min_a' Q_target(child)`, clamped to 0 where the child is solved; target refreshed every
500 steps, lr 2e-5, batch 768, anchors still in every batch but at **weight 2.0**.

Two things about it that were only understood on 2026-08-22:

- **It must be warm-started.** From a near-scratch init the bootstrap inherits the target
  net's flatness at depth and there is nothing to propagate outward. Same code, opposite
  verdict, decided entirely by the init.
- **It must be SHORT.** ~2,000 steps, not 20,000. See `RESULTS.md` s2 — the deployed 20k
  checkpoint is the worst of six measured.

Note the Bellman phase regresses **all 30 columns** (`22_bellman.py:119`, target shape
`(b,30)`), unlike pretraining which supervises 2. Proposals premised on "28 unsupervised
columns" therefore do not apply to any refined checkpoint.

## 4. The beam, and why each flag

Deployed configuration:

    --beams 2097152 --max-steps 300 --bf16 --compile --history-depth 4
    --no-backtrack --endgame-depth 5 --frames 0,7,19,33,41,47
    --internal-batch-size 262144 --fallback ... --resume

| flag | why |
|---|---|
| `--beams 2097152` | 2^21. Wider is **worse** — see `RESULTS.md` s3 |
| `--max-steps 300` | a beam finds a length-L solution *at step L*, so the cap only truncates. Solutions run 118-250 |
| `--compile` | 1.35x throughput, byte-identical output, **requires** padded fixed chunks |
| `--history-depth 4` | -14% length. Saturates at 4: all 30 generators are ODD, so the graph is parity-bipartite and revisits occur only at even lags |
| `--no-backtrack` | neutral; `history>=1` already drops those, ~3% of the shortlist |
| `--endgame-depth 5` | widens the goal from 1 state to 10,739,017, with an exact tail spliced on arrival |
| `--frames a,b,c,...` | each frame a near-independent draw; solve rate `1-(1-q)^k`. Early exit makes extras nearly free |
| `--fallback` | records every *attempted* pid, so `--resume` cannot re-grind failures |

**Symmetry frames are the best cheap lever.** Per-frame success is ~33-50%, so 6 frames
gives 88-99%. Frames are independent *within* a round but decay *across* rounds
(0.84 -> 0.65 -> 0.50 measured), because the residue is enriched in genuinely hard pids.

## 5. What did NOT work — measured, do not re-spend

| lever | verdict |
|---|---|
| **wider beam** | 2^22 solved **fewer** pids than 2^21; 2^23 solved none. The scorer cannot rank the extra candidates it admits |
| **qv-consistency** | not a length lever: +8.0 moves on 5 matched pids, 1 win in 5, and it drops pids. Twice rejected |
| **more Bellman** | 20k is the worst of six checkpoints; 4k/6k/8k are indistinguishable from each other |
| **min-bias fix (double-Q)** | the bias is +0.138 moves. Nothing to fix |
| **capacity 24.8M vs 14.2M** | no measurable difference at matched budget (but see the caveat in `RESULTS.md` s5) |
| **commutation reduction on beam paths** | +0.00%. All its value is on raw scramble words |
| **shortcut splicing from d<=5** | -0.74% |
| **window compression** | a window longer than our solve length compresses to that length, so the optimal window is the whole path, i.e. the global solve |
| **path labels, sorted profiles** | rejected on the sibling puzzle; measure-zero slice, the beam lives off-path |

## 6. Instrumentation rules, learned the hard way

- **Gate on beam LENGTH and coverage, never on loss, `E[target]`, or top-1.** On the 6x6x6
  sibling, `E[target]` rose monotonically for 20k steps while the beam it serves went
  13 -> 7 -> 6 -> 1. In this project's own sub-sweep the loss fell 2.86 -> 0.61 while the
  beam got worse. And top-1 was bit-identical across qvc lambdas that moved the beam 30 moves.
- **Training is NON-DETERMINISTIC; the beam is not.** Two runs with identical init, seed and
  hyperparameters produce weights differing by up to 5.5e-5, which changes beam output by
  **~10.5 moves per pid** and flips solvability on ~7 of 24 pids. Any A/B smaller than that
  is unmeasurable from a single run. Given a fixed checkpoint the beam is fully deterministic.
- **Run the matched control in the same invocation**, same pids and flags. Byte-identical
  results across a config change mean the flag is **not wired**, not that it is neutral.
- **A hash hit is not proof of identity.** The endgame membership test once compared 64-bit
  hashes only; at 10.7M entries and ~1e11 lookups a false positive is expected, and one
  killed a 29-hour run. Compare the state; make the residual non-fatal.
- **Selection bias is real here.** Picking the best of N checkpoints on one small pid set
  has produced a "winner" that later came back tied. Confirm on a disjoint set.
- **Do not extrapolate a partial run ordered by difficulty.** A held-out list sorted deepest
  -first understated a total by 18% when scaled by item count.
