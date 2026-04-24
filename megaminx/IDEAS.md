# Megaminx experiment backlog

Ideas ranked by **expected payoff per hour of work**. Check off as done and point to the
EXPERIMENTS.md entry.

**Current best (submitted)**: none yet. Local best (not submitted, matches pp_fallback): 457,802.
**Target (Rokicki, #3)**: 93,606. **Target (leader Kuznetsov)**: 79,971.

---

## HIGHEST priority untried

1. [ ] **Bigger model (hidden [2048, 512], 2 res blocks) at k_max=80** (`m03`). Goal: halve heuristic noise so beam search doesn't drift on medium scrambles. Run on Kaggle (P100) in parallel with local solves; expected ~1–2h for 4000 ep. **EV: unlock buckets 1-5 in next solve** (currently 0/10 solved). First leaderboard submission likely lands here.

2. [ ] **Q-function head** (`m04`). Train output_dim=24, loss = MSE of Q(s,a) vs V_teacher(apply(s,a)) distilled from m02 or m03. Beam expansion does ONE forward per parent instead of 24. 10–20× faster per step, and more discriminative because the 24 outputs share a common embedding. IHES project already has this code (`scripts/06_train_qfunction.py`). **EV: similar solve rate at much bigger beam within same time budget.**

3. [ ] **Wider beam run** (131k or 262k) on existing m02 checkpoint, targeting unsolved medium-hard puzzles from m02's full solve. Re-runs only the hard-end (pid ≥ 200) so cost is bounded. Needs the GCP L4 VM (24 GB VRAM) because 4090 Laptop 16 GB OOMs at beam 262k with state_size=120. **EV: few hundred to few thousand moves saved without retraining.**

4. [ ] **NISS (inverse-scramble search)**. Apply `invert_state` to the scrambled puzzle, solve the inverse, then `invert_path` the output. Directional anisotropy: if forward solve runs out of beam, inverse side may find shortcut. IHES saw -432 moves from NISS-on-ensemble. `megaminx.Megaminx` already has `invert_state` / `invert_path`. **EV: ~1–3% move reduction, trivial to implement.**

5. [ ] **Multi-seed ensemble** of m03. Train 3–5 big models with different seeds on Kaggle in parallel, solve each, keep the min per puzzle. IHES saw 3–5% gain per ensemble step. **EV: high, but gated by m03 succeeding first.**

## MEDIUM priority untried

6. [ ] **Adaptive beam per puzzle**. First pass beam 16k max_steps 60 (catches easy); for unsolved, retry beam 65k max_steps 150. Saves GPU time on easy puzzles while giving hard puzzles the compute they need. Pure code change, no retraining.

7. [ ] **BFS-d5 post-processing table for Megaminx** (≈8M states, ~500 MB). Shortcut window replacement like IHES's bfs_table_d5.pkl. d4 (331k states) as cheap first step to gauge whether window shortcuts exist at all in Megaminx paths. **EV: unknown for Megaminx — IHES gained 30-80 moves/submission; order-5 face structure may leave less to cancel.**

8. [ ] **Icosahedral symmetry augmentation**. Dodecahedron has 60 rotational symmetries. Augment training data by random rotation. Requires deriving the 60 whole-puzzle rotations acting on the 120-sticker state — non-trivial. IHES version (24-symmetry) was proposed but not done. **EV: 5–10% loss floor reduction, but high implementation cost.**

9. [ ] **Longer k_max (120 or uniform over 1..150)** on m03 arch. The full test set has scrambles up to 1000 moves — current k_max=80 still OOD for bucket 8+. Trade: wider range = harder to fit, noisier predictions at short range too.

10. [ ] **Bellman refinement** of m03 checkpoint. `y = 1 + min_a V(apply(s,a))`. IHES saw -42 moves from this step alone.

## LOW priority / uncertain

11. [ ] **Weighted A* (`f = w·g + h`)** in khoruzhii searcher. Currently pure h-greedy. Adding depth cost might help on drifty paths.

12. [ ] **Community submissions** for Megaminx (e.g., `alexandervc/cayleypy-submissions` if it has megaminx). Min-merge would give a strong floor but per IHES policy (user 2026-04-20) we don't submit community-merged results until our own work beats them — so this is internal-only signal.

13. [ ] **Transformer model**. Chat-reported beat MLP on IHES, but weeks of work and uncertain payoff on a different geometry.

14. [ ] **Kociemba-equivalent two-phase solver for Megaminx**. Doesn't exist off the shelf. Would be massive effort (weeks to months); deprioritized until ML approaches plateau far from leader.

## DONE

- [x] **Port puzzle class, training loop, beam search** (m01, 2026-04-24). State/generator loader via duck-typed Megaminx class; shared cayley.* modules accept it unchanged.
- [x] **Same-face run reduction post-processing** (2026-04-24). X^5→id, X^4→-X, X^3→-X·-X. Sample post-processed 500,572 → 457,810 (-8.54%). See EXPERIMENTS.md.
- [x] **Stratified sampling (k per 100-bucket) for smoke tests** (2026-04-24). Avoids overfitting evaluation to easy prefix.

