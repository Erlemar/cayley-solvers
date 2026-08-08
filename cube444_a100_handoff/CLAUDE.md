# Project: CayleyPy 4x4x4 colour cube (A100 box)

Copy this to the repo root on the A100 as `CLAUDE.md`. It is the standing instruction set;
`README.md` in the handoff package is the orientation doc.

Competition: https://www.kaggle.com/competitions/cayley-py-444-cube — deadline 2026-09-22.

## Non-negotiable rules

1. **`num_classes = 6`, not `state_size`.** Every model constructor in `src/cayley/`
   defaults it to `state_size`, which is correct for a permutation puzzle and wrong here.
   Pass it explicitly at every construction site. It does not error — it silently builds a
   96-way embedding and trains a larger, worse model.

2. **This is a COLOUR cube, not a permutation puzzle.** A state is a colouring; it pins
   the group element only up to the `4!^6 = 191,102,976` stabiliser of solved. Therefore:
   **no `invert_state`**, no NISS, no inverse symmetry frames, no bidirectional search.
   24 rotation frames, not 48.

3. **Symmetry needs a recolour:** `sym(s,R) = color_map_R[s[rotation_R]]`. The
   slot-permutation-only form produces a legal-looking colouring and is wrong. Silent.

4. **Judge a beam run by the per-pid MIN against the floor, never the standalone mean.**
   Our 2^20 run scored 55,846 standalone against a 54,754 floor — worse — and was worth
   ~1,300 moves in the merge.

5. **Run the matched control before quoting any A/B delta.** Same pid set, machine, flags
   AND checkpoint. If the control does not exist, run it — it is cheaper than the wrong
   conclusion. **Byte-identical results across a config change mean the flag is not
   wired**, not that the feature is neutral.

6. **Verify before quoting or submitting.** `73_verify_submission.py` replays every path
   against `test.csv`; expect 1043/1043. Never trust a file's own reported total.

7. **Loss is not a proxy for beam quality.** Bench a real beam every ~50 epochs on a fixed
   pid set and stop on the bench. On this puzzle **SNR is a catastrophe detector, not a
   progress meter** — solve rate went 12/18 -> 18/18 with flat SNR.

8. **V scale-collapse is BENIGN here.** Bellman drives `V@d80` 43 -> 17 and the more
   collapsed checkpoint beams better. Gate on discrimination, never absolute scale. This
   is a real exception to the saturation rule that holds on our other puzzles.

9. **Exact anchors in every batch** (`n_anchor_v0=32`, `n_anchor_d1=4`) or the Bellman
   bootstrap settles at `V(solved) ~ 2`.

10. **ASCII-only in printed output; `encoding="utf-8"` on every `open()`.** The origin
    machine is Windows/cp932 and code should be able to travel back.

11. **Don't `torch.compile` beam inference without padding to a fixed batch size** —
    measured 5.8x slowdown from reshape recompiles. Training has fixed shapes and is fine.

12. **`pgrep -f` / `pkill -f` self-match** the checking command's own arguments. Bracket
    the pattern (`[f]oo`) — necessary but NOT sufficient, since it still fires if the name
    appears unbracketed elsewhere in the same command. The reliable form is two calls:
    read the pids, then `kill <explicit numeric pids>` in a command mentioning no pattern.

13. **Search merge sources by CONTENT**, not filename or location: check the header AND
    that the move alphabet matches this puzzle's generators.

14. **A timeout is not a proof.** Log hit / none / timeout separately, or "0 improvements"
    silently means "0 candidates searched".

## Conventions

* Layout: `src/cayley/`, `cube444/{src,scripts,configs,data,models,logs}`. Scripts resolve
  `PROJECT = Path(__file__).resolve().parents[2]`.
* Configs in `cube444/configs/*.yaml`, checkpoints in `cube444/models/<name>/`.
* Launch long runs detached (`nohup ... &` to a log); checkpoint every 25 epochs.
* Data files over ~500 MB (`solved_ball_d6.npz`, `outer_ball_d8.npz`) are rebuilt with
  `01_build_bfs.py`, never transferred.

## Where the value is

The scorer is the bottleneck on this puzzle by ~5 moves/pid — a transformer beats our
ResMLP-V beam on 87.8% of pids, uniformly across difficulty bands. **The open question is
whether that is the architecture or the sparse-Q objective**; see `04_TRAIN_SPARSE_Q.md`
for the 2x2 that settles it. Post-processing, the merge axis, HTM two-phase, and corner
PDBs are all measured and exhausted — do not spend time there.
