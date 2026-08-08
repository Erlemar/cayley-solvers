# 06 — Gotchas

Each of these cost real time on the origin machine. They are ordered by how likely they
are to bite you and by how silent the failure is.

## 1. `num_classes = 6`, not `state_size` — SILENT

Every model constructor in the shared library defaults `num_classes = state_size`, which
is right for a permutation puzzle and wrong here. **Pass it explicitly at every
construction site.** It does not error; it builds a 96-way embedding instead of a 6-way
one and trains a larger, worse model that looks fine in the logs.

This is the single most common way to waste a run on this puzzle.

## 2. The colour-cube symmetry needs a RECOLOUR — SILENT

```python
sym(s, R) = color_map_R[ s[rotation_R] ]     # correct
s[rotation_R]                                # WRONG, and looks legal
```

The wrong form produces a valid-looking colouring and silently degrades every
symmetry-frame beam. `cube444/scripts/test_symmetry.py` checks this; run it before any
training. (`00_verify.py` / `73_verify_submission.py` are submission verifiers, not this.)

## 3. Judge a beam run by the per-pid MIN, not the standalone mean

Our 2^20 run scored 55,846 standalone against a 54,754 floor — worse — and was worth
~1,300 moves in the merge. See `05_BEAM_SEARCH.md`. Never conclude from a standalone mean.

## 4. Run the matched control before quoting any A/B delta

A total is only meaningful against the same pid set, machine, flags **and** checkpoint.
Any other difference gets silently attributed to the variable under test. On tetraminx a
full day of results had to be retracted for exactly this.

Corollaries, all learned the hard way:
* If the control does not exist, **run it** — it is cheaper than the wrong conclusion.
* **Byte-identical results across a config change mean the flag is not wired**, not that
  the feature is neutral. We shipped a dead `--history-depth` for weeks this way.
* Never generalise an interaction from one measured pair to an unmeasured one.

## 5. Loss is not a proxy for beam quality

A megaminx model trained 184 epochs had lower loss than its 50-epoch checkpoint and beamed
strictly worse (solve rate collapsed to 0% on a whole pid band). **Bench against a real
beam every ~50 epochs** and stop on the bench, not the loss.

On this puzzle specifically: **SNR is a catastrophe detector, not a progress meter.** Solve
rate went 12/18 -> 18/18 across 100 epochs with completely flat SNR. Do not early-stop on
it.

## 6. V scale-collapse is BENIGN here — do not gate on it

Bellman drives `V@d80` from 43 -> 17. On megaminx and the IHES cube that signature meant a
dead model. **Here the more-collapsed checkpoint beams better.** Gate on discrimination,
never on absolute scale. This is a genuine cross-puzzle exception — do not import the
saturation gate from the other projects.

## 7. Don't `torch.compile` beam inference without padding to a fixed batch

Naive `model(candidates)` recompiles on every shape change — measured **5.8x slowdown**.
Either pad every forward to a fixed `internal_batch_size` and pre-warm the compiled graph
at that shape, or leave compile off for inference. Training has fixed shapes, so
compile-for-training is always fine.

## 8. Exact anchors in every batch, or V(solved) drifts to ~2

`n_anchor_v0=32`, `n_anchor_d1=4`. Without them the Bellman bootstrap never learns that
solved is zero and every beam steers by a broken origin.

## 9. Process management (if you script around long runs)

* `pgrep -f <pat>` / `pkill -f <pat>` **self-match** the checking command's own arg list,
  so a bare `pgrep -f foo` always finds itself. Bracket the pattern (`[f]oo`) — but note
  that is necessary, **not sufficient**: it still fires if the name appears unbracketed
  anywhere else in the same command (a file path, a launch line). The only reliable form
  is **two calls**: read the pids, then `kill <explicit numeric pids>` in a separate
  command that mentions no pattern.
* Never put a kill and a launch of the same binary in one command.

## 10. ASCII-only output, `encoding="utf-8"` on every `open()`

The origin machine is Windows/cp932. A single non-ASCII glyph in a log line crashed a
training run mid-epoch. Costs nothing to keep, and means code can travel back.

## 11. A timeout is not a proof

If a search is bounded by wall clock, log **hit / none / timeout separately**. Otherwise
"0 improvements found" silently means "0 candidates actually searched", and you will
record a negative result that was never tested.

## 12. Search by CONTENT when merging, not by filename or location

Check the header **and** that the move alphabet matches this puzzle's generators. Files
from a sibling puzzle pass the header test and fail the second; on tetraminx three such
files sitting in an unrelated folder were worth -22 moves once found.
