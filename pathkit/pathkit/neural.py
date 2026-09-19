"""Neural bridge compression: re-solve a path WINDOW with the full solver, not a table.

MECHANISM. For a window [i, j) of a path, the residual sub-problem is the state

    X = inv(s_j)[s_i]

because a word v that solves X to the identity has exactly the composite permutation that
takes s_i to s_j. Hand X to any solver you already own (beam search, a Q-guided search,
whatever), and if it returns a word shorter than j - i, splice it in. Validity is
preserved by construction and every splice is replayed anyway, so this pass can never
regress -- the worst case is that it spends compute and returns the input.

This is the only method here that can look further than an exact table or ball can reach,
because the residual is solved by the same machinery that solved the puzzle in the first
place. That is also its weakness.

WHERE IT PAYS, AND THE FAILURE MODE THAT DEFINES THE BOUNDARY. Measured on megaminx: on
our own loose paths it was worth 128 moves at a ~3.5 pct win rate over ~2,200 attempts.
On already min-merged community paths it collapsed to under 1 pct, and on the hardest pids
to exactly zero at every configuration tried, including a TPU beam of width 1M with
symmetry ensembling. The cause is not the solver, it is the SCORER used to pick windows:
a distance model saturates at roughly the puzzle diameter, so for a deep residual it
predicts the same value for everything and the "predicted saving" that selects windows
becomes pure false positives.

So the rule is: measure the saturation horizon FIRST (`estimate_saturation`), then only
select windows whose residual could plausibly sit inside it. `select_windows` enforces
that rather than trusting the score. On community-floor mid pids a cheap single-pass beam
captured everything a 1M-width ensemble did -- when this method works at all, it works
cheaply; when it needs a bigger hammer, the hammer does not help.
"""
from __future__ import annotations

import numpy as np


def residual_state(puzzle, s_i, s_j):
    """X with the property: any word solving X to the target also takes s_i to s_j.

    Colour puzzles have no inverse state, so there is no residual to hand a
    solved-state-seeking solver; the analogous move there is to re-solve a SUFFIX, which
    is the one sub-problem whose target really is the solved state (see suffix.py).
    """
    assert puzzle.is_permutation, (
        "residual extraction needs distinct labels; on a colour puzzle use suffix.py "
        "or the state-space ball sweep in window.py")
    inv_sj = np.empty(puzzle.state_size, dtype=np.int64)
    inv_sj[s_j] = np.arange(puzzle.state_size)
    return inv_sj[s_i]


def estimate_saturation(puzzle, scorer, rng=None, depths=(5, 10, 20, 40, 80), n=64):
    """Measure where the scorer stops growing with true distance.

    Returns {depth: mean score} plus `horizon`, the largest depth at which the score is
    still rising meaningfully. Beyond the horizon the scorer cannot rank residuals, and a
    selection rule built on it produces false positives rather than candidates.

    This doubles as the health check for a distance model in general: a model whose score
    keeps climbing past the puzzle diameter is predicting WALK depth, not distance, and
    will fail beam search even when its shallow calibration looks perfect.
    """
    rng = rng or np.random.default_rng(0)
    means = {}
    for d in depths:
        vals = []
        for _ in range(n):
            s, _w = puzzle.random_scramble(d, rng)
            vals.append(float(scorer(s[None, :])[0]))
        means[d] = float(np.mean(vals))
    horizon = depths[0]
    ds = sorted(means)
    for a, b in zip(ds, ds[1:]):
        if means[b] - means[a] >= 0.5 * (b - a) * 0.25:      # still climbing meaningfully
            horizon = b
    return {"means": means, "horizon": horizon}


def select_windows(puzzle, s0, word, scorer, grains=(8, 12, 16, 20, 25, 30),
                   top_k=16, horizon=None, min_predicted_save=1.0):
    """Rank candidate windows by predicted saving, refusing the unscoreable ones.

    predicted saving = (j - i) - scorer(residual)

    The `horizon` guard is the load-bearing part. Without it, every deep window scores as
    a huge saving (the scorer has saturated and returns its ceiling for all of them) and
    the run spends its whole budget on candidates that cannot win.
    """
    states = puzzle.path_states(s0, word)
    L = len(word)
    cands = []
    for g in grains:
        if horizon is not None and g > horizon:
            continue                       # residual is past what the scorer can rank
        for i in range(0, L - g + 1):
            j = i + g
            cands.append((i, j))
    if not cands:
        return []
    X = np.stack([residual_state(puzzle, states[i], states[j]) for i, j in cands])
    pred = np.asarray(scorer(X), dtype=float)
    save = np.array([j - i for i, j in cands], dtype=float) - pred
    order = np.argsort(-save)
    out = []
    for k in order:
        if save[k] < min_predicted_save:
            break
        i, j = cands[k]
        out.append({"i": i, "j": j, "predicted_save": float(save[k]),
                    "predicted_depth": float(pred[k])})
        if len(out) >= top_k:
            break
    return out


def compress_path(puzzle, s0, word, solver, scorer=None, grains=(8, 12, 16, 20, 25, 30),
                  top_k=16, horizon=None, rounds=2, stats=None, log=None):
    """Select windows, re-solve each residual, splice every verified win. Never regresses.

    `solver(state) -> word | None` is whatever you already use to solve a scramble. Only
    strictly shorter, replay-verified replacements are accepted.
    """
    cur = list(word)
    for _r in range(rounds):
        states = puzzle.path_states(s0, cur)
        if scorer is not None:
            picks = select_windows(puzzle, s0, cur, scorer, grains, top_k, horizon)
        else:                                   # no scorer: sweep the grains directly
            picks = [{"i": i, "j": i + g, "predicted_save": None}
                     for g in grains for i in range(0, len(cur) - g + 1)][:top_k]
        gained = 0
        used = np.zeros(len(cur) + 1, dtype=bool)
        splices = []
        for p in picks:
            i, j = p["i"], p["j"]
            if used[i:j].any():
                continue
            X = residual_state(puzzle, states[i], states[j])
            _bump(stats, "attempts")
            v = solver(X)
            if not v or len(v) >= j - i:
                continue
            if not np.array_equal(puzzle.apply_word(states[i], v), states[j]):
                _bump(stats, "rejected_invalid")
                continue
            splices.append((i, j, list(v)))
            used[i:j] = True
            gained += (j - i) - len(v)
        if not splices:
            break
        splices.sort()
        out, prev = [], 0
        for i, j, v in splices:
            out.extend(cur[prev:i])
            out.extend(v)
            prev = j
        out.extend(cur[prev:])
        cur = out
        _bump(stats, "wins", len(splices))
        _bump(stats, "saved", gained)
        if log:
            log("  round %d: %d splices, %d moves" % (_r + 1, len(splices), gained))
        assert puzzle.solves(s0, cur), "bridge compression broke the path"
    return cur


def should_run(path_length, from_merged_floor, log=print):
    """The measured rule of thumb, encoded so it is not re-litigated every run.

    Long paths that are OUR OWN output are the regime where this pays; anything coming out
    of an n-way community min-merge is already too tight to be worth the compute.
    """
    if from_merged_floor:
        log("bridge compression on a min-merged floor: expect <1 pct win rate; "
            "run a smoke test only")
        return False
    if path_length < 80:
        log("path is short (%d); expected yield is small -- prefer an exact ball sweep"
            % path_length)
        return False
    return True


def _bump(stats, key, by=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + by
