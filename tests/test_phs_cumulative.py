"""Unit tests for PHS cumulative path scoring in the beam_lab solvers.

The cumulative score is score(child) = V(child) + lambda * sum_{t<=d} (-log pi(a_t|s_t)),
threaded through each surviving beam state via `self._phs_cum`. The core invariant
tested here is the per-step recurrence:

    cum_after[survivor i] == cum_before[parent(i)] + (-log pi(move(i) | parent_state))

verified at every step of a real solve, plus the per-attempt reset, the local-vs-
cumulative distinction, and the misconfiguration guard.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_phs_cumulative.py -v

Or as a script:
    .venv/Scripts/python.exe tests/test_phs_cumulative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "megaminx" / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "megaminx" / "beam_lab"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beam_search import KhoruzhiiSearchConfig, KhoruzhiiSolver, Profile
from megaminx.puzzle import Megaminx


PUZZLE = Megaminx.load(ROOT / "megaminx" / "data" / "puzzle_info.json")
SOLVED = PUZZLE.solved_state
N_GEN = len(PUZZLE.move_names)
STATE_SIZE = len(SOLVED)
DEVICE = "cpu"


class FakeV(nn.Module):
    """Deterministic admissible-ish heuristic: Hamming distance to solved. Good
    enough to steer the beam to a solution on a short scramble so the solve runs
    multiple real steps."""

    output_dim = 1

    def __init__(self, solved):
        super().__init__()
        self.register_buffer("solved", torch.tensor(list(solved), dtype=torch.long))

    def forward(self, x):  # x: (B, S)
        return (x.long() != self.solved).sum(dim=1).float()


class FakePolicy(nn.Module):
    """Fixed random linear projection -> non-uniform but deterministic logits."""

    def __init__(self, state_size, n_gen, seed=0):
        super().__init__()
        self.output_dim = n_gen
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(state_size, n_gen, generator=g))

    def forward(self, x):  # x: (B, S) -> (B, n_gen)
        return x.float() @ self.W


class FakeQ(nn.Module):
    """Q-head student for the qshort path: one score per action (output_dim=n_actions)."""

    def __init__(self, state_size, n_actions, seed=2):
        super().__init__()
        self.output_dim = n_actions
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(state_size, n_actions, generator=g))

    def forward(self, x):  # x: (B, S) -> (B, n_actions)
        return x.float() @ self.W


def _scramble(k: int, seed: int):
    rng = np.random.default_rng(seed)
    cur = SOLVED
    for _ in range(k):
        m = PUZZLE.move_names[int(rng.integers(0, N_GEN))]
        cur = PUZZLE.apply_move(cur, m)
    return cur


def _build_solver(lam: float, phs: bool, seed: int = 0):
    return KhoruzhiiSolver(
        PUZZLE,
        FakeV(SOLVED),
        device=DEVICE,
        internal_batch_size=4096,
        random_seed=seed,
        profile=False,
        policy_model=FakePolicy(STATE_SIZE, N_GEN, seed=1),
        lambda_policy=lam,
        phs_cumulative=phs,
    )


def _run_with_capture(solver, scramble, beam=2048, steps=12):
    """Run solve, capturing _phs_cum before/after each beam step."""
    records = []
    orig = solver._do_greedy_step  # bound method captured before patching

    def wrapped(states, states_hashed, states_bad_hashed, B, prof, *a, **k):
        parent_cum = solver._phs_cum.clone()
        parent_states = states.clone()
        out = orig(states, states_hashed, states_bad_hashed, B, prof, *a, **k)
        next_states, _values, moves, parents, _hashes = out
        child_cum = solver._phs_cum.clone()
        records.append(
            {
                "parent_cum": parent_cum,
                "parent_states": parent_states,
                "child_cum": child_cum,
                "moves": moves.clone(),
                "parents": parents.clone(),
                "n_survivors": int(next_states.shape[0]),
            }
        )
        return out

    solver._do_greedy_step = wrapped
    cfg = KhoruzhiiSearchConfig(beam_width=beam, num_steps=steps, num_attempts=1)
    found, plen, names, _prof = solver.solve(scramble, cfg)
    return found, plen, names, records


def test_recurrence_holds_every_step():
    """cum_after[i] == cum_before[parent(i)] + (-log pi(move(i) | parent_state))."""
    solver = _build_solver(lam=0.1, phs=True)
    policy = solver.policy_model
    _found, _plen, _names, records = _run_with_capture(solver, _scramble(4, seed=3))

    checked_steps = 0
    for rec in records:
        moves = rec["moves"]
        if moves.numel() == 0:  # beam died this step; _phs_cum not advanced
            continue
        parent_states = rec["parent_states"]
        with torch.inference_mode():
            log_pi = F.log_softmax(policy(parent_states.long()).float(), dim=-1)
        neg = (-log_pi).float()
        parent_cum = rec["parent_cum"]
        child_cum = rec["child_cum"]
        parents = rec["parents"]
        # Vectorized recurrence check.
        expected = parent_cum[parents] + neg[parents, moves]
        max_err = (child_cum.float() - expected).abs().max().item()
        assert max_err < 1e-3, f"recurrence violated, max_err={max_err}"
        # -log pi is non-negative, so cumulative cost never decreases along a path.
        assert (child_cum >= parent_cum[parents] - 1e-4).all()
        checked_steps += 1

    assert checked_steps >= 2, f"expected >=2 real steps, got {checked_steps}"


def test_root_resets_to_zero():
    """First step's parent_cum is the single-state root: exactly [0.0]."""
    solver = _build_solver(lam=0.1, phs=True)
    _f, _p, _n, records = _run_with_capture(solver, _scramble(3, seed=5))
    assert len(records) >= 1
    root_cum = records[0]["parent_cum"]
    assert root_cum.numel() == 1
    assert abs(float(root_cum[0])) < 1e-6


def test_cumulative_differs_from_local_after_depth():
    """By step >=2 the cumulative cost on a survivor strictly exceeds a single
    step's -log pi for at least some survivors (i.e. it accumulated history),
    which is precisely what the local penalty cannot do."""
    solver = _build_solver(lam=0.1, phs=True)
    policy = solver.policy_model
    _f, _p, _n, records = _run_with_capture(solver, _scramble(4, seed=11))
    # Find a step at depth >= 2 (its parents already carry >0 accumulated cost).
    deep = [r for r in records if r["moves"].numel() > 0 and (r["parent_cum"] > 1e-6).any()]
    assert deep, "no step had parents with accumulated cost > 0"
    rec = deep[0]
    with torch.inference_mode():
        log_pi = F.log_softmax(policy(rec["parent_states"].long()).float(), dim=-1)
    neg = (-log_pi).float()
    local_only = neg[rec["parents"], rec["moves"]]      # what the local penalty would use
    cumulative = rec["child_cum"].float()                # what PHS cumulative uses
    # At least one survivor's cumulative cost exceeds its single-step cost.
    assert (cumulative > local_only + 1e-4).any()


def test_phs_requires_policy():
    """phs_cumulative=True without a policy model is a hard configuration error."""
    raised = False
    try:
        KhoruzhiiSolver(
            PUZZLE, FakeV(SOLVED), device=DEVICE, profile=False,
            policy_model=None, lambda_policy=0.0, phs_cumulative=True,
        )
    except ValueError:
        raised = True
    assert raised, "expected ValueError when phs_cumulative=True without policy"


def test_solved_path_is_valid_when_found():
    """If the cumulative-scored solve returns a path, it must verify."""
    solver = _build_solver(lam=0.05, phs=True)
    scramble = _scramble(4, seed=3)
    found, plen, names, _records = _run_with_capture(solver, scramble, beam=4096, steps=14)
    if found:
        end = PUZZLE.apply_path(list(scramble), names)
        assert tuple(end) == SOLVED, "returned cumulative-scored path did not solve"
        assert plen == len(names)


def test_qshort_recurrence_holds_every_step():
    """Same recurrence, but through the production two-stage qshort path
    (_do_qshort_step applies the policy cost at both the shortlist gate and the
    teacher rerank, then advances _phs_cum on the chosen survivors)."""
    from beam_search_qshort import QShortlisterSolver

    solver = QShortlisterSolver(
        PUZZLE,
        teacher=FakeV(SOLVED),
        student=FakeQ(STATE_SIZE, N_GEN),
        device=DEVICE,
        internal_batch_size=4096,
        random_seed=0,
        alpha=2.0,
        policy_model=FakePolicy(STATE_SIZE, N_GEN, seed=1),
        lambda_policy=0.1,
        phs_cumulative=True,
    )
    policy = solver.policy_model

    records = []
    orig = solver._do_qshort_step

    def wrapped(states, states_hashed, states_bad_hashed, B, prof):
        parent_cum = solver._phs_cum.clone()
        parent_states = states.clone()
        out = orig(states, states_hashed, states_bad_hashed, B, prof)
        next_states, _v, moves, parents, _h = out
        records.append(
            {
                "parent_cum": parent_cum,
                "parent_states": parent_states,
                "child_cum": solver._phs_cum.clone(),
                "moves": moves.clone(),
                "parents": parents.clone(),
            }
        )
        return out

    solver._do_qshort_step = wrapped
    cfg = KhoruzhiiSearchConfig(beam_width=2048, num_steps=12, num_attempts=1)
    solver.solve(_scramble(4, seed=3), cfg)

    checked = 0
    for rec in records:
        moves = rec["moves"]
        if moves.numel() == 0:
            continue
        with torch.inference_mode():
            log_pi = F.log_softmax(policy(rec["parent_states"].long()).float(), dim=-1)
        neg = (-log_pi).float()
        expected = rec["parent_cum"][rec["parents"]] + neg[rec["parents"], moves]
        max_err = (rec["child_cum"].float() - expected).abs().max().item()
        assert max_err < 1e-3, f"qshort recurrence violated, max_err={max_err}"
        checked += 1
    assert checked >= 2, f"expected >=2 real qshort steps, got {checked}"


if __name__ == "__main__":
    tests = [
        ("recurrence holds every step", test_recurrence_holds_every_step),
        ("root resets to zero", test_root_resets_to_zero),
        ("cumulative differs from local after depth", test_cumulative_differs_from_local_after_depth),
        ("phs requires policy", test_phs_requires_policy),
        ("solved path valid when found", test_solved_path_is_valid_when_found),
        ("qshort recurrence holds every step", test_qshort_recurrence_holds_every_step),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001 - surface setup errors as failures
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
