"""Production-stack residual solver for bridge compression.

Wraps the residual solve with optional:
  - A1: sym-ensemble (K rotations, take min)
  - A2: qshort prefilter (m23 / m23_v3 student model)
  - A3: NISS (solve inverse residual too, take min)

Implementation mirrors the production solver in `megaminx/scripts/03_solve.py`
but specialized for the bridge use case (small residual states, no fallback,
no full-1001 bookkeeping).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.verify import verify_path


def apply_rotation_to_state(state, R, R_inv):
    """Compute R · state · R_inv. out[i] = R[state[R_inv[i]]]."""
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))


def compute_conjugation_map(R, R_inv, generators_dict, gen_names):
    """For each generator name n in the rotated frame, find the original-frame
    name whose generator equals R_inv · g_n · R. Used to translate solver paths
    found on a rotated state back to the original state's frame."""
    n = len(R)
    perm_to_name = {tuple(g): nm for nm, g in generators_dict.items()}
    out = {}
    for nm in gen_names:
        g = generators_dict[nm]
        conj = tuple(R_inv[g[R[i]]] for i in range(n))
        if conj not in perm_to_name:
            raise ValueError(f"rotation is not a symmetry: R_inv * g_{nm} * R is not a generator")
        out[nm] = perm_to_name[conj]
    return out


def _ensure_beam_lab_on_path():
    """beam_lab is a top-level dir, not a package. Inject its path so its
    modules are importable as flat top-level names (beam_search, etc.)."""
    beam_lab_dir = str(PROJECT / "beam_lab")
    if beam_lab_dir not in sys.path:
        sys.path.insert(0, beam_lab_dir)


class _QShortAdapter:
    """Adapt beam_lab's 4-tuple solver API to the 3-tuple API."""
    def __init__(self, inner, internal_batch_size: int):
        self._inner = inner
        self._ibs = internal_batch_size

    def solve(self, state, cfg):
        _ensure_beam_lab_on_path()
        from beam_search import KhoruzhiiSearchConfig as _BLConfig  # noqa: E402
        bl_cfg = _BLConfig(
            beam_width=cfg.beam_width,
            num_steps=cfg.num_steps,
            num_attempts=cfg.num_attempts,
            internal_batch_size=self._ibs,
        )
        found, plen, names, _prof = self._inner.solve(state, bl_cfg)
        return found, plen, names


def load_sym_rotations(rot_path: Path, K: int, puzzle, sym_seed: int = 0):
    """Load K rotations from rotations.npy. Always includes identity.
    Returns list of (R, R_inv, conj_map) tuples, length K.
    """
    if K <= 0:
        return []
    rot_arr = np.load(rot_path)
    if rot_arr.shape[1] != len(puzzle.solved_state):
        raise ValueError(f"rotations cols={rot_arr.shape[1]} != state_size {len(puzzle.solved_state)}")
    if K > rot_arr.shape[0]:
        raise ValueError(f"K={K} > #rotations {rot_arr.shape[0]}")
    identity = np.arange(rot_arr.shape[1], dtype=rot_arr.dtype)
    identity_idx = None
    for i in range(rot_arr.shape[0]):
        if np.array_equal(rot_arr[i], identity):
            identity_idx = i
            break
    if identity_idx is None:
        raise ValueError("rotations file missing identity row")
    rng = np.random.default_rng(sym_seed)
    other_idxs = [i for i in range(rot_arr.shape[0]) if i != identity_idx]
    chosen_idxs = [identity_idx]
    if K > 1:
        chosen_idxs += list(rng.choice(other_idxs, size=K - 1, replace=False).tolist())
    out = []
    for idx in chosen_idxs:
        R = tuple(int(x) for x in rot_arr[idx])
        R_inv = tuple(int(x) for x in np.argsort(rot_arr[idx]))
        cm = compute_conjugation_map(R, R_inv, puzzle.generators, puzzle.move_names)
        out.append((R, R_inv, cm))
    return out


class ProductionResidualSolver:
    """Bridge residual solver supporting sym-ensemble + NISS, optionally qshort."""

    def __init__(self, puzzle, V_model, device: str,
                 internal_batch_size: int = 16384,
                 state_dtype: torch.dtype = torch.int8,
                 sym_rotations: Optional[list] = None,
                 use_niss: bool = False,
                 qshort_student: Optional[torch.nn.Module] = None,
                 qshort_alpha: int = 2):
        """Initialize the solver.

        Parameters
        ----------
        puzzle : Megaminx
        V_model : torch.nn.Module (teacher V)
        device : str
        sym_rotations : list of (R, R_inv, conj_map) tuples, or empty list. If
            empty / None, only identity is used.
        use_niss : whether to also solve inverse(residual) per rotation
        qshort_student : optional qshort student model. If given, use
            QShortlisterSolver with V_model as teacher + this as student.
        qshort_alpha : qshort top-k multiplier
        """
        self.puzzle = puzzle
        self.device = device
        self.sym_rotations = list(sym_rotations) if sym_rotations else []
        self.use_niss = use_niss
        self.internal_batch_size = internal_batch_size

        if qshort_student is not None:
            _ensure_beam_lab_on_path()
            from beam_search import setup_model_for_inference as _bl_setup_inference  # noqa: E402
            from beam_search_qshort import QShortlisterSolver  # noqa: E402
            _bl_setup_inference(V_model)
            _bl_setup_inference(qshort_student)
            inner = QShortlisterSolver(
                puzzle, teacher=V_model, student=qshort_student,
                device=device,
                internal_batch_size=internal_batch_size,
                random_seed=0, state_dtype=state_dtype,
                alpha=qshort_alpha,
                pad_to_batch_size=False,
            )
            self.solver = _QShortAdapter(inner, internal_batch_size)
        else:
            self.solver = KhoruzhiiSolver(
                puzzle, V_model, device=device,
                internal_batch_size=internal_batch_size,
                random_seed=0, state_dtype=state_dtype,
            )

    def solve_residual(self, residual_state: list[int], beam: int, max_steps: int,
                       num_attempts: int = 1):
        """Solve the residual state. Returns (found, bridge_len, bridge_moves).
        Tries all sym rotations + (optionally) NISS, returns shortest verified bridge.
        """
        candidates: list[list[str]] = []
        cfg = KhoruzhiiSearchConfig(
            beam_width=beam, num_steps=max(max_steps, 1),
            num_attempts=num_attempts,
            internal_batch_size=self.internal_batch_size,
        )
        residual_state = tuple(residual_state)
        rot_iter = self.sym_rotations if self.sym_rotations else [(None, None, None)]
        for R, R_inv, conj_map in rot_iter:
            # 1) Forward solve (rotated)
            s_to_solve = (residual_state if R is None
                          else apply_rotation_to_state(residual_state, R, R_inv))
            found, _, raw = self.solver.solve(s_to_solve, cfg)
            if found:
                p_orig = [conj_map[m] for m in raw] if conj_map is not None else raw
                vr = verify_path(self.puzzle, residual_state, p_orig)
                if vr.ok:
                    candidates.append(p_orig)
            # 2) NISS: solve inverse_state(rotated_residual), invert path
            if self.use_niss:
                inv_state = self.puzzle.invert_state(s_to_solve)
                found2, _, raw2 = self.solver.solve(inv_state, cfg)
                if found2:
                    inverted_back = self.puzzle.invert_path(raw2)
                    p_orig2 = ([conj_map[m] for m in inverted_back]
                               if conj_map is not None else inverted_back)
                    vr2 = verify_path(self.puzzle, residual_state, p_orig2)
                    if vr2.ok:
                        candidates.append(p_orig2)
        if not candidates:
            return False, 0, []
        best = min(candidates, key=len)
        return True, len(best), best
