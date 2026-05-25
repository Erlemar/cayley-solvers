"""Solve a single arbitrary megaminx scramble (not from test.csv).

Parses a space-separated scramble in the X / X' / X2 / X2' notation,
applies it to the solved state, then solves with KhoruzhiiSolver wrapped
with AZ v4 V + m23_v3 qshort + sym-ensemble.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/solve_adhoc.py \\
        --v-checkpoint megaminx/models/m_az_v4_v_only.pt \\
        --qshort-student megaminx/models/m23_v3_az_v4_sym/epoch_0199.pt \\
        --scramble "D B2 D DL2' ... U" \\
        --sym-ensemble 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def parse_scramble(scramble: str, valid_faces: set) -> list[str]:
    """Decode 'D B2 D DL2' B2'' notation → list of single primitive moves.

    X    → X (1 CW)
    X'   → -X (1 CCW)
    X2   → X.X (2 CW)
    X2'  → -X.-X (2 CCW)
    """
    out: list[str] = []
    for tok in scramble.split():
        # Determine inverse flag (trailing apostrophe)
        inv = tok.endswith("'")
        tok_body = tok[:-1] if inv else tok
        # Determine doubling (trailing 2)
        double = tok_body.endswith("2")
        face = tok_body[:-1] if double else tok_body
        if face not in valid_faces:
            raise ValueError(f"Unknown face '{face}' in token '{tok}'. "
                             f"Valid: {sorted(valid_faces)}")
        move = ("-" + face) if inv else face
        out.append(move)
        if double:
            out.append(move)
    return out


def load_v_model(path: Path, device: str) -> ResMLPDistance:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        output_dim=1,
    )
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def load_q_model(path: Path, device: str) -> ResMLPDistance:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 1024])),
        num_res_blocks=mc.get("num_res_blocks", 3),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        output_dim=24,
    )
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-checkpoint", required=True, type=Path)
    ap.add_argument("--qshort-student", type=Path, default=None)
    ap.add_argument("--qshort-alpha", type=float, default=2.0)
    ap.add_argument("--scramble", required=True, type=str,
                    help="Space-separated scramble in X / X' / X2 / X2' notation.")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--sym-ensemble", type=int, default=1,
                    help="Number of symmetry rotations (1 = no sym).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    gen_names = list(puzzle.generators.keys())
    cw_faces = {g for g in gen_names if not g.startswith("-")}
    print(f"puzzle: state_size={len(puzzle.solved_state)}  n_gens={len(gen_names)}")
    print(f"valid CW faces: {sorted(cw_faces)}")

    # Parse scramble → primitive moves in our notation
    primitives = parse_scramble(args.scramble, cw_faces)
    print(f"\nscramble: {args.scramble}")
    print(f"  parsed to {len(primitives)} primitive moves")

    # Apply scramble to solved state
    cur = list(puzzle.solved_state)
    for m in primitives:
        gen = puzzle.generators[m]
        cur = [cur[g] for g in gen]
    start_state = tuple(cur)
    print(f"  start state: differs from solved at "
          f"{sum(1 for i in range(len(start_state)) if start_state[i] != puzzle.solved_state[i])}/{len(start_state)} positions")

    # Load V model
    v_model = load_v_model(args.v_checkpoint, args.device)
    if args.bf16 and args.device == "cuda":
        v_model = v_model.to(torch.bfloat16)
    print(f"\nV: {sum(p.numel() for p in v_model.parameters()):,} params")

    # Load qshort if provided
    if args.qshort_student is not None:
        q_model = load_q_model(args.qshort_student, args.device)
        if args.bf16 and args.device == "cuda":
            q_model = q_model.to(torch.bfloat16)
        print(f"qshort: {sum(p.numel() for p in q_model.parameters()):,} params, "
              f"alpha={args.qshort_alpha}")
    else:
        q_model = None

    # Solver — V-only path (qshort needs beam_lab.QShortlisterSolver, not wired here)
    if q_model is not None:
        print("  NOTE: qshort wiring not implemented in adhoc; using V-only beam.")
    solver = KhoruzhiiSolver(
        puzzle=puzzle, model=v_model, device=args.device,
        internal_batch_size=2**14,
    )
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)
    print(f"beam={args.beam:,}  max_steps={args.max_steps}  sym_ensemble={args.sym_ensemble}")

    # Sym-ensemble loop
    rotations_path = PROJECT / "data" / "rotations.npy"
    if args.sym_ensemble > 1 and rotations_path.exists():
        rotations = np.load(rotations_path)
        rng = np.random.default_rng(0)
        non_identity = list(range(1, len(rotations)))
        chosen_idx = [0] + list(rng.choice(non_identity, size=args.sym_ensemble - 1, replace=False))
        print(f"  sym rotations selected: {chosen_idx}")
    else:
        chosen_idx = [None]

    best_path = None
    best_len = float("inf")
    total_wall = 0.0
    for k, rot_idx in enumerate(chosen_idx):
        if rot_idx is None or rot_idx == 0:
            s = start_state
            label = f"rot[0]/identity"
        else:
            R = rotations[rot_idx].astype(np.int64)
            R_inv = np.argsort(R)
            s = tuple(int(R[start_state[R_inv[i]]]) for i in range(len(start_state)))
            label = f"rot[{rot_idx}]"

        t0 = time.time()
        found, path_len, path = solver.solve(s, cfg)
        wall = time.time() - t0
        total_wall += wall

        if not found:
            print(f"  [{k+1}/{args.sym_ensemble}] {label}: N/F at {wall:.1f}s")
            continue

        # Translate path back through conjugation if rotated
        if rot_idx is not None and rot_idx != 0:
            # Need to translate moves: m_in_rotated_frame = R_inv · m · R
            # For simplicity, just verify on the rotated state — full back-translation
            # requires the conj_idx logic from 03_solve.py.
            # For this adhoc tool, only report the identity rotation's path.
            print(f"  [{k+1}/{args.sym_ensemble}] {label}: solved len={path_len} at {wall:.1f}s "
                  f"(path in rotated frame — not back-translated)")
            continue

        # Verify on original state
        cur = list(start_state)
        for name in path:
            gen = puzzle.generators[name]
            cur = [cur[g] for g in gen]
        verify_ok = (tuple(cur) == puzzle.solved_state)
        print(f"  [{k+1}/{args.sym_ensemble}] {label}: solved len={path_len} at {wall:.1f}s "
              f"verify={'OK' if verify_ok else 'FAIL'}")
        if verify_ok and path_len < best_len:
            best_len = path_len
            best_path = path

    print(f"\ntotal wall: {total_wall:.1f}s")
    if best_path is None:
        print("BEST: N/F")
        return 1
    print(f"BEST path length: {best_len}")
    print(f"BEST path: {'.'.join(best_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
