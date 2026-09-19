"""Package a chosen IHES transformer checkpoint for wide-beam (256M+) runs by others.

    python scripts/88_package_ihes_tf.py --checkpoint models/ihes_tf_b_s3/step_02000.pt \
        --name ihes_tf_v1 --out-dir exports/ihes_tf_v1 --gate-json submissions/eval/.../e2000.json

Writes <out-dir>/
    <name>.pt                  the PyTorch checkpoint (tetraminx.models.PieceTransformerQ)
    <name>.npz                 torch-free weights in the cube444 256M kernel's npz layout
    ihes_piece_layout.json     26-piece layout (scripts/50 derived, block-system verified)
    puzzle_info.json           move order the Q head indexes (18 moves)
    README.md                  input/output contract, activation, constants, gate numbers
    MANIFEST.json              sha256 of every file
The npz is parity-checked against PyTorch (scripts/87) before anything is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

README = """# {name}: IHES picture-cube PieceTransformer Q head

Trained {date} for the CayleyPy IHES cube (kaggle.com/competitions/cayleypy-ihes-cube).

## Contract

- **Input:** a (B, 72) integer state. Entry i is the facelet label (0..71) currently in slot i,
  in the convention of `puzzle_info.json`. Solved is the identity. A move applies as
  `new[i] = state[gen[i]]`.
- **Output:** (B, 18). `Q(s, a)` estimates the distance to solved of `apply(s, a)`. Lower is
  better. The column order is the move order of `puzzle_info.json`:
  `{moves}`.
- **Beam use:** score B parents once and take a global top-B over all (parent, move) pairs.
  `min_a Q(s, a) = d(s) - 1`.
- **Value head** (AZ): V(s) ~ d(s) from the same trunk. It is optional and can be used for
  qv-consistency.

## Architecture (all constants are also in `meta/*` inside the npz)

| | |
|---|---|
| tokens | 26 physical pieces + CLS = 27; pieces hold up to 4 facelets (`ihes_piece_layout.json`) |
| embedding | (max_piece_size x 72) value table, folded with piece_projection; piece position + type embeddings |
| trunk | d_model 256, 8 heads, 4 pre-norm blocks, ff 1024, LayerNorm eps 1e-5, CLS pooling |
| **activation** | **SiLU (x * sigmoid(x)), NOT ReLU** |
| params | {params:,} |

**The cube444 256M kernel hard-codes ReLU and cube444 constants** (6 colours, 96 slots,
24 actions, 56 pieces, max piece size 3). Loading this npz into that module unchanged runs
without error and gives wrong scores. Use `ihes_jax_q_models.py` (`kaggle_notebooks/tpu_beam_ihes_tf/`),
which reads `meta/silu` and the other constants. Its `apply_piece_transformer_mixed`
reproduces the kernel's CLS-only final block and HIGHEST-precision readout.

## Provenance

- checkpoint: `{ckpt}` (epoch {epoch}, bellman_step {bstep})
- parity (JAX npz vs PyTorch, fp32, {nprobe} states): see MANIFEST.json `parity`
- gate: {gate}
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Package an IHES transformer checkpoint.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--gate-json", type=Path, default=None,
                    help="84_solve_tf.py summary for this checkpoint (quoted in the README)")
    ap.add_argument("--note", action="append", default=[],
                    help="extra measured result lines for the README (repeatable)")
    args = ap.parse_args()

    import datetime
    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = args.out_dir / f"{args.name}.npz"
    r = subprocess.run([sys.executable, str(PROJECT / "scripts" / "87_export_ihes_tf_npz.py"),
                        "--checkpoint", str(args.checkpoint), "--out", str(npz)],
                       capture_output=True, text=True, encoding="utf-8")
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("parity check failed -- nothing packaged")
    parity = [ln for ln in r.stdout.splitlines() if "vs torch" in ln]

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # weights-only copy: the training checkpoint also carries the AdamW state (~3x the size)
    slim = {k: ck[k] for k in ("state_dict", "model_config", "epoch", "bellman_step", "init",
                               "train_config", "bellman_config") if k in ck}
    slim["model_config"] = {**slim["model_config"], "layout_path": "ihes_piece_layout.json"}
    torch.save(slim, args.out_dir / f"{args.name}.pt")
    shutil.copy2(PROJECT / "data" / "ihes_piece_layout.json", args.out_dir / "ihes_piece_layout.json")
    shutil.copy2(PROJECT / "data" / "puzzle_info.json", args.out_dir / "puzzle_info.json")
    info = json.loads((PROJECT / "data" / "puzzle_info.json").read_text(encoding="utf-8"))
    gate = "not given"
    if args.gate_json and args.gate_json.exists():
        g = json.loads(args.gate_json.read_text(encoding="utf-8"))
        c = g.get("config", {})
        gate = (f"{g['n']}-pid held-out gate, B={c.get('beam')}, {c.get('sym_frames')} frame, "
                f"endgame d{c.get('endgame')}: total {g['total']} vs floor {g['floor_same']} "
                f"({g['excess']:+d}), ties {g['ties']}, solved {g['solved']}/{g['n']}")
    if args.note:
        gate += "".join(f"\n- {n}" for n in args.note)
    n_params = sum(v.numel() for v in ck["state_dict"].values())
    (args.out_dir / "README.md").write_text(README.format(
        name=args.name, date=datetime.date.today().isoformat(),
        moves=" ".join(info["generators"].keys()), params=n_params,
        ckpt=str(args.checkpoint).replace("\\", "/"), epoch=ck.get("epoch"),
        bstep=ck.get("bellman_step", "-"), nprobe="656", gate=gate), encoding="utf-8")
    files = sorted(p for p in args.out_dir.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    manifest = {"name": args.name, "files": {p.name: sha(p) for p in files},
                "parity": parity, "gate": gate}
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
