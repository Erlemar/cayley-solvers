"""IHES beam solve for Q-head (transformer) and V (ResMLP) checkpoints.

Fork of `tetraminx/scripts/30_solve.py` for the IHES picture cube (72 facelets, 18 moves,
48 spatial frames x inverse antisymmetry). Same three levers:

  1. SYM FRAMES -- solve conjugated / inverted copies and keep the shortest. Frame order
     interleaves the inverse, so --sym-frames 2 = forward + inverse of the identity frame.
  2. EXACT ENDGAME -- the goal test is "inside the d<=6 Zobrist table"
     (data/bfs_table_d6_hash.npz); the tail is the table's optimal descent.
  3. FLOOR -- per-pid min against a reference CSV unless --no-merge (benching).

Scorer is auto-detected: an all-actions Q head (output_dim == 18, e.g. the
PieceTransformer from 51_train_sparse_q.py) scores all children with one forward per
parent; a scalar V (E6 ResMLPDistance) scores every child. Both run through the SAME
beam, dedup, endgame and frames, which is what makes E6 a matched control.

Every emitted path is replayed against the original test state before it is written.

    python scripts/84_solve_tf.py --checkpoint models/ihes_tf_a/epoch_0200.pt \
        --pid-file data/ihes_gate54.json --beam 65536 --sym-frames 1 --no-merge \
        --out submissions/eval/ihes_tf_a_e0200.csv --summary-json submissions/eval/ihes_tf_a_e0200.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.model import ResMLPDistance
from cayley.puzzle import PictureCube


class EndgameTable:
    """Zobrist-hashed BFS table: membership test + optimal descent to solved."""

    def __init__(self, path: Path, device: str):
        d = np.load(path, allow_pickle=True)
        self.hashes = torch.from_numpy(d["hashes"]).to(device)      # sorted int64
        self.depths = torch.from_numpy(d["depths"]).to(device)      # uint8
        self.ztab = torch.from_numpy(d["ztab"]).to(device)          # (S, S) int64
        self.max_depth = int(d["max_depth"])
        self.device = device

    def hash(self, states: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(states.shape[0], dtype=torch.int64, device=states.device)
        s = states.to(torch.int64)
        for i in range(states.shape[1]):
            h ^= self.ztab[i][s[:, i]]
        return h

    def lookup(self, states: torch.Tensor) -> torch.Tensor:
        """Return depth per state, or -1 when absent."""
        h = self.hash(states)
        pos = torch.searchsorted(self.hashes, h).clamp(max=self.hashes.numel() - 1)
        hit = self.hashes[pos] == h
        out = torch.full((states.shape[0],), -1, dtype=torch.int64, device=states.device)
        out[hit] = self.depths[pos[hit]].to(torch.int64)
        return out

    def descend(self, state: np.ndarray, gens: np.ndarray, move_names: list[str]) -> list[str]:
        """Optimal path from a table state to solved, one exact step at a time."""
        cur = torch.from_numpy(np.asarray(state)).to(self.device)
        d = int(self.lookup(cur[None, :])[0].item())
        assert d >= 0, "descend() called on a state outside the table"
        out: list[str] = []
        gens_t = torch.from_numpy(gens).to(self.device)
        while d > 0:
            children = cur[gens_t]                       # (n_gen, S)
            child_d = self.lookup(children)
            nxt = int(torch.nonzero(child_d == d - 1, as_tuple=True)[0][0].item())
            out.append(move_names[nxt])
            cur = children[nxt]
            d -= 1
        return out


def load_model(ckpt_path: Path, device: str, bf16: bool, chunk: int):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    if "arch" in cfg:
        # tetraminx.models checkpoint (PieceTransformerQ / ResMLPQ, optional AZ head).
        from tetraminx.models import model_from_config
        lp = cfg.get("layout_path")
        if lp and not Path(lp).is_absolute() and not Path(lp).exists():
            rel = Path(str(lp).replace("\\", "/"))
            # repo-relative (training checkpoints) or bundle-relative (exports/*/*.pt)
            for cand in (PROJECT / rel, Path(ckpt_path).parent / rel.name):
                if cand.exists():
                    cfg["layout_path"] = str(cand)
                    break
        model = model_from_config(cfg).to(device).eval()
        sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt.get("state_dict", ckpt).items()}
        model.load_state_dict(sd)
        model.return_value = False          # beam reads the Q head only
        return (model.to(torch.bfloat16) if bf16 else model), ckpt
    model = ResMLPDistance(
        state_size=cfg.get("state_size", 72),
        num_classes=cfg.get("num_classes", 72),
        hidden_dims=tuple(cfg.get("hidden_dims", (1024, 256))),
        num_res_blocks=cfg.get("num_res_blocks", 1),
        encoding=cfg.get("encoding", "embedding"),
        embed_dim=cfg.get("embed_dim", 16),
        output_dim=cfg.get("output_dim", cfg.get("n_actions", 1)),
        inference_chunk_size=chunk,
    )
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device).eval()
    if bf16:
        model = model.to(torch.bfloat16)
    return model, ckpt


def frame_list(n_frames: int, n_syms: int) -> list[tuple[int, bool]]:
    """Interleave the inverse frame so small ensembles still get antisymmetry."""
    return [(f // 2 % n_syms, f % 2 == 1) for f in range(n_frames)]


def parse_pids(spec: str) -> list[int]:
    pids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            pids.extend(range(int(a), int(b) + 1))
        else:
            pids.append(int(part))
    return pids


def main() -> int:
    ap = argparse.ArgumentParser(description="IHES beam solve (Q transformer or V ResMLP).")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--blend", nargs="+", type=Path, default=None,
                    help="extra Q checkpoints averaged with --checkpoint at every step")
    ap.add_argument("--blend-weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="write per-pid lengths, floor and totals here")
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--floor", type=Path,
                    default=PROJECT / "submissions" / "ihes_20260912_1410_verified.csv")
    ap.add_argument("--endgame", type=Path, default=None,
                    help="Zobrist BFS npz (default <data-dir>/bfs_table_d6_hash.npz; 'none' disables)")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--sym-frames", type=int, default=1)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--history-depth", type=int, default=0)
    ap.add_argument("--pids", type=str, default="", help="e.g. 0-99 or 3,7,11")
    ap.add_argument("--pid-file", type=Path, default=None,
                    help="JSON with a 'pids' list (e.g. data/ihes_gate54.json)")
    ap.add_argument("--limit", type=int, default=None)
    # 4096: peak transient memory is linear in the chunk; a larger chunk silently pages
    # on a 16 GB Windows card (memory transformer_chunk_size_sdpa_paging).
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="skip pids already in --out")
    ap.add_argument("--qv-consistency", type=float, default=0.0,
                    help="lambda on |Q(s,a) - (V(s)-1)|, V from the same trunk pass (AZ head)")
    ap.add_argument("--no-merge", action="store_true",
                    help="emit pure beam output (floor still loaded and reported)")
    args = ap.parse_args()

    puzzle = PictureCube.load(args.data_dir / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    gens = np.array([puzzle.generators[nm] for nm in move_names], dtype=np.int64)
    inv_idx = np.array([name_to_idx[puzzle.inverse_name(nm)] for nm in move_names])
    solved = np.array(puzzle.solved_state, dtype=np.int64)

    sym = np.load(args.data_dir / "cube_symmetries.npy").astype(np.int64)
    sym_inv = np.load(args.data_dir / "cube_symmetries_inv.npy").astype(np.int64)
    # built by scripts/80_build_ihes_move_relabel.py in THIS conjugation convention
    relabel = np.load(args.data_dir / "cube_move_relabel.npy").astype(np.int64)

    endgame = None
    eg_path = args.endgame if args.endgame is not None else args.data_dir / "bfs_table_d6_hash.npz"
    if str(eg_path).lower() != "none":
        if Path(eg_path).exists():
            endgame = EndgameTable(Path(eg_path), args.device)
            print(f"endgame table: {endgame.hashes.numel():,} states <= d{endgame.max_depth}",
                  flush=True)
        else:
            print(f"[warn] no endgame table at {eg_path} -- solving to exact goal", flush=True)

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states_by_pid = {int(r["initial_state_id"]):
                         np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                         for r in csv.DictReader(f)}
    assert all(v.shape == (72,) for v in states_by_pid.values()), "test.csv state parse failed"

    floor: dict[int, list[str]] = {}
    if str(args.floor).lower() != "none" and args.floor.exists():
        with open(args.floor, encoding="utf-8", newline="") as f:
            floor = {int(r["initial_state_id"]): r["path"].split(".")
                     for r in csv.DictReader(f) if r["path"]}
        print(f"floor: {len(floor)} pids, {sum(len(v) for v in floor.values()):,} moves "
              f"({args.floor.name})", flush=True)

    if args.pid_file is not None:
        pids = [int(p) for p in json.loads(args.pid_file.read_text(encoding="utf-8"))["pids"]]
    elif args.pids:
        pids = parse_pids(args.pids)
    else:
        pids = sorted(states_by_pid)
    if args.limit:
        pids = pids[:args.limit]

    done: dict[int, str] = {}
    if args.resume and args.out.exists():
        with open(args.out, encoding="utf-8", newline="") as f:
            done = {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(f)}
        print(f"resume: {len(done)} pids already in {args.out}", flush=True)

    if args.blend:
        from tetraminx.models import BlendedQ
        members = [load_model(p, args.device, args.bf16, args.chunk_size)[0]
                   for p in [args.checkpoint, *args.blend]]
        model = BlendedQ(members, weights=args.blend_weights).to(args.device).eval()
        if args.bf16:
            model = model.to(torch.bfloat16)
        print(f"blend: {len(members)} models, weights "
              + ", ".join(f"{w:.3f}" for w in model.weights.tolist()), flush=True)
        ck_meta = {}
    else:
        model, ck = load_model(args.checkpoint, args.device, args.bf16, args.chunk_size)
        ck_meta = {"epoch": ck.get("epoch"), "loss": ck.get("loss"), "val_loss": ck.get("val_loss")}
    if args.qv_consistency != 0.0 and not getattr(model, "has_value_head", False):
        raise SystemExit("--qv-consistency needs a checkpoint trained with az_head=true")
    is_q = int(getattr(model, "output_dim", 1)) == len(move_names)
    print(f"scorer: {'Q head (1 forward per parent)' if is_q else 'V (1 forward per child)'} "
          f"| checkpoint {args.checkpoint} {ck_meta}", flush=True)
    solver = KhoruzhiiSolver(puzzle, model, device=args.device,
                             internal_batch_size=args.chunk_size,
                             use_q_function=is_q,
                             qv_consistency_lambda=args.qv_consistency)
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps,
                                internal_batch_size=args.chunk_size,
                                num_attempts=args.num_attempts,
                                history_depth=args.history_depth)
    frames = frame_list(args.sym_frames, sym.shape[0])
    print(f"frames: {frames} | beam {args.beam} | max_steps {args.max_steps} | "
          f"bf16 {args.bf16} | merge {not args.no_merge}", flush=True)

    goal_fn = None
    if endgame is not None:
        def goal_fn(s):
            """Accept only the SHALLOWEST table hit in this beam step (see 30_solve.py)."""
            d = endgame.lookup(s)
            hit = d >= 0
            if not bool(hit.any()):
                return hit
            return hit & (d == d[hit].min())

    def replay(s0: np.ndarray, names: list[str]) -> np.ndarray:
        cur = s0
        for nm in names:
            cur = cur[gens[name_to_idx[nm]]]
        return cur

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_pid: dict[int, dict] = {}
    t_start = time.time()
    for pid in pids:
        if pid in done:
            per_pid[pid] = {"len": len(done[pid].split(".")) if done[pid] else 0,
                            "floor": len(floor.get(pid, [])), "src": "resume"}
            continue
        s0 = states_by_pid[pid]
        best = [] if args.no_merge else list(floor.get(pid, []))
        best_src = "floor" if best else "none"
        t0 = time.time()
        frame_lens: dict[str, int | None] = {}
        for (k, inverted) in frames:
            t = s0
            if inverted:
                t = np.argsort(t)                       # group inverse of the state
            u = sym_inv[k][t[sym[k]]]                   # conj_k(t)
            found, _, path = solver.solve(u, cfg, goal_check_fn=goal_fn)
            key = f"k{k}{'i' if inverted else 'f'}"
            if not found:
                frame_lens[key] = None
                continue
            if endgame is not None:
                path = path + endgame.descend(replay(u, path), gens, move_names)
            q = [move_names[relabel[k][name_to_idx[nm]]] for nm in path]
            if inverted:
                q = [move_names[inv_idx[name_to_idx[nm]]] for nm in reversed(q)]
            if not np.array_equal(replay(s0, q), solved):
                print(f"  pid {pid} frame {key}: path FAILED to solve -- discarded", flush=True)
                frame_lens[key] = None
                continue
            frame_lens[key] = len(q)
            if not best or len(q) < len(best):
                best = q
                best_src = f"beam({key})"
        floor_len = len(floor.get(pid, []))
        secs = time.time() - t0
        if not best and floor_len and not (states_by_pid[pid] == solved).all():
            print(f"pid {pid:4d}: NO PATH (floor {floor_len}) frames {frame_lens} {secs:.1f}s",
                  flush=True)
            per_pid[pid] = {"len": None, "floor": floor_len, "frames": frame_lens, "secs": secs}
            continue
        assert np.array_equal(replay(s0, best), solved), f"pid {pid}: emitted path is wrong"
        done[pid] = ".".join(best)
        per_pid[pid] = {"len": len(best), "floor": floor_len, "src": best_src,
                        "frames": frame_lens, "secs": round(secs, 2)}
        tag = "WIN" if len(best) < floor_len else ("tie" if len(best) == floor_len else "")
        print(f"pid {pid:4d}: {len(best):3d} moves  ({best_src}, floor {floor_len}) {tag:3s} "
              f"{secs:.1f}s", flush=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["initial_state_id", "path"])
            for p in sorted(done):
                w.writerow([p, done[p]])

    solved_rows = {p: r for p, r in per_pid.items() if r["len"] is not None}
    ours = sum(r["len"] for r in solved_rows.values())
    floor_same = sum(r["floor"] for r in solved_rows.values())
    excess = ours - floor_same
    ties = sum(1 for r in solved_rows.values() if r["len"] == r["floor"])
    wins = sum(1 for r in solved_rows.values() if r["len"] < r["floor"])
    unsolved = sorted(p for p, r in per_pid.items() if r["len"] is None)
    wall = time.time() - t_start
    print(f"\nSUMMARY {args.checkpoint.name}: solved {len(solved_rows)}/{len(per_pid)} | "
          f"total {ours} vs floor {floor_same} (excess {excess:+d}) | ties {ties} wins {wins} | "
          f"unsolved {unsolved} | {wall:.0f}s", flush=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({
            "checkpoint": str(args.checkpoint).replace("\\", "/"),
            "checkpoint_meta": ck_meta,
            "config": {"beam": args.beam, "max_steps": args.max_steps,
                       "sym_frames": args.sym_frames, "bf16": args.bf16,
                       "history_depth": args.history_depth, "qv": args.qv_consistency,
                       "chunk": args.chunk_size, "merge": not args.no_merge,
                       "endgame": None if endgame is None else endgame.max_depth},
            "n": len(per_pid), "solved": len(solved_rows), "total": ours,
            "floor_same": floor_same, "excess": excess, "ties": ties, "wins": wins,
            "unsolved": unsolved, "wall_s": round(wall, 1),
            "per_pid": {str(p): r for p, r in sorted(per_pid.items())},
        }, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
