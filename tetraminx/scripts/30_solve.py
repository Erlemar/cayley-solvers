"""Tetraminx beam solve: sym-ensemble frames + exact BFS endgame + floor merge.

Three levers on top of a plain beam:

  1. SYM-ENSEMBLE -- each puzzle is solved in several of the 48 distance-preserving
     frames (24 spatial symmetries x optional inverse antisymmetry).  Frames are
     independent searches over the same puzzle; we keep the shortest.  Frame order
     interleaves the inverse so even --sym-frames 2 gets the antisymmetry.

  2. EXACT ENDGAME -- the beam's goal test is "within the BFS table" (d <= 6), not
     "solved".  The search stops ~6 steps early and the tail is replaced by the
     table's optimal descent, so the last moves of every solve are provably
     optimal.  This is also where the beam is narrowest, so it saves real width.

  3. FLOOR MERGE -- per-pid min against a reference CSV.  Output is never worse
     than the floor.  (CLAUDE.md Rule 26: judge any result against the n-way min
     over every CSV, not against a single base.)

Every emitted path is replayed against the ORIGINAL test state and asserted to
solve before it is written.

    python3 tetraminx/scripts/30_solve.py \
        --checkpoint tetraminx/models/tv0_bellman/best.pt \
        --out tetraminx/submissions/run1.csv \
        --beam 65536 --sym-frames 4 --bf16
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.model import ResMLPDistance
from tetraminx.puzzle import Tetraminx


class EndgameTable:
    """Zobrist-hashed BFS table: membership test + optimal descent to solved."""

    def __init__(self, path: Path, device: str):
        d = np.load(path, allow_pickle=True)
        self.hashes = torch.from_numpy(d["hashes"]).to(device)      # sorted int64
        self.depths = torch.from_numpy(d["depths"]).to(device)      # int8
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

    def contains(self, states: torch.Tensor) -> torch.Tensor:
        return self.lookup(states) >= 0

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
    cfg = ckpt.get("model_config", {})
    if "arch" in cfg:
        # Architecture-matrix checkpoint (tetraminx.models): resmlp / gflownet /
        # transformer, with or without an AZ value head. ResMLPDistance cannot hold
        # these -- an AZ checkpoint dies on "Unexpected key(s): value_head.weight".
        from tetraminx.models import model_from_config
        model = model_from_config(dict(cfg)).to(device).eval()
        sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt.get("state_dict", ckpt).items()}
        model.load_state_dict(sd)
        model.return_value = False          # beam reads the Q head only
        return model.to(torch.bfloat16) if bf16 else model
    model = ResMLPDistance(
        state_size=cfg.get("state_size", 88),
        num_classes=cfg.get("num_classes", 88),
        hidden_dims=tuple(cfg.get("hidden_dims", (2048, 512))),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        encoding=cfg.get("encoding", "embedding"),
        embed_dim=cfg.get("embed_dim", 16),
        # output_dim 24 = an all-neighbours Q head (51_train_sparse_q.py); the solver
        # auto-detects it below and scores one forward per parent instead of 24 per child.
        # Checkpoints written through models.build_model call the same field `n_actions`
        # and omit `output_dim`; without the fallback those load a scalar head and die
        # with "size mismatch for head.weight: [24, 512] vs [1, 512]". The two classes
        # are weight-compatible (same head key names), so ResMLPDistance takes both.
        output_dim=cfg.get("output_dim", cfg.get("n_actions", 1)),
        inference_chunk_size=chunk,
    )
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device).eval()
    if bf16:
        model = model.to(torch.bfloat16)
    return model


def frame_list(n_frames: int, n_syms: int) -> list[tuple[int, bool]]:
    """Interleave the inverse frame so small ensembles still get antisymmetry."""
    frames = []
    for f in range(n_frames):
        frames.append((f // 2 % n_syms, f % 2 == 1))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--blend", nargs="+", type=Path, default=None,
                    help="extra checkpoints to average with --checkpoint at every step "
                         "(output-space ensemble; costs one forward per member)")
    ap.add_argument("--blend-weights", nargs="+", type=float, default=None,
                    help="per-member weights for --blend, in --checkpoint-then--blend "
                         "order (default: uniform)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--floor", type=Path,
                    default=PROJECT / "tetraminx" / "submissions" / "floor_public_29622.csv")
    ap.add_argument("--endgame", type=Path, default=None,
                    help="BFS endgame npz (default <data-dir>/bfs_endgame.npz; "
                         "pass 'none' to disable)")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=45)
    ap.add_argument("--sym-frames", type=int, default=1)
    ap.add_argument("--num-attempts", type=int, default=1,
                    help="retry a stagnated beam with the stuck hashes blacklisted")
    ap.add_argument("--history-depth", type=int, default=0,
                    help="exclude candidates seen in the last N layers (CayleyPy's "
                         "beam uses 10); 0 = dedup within the current layer only")
    ap.add_argument("--pids", type=str, default="", help="e.g. 0-99 or 3,7,11")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=32768)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="skip pids already in --out")
    ap.add_argument("--value-head", action="store_true",
                    help="score with the checkpoint's AZ VALUE head instead of its Q head "
                         "(the megaminx m_az_v4_v_only arrangement)")
    ap.add_argument("--qv-rerank", action="store_true",
                    help="Q head shortlists alpha*B candidates, the same checkpoint's "
                         "value head picks the final B")
    ap.add_argument("--qv-alpha", type=float, default=2.0)
    ap.add_argument("--qv-band-lo", type=float, default=0.0,
                    help="auto-accept the top band-lo*B by Q and rerank only the "
                         "uncertain band above it (0 = rerank the whole shortlist)")
    ap.add_argument("--qv-consistency", type=float, default=0.0,
                    help="lambda on |Q(s,a) - (V(s)-1)| using V(parent) from the SAME "
                         "trunk pass -- costs no extra forwards")
    ap.add_argument("--no-merge", action="store_true",
                    help="do not seed from the floor -- emit pure beam output "
                         "(floor is still loaded and reported, for benching)")
    args = ap.parse_args()

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    gens = np.array([puzzle.generators[nm] for nm in move_names], dtype=np.int64)
    inv_idx = np.array([name_to_idx[puzzle.inverse_name(nm)] for nm in move_names])
    solved = np.array(puzzle.solved_state, dtype=np.int64)

    sym = np.load(args.data_dir / "tetra_symmetries.npy").astype(np.int64)
    sym_inv = np.load(args.data_dir / "tetra_symmetries_inv.npy").astype(np.int64)
    relabel = np.load(args.data_dir / "tetra_move_relabel.npy").astype(np.int64)

    endgame = None
    eg_path = args.endgame if args.endgame is not None else args.data_dir / "bfs_endgame.npz"
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

    floor: dict[int, list[str]] = {}
    if str(args.floor).lower() == "none":
        print("floor: disabled -- output is pure beam", flush=True)
    elif args.floor and args.floor.exists():
        with open(args.floor, encoding="utf-8", newline="") as f:
            floor = {int(r["initial_state_id"]): r["path"].split(".")
                     for r in csv.DictReader(f) if r["path"]}
        print(f"floor: {len(floor)} pids, {sum(len(v) for v in floor.values()):,} moves",
              flush=True)

    if args.pids:
        pids = []
        for part in args.pids.split(","):
            if "-" in part:
                a, b = part.split("-")
                pids.extend(range(int(a), int(b) + 1))
            else:
                pids.append(int(part))
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
        # Output-space ensemble: K forwards per step, so this costs ~K x a single
        # model. Compare it against a SINGLE model at K x the beam width, not against
        # a single model at the same width -- otherwise the blend is being credited
        # with compute it was simply handed. Weight-space souping (54_model_soup.py)
        # is the free alternative and should be tried first.
        from tetraminx.models import BlendedQ
        members = [load_model(p, args.device, args.bf16, args.chunk_size)
                   for p in [args.checkpoint, *args.blend]]
        model = BlendedQ(members, weights=args.blend_weights).to(args.device).eval()
        if args.bf16:
            model = model.to(torch.bfloat16)
        print(f"blend: {len(members)} models averaged per step, weights "
              + ", ".join(f"{w:.3f}" for w in model.weights.tolist())
              + f" (costs ~{len(members)}x a single model)", flush=True)
    else:
        model = load_model(args.checkpoint, args.device, args.bf16, args.chunk_size)
    rerank = None
    if args.qv_consistency != 0.0 and not getattr(model, "has_value_head", False):
        raise SystemExit("--qv-consistency needs a checkpoint trained with az_head=true")
    if args.value_head or args.qv_rerank:
        from tetraminx.models import ValueHeadOnly
        if not getattr(model, "has_value_head", False):
            raise SystemExit("--value-head/--qv-rerank need a checkpoint trained with az_head=true")
        vh = ValueHeadOnly(model).to(args.device).eval()
        if args.bf16:
            vh = vh.to(torch.bfloat16)
        if args.value_head:
            model = vh          # V-only: score children with the value head
        else:
            rerank = vh         # Q shortlists, V reranks
    is_q = int(getattr(model, "output_dim", 1)) == len(puzzle.move_names)
    print(f"scorer: {'Q head (1 forward per parent)' if is_q else 'V (1 forward per child)'}",
          flush=True)
    solver = KhoruzhiiSolver(puzzle, model, device=args.device,
                             internal_batch_size=args.chunk_size,
                             use_q_function=is_q,
                             qv_rerank_model=rerank, qv_alpha=args.qv_alpha,
                             qv_band_lo=args.qv_band_lo,
                             qv_consistency_lambda=args.qv_consistency)
    if rerank is not None:
        print(f"rerank: Q shortlist alpha={args.qv_alpha} -> value head", flush=True)
    # history_depth was parsed but never forwarded, so --history-depth was a silent
    # no-op: every "hd=1" local run was really hd=0, and comparing those against the
    # TPU driver (which passes 1) produced a phantom platform gap. Caught 2026-08-02
    # when a local hd=1 run reproduced the hd=0 control byte-for-byte.
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps,
                                internal_batch_size=args.chunk_size,
                                num_attempts=args.num_attempts,
                                history_depth=args.history_depth)
    frames = frame_list(args.sym_frames, sym.shape[0])
    print(f"frames: {frames}", flush=True)

    goal_fn = None
    if endgame is not None:
        def goal_fn(s):
            """Hit the table -- but only accept the SHALLOWEST hit in this beam.

            The searcher reconstructs to the first index its goal mask marks, so
            masking every hit would let a d=6 node win over a d=2 node found in
            the same step and cost up to 4 moves.  Stopping at the first step with
            any hit is itself free: a node at depth d found at step k gives k+d,
            and one more beam step can at best reach depth d-1 for the same total.
            """
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
    t_start = time.time()
    n_beat = 0
    total = 0
    for pid in pids:
        if pid in done:
            total += len(done[pid].split("."))
            continue
        s0 = states_by_pid[pid]
        best = [] if args.no_merge else list(floor.get(pid, []))
        best_src = "floor" if best else "none"
        t0 = time.time()
        for (k, inverted) in frames:
            t = s0
            if inverted:
                t = np.argsort(t)                       # group inverse of the state
            u = sym_inv[k][t[sym[k]]]                   # conj_k(t)
            found, _, path = solver.solve(u, cfg, goal_check_fn=goal_fn)
            if not found:
                continue
            if endgame is not None:
                reached = replay(u, path)
                path = path + endgame.descend(reached, gens, move_names)
            # frame -> original: sigma relabel, then undo the inverse frame
            q = [move_names[relabel[k][name_to_idx[nm]]] for nm in path]
            if inverted:
                q = [move_names[inv_idx[name_to_idx[nm]]] for nm in reversed(q)]
            if not np.array_equal(replay(s0, q), solved):
                print(f"  pid {pid} frame (k={k}, inv={inverted}): path FAILED to solve "
                      f"-- discarded", flush=True)
                continue
            if not best or len(q) < len(best):
                best = q
                best_src = f"beam(k={k},inv={int(inverted)})"
                n_beat += 1
        if not best:
            print(f"  pid {pid}: NO PATH (no floor, all frames failed)", flush=True)
            continue
        assert np.array_equal(replay(s0, best), solved), f"pid {pid}: emitted path is wrong"
        done[pid] = ".".join(best)
        total += len(best)
        floor_len = len(floor.get(pid, []))
        print(f"pid {pid:4d}: {len(best):3d} moves  ({best_src}, floor {floor_len})  "
              f"{time.time() - t0:.1f}s", flush=True)

        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["initial_state_id", "path"])
            for p in sorted(done):
                w.writerow([p, done[p]])

    ours = sum(len(v.split(".")) for v in done.values())
    floor_total = sum(len(floor[p]) for p in done if p in floor)
    # Mean alongside the total -- a total is only comparable against runs over the
    # SAME pid set, and `avg>1` drops trivially-short pids (pid 0 is 1 move) that
    # otherwise pull the mean below the real per-puzzle cost.
    _lens = [len(v.split(".")) for v in done.values()]
    _nz = [v for v in _lens if v > 1]
    _avg = ours / len(done) if done else 0.0
    _avg_nz = sum(_nz) / len(_nz) if _nz else 0.0
    print(f"\nwrote {args.out}: {len(done)} pids, {ours:,} moves "
          f"(avg {_avg:.2f}, avg>1 {_avg_nz:.2f} over n={len(_nz)}) "
          f"({n_beat} frame-wins over running best) in {time.time() - t_start:.0f}s",
          flush=True)
    if floor_total:
        delta = ours - floor_total
        print(f"vs floor on the same pids: {floor_total:,} -> {ours:,} "
              f"({delta:+,} moves, {delta / max(len(done), 1):+.2f}/pid)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
