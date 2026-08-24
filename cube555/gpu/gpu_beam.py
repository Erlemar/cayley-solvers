"""Single-GPU cube555 beam worker. One process owns one device and one task list.

This is `cube555/scripts/30_solve.py`'s solve loop, restructured so that N of them can
run concurrently on N GPUs with no communication between them. It is deliberately NOT a
sharded beam -- see run_multi_gpu.py for why that would be the wrong build on this
puzzle.

Differences from 30_solve.py, all forced by the target hardware or by the audit:

  fp16, not bf16       T4 is Turing (sm75) and has no native bf16, so bf16 is emulated.
                       Measured on 512 real test states against fp32:
                           fp16  max|dQ| 0.119  argmin 501/512
                           bf16  max|dQ| 0.876  argmin 427/512
                       7x tighter, and it matches what the reference 2xT4 engine picks.

  endgame ball held    30_solve.py:203/215 keeps `eg_states` AND its sorted copy alive,
  once                 so the d<=5 ball costs ~3.2 GB instead of 1.6 GB. Free on an
                       80 GB A100, not free on a 16 GB T4. Fixed here.

  best-of-frames       30_solve.py:269 breaks out of the frame loop on the first frame
                       that solves. Every frame is run here and the per-pid minimum is
                       kept -- RESULTS.md s7 lists that as untested and "strictly
                       better". It costs wall time, which is exactly what a second GPU
                       is buying back.

  fail-soft endgame    a 64-bit hash hit is not proof. `eg_descend` returns None instead
                       of raising, so a false positive costs one frame, not the session.

    python gpu_beam.py --device cuda:0 --tasks tasks.json --out results_0.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def _add_paths(handoff: Path) -> None:
    """Put the HANDOFF's `cayley` ahead of the repo's -- they have diverged.

    The repo also ships src/cayley, and it is the OLDER one: 641 lines against the
    handoff's 761, missing `no_backtrack` on KhoruzhiiSearchConfig among other things.
    sys.path.insert(0, ...) in a loop puts the LAST entry first, so listing the repo
    last here is what makes the handoff win. Getting this backwards does not fail
    loudly -- it silently runs a different searcher, and the first symptom was a
    TypeError on a flag that exists in one copy and not the other.
    """
    for p in (PROJECT / "src", handoff.parent / "src", handoff / "src"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


class Cube555Beam:
    """Everything that is per-device and expensive: model, endgame ball, solver."""

    def __init__(self, handoff: Path, assets: Path, checkpoint: Path, device: str,
                 endgame_depth: int = 5, internal_batch_size: int = 65536,
                 dtype: str = "fp16", history_depth: int = 4, state_dtype: str = "uint8",
                 no_backtrack: bool = False, qv_consistency: float = 0.0):
        import torch

        _add_paths(handoff)
        from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
        from cube555.models import load_model
        from cube555.puzzle import Cube555

        self.torch = torch
        self.KhoruzhiiSearchConfig = KhoruzhiiSearchConfig
        self.device = device
        self.history_depth = history_depth
        self.no_backtrack = no_backtrack

        self.puz = Cube555.load(assets / "puzzle_info.json")
        self.names = list(self.puz.move_names)
        self.name_i = {n: i for i, n in enumerate(self.names)}
        self.G = {n: np.array(v, dtype=np.int64) for n, v in self.puz.generators.items()}
        self.all_moves_np = np.array([self.puz.generators[n] for n in self.names],
                                     dtype=np.int64)
        self.central = np.array(self.puz.solved_state, dtype=np.int64)
        self.inv_idx = np.array([self.name_i[self.puz.inverse_name(n)]
                                 for n in self.names], dtype=np.int64)

        self.SYM = np.load(assets / "cube555_sym.npy").astype(np.int64)
        self.SYM_INV = np.load(assets / "cube555_sym_inv.npy").astype(np.int64)
        # FRAME move -> ORIGINAL move. The sibling table goes the other way and would
        # yield right-length paths that solve the conjugate and not the original.
        self.RELABEL = np.load(assets / "cube555_move_relabel_inv.npy").astype(np.int64)

        td = {"fp16": torch.float16, "fp32": torch.float32,
              "bf16": torch.bfloat16}[dtype]
        self.model = load_model(checkpoint, device=device, dtype=td)
        self.model.return_value = bool(qv_consistency)
        self.model.inference_chunk_size = internal_batch_size

        # ---- endgame ball -----------------------------------------------------
        self.eg = None
        self.goal_fn = None
        if endgame_depth > 0:
            blob = torch.load(str(assets.parent / f"anchors_d{endgame_depth}.pt")
                              if (assets.parent / f"anchors_d{endgame_depth}.pt").exists()
                              else str(handoff / "data" / f"anchors_d{endgame_depth}.pt"),
                              map_location="cpu", weights_only=False)
            hv = torch.randint(-(2 ** 62), 2 ** 62, (150,), dtype=torch.int64,
                               device=device,
                               generator=torch.Generator(device=device).manual_seed(555))
            st = blob["states"].to(device)
            # Chunked for the same reason _lookup is: `st.long()` over the whole d<=5
            # table is 10,739,017 x 150 x 8 B = 12.9 GiB in one allocation, which is
            # the "Tried to allocate 12.00 GiB" that killed the first 2xT4 run. It
            # survived locally only because Windows WDDM spills CUDA allocations into
            # host memory -- an impossible 25.55 GiB peak reading on a 16 GB card was
            # this, and I wrote it off as broken instrumentation. Linux/T4 does not
            # spill, so it hard-fails there.
            h = torch.empty(st.shape[0], dtype=torch.int64, device=device)
            _hc = 1 << 20
            for _i in range(0, st.shape[0], _hc):
                h[_i:_i + _hc] = (st[_i:_i + _hc].to(torch.int64) * hv).sum(1)
            order = torch.argsort(h)
            eg_hash = h[order].contiguous()
            eg_q = blob["q"].to(device)[order].contiguous()
            eg_depth = blob["depth"].to(device)[order].contiguous()
            eg_sorted = st[order].contiguous()
            # Drop the unsorted copy: on a 16 GB card the d<=5 ball is 1.6 GB and
            # keeping both is 3.2 GB for nothing.
            del st, h, order
            torch.cuda.empty_cache()

            eg_chunk = max(1 << 18, internal_batch_size)

            def _lookup(states_t):
                """EXACT membership, chunked and without an int64 blow-up.

                A 64-bit hash hit is not proof -- at 10.7M entries and ~1e11 lookups a
                false positive is expected, and one killed a 29 h run -- so the state
                itself is compared. The naive way to write that is
                `eg_sorted[pos].to(int64) == states_t.long()`, and it is what OOM'd a
                T4: the Q path hands this the whole progressive-top-k shortlist
                (k = max(2B, q_topk_min), i.e. ~4.2M rows at B=2^21), and every
                (N,150) int64 temporary is 4.69 GiB -- the exact figure in the OOM.
                Comparing in the TABLE's dtype is 8x smaller, and chunking bounds the
                hash temporary regardless of how large the shortlist gets.
                """
                n = states_t.shape[0]
                pos_o = torch.empty(n, dtype=torch.int64, device=device)
                hit_o = torch.empty(n, dtype=torch.bool, device=device)
                for i in range(0, n, eg_chunk):
                    s = states_t[i:i + eg_chunk]
                    hh = (s.to(torch.int64) * hv).sum(1)
                    q = torch.searchsorted(eg_hash, hh).clamp_max(eg_hash.numel() - 1)
                    h = eg_hash[q] == hh
                    h &= (eg_sorted[q] == s.to(eg_sorted.dtype)).all(dim=1)
                    pos_o[i:i + eg_chunk] = q
                    hit_o[i:i + eg_chunk] = h
                return pos_o, hit_o

            self.eg = (_lookup, eg_q, eg_depth)
            self.goal_fn = lambda s: _lookup(s)[1]
            self.eg_n = int(eg_hash.numel())

        self.solver = KhoruzhiiSolver(
            self.puz, self.model, device=device,
            internal_batch_size=internal_batch_size,
            use_q_function=True, qv_consistency_lambda=qv_consistency,
            # NOT int8: classes 128..149 would wrap negative into nn.Embedding.
            # uint8 holds 0..255 and HALVES the dominant allocation -- KhoruzhiiSolver
            # materialises the whole child block, (B, 30, 150) in states.dtype, which
            # is 17.6 GiB at 2^21 in int16 and 8.8 GiB in uint8. On an 80 GB A100 that
            # was free; on a 14.56 GiB T4 it is the difference between running and not.
            state_dtype={"int16": torch.int16, "uint8": torch.uint8}[state_dtype],
        )
        self.inv_move_tbl = (torch.as_tensor(self.inv_idx, dtype=torch.int32,
                                             device=device) if no_backtrack else None)

    # ---- frame algebra (mirrors cube555/scripts/30_solve.py) ------------------
    def to_frame(self, s0, k, inverted):
        t = np.array(self.puz.invert_state(s0), dtype=np.int64) if inverted else s0
        return self.SYM_INV[k][t[self.SYM[k]]] if k else t

    def from_frame(self, path_idx, k, inverted):
        q = [int(self.RELABEL[k][m]) for m in path_idx] if k else list(path_idx)
        return [int(self.inv_idx[m]) for m in reversed(q)] if inverted else q

    def replay(self, s0, path_idx):
        cur = s0
        for m in path_idx:
            cur = cur[self.all_moves_np[m]]
        return cur

    def eg_descend(self, state):
        """Exact tail from a ball state to solved. None on a hash false positive."""
        torch = self.torch
        _lookup, eg_q, eg_depth = self.eg
        out, cur = [], state
        for _ in range(64):
            st = torch.tensor(cur, dtype=torch.int64, device=self.device).unsqueeze(0)
            pos, hit = _lookup(st)
            if not bool(hit[0]):
                return None
            if int(eg_depth[pos[0]]) == 0:
                return out if np.array_equal(cur, self.central) else None
            a = int(eg_q[pos[0]].argmin())
            out.append(a)
            cur = cur[self.all_moves_np[a]]
        return None

    def solve_one(self, state, pid, k, inverted, beams, steps):
        """Return (path_idx_in_original_coords | None, wall_s, note)."""
        self.torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        u = self.to_frame(state, k, inverted)
        for B, ns in zip(beams, steps):
            cfg = self.KhoruzhiiSearchConfig(
                beam_width=B, num_steps=ns, num_attempts=1,
                internal_batch_size=self.model.inference_chunk_size,
                history_depth=self.history_depth, no_backtrack=self.no_backtrack,
            )
            found, _, raw = self.solver.solve(u, cfg, goal_check_fn=self.goal_fn)
            if not found:
                continue
            # KhoruzhiiSolver returns move NAMES; everything below this line works in
            # move INDICES (the frame tables are indexed, and so is eg_descend).
            frame_path = [self.name_i[m] for m in raw]
            if self.eg is not None:
                reached = self.replay(u, frame_path)
                if not np.array_equal(reached, self.central):
                    tail = self.eg_descend(reached)
                    if tail is None:
                        return None, time.time() - t0, "endgame_false_positive"
                    frame_path = frame_path + tail
            orig = self.from_frame(frame_path, k, inverted)
            if not np.array_equal(self.replay(state, orig), self.central):
                # A wrong RELABEL direction looks exactly like this and nothing else does.
                return None, time.time() - t0, "translated_path_does_not_solve"
            return orig, time.time() - t0, "ok"
        return None, time.time() - t0, "not_found"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", type=Path,
                    default=PROJECT / "cube555_pull" / "cube555_handoff_2026_08_22" / "cube555")
    ap.add_argument("--assets", type=Path, default=PROJECT / "cube555" / "tpu" / "kaggle_dataset")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", type=Path, required=True,
                    help="JSON list of {pid, sym, inverted}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beams", default="2097152")
    ap.add_argument("--max-steps", default="300")
    ap.add_argument("--endgame-depth", type=int, default=5)
    ap.add_argument("--internal-batch-size", type=int, default=65536)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--state-dtype", default="uint8", choices=["uint8", "int16"],
                    help="uint8 halves the child block; int16 is the upstream default")
    ap.add_argument("--history-depth", type=int, default=4)
    ap.add_argument("--no-backtrack", action="store_true")
    ap.add_argument("--qv-consistency", type=float, default=0.0)
    ap.add_argument("--test-csv", type=Path, default=None)
    args = ap.parse_args()

    ck = args.checkpoint or (args.assets / "q555_2k_BEST.pt")
    tests = {}
    src = args.test_csv or (args.handoff / "data" / "test.csv")
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tests[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    beams = [int(x) for x in args.beams.split(",")]
    steps = [int(x) for x in args.max_steps.split(",")]

    beam = Cube555Beam(
        args.handoff, args.assets, ck, args.device,
        endgame_depth=args.endgame_depth,
        internal_batch_size=args.internal_batch_size, dtype=args.dtype,
        history_depth=args.history_depth, no_backtrack=args.no_backtrack,
        qv_consistency=args.qv_consistency, state_dtype=args.state_dtype)
    print(f"[{args.device}] ready: {ck.name} dtype={args.dtype} "
          f"endgame={getattr(beam, 'eg_n', 0):,} tasks={len(tasks)}", flush=True)

    results = []
    for i, t in enumerate(tasks):
        pid, k, inv = int(t["pid"]), int(t["sym"]), bool(t["inverted"])
        path, wall, note = beam.solve_one(tests[pid], pid, k, inv, beams, steps)
        import torch as _t
        peak = _t.cuda.max_memory_allocated(args.device) / 2 ** 30
        rec = {"pid": pid, "sym": k, "inverted": inv, "wall_s": round(wall, 1),
               "note": note, "device": args.device, "checkpoint": ck.name,
               "peak_gib": round(peak, 2), "state_dtype": args.state_dtype}
        if path is not None:
            rec["path_len"] = len(path)
            rec["path"] = ".".join(beam.names[m] for m in path)
        results.append(rec)
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"[{args.device}] {i+1}/{len(tasks)} pid={pid} k={k} inv={int(inv)}: "
              f"{rec.get('path_len', note)}  {wall:.0f}s  peak {peak:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
