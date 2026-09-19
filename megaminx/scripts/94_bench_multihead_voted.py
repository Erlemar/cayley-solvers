"""Bench a multi-head orbit V checkpoint with head-voted beam selection."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver, _state_hash
from cayley.model import ResBlock
from megaminx.puzzle import Megaminx


class ResMLPMultiHeadOrbit(nn.Module):
    def __init__(
        self,
        state_size: int = 120,
        num_classes: int = 120,
        hidden_dims: tuple[int, ...] = (2048, 512),
        num_res_blocks: int = 2,
        encoding: str = "embedding",
        embed_dim: int = 16,
        n_heads: int = 4,
    ):
        super().__init__()
        if encoding != "embedding":
            raise NotImplementedError("only embedding encoding is supported")
        self.state_size = state_size
        self.num_classes = num_classes
        self.encoding = encoding
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.output_dim = n_heads
        in_dim = state_size * embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])
        self.value_heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_heads)])
        self.uncertainty_head = nn.Linear(prev, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.input_stack[0].weight.dtype
        h = self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.value_heads], dim=1)

    def forward_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        values = torch.cat([head(h) for head in self.value_heads], dim=1)
        uncertainty = torch.nn.functional.softplus(self.uncertainty_head(h).squeeze(-1))
        return values, uncertainty


def load_multihead(path: Path, device: str, dtype=torch.bfloat16) -> ResMLPMultiHeadOrbit:
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()}
    mc = ck["model_config"]
    model = ResMLPMultiHeadOrbit(
        state_size=mc["state_size"],
        num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]),
        num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_heads=mc.get("n_heads", 4),
    )
    model.load_state_dict(sd)
    return model.to(device=device, dtype=dtype).eval()


def multihead_predict(model: nn.Module, states: torch.Tensor, batch_size: int,
                      with_uncertainty: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    model.eval()
    n_heads = getattr(model, "n_heads", getattr(model, "output_dim", 4))
    out = torch.empty((states.size(0), n_heads), dtype=torch.float16, device=states.device)
    unc = torch.empty(states.size(0), dtype=torch.float16, device=states.device) if with_uncertainty else None
    with torch.no_grad():
        for i in range(0, states.size(0), batch_size):
            chunk = states[i : i + batch_size]
            if with_uncertainty and hasattr(model, "forward_with_uncertainty"):
                vals, u = model.forward_with_uncertainty(chunk)
                out[i : i + batch_size] = vals.to(torch.float16)
                unc[i : i + batch_size] = u.to(torch.float16)
            else:
                out[i : i + batch_size] = model(chunk).to(torch.float16)
    return out, unc


class HeadVotedSolver(KhoruzhiiSolver):
    def __init__(self, *args, selection_mode: str = "voted", u_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_mode = selection_mode
        self.u_lambda = float(u_lambda)

    def _choose_candidates(self, values: torch.Tensor, uncertainty: torch.Tensor | None,
                           B: int) -> torch.Tensor:
        n, n_heads = values.shape
        target = min(B, n)
        score = values.min(dim=1).values
        if self.selection_mode == "min":
            _, chosen = torch.topk(score, target, largest=False, sorted=False)
            return chosen
        if self.selection_mode in ("head0", "head0-u"):
            score = values[:, 0]
            if self.selection_mode == "head0-u" and uncertainty is not None:
                score = score + self.u_lambda * uncertainty
            _, chosen = torch.topk(score, target, largest=False, sorted=False)
            return chosen
        per_head = max(1, B // n_heads)
        picks = []
        for h in range(n_heads):
            k = min(per_head, n)
            _, idx = torch.topk(values[:, h], k, largest=False, sorted=False)
            picks.append(idx)
        chosen = torch.unique(torch.cat(picks)) if picks else torch.empty(0, dtype=torch.long, device=values.device)
        if chosen.numel() > target:
            _, order = torch.topk(score[chosen], target, largest=False, sorted=False)
            chosen = chosen[order]
        elif chosen.numel() < target:
            mask = torch.ones(n, dtype=torch.bool, device=values.device)
            if chosen.numel() > 0:
                mask[chosen] = False
            remain = torch.nonzero(mask, as_tuple=True)[0]
            k = target - chosen.numel()
            _, fill_order = torch.topk(score[remain], k, largest=False, sorted=False)
            chosen = torch.cat([chosen, remain[fill_order]])
        return chosen

    def _do_greedy_step(
        self, states: torch.Tensor, states_bad_hashed: torch.Tensor, B: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = states.size(0)
        bs = self.internal_batch_size
        idx0 = torch.arange(n, device=self.device).repeat_interleave(self.n_actions)
        moves = torch.arange(self.n_actions, device=self.device).repeat(n)

        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, n, bs):
            chunk_states = states[i : i + bs]
            chunk_neighbors = self._get_neighbors(chunk_states).flatten(end_dim=1)
            neighbors_hashed[i * self.n_actions : (i + chunk_states.size(0)) * self.n_actions] = _state_hash(
                chunk_neighbors, self.hash_vec, bs
            )
        idx1 = self._unique_hashed_idx(neighbors_hashed, states_bad_hashed)
        if idx1.numel() == 0:
            empty_s = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_v = torch.empty(0, dtype=torch.float16, device=self.device)
            empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty_s, empty_v, empty_i, empty_i

        candidate_states = self._apply_move(states[idx0[idx1]], moves[idx1])
        want_unc = self.selection_mode.endswith("-u")
        values, uncertainty = multihead_predict(self.model, candidate_states, bs, with_uncertainty=want_unc)
        idx2 = self._choose_candidates(values, uncertainty, B)
        chosen_idx1 = idx1[idx2]
        next_states = candidate_states[idx2]
        value = values[idx2].min(dim=1).values
        return next_states, value, moves[chosen_idx1], idx0[chosen_idx1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--pids", default="0,100")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--mode", choices=["voted", "min", "head0", "head0-u"], default="voted")
    ap.add_argument("--u-lambda", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional partial CSV of verified solved paths; requires one checkpoint")
    args = ap.parse_args()
    if args.out is not None and len(args.checkpoints) != 1:
        print("--out is only supported with exactly one checkpoint", file=sys.stderr)
        return 2

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv", encoding="utf-8")))
    pids = [int(x) for x in args.pids.split(",")]
    results = {}
    out_rows: list[dict[str, str]] = []
    for spec in args.checkpoints:
        path, label = (spec.rsplit(":", 1) + [None])[:2]
        path = Path(path)
        label = label or path.stem
        print(f"\n--- {label} ({path}) ---", flush=True)
        model = load_multihead(path, device="cuda")
        solver_cls = HeadVotedSolver
        solver = solver_cls(
            puzzle=puzzle,
            model=model,
            device="cuda",
            internal_batch_size=2**14,
            selection_mode=args.mode,
            u_lambda=args.u_lambda,
        )
        cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)
        n_solved = 0
        total = 0
        for pid in pids:
            s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
            t0 = time.time()
            found, path_len, path_names = solver.solve(s0, cfg)
            wall = time.time() - t0
            ok = False
            if found and path_names:
                cur = list(s0)
                for name in path_names:
                    g = puzzle.generators[name]
                    cur = [cur[gi] for gi in g]
                ok = (tuple(cur) == puzzle.solved_state)
            print(f"  pid {pid:4d}: {path_len if ok else '-':>5} moves  "
                  f"({'OK' if ok else 'N/F'}, {wall:.1f}s)", flush=True)
            if ok:
                n_solved += 1
                total += int(path_len)
                if args.out is not None:
                    out_rows.append({"initial_state_id": str(pid), "path": ".".join(path_names)})
        print(f"  bench: {n_solved}/{len(pids)} solved | total {total} | avg {total/max(1,n_solved):.1f}",
              flush=True)
        results[label] = (n_solved, total, len(pids))
        del model, solver
        torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for label, (s, t, n) in results.items():
        print(f"  {label}: {s}/{n} solved, total {t}, avg {t/max(1,s):.1f}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
            w.writeheader()
            for row in sorted(out_rows, key=lambda r: int(r["initial_state_id"])):
                w.writerow(row)
        print(f"wrote {args.out} ({len(out_rows)} solved rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
