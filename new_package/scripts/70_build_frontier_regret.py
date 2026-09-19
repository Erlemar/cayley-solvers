"""Build on-policy frontier-regret labels for Tetraminx.

States come from two deployment distributions:

* greedy roll-ins, including low-margin decisions and pre-cycle states;
* near-cutoff and low-margin states from a modest Q beam.

For every state, a stronger short-horizon pooled beam propagates leaf costs back
to the first action.  Exact d<=6 hits replace the neural leaf estimate.  The
result is a masked 24-way cost target suitable for listwise policy distillation.

This is intentionally a probe-sized builder.  Scale only if held-out teacher
regret and greedy/tiny-search evaluation improve.
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

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver, _state_hash
from tetraminx.models import BlendedQ, model_from_config
from tetraminx.puzzle import Tetraminx
from tetraminx.search_free import lookahead_root_costs, q_policy_scores


class HashDepthTable:
    def __init__(self, path: Path, device: str):
        z = np.load(path)
        self.hashes = torch.from_numpy(z["hashes"]).to(device)
        self.depths = torch.from_numpy(z["depths"]).to(device)
        self.ztab = torch.from_numpy(z["ztab"]).to(device)
        self.max_depth = int(z["max_depth"])

    def lookup(self, states: torch.Tensor) -> torch.Tensor:
        s = states.long()
        h = torch.zeros(s.size(0), dtype=torch.int64, device=s.device)
        for i in range(s.size(1)):
            h ^= self.ztab[i].index_select(0, s[:, i])
        pos = torch.searchsorted(self.hashes, h).clamp_max(self.hashes.numel() - 1)
        hit = self.hashes.index_select(0, pos) == h
        out = torch.full((s.size(0),), -1, dtype=torch.int64, device=s.device)
        out[hit] = self.depths.index_select(0, pos[hit]).long()
        return out


def load_arch_model(path: Path, device: str, bf16: bool):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    if "arch" not in cfg:
        raise ValueError(f"{path} is not a tetraminx.models architecture checkpoint")
    model = model_from_config(cfg).to(device).eval()
    state = {k.removeprefix("_orig_mod."): v
             for k, v in ckpt.get("state_dict", ckpt).items()}
    model.load_state_dict(state)
    model.return_value = False
    return model.to(torch.bfloat16) if bf16 else model


def parse_pids(spec: str, default_path: Path) -> list[int]:
    if not spec:
        spec = default_path.read_text(encoding="utf-8").strip()
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--blend", nargs="*", type=Path, default=[])
    ap.add_argument("--blend-weights", nargs="*", type=float, default=None)
    ap.add_argument("--rollin-checkpoint", type=Path, default=None,
                    help="optional student policy used for greedy DAgger roll-ins; "
                         "teacher labels still use --checkpoint/--blend")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--pids", default="")
    ap.add_argument("--limit-pids", type=int, default=None)
    ap.add_argument("--frontier-beam", type=int, default=1024)
    ap.add_argument("--frontier-steps", type=int, default=18)
    ap.add_argument("--uncertain-per-step", type=int, default=2)
    ap.add_argument("--cutoff-per-step", type=int, default=1)
    ap.add_argument("--greedy-steps", type=int, default=36)
    ap.add_argument("--greedy-stride", type=int, default=3)
    ap.add_argument("--query-gap", type=float, default=0.35)
    ap.add_argument("--max-states-per-pid", type=int, default=32)
    ap.add_argument("--teacher-beam", type=int, default=4096)
    ap.add_argument("--teacher-depth", type=int, default=5)
    ap.add_argument("--verified-teacher-beam", type=int, default=0,
                    help="if >0, replace bounded-lookahead targets with multi-positive "
                         "root actions from a successful full beam of this width")
    ap.add_argument("--verified-teacher-steps", type=int, default=45)
    ap.add_argument("--history-depth", type=int, default=1)
    ap.add_argument("--qv-consistency", type=float, default=0.3)
    ap.add_argument("--chunk-size", type=int, default=32768)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    with (args.data_dir / "test.csv").open(encoding="utf-8", newline="") as fh:
        states_by_pid = {
            int(row["initial_state_id"]):
            np.asarray([int(x) for x in row["initial_state"].split(",")], dtype=np.int64)
            for row in csv.DictReader(fh)
        }
    pids = parse_pids(args.pids, args.data_dir / "ablation_pids.txt")
    if args.limit_pids is not None:
        pids = pids[:args.limit_pids]

    members = [load_arch_model(p, args.device, args.bf16)
               for p in [args.checkpoint, *args.blend]]
    if len(members) == 1:
        model = members[0]
    else:
        weights = args.blend_weights if args.blend_weights else None
        model = BlendedQ(members, weights=weights).to(args.device).eval()
        if args.bf16:
            model = model.to(torch.bfloat16)
    if not getattr(model, "has_value_head", False) and args.qv_consistency != 0.0:
        raise ValueError("--qv-consistency requires a value-head checkpoint/blend")

    solver = KhoruzhiiSolver(
        puzzle, model, device=args.device, internal_batch_size=args.chunk_size,
        use_q_function=True, qv_consistency_lambda=args.qv_consistency)
    rollin_solver = solver
    if args.rollin_checkpoint is not None:
        rollin_model = load_arch_model(args.rollin_checkpoint, args.device, args.bf16)
        rollin_solver = KhoruzhiiSolver(
            puzzle, rollin_model, device=args.device,
            internal_batch_size=args.chunk_size, use_q_function=True,
            qv_consistency_lambda=0.0)
        print(f"greedy DAgger roll-in: {args.rollin_checkpoint}", flush=True)
    table = HashDepthTable(args.data_dir / "bfs_endgame.npz", args.device)
    print(f"pids={pids} | teacher B={args.teacher_beam} H={args.teacher_depth} | "
          f"endgame=d{table.max_depth}", flush=True)

    inv_idx = torch.empty(solver.n_gen, dtype=torch.int64, device=args.device)
    name_to_idx = {name: i for i, name in enumerate(solver.move_names)}
    for i, name in enumerate(solver.move_names):
        inv_idx[i] = name_to_idx[name[1:] if name.startswith("-") else "-" + name]

    rows: list[dict] = []
    seen: set[bytes] = set()

    for pid in pids:
        candidates: list[dict] = []
        pid_seen: set[bytes] = set()

        def add_state(pid: int, state: torch.Tensor, source: int, depth: int, margin: float):
            host_state = state.detach().cpu().to(torch.uint8)
            key = host_state.numpy().tobytes()
            if key in seen or key in pid_seen:
                return
            pid_seen.add(key)
            candidates.append(dict(pid=pid, state=host_state,
                                   source=source, depth=depth, margin=margin))

        initial = torch.as_tensor(
            states_by_pid[pid], dtype=solver.state_dtype, device=args.device)

        # Greedy roll-in: periodic states plus every low-margin/cycle-adjacent state.
        state = initial.clone()
        last_action: int | None = None
        visited = [state.clone()]
        for depth in range(args.greedy_steps):
            if int(table.lookup(state.unsqueeze(0))[0]) >= 0:
                break
            _, score = q_policy_scores(rollin_solver, state.unsqueeze(0))
            score = score[0]
            children = rollin_solver._apply_move(
                state.unsqueeze(0).expand(rollin_solver.n_actions, -1),
                torch.arange(rollin_solver.n_actions, device=args.device))
            legal = torch.ones(rollin_solver.n_actions, dtype=torch.bool, device=args.device)
            if last_action is not None:
                legal[inv_idx[last_action]] = False
            ch = _state_hash(
                children, rollin_solver.hash_vec, rollin_solver.internal_batch_size)
            vh = _state_hash(
                torch.stack(visited), rollin_solver.hash_vec, rollin_solver.internal_batch_size)
            legal &= ~torch.isin(ch, vh)
            order = torch.argsort(score.masked_fill(~legal, float("inf")))
            live = order[legal[order] & torch.isfinite(score[order])]
            if live.numel() == 0:
                add_state(pid, state, 0, depth, 0.0)
                break
            gap = (float(score[live[1]] - score[live[0]])
                   if live.numel() > 1 else float("inf"))
            if depth % max(args.greedy_stride, 1) == 0 or gap < args.query_gap:
                add_state(pid, state, 0, depth, gap)
            last_action = int(live[0])
            state = children[last_action]
            visited.append(state.clone())

        # Modest beam frontier: uncertainty samples and states at the retention edge.
        states = initial.unsqueeze(0)
        recent_hashes: list[torch.Tensor] = []
        for depth in range(args.frontier_steps):
            _, score = q_policy_scores(solver, states)
            ordered = torch.sort(score, dim=1).values
            margin = ordered[:, 1] - ordered[:, 0]
            n_uncertain = min(args.uncertain_per_step, states.size(0))
            if n_uncertain:
                for idx in torch.argsort(margin)[:n_uncertain].tolist():
                    add_state(pid, states[idx], 1, depth, float(margin[idx]))
            n_cutoff = min(args.cutoff_per_step, states.size(0))
            if n_cutoff and states.size(0) > n_uncertain:
                for idx in range(states.size(0) - n_cutoff, states.size(0)):
                    add_state(pid, states[idx], 2, depth, float(margin[idx]))
            excluded = (torch.cat(recent_hashes[-args.history_depth:])
                        if args.history_depth > 0 and recent_hashes else
                        torch.empty(0, dtype=torch.int64, device=args.device))
            previous_hash = _state_hash(states, solver.hash_vec, solver.internal_batch_size)
            states, _, _, _ = solver._do_greedy_step(states, excluded, args.frontier_beam)
            if states.numel() == 0:
                break
            recent_hashes.append(previous_hash)
            if bool((table.lookup(states) >= 0).any()):
                break

        # Keep both the per-pid budget and the three deployment sources balanced.
        # Greedy roll-ins are numerous and otherwise consume the cap before the
        # true frontier rows are even considered.
        quota = max(1, args.max_states_per_pid // 3)
        selected: list[dict] = []
        selected_keys: set[bytes] = set()
        for source in (0, 1, 2):
            for row in (r for r in candidates if r["source"] == source):
                if len(selected) >= args.max_states_per_pid:
                    break
                key = row["state"].numpy().tobytes()
                if key not in selected_keys and sum(x["source"] == source for x in selected) < quota:
                    selected.append(row)
                    selected_keys.add(key)
        for row in candidates:
            if len(selected) >= args.max_states_per_pid:
                break
            key = row["state"].numpy().tobytes()
            if key not in selected_keys:
                selected.append(row)
                selected_keys.add(key)
        rows.extend(selected)
        seen.update(selected_keys)
        counts = {source: sum(r["source"] == source for r in selected) for source in (0, 1, 2)}
        print(f"harvest pid {pid}: {len(selected)} states by source {counts}", flush=True)

    if not rows:
        raise RuntimeError("no frontier states harvested")

    states_out: list[torch.Tensor] = []
    costs_out: list[torch.Tensor] = []
    current_rank_out: list[int] = []
    teacher_best_out: list[torch.Tensor] = []
    kept_rows: list[dict] = []
    teacher_diag: list[dict] = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        state = row["state"].to(args.device, dtype=solver.state_dtype)
        _, current = q_policy_scores(rollin_solver, state.unsqueeze(0))
        if args.verified_teacher_beam > 0:
            def goal_fn(s):
                d = table.lookup(s)
                hit = d >= 0
                if not bool(hit.any()):
                    return hit
                return hit & (d == d[hit].min())

            search_stats = {"ancestor_depth": 1}
            verified_cfg = KhoruzhiiSearchConfig(
                beam_width=args.verified_teacher_beam,
                num_steps=args.verified_teacher_steps,
                internal_batch_size=args.chunk_size,
                history_depth=args.history_depth,
            )
            found, _, path = solver.solve(
                state.detach().cpu().numpy(), verified_cfg,
                goal_check_fn=goal_fn, stats=search_stats)
            if not found or not path or "goal_ancestor_moves" not in search_stats:
                continue
            cur = state
            for name in path:
                cur = solver._apply_move(
                    cur.unsqueeze(0),
                    torch.tensor([name_to_idx[name]], device=args.device))[0]
            tail_depth = int(table.lookup(cur.unsqueeze(0))[0])
            if tail_depth < 0:
                raise RuntimeError("verified teacher path did not end inside exact table")
            total_cost = float(len(path) + tail_depth)
            best = torch.zeros(solver.n_actions, dtype=torch.bool, device=args.device)
            root_actions = torch.unique(search_stats["goal_ancestor_moves"].to(args.device))
            best[root_actions] = True
            if not bool(best[name_to_idx[path[0]]]):
                raise RuntimeError(
                    "teacher path first action disagrees with propagated goal root")
            costs = torch.full(
                (solver.n_actions,), float("inf"), dtype=torch.float32, device=args.device)
            costs[best] = total_cost
            diag = {
                "verified": True,
                "positive_roots": int(best.sum()),
                "solution_cost": total_cost,
            }
        else:
            costs, diag = lookahead_root_costs(
                solver, state, beam_width=args.teacher_beam, depth=args.teacher_depth,
                goal_depth_fn=table.lookup)
            finite = torch.isfinite(costs)
            if int(finite.sum()) < 2:
                continue
            best_cost = costs[finite].min()
            best = finite & (costs <= best_cost + 1.0e-5)
        order = torch.argsort(current[0])
        rank = int(torch.nonzero(best[order], as_tuple=True)[0][0]) + 1
        states_out.append(row["state"])
        costs_out.append(costs.detach().cpu())
        current_rank_out.append(rank)
        teacher_best_out.append(best.detach().cpu())
        kept_rows.append(row)
        teacher_diag.append(diag)
        if i % 10 == 0 or i == len(rows):
            print(f"label {i:4d}/{len(rows)} | kept={len(states_out)} | "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)

    if not states_out:
        raise RuntimeError("teacher produced fewer than two live root actions for every state")
    assert len(kept_rows) == len(states_out)

    cost_tensor = torch.stack(costs_out).to(torch.float16)
    rank_tensor = torch.tensor(current_rank_out, dtype=torch.int16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "states": torch.stack(states_out),
        "teacher_costs": cost_tensor,
        "teacher_best_mask": torch.stack(teacher_best_out),
        "pids": torch.tensor([r["pid"] for r in kept_rows], dtype=torch.int16),
        "sources": torch.tensor([r["source"] for r in kept_rows], dtype=torch.int8),
        "depths": torch.tensor([r["depth"] for r in kept_rows], dtype=torch.int8),
        "policy_margins": torch.tensor([r["margin"] for r in kept_rows], dtype=torch.float16),
        "current_teacher_best_rank": rank_tensor,
        "teacher_diagnostics": teacher_diag,
        "meta": {
            "checkpoint": str(args.checkpoint),
            "blend": [str(x) for x in args.blend],
            "blend_weights": args.blend_weights,
            "rollin_checkpoint": (str(args.rollin_checkpoint)
                                  if args.rollin_checkpoint is not None else None),
            "teacher_beam": args.teacher_beam,
            "teacher_depth": args.teacher_depth,
            "verified_teacher_beam": args.verified_teacher_beam,
            "verified_teacher_steps": args.verified_teacher_steps,
            "frontier_beam": args.frontier_beam,
            "qv_consistency": args.qv_consistency,
            "pids": pids,
            "source_names": {0: "greedy", 1: "frontier_uncertain", 2: "frontier_cutoff"},
        },
    }
    torch.save(payload, args.out)
    ranks = rank_tensor.float()
    print(f"wrote {args.out}: {len(states_out)} rows | teacher-best current "
          f"top1={(ranks <= 1).float().mean():.3f} top2={(ranks <= 2).float().mean():.3f} "
          f"top4={(ranks <= 4).float().mean():.3f} mean-rank={ranks.mean():.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
