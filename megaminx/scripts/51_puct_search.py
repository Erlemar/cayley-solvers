"""PUCT (AlphaZero-style) search for megaminx.

Sequential per-descent selection (Python loop over batch_size). Each descent
applies virtual loss visible to subsequent descents within the same batch —
this is what gives lockstep-vectorization trouble. Slow per descent (Python),
but correct exploration. Use GPU for fast model forwards on batched leaves.

Q-formulation: Q(parent, action) = min(V over child subtree). NOT depth+V.
That formulation incentivizes shallow search.

Usage:
    python megaminx/scripts/51_puct_search.py \\
        --v-checkpoint megaminx/models/m_curr_v3/epoch_0499.pt \\
        --pi-checkpoint megaminx/models/m_pi_v2/epoch_0199.pt \\
        --pids 0,1,2,3,4 \\
        --max-nodes 1000000 --batch-size 256 \\
        --out megaminx/submissions/puct_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


VIRTUAL_LOSS = 3


class PUCTTree:
    def __init__(self, max_nodes, state_size, n_gen, device):
        self.max_nodes = max_nodes
        self.state_size = state_size
        self.n_gen = n_gen
        self.device = device

        self.states = torch.zeros((max_nodes, state_size), dtype=torch.int8, device=device)
        self.parent = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
        self.parent_action = torch.full((max_nodes,), -1, dtype=torch.int8, device=device)
        self.depth = torch.zeros(max_nodes, dtype=torch.int32, device=device)
        self.V = torch.zeros(max_nodes, dtype=torch.float32, device=device)

        self.children = torch.full((max_nodes, n_gen), -1, dtype=torch.int32, device=device)
        self.visits = torch.zeros((max_nodes, n_gen), dtype=torch.int32, device=device)
        self.virtual_visits = torch.zeros((max_nodes, n_gen), dtype=torch.int32, device=device)
        self.Q = torch.full((max_nodes, n_gen), 1e6, dtype=torch.float32, device=device)
        self.pi = torch.full((max_nodes, n_gen), 1.0 / n_gen, dtype=torch.float32, device=device)

        self.n_nodes = 0

    def add(self, state, parent, action, depth):
        idx = self.n_nodes
        self.states[idx] = state
        self.parent[idx] = parent
        self.parent_action[idx] = action
        self.depth[idx] = depth
        self.n_nodes += 1
        return idx


def _model_predict(model, states, chunk=4096):
    out = None
    with torch.no_grad():
        for i in range(0, states.size(0), chunk):
            s = states[i : i + chunk].long()
            v = model(s)
            if out is None:
                if v.dim() == 1:
                    out = torch.empty(states.size(0), dtype=torch.float32, device=states.device)
                else:
                    out = torch.empty((states.size(0), v.size(-1)), dtype=torch.float32, device=states.device)
            out[i : i + chunk] = v.float()
    return out


def puct_solve(
    initial_state, V_model, pi_model, all_moves, V0, *,
    c_puct=1.0, max_nodes=1_000_000, batch_size=256, max_iterations=100_000,
    device="cuda", verbose=False,
):
    """Solve via PUCT. Sequential descent + batched eval + Python backup."""
    state_size = initial_state.numel()
    n_gen = all_moves.size(0)

    initial_state = initial_state.to(device).to(torch.int8)
    V0 = V0.to(device).to(torch.int8)

    if torch.equal(initial_state, V0):
        return [], {"n_nodes": 0, "iterations": 0, "wall_s": 0.0}

    tree = PUCTTree(max_nodes, state_size, n_gen, device)
    root_idx = tree.add(initial_state, parent=-1, action=-1, depth=0)

    # Expand root
    V_root = _model_predict(V_model, initial_state.unsqueeze(0))[0].item()
    pi_logits_root = _model_predict(pi_model, initial_state.unsqueeze(0))[0]
    pi_root = torch.softmax(pi_logits_root, dim=-1)
    tree.V[root_idx] = V_root
    tree.pi[root_idx] = pi_root

    if verbose:
        print(f"[puct] root V={V_root:.3f}, top-3 pi: {torch.topk(pi_root, 3).values.tolist()}",
              flush=True)

    t_start = time.time()
    found_idx = -1

    for it in range(max_iterations):
        # Phase 1: Sequential descent — each descent applies VL visible to subsequent ones
        leaf_parents: list[int] = []
        leaf_actions: list[int] = []
        leaf_states: list[torch.Tensor] = []
        claimed: dict[tuple[int, int], int] = {}
        virtual_paths: list[list[tuple[int, int]]] = []

        for _ in range(batch_size):
            if tree.n_nodes >= max_nodes - batch_size:
                break
            cur = root_idx
            path = []
            while True:
                children_row = tree.children[cur]
                visits_row = (tree.visits[cur] + tree.virtual_visits[cur]).float()
                pi_row = tree.pi[cur]
                Q_row = tree.Q[cur]

                N_total = visits_row.sum().item() + 1.0
                sqrt_N = math.sqrt(N_total)
                init_q = float(tree.V[cur].item())  # Q for unexpanded edges = parent V
                Q_use = torch.where(children_row >= 0, Q_row,
                                    torch.tensor(init_q, device=device, dtype=torch.float32))
                bonus = c_puct * pi_row * sqrt_N / (1.0 + visits_row)
                score = Q_use - bonus
                a = int(score.argmin().item())

                path.append((cur, a))
                tree.virtual_visits[cur, a] += VIRTUAL_LOSS

                child_idx = int(children_row[a].item())
                if child_idx < 0:
                    if (cur, a) in claimed:
                        break
                    claimed[(cur, a)] = len(leaf_parents)
                    child_state = tree.states[cur][all_moves[a]]
                    leaf_parents.append(cur)
                    leaf_actions.append(a)
                    leaf_states.append(child_state)
                    break
                cur = child_idx
            virtual_paths.append(path)

        if not leaf_states:
            # Undo any VL still applied
            for path in virtual_paths:
                for cur, a in path:
                    tree.virtual_visits[cur, a] -= VIRTUAL_LOSS
            break

        # Phase 2: Batch eval
        states_batch = torch.stack(leaf_states)
        V_vals = _model_predict(V_model, states_batch).flatten()
        pi_logits = _model_predict(pi_model, states_batch)
        pi_vals = torch.softmax(pi_logits, dim=-1)

        # Phase 3: Add nodes, link, backup
        for i, (parent_idx, action, child_state) in enumerate(zip(leaf_parents, leaf_actions, leaf_states)):
            new_idx = tree.add(child_state, parent_idx, action,
                                int(tree.depth[parent_idx].item()) + 1)
            v_leaf = float(V_vals[i].item())
            tree.V[new_idx] = v_leaf
            tree.pi[new_idx] = pi_vals[i]
            tree.children[parent_idx, action] = new_idx

            if torch.equal(child_state, V0):
                found_idx = new_idx
                break

            # Backup V along path (min-update)
            cur_node = new_idx
            while True:
                p = int(tree.parent[cur_node].item())
                if p < 0:
                    break
                pa = int(tree.parent_action[cur_node].item())
                tree.visits[p, pa] += 1
                if v_leaf < float(tree.Q[p, pa].item()):
                    tree.Q[p, pa] = v_leaf
                cur_node = p

        # Undo VL
        for path in virtual_paths:
            for cur, a in path:
                tree.virtual_visits[cur, a] -= VIRTUAL_LOSS

        if found_idx >= 0:
            break

        if verbose and (it + 1) % 20 == 0:
            best_root_q = float(tree.Q[root_idx].min().item())
            print(f"[puct] iter {it+1:>5} nodes={tree.n_nodes:>7} root_min_Q={best_root_q:.3f} "
                  f"wall={time.time()-t_start:.1f}s", flush=True)

    wall = time.time() - t_start

    if found_idx < 0:
        return None, {"n_nodes": tree.n_nodes, "iterations": it + 1, "wall_s": wall}

    path = []
    cur = found_idx
    while cur != root_idx:
        path.append(int(tree.parent_action[cur].item()))
        cur = int(tree.parent[cur].item())
    path.reverse()
    return path, {"n_nodes": tree.n_nodes, "iterations": it + 1, "wall_s": wall, "found_depth": len(path)}


def load_model(path, device, output_dim=1):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        output_dim=mc.get("output_dim", output_dim),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.to(torch.bfloat16)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-checkpoint", required=True, type=Path)
    ap.add_argument("--pi-checkpoint", required=True, type=Path)
    ap.add_argument("--pids", type=str, default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nodes", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-iterations", type=int, default=100_000)
    ap.add_argument("--c-puct", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))

    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    print(f"PUCT solving {len(pids)} pids: {pids[:10]}{'...' if len(pids) > 10 else ''}", flush=True)
    print(f"  V model:  {args.v_checkpoint}")
    print(f"  pi model: {args.pi_checkpoint}")
    print(f"  c_puct={args.c_puct}, batch_size={args.batch_size}, max_nodes={args.max_nodes:,}",
          flush=True)

    V_model = load_model(args.v_checkpoint, args.device, output_dim=1)
    pi_model = load_model(args.pi_checkpoint, args.device, output_dim=24)

    n_gen = len(puzzle.move_names)
    state_size = len(puzzle.solved_state)
    all_moves = torch.zeros((n_gen, state_size), dtype=torch.int64, device=args.device)
    for i, name in enumerate(puzzle.move_names):
        all_moves[i] = torch.tensor(puzzle.generators[name], dtype=torch.int64)
    V0 = torch.tensor(puzzle.solved_state, dtype=torch.int8, device=args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.out, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["initial_state_id", "path"])

    n_solved = 0
    total_moves = 0
    t_total_start = time.time()

    for pid in pids:
        s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
        initial_state = torch.tensor(s0, dtype=torch.int8)

        t0 = time.time()
        path_idx, stats = puct_solve(
            initial_state, V_model, pi_model, all_moves, V0,
            c_puct=args.c_puct, max_nodes=args.max_nodes,
            batch_size=args.batch_size, max_iterations=args.max_iterations,
            device=args.device, verbose=args.verbose,
        )
        wall = time.time() - t0

        if path_idx is None:
            print(f"  pid={pid:>4} NOT FOUND  nodes={stats['n_nodes']:,} wall={wall:.1f}s", flush=True)
            writer.writerow([pid, ""])
        else:
            cur = list(s0)
            for m in path_idx:
                gen = puzzle.generators[puzzle.move_names[m]]
                cur = [cur[g] for g in gen]
            verify_ok = (tuple(cur) == puzzle.solved_state)
            path_str = ".".join(puzzle.move_names[m] for m in path_idx)
            print(f"  pid={pid:>4} found len={len(path_idx):>3} verify={verify_ok} "
                  f"nodes={stats['n_nodes']:,} wall={wall:.1f}s", flush=True)
            if verify_ok:
                writer.writerow([pid, path_str])
                n_solved += 1
                total_moves += len(path_idx)
            else:
                writer.writerow([pid, ""])

        f_csv.flush()

    f_csv.close()
    t_total = time.time() - t_total_start

    print(f"\n== summary ==", flush=True)
    print(f"solved: {n_solved}/{len(pids)}", flush=True)
    print(f"total moves: {total_moves:,}", flush=True)
    print(f"total wall: {t_total:.1f}s ({t_total/max(1,len(pids)):.1f}s/pid avg)", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
