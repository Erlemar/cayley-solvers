# Megaminx Capacity-Axis Brainstorm — Companion Plan

> Companion to `i-want-you-to-gentle-crayon.md` (data-signal axis). This file covers
> the capacity axis: ways to gain effective capacity beyond raw param count or single-model
> ResMLP scaling.

## Context

The "data-bound, not capacity-bound" claim in HANDOFF was specifically: scaling **the same
ResMLPDistance** at **the same Bellman recipe** plateaus (m05 6M → m26 12M → m26b 13M, all
hit Bellman loss ~0.094-0.10; RW MSE plateaus at ~64). That precisely rules out one thing
and leaves six axes intact:

1. **Different inductive bias** (transformer, GNN, equivariant, MoE)
2. **Different output representation** (distributional, ranking, uncertainty)
3. **Effective capacity at inference** (TTT, per-puzzle adaptation, MCTS, best-of-N)
4. **Mixture-of-specialists** (per-bucket, snapshot, MoE)
5. **Auxiliary supervision** (Bellman-during-RW, cycle decomp, piece distance)
6. **Better optimization** (SWA/EMA — Muon already tried)

The most expensive misdiagnosis would be to conclude "model capacity exhausted" when only
"this architecture × this loss × this signal" is exhausted. This document specifies the
full menu of capacity-axis levers we have NOT yet tried, with code sketches for each.

---

## Tier and ranking summary

| Tier | What it covers | When to use |
|---|---|---|
| **A** | Highest EV / cheapest spike | First experiments; cost ≤ 1 day each |
| **B** | Architecture changes worth trying | Pick 1-2 in parallel with Tier A; cost 1-3 days each |
| **C** | Multi-task aux supervision | Hedge with body-shared training; cost 1-2 days each |
| **D** | Inference-time compute beyond TTT | High ceiling, larger surgery |
| **E** | Optimization regime experiments | Low confidence; fold into other training runs |

**Top-3 first experiments**: A1 (listwise ranking loss), A3 (test-time Bellman refinement), A2 (distributional V).

---

## Tier A — Highest EV / cheapest spike

### A1 — Listwise ranking loss instead of MSE on V

**Hypothesis.** Beam search at every step uses ONLY the relative ordering of the 24 children
of each parent. We currently train MSE against a noisy walk-depth upper bound, then take
argmin at inference. This is an objective↔inference mismatch. Train via pairwise / listwise
rank loss on the 24-child tuple per parent. The model only needs to put the right child first
— not predict the absolute distance.

**Why it might break the plateau.** MSE-64 plateau is data-noise-bound; ranking loss is
robust to per-sample label scale (only relative order matters). The label noise in walk
depth IS the source of the MSE plateau, but it's largely scale noise, not order noise.

**Surgery.** Modify `02_train.py` and `05_bellman_refine.py` loss. New loss function:

```python
import torch
import torch.nn.functional as F


def listwise_rank_loss(
    parents: torch.Tensor,         # (B, S) state batch
    target_model,                  # frozen target net for true Bellman ranking
    student_model,                 # the model we're training
    generators: torch.Tensor,      # (n_gen, S)
    chunk_size: int = 8192,
    temperature: float = 1.0,
) -> torch.Tensor:
    """ListNet-style soft ranking loss. For each parent state, we have 24 children.
    The "true" ranking is by `1 + V_target(child)` (lower = better). The student must
    produce a softmax over its V predictions matching the target softmax.
    """
    B, S = parents.shape
    n_gen = generators.shape[0]

    # Generate 24 children per parent.
    children = torch.gather(
        parents.unsqueeze(1).expand(B, n_gen, S), 2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    ).reshape(B * n_gen, S)

    # Target ranking: V_target(child) (lower = better).
    with torch.no_grad():
        v_target = torch.empty(B * n_gen, dtype=torch.float32, device=parents.device)
        for i in range(0, B * n_gen, chunk_size):
            v_target[i : i + chunk_size] = target_model(
                children[i : i + chunk_size]
            ).flatten().to(torch.float32)
        v_target = v_target.view(B, n_gen)
        # Lower V is better; convert to logits where higher = better.
        target_logits = -v_target / temperature
        target_probs = F.softmax(target_logits, dim=1)  # (B, 24)

    # Student predictions over the same 24 children.
    v_student = torch.empty(B * n_gen, dtype=torch.float32, device=parents.device)
    for i in range(0, B * n_gen, chunk_size):
        v_student[i : i + chunk_size] = student_model(
            children[i : i + chunk_size]
        ).flatten().to(torch.float32)
    v_student = v_student.view(B, n_gen)
    student_logits = -v_student / temperature
    student_log_probs = F.log_softmax(student_logits, dim=1)

    # Cross-entropy between target and student soft rankings.
    return -(target_probs * student_log_probs).sum(dim=1).mean()
```

Then in the training loop, replace `loss = F.mse_loss(pred, target)` with
`loss = listwise_rank_loss(batch_states, target_model, model, gens_t)`.

**Wall.** ~25 extra forward passes per batch (24 children + 1 parent). Effective ~25× cost
on the target net, but the student network's batch is the same size — total ~2× wall vs
MSE Bellman.

**Acceptance gate.** Standard (≥+3 strat-5 solves AND ≤0.95× mean).

**Risk.** Low. Worst case: comparable to m05.

---

### A2 — Distributional V (C51 categorical / IQN quantile)

**Hypothesis.** Walk-depth labels are upper bounds with high variance. MSE collapses the
noise into the mean. Distributional output (Bellemare et al C51, IQN) lets the model learn
the noise structure rather than averaging through it. Beam can use either argmin of mean or
UCB-style `μ - β·σ`.

**Why it might break the plateau.** Plateau likely has irreducible aleatoric noise; modeling
it explicitly adds capacity in the output dimension without touching the body.

**Surgery.** Replace scalar head with categorical head (C51). Atoms: V_min=0, V_max=120,
n_atoms=51 (so atom_step ≈ 2.4).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class C51Head(nn.Module):
    """Categorical distribution over distance bins. Replaces ResMLPDistance's scalar head.

    Body output stays the same; this wraps it. n_atoms=51 → 51-way softmax per state.
    Atom locations: linspace(v_min, v_max, n_atoms).
    """
    def __init__(self, body: nn.Module, body_out_dim: int,
                 v_min: float = 0.0, v_max: float = 120.0, n_atoms: int = 51):
        super().__init__()
        self.body = body
        self.proj = nn.Linear(body_out_dim, n_atoms)
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        self.v_min, self.v_max, self.n_atoms = v_min, v_max, n_atoms

    def forward(self, s):
        h = self.body.forward_features(s)        # (B, body_out_dim) — needs body hook
        logits = self.proj(h)                    # (B, n_atoms)
        return logits  # raw logits; mean/sample handled at loss/inference site

    def expected_value(self, logits):
        probs = F.softmax(logits, dim=-1)
        return (probs * self.atoms).sum(dim=-1)  # (B,) — for beam scoring


def c51_bellman_loss(
    student_logits: torch.Tensor,    # (B, n_atoms)
    target_distribution: torch.Tensor,  # (B, n_atoms) — projected
) -> torch.Tensor:
    """Cross-entropy between student categorical and projected target distribution."""
    log_probs = F.log_softmax(student_logits, dim=-1)
    return -(target_distribution * log_probs).sum(dim=-1).mean()


def project_target_distribution(
    target_logits: torch.Tensor,  # (B, n_atoms) — target net's distribution at child
    atoms: torch.Tensor,          # (n_atoms,)
    v_min: float, v_max: float, n_atoms: int,
) -> torch.Tensor:
    """Standard C51 projection: shift child distribution by +1 (one-step Bellman),
    clamp to [v_min, v_max], project onto atoms via linear interpolation."""
    target_probs = F.softmax(target_logits, dim=-1)
    # Shift atoms by +1 step (Bellman).
    shifted = (atoms + 1.0).clamp(v_min, v_max)            # (n_atoms,)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    b = (shifted - v_min) / delta_z                        # fractional bin index
    lower, upper = b.floor().long(), b.ceil().long()
    # Distribute mass between lower and upper bins.
    proj = torch.zeros_like(target_probs)
    proj.scatter_add_(1, lower.unsqueeze(0).expand_as(target_probs),
                      target_probs * (upper.float() - b).unsqueeze(0))
    proj.scatter_add_(1, upper.unsqueeze(0).expand_as(target_probs),
                      target_probs * (b - lower.float()).unsqueeze(0))
    return proj
```

**Beam integration.** In `03_solve.py`, when scoring children replace `model(child)` with
`model.expected_value(model(child))`. Or experiment with UCB scoring `μ - β·σ` where σ comes
from the categorical's variance.

**Wall.** Negligible — single linear projection adds ~0.1% to forward.

**Acceptance gate.** Standard.

**Risk.** Low. Needs body to expose `forward_features()` — ~5 line surgery in cayley's
ResMLPDistance.

---

### A3 — Test-time Bellman refinement (per-puzzle TTT)

**Hypothesis.** Global m05 has Bellman loss ~0.094 averaged over the entire state space.
For any specific test puzzle's neighborhood (states reachable in ≤K random walks from
its initial state), local error is likely much higher. K-step gradient adaptation locally
sharpens the heuristic before beam runs.

**Evidence.** Akyurek et al "Surprising effectiveness of test-time training for abstract
reasoning" (2024). Sun et al CVPR 2020. OpenAI o1 paradigm: trade inference compute for
quality.

**Surgery.** New helper invoked from `03_solve.py` per puzzle, before beam:

```python
import copy
import torch
import torch.nn.functional as F


@torch.enable_grad()
def adapt_to_puzzle(
    model, puzzle, initial_state, generators, solved_state,
    n_walks: int = 16, walk_depth: int = 30,
    n_steps: int = 100, lr: float = 1e-4,
    device: str = "cuda",
) -> torch.nn.Module:
    """Fork the model, run K self-supervised Bellman steps on a synthetic dataset
    of states reachable from `initial_state` via short random walks. Return the
    adapted model. Original model is untouched.

    No ground truth needed: enforce f(s) ≈ 1 + min_a f(apply(s, a)).
    """
    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    optim = torch.optim.AdamW(adapted.parameters(), lr=lr)

    # Generate synthetic local data via random walks from initial_state.
    n_gen = generators.shape[0]
    s = torch.tensor(list(initial_state), dtype=torch.int64, device=device)
    walks_states = torch.empty((n_walks * walk_depth, len(initial_state)),
                                dtype=torch.int64, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    for w in range(n_walks):
        cur = s.clone()
        for d in range(walk_depth):
            a = torch.randint(0, n_gen, (1,), generator=g, device=device).item()
            cur = cur[generators[a]]
            walks_states[w * walk_depth + d] = cur

    for step in range(n_steps):
        # Compute Bellman target with frozen self.
        with torch.no_grad():
            children = torch.gather(
                walks_states.unsqueeze(1).expand(-1, n_gen, -1), 2,
                generators.unsqueeze(0).expand(walks_states.size(0), -1, -1),
            ).reshape(-1, walks_states.size(1))
            v_child = adapted(children).flatten().to(torch.float32)
            v_child = torch.where(
                (children == solved_state).all(dim=1),
                torch.zeros_like(v_child), v_child,
            )
            target = 1.0 + v_child.view(walks_states.size(0), n_gen).min(dim=1).values

        pred = adapted(walks_states).flatten().to(torch.float32)
        loss = F.mse_loss(pred, target)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    adapted.eval()
    return adapted
```

Wire into `03_solve.py`:

```python
ap.add_argument("--ttt-steps", type=int, default=0,
                help="If >0, do K test-time-training Bellman steps per puzzle "
                     "before beam. Adds ~3s/puzzle.")
# ... and per puzzle:
if args.ttt_steps > 0:
    adapted = adapt_to_puzzle(model, puzzle, state, gens_t, solved_t,
                              n_steps=args.ttt_steps, device=args.device)
    solver.model = adapted   # swap; restore after solve
```

**Wall.** ~2-3 sec/puzzle for 100 steps. Total ~50 min on 1001 puzzles.

**Acceptance gate.** Compare full-submission total moves vs current best 88,195. Standard
strat-5 won't show this well — TTT is most useful on hard puzzles where global model is weakest.

**Risk.** Moderate — could overfit to noise if too many steps. Start n_steps=50, lr=1e-4
conservative.

---

### A4 — Per-bucket difficulty specialists

**Hypothesis.** Test ordering is by random-walk LENGTH (pid 0 has 72-move walk; pid 1000 has
925-move walk). Stratifying by `pid // 100` gives 10 difficulty buckets. The data signal is
fundamentally different across buckets (random walks of length 20 give nearly-i.i.d. labels;
length 80 walks have heavy upper-bound bias). One model averages over all of this.

**Evidence.** MoE / specialist literature (Switch Transformer, Lepikhin GShard). Per-bucket
training avoids the "average everything" failure mode.

**Surgery.** Train 10 specialist V models. Each filtered to a different `k_max` range:
- bucket 0: k_max ∈ [10, 30]
- bucket 5: k_max ∈ [50, 70]
- bucket 9: k_max ∈ [80, 100]

Reuse `02_train.py` with a new `--k-min K --k-max K` arg. At inference, route by `pid // 100`:

```python
SPECIALISTS = {
    bucket: f"models/m30_specialist_b{bucket}/epoch_0499.pt"
    for bucket in range(10)
}

def route_specialist(pid: int):
    bucket = pid // 100
    return SPECIALISTS.get(bucket, SPECIALISTS[5])  # fallback to mid-bucket
```

Then in `03_solve.py`, lazy-load each specialist on first use of its bucket (cache dict),
swap into `solver.model` per pid. To save memory, only keep N specialists in VRAM at once
(LRU eviction).

**Wall.** 10× training (~50h sequential on 4090; cheaper with Kaggle parallel). Inference
adds one model load per bucket transition.

**Acceptance gate.** Per-bucket: each specialist must beat m05 on ITS bucket's strat-5
subset. Aggregate: total over 1001 puzzles must beat current 88,195.

**Risk.** Moderate. Boundary buckets (0, 9) may have too little useful walk-depth signal.

---

### A5 — SWA / EMA weight averaging on m05 cohort

**Hypothesis.** Loss-landscape connectivity (Garipov, Frankle) shows neighboring SGD minima
are linearly connectable. SWA averages weights across the last K checkpoints; the average
typically generalizes better than any single member.

**Evidence.** Izmailov et al. 2018 SWA paper. Standard technique in modern training.

**Surgery.** No retrain. Take m05 checkpoints at epochs 400, 425, 450, 475, 499 and average:

```python
import torch


def swa_average_checkpoints(checkpoint_paths: list, output_path: str,
                            device: str = "cpu") -> None:
    """Average state_dicts of multiple checkpoints. Common keys only; first checkpoint
    contributes its non-overlapping keys (e.g., for new-arch retrains)."""
    ckpts = [torch.load(p, map_location=device, weights_only=False) for p in checkpoint_paths]
    sds = [ck["state_dict"] for ck in ckpts]
    avg_sd = {}
    for k in sds[0]:
        if all(k in sd for sd in sds):
            avg_sd[k] = torch.stack([sd[k].float() for sd in sds]).mean(dim=0)
        else:
            avg_sd[k] = sds[0][k]
    torch.save({
        "state_dict": avg_sd,
        "swa_from": [str(p) for p in checkpoint_paths],
        "model_config": ckpts[0]["model_config"],
    }, output_path)
    print(f"wrote SWA average of {len(checkpoint_paths)} ckpts to {output_path}")


if __name__ == "__main__":
    paths = [f"models/m05_bellman_warm/epoch_{e:04d}.pt"
             for e in (399, 424, 449, 474, 499)]
    swa_average_checkpoints(paths, "models/m05_swa_5/epoch_0499.pt")
```

Then strat-5 the SWA model. If it wins, deploy.

**Wall.** 30 min code + 20 min strat-5 eval.

**Acceptance gate.** Standard.

**Risk.** Lowest possible. Truly free shot.

---

## Tier B — Architecture changes worth trying

Pick 1-2 in parallel with Tier A. Each is a Tier-1 retraining commitment (~12h on 4090).

### B1 — m18 transformer retry on 4090 (the rejection was wall-cost, not quality)

**Hypothesis.** HANDOFF rejected m18 because P100 took 225s/epoch, blowing the 12h Kaggle
limit. On 4090 the wall is ~90s/epoch (4090 ≈ 2.5× P100), so 500 epochs = 12.5h. **Feasible.**
HANDOFF explicitly says "Not infeasible in principle, just slower-per-wall."

**Concrete arch.** nanoGPT-style 6 layers, d_model=384, n_heads=6, ~6.5M params.

```python
import torch
import torch.nn as nn


class GPTValueModel(nn.Module):
    """Compact GPT-style transformer. Drop-in for ResMLPDistance: forward(s) → (B,) scalar."""
    def __init__(self, n_state: int = 120, vocab_size: int = 120,
                 d_model: int = 384, n_heads: int = 6, n_layers: int = 6,
                 d_ff: int = 1536, output_dim: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_state + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, s):  # s: (B, 120) int
        x = self.embed(s.long())
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb
        x = self.transformer(x)
        out = self.head(x[:, 0, :])
        return out.squeeze(-1) if self.output_dim == 1 else out
```

**Surgery.** New script `scripts/15_train_transformer.py` (clones `02_train.py` but
instantiates `GPTValueModel` instead of `ResMLPDistance`). New config
`configs/m29_transformer.yaml`.

**Wall.** ~12.5h for 500 epochs on 4090.

**Acceptance gate.** Standard.

**Risk.** Moderate. Attention on permutations is under-explored; rotary embeddings or learned
positional may matter.

---

### B2 — GNN over hardcoded dodecahedron piece adjacency

**Hypothesis.** Pieces that physically touch on the Megaminx surface are most likely to
interact during moves. Hard-code this geometry as edges. ResMLP treats all 120 positions
symmetrically; GNN imposes the puzzle's actual topology.

**Concrete arch.** GIN, 4 layers, d_hidden=128, ~2.5M params. Faster than m05 (40-60s/epoch
vs 25s/epoch with smaller batch).

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv  # pip install torch_geometric


class PieceGNN(nn.Module):
    """GNN over piece adjacency graph. Static edge_index from dodecahedron geometry."""
    def __init__(self, edge_index: torch.Tensor, d_hidden: int = 128, n_layers: int = 4,
                 vocab_size: int = 120, output_dim: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_hidden)
        self.register_buffer("edge_index", edge_index)
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(d_hidden, 2 * d_hidden),
                nn.BatchNorm1d(2 * d_hidden),
                nn.ReLU(),
                nn.Linear(2 * d_hidden, d_hidden),
            )) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_hidden, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, s):  # s: (B, 120)
        B, N = s.shape
        x = self.embed(s.long())          # (B, 120, d_hidden)
        # GINConv expects (B*N, d) with edge_index batched. Build batch index.
        x = x.reshape(B * N, -1)
        offsets = torch.arange(B, device=s.device).repeat_interleave(self.edge_index.size(1)) * N
        ei = self.edge_index.repeat(1, B) + offsets.unsqueeze(0)
        for conv in self.convs:
            x = conv(x, ei).relu()
        x = x.view(B, N, -1).mean(dim=1)
        out = self.head(x)
        return out.squeeze(-1) if self.output_dim == 1 else out


def build_dodecahedron_edges() -> torch.Tensor:
    """Return edge_index (2, E) for the 120-cell adjacency graph.

    Megaminx has 12 faces × 11 cells/face = 132, BUT the competition's puzzle_info has a
    120-element state. This means each face shares cells with neighbors. This builder is
    a placeholder; real implementation needs to read the actual cell-to-position mapping
    from the puzzle's geometry. For first-pass: connect each cell to 5 nearest neighbors
    based on a fixed dodecahedron face graph.
    """
    # Placeholder — needs real mapping from cell index to (face, position) on dodecahedron.
    # Approach: parse generators from puzzle_info.json; two cells are adjacent iff some
    # generator swaps or rotates them together.
    edges: list[tuple[int, int]] = []
    # ... (build from puzzle.generators by inspecting which cells move together)
    if not edges:
        edges = [(i, (i + 1) % 120) for i in range(120)]  # ring fallback
    e = torch.tensor(edges, dtype=torch.long).t()
    return torch.cat([e, e.flip(0)], dim=1)  # symmetric
```

**Surgery.** Train via new script `scripts/16_train_gnn.py`. The edge construction is the
real engineering — build it from `puzzle.generators` (cells that move together under any
generator share an edge).

**Wall.** ~10h for 500 epochs on 4090.

**Acceptance gate.** Standard.

**Risk.** Low-moderate. Worst case: GNN underperforms, signaling that position semantics
dominate physical adjacency.

---

### B3 — Mamba / bidirectional SSM

**Hypothesis.** Linear-in-n alternative to attention. State-space update may capture the
sequential algebraic structure of the puzzle (state evolves under generator action).

**Concrete arch.** d_model=512, n_layers=6, bidirectional, ~5M params, 60-80s/epoch.

```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba   # pip install mamba-ssm


class BidirMambaValue(nn.Module):
    def __init__(self, n_state: int = 120, vocab_size: int = 120,
                 d_model: int = 512, n_layers: int = 6, output_dim: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fwd = nn.ModuleList([Mamba(d_model) for _ in range(n_layers)])
        self.bwd = nn.ModuleList([Mamba(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 256), nn.GELU(),
            nn.Linear(256, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, s):  # s: (B, 120)
        x = self.embed(s.long())
        f = x
        for layer in self.fwd:
            f = layer(f)
        b = torch.flip(x, dims=[1])
        for layer in self.bwd:
            b = layer(b)
        b = torch.flip(b, dims=[1])
        # Mean-pool each direction.
        pooled = torch.cat([f.mean(dim=1), b.mean(dim=1)], dim=-1)
        out = self.head(pooled)
        return out.squeeze(-1) if self.output_dim == 1 else out
```

**Surgery.** Train via `scripts/17_train_mamba.py`. Requires `mamba-ssm` installed.

**Wall.** ~10h for 500 epochs on 4090.

**Risk.** Low (production-ready library) but unclear fit for this puzzle.

---

### B4 — Permutation-equivariant DeepSets variant

**Hypothesis.** Position-invariant by construction: V(π·s) = V(s) for any π ∈ S_120. This is
WRONG for Megaminx (positions matter), but useful as a probe — if it underperforms badly,
confirms position semantics matter.

**Skip in production. Worth it as a one-off ablation.**

---

### B5 — Per-piece additive heuristic (admissible, unlocks A*)

**Hypothesis.** V(s) = Σ_i V_unary(s[i], i). Sum-of-subset is admissible (Korf PDB). Lower
raw capacity but enables true A*/IDA* search (Tier-3 in data plan).

**Concrete arch.** ~2.5M params, very fast. The unary function is a small MLP over
(piece_value, position_id).

```python
import torch
import torch.nn as nn


class AdditiveValue(nn.Module):
    """V(s) = Σ_i V_unary(s[i], i). Naturally admissible if V_unary ≥ 0 trained against
    BFS distances on subsets."""
    def __init__(self, vocab_size: int = 120, n_positions: int = 120, d_hidden: int = 64):
        super().__init__()
        self.value_emb = nn.Embedding(vocab_size, d_hidden)
        self.pos_emb = nn.Embedding(n_positions, d_hidden)
        self.unary = nn.Sequential(
            nn.Linear(2 * d_hidden, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.softplus = nn.Softplus()  # ensure non-negative for admissibility
        self.output_dim = 1

    def forward(self, s):  # s: (B, 120)
        B, N = s.shape
        pos = torch.arange(N, device=s.device).unsqueeze(0).expand(B, -1)
        v = self.value_emb(s.long())  # (B, N, d_hidden)
        p = self.pos_emb(pos)         # (B, N, d_hidden)
        x = torch.cat([v, p], dim=-1)
        per_pos = self.unary(x).squeeze(-1)        # (B, N)
        return self.softplus(per_pos).sum(dim=-1)  # (B,) — non-negative additive sum
```

**Surgery.** Train against BFS-d6 distances first (admissible by construction since BFS is
exact). Then optionally relax to Bellman targets for better sharpness.

**Wall.** ~3h for 500 epochs.

**Risk.** Low capacity — may underfit. Best paired with downstream A* search use.

---

## Tier C — Multi-task auxiliary supervision

### C1 — Bellman consistency aux during RW pretrain

**Hypothesis.** m05's recipe is two-phase: RW pretrain → Bellman fine-tune. Integrating
Bellman from epoch 1 (not just last phase) may break the RW MSE-64 plateau by reshaping
the pretrain loss landscape.

**Surgery.** Modify `02_train.py` (RW pretrain script) to add a secondary loss:

```python
# In the training loop, after computing the primary RW MSE loss:
loss_rw = F.mse_loss(pred, target_walk_depth)

# Add Bellman consistency as auxiliary:
with torch.no_grad():
    children = torch.gather(
        bs.unsqueeze(1).expand(-1, n_gen, -1), 2,
        gens_t.unsqueeze(0).expand(bs.size(0), -1, -1),
    ).reshape(-1, bs.size(1))
    v_child = model(children).flatten().to(torch.float32)  # use SELF, not target net
    bellman_target = 1.0 + v_child.view(bs.size(0), n_gen).min(dim=1).values
loss_bellman_aux = F.mse_loss(pred, bellman_target.detach())

loss = loss_rw + args.bellman_aux_weight * loss_bellman_aux
```

Pass `--bellman-aux-weight 0.1` initially; sweep 0.05 → 0.5.

**Wall.** +40% per epoch (24 extra forwards per batch).

**Acceptance gate.** Cheap-eval RMSE ≤ 7.5 (m07's plateau is ~8). Then run Bellman fine-tune
on this body, gate on strat-5.

**Risk.** Moderate — early Bellman with self-bootstrap can be unstable. Mitigate via small
weight, scheduled increase.

---

### C2 — V + policy multi-task (m24 redesigned with V-head)

**Already covered as T2.C in the data plan.** Cross-reference: the dual-head implementation
in `i-want-you-to-gentle-crayon.md` Sketch — T2.C. Body shared, V trained against Bellman
target, π trained against inverse-walk action labels. Beam scoring becomes
`V(child) + λ·(-log π(a|parent))`.

---

### C3 — V + per-piece distance auxiliary

**Hypothesis.** Cheap aux signal (Hamming distance per piece) provides regularization. Body
shared between V and per-piece head.

**Surgery.** Auxiliary head outputs 120 scalars (one per piece's "distance from solved" — a
proxy via permutation cycle structure). Loss: `MSE_V + 0.05·MSE_per_piece`.

```python
# Assuming a body that exposes forward_features():
class VPlusPerPieceHead(nn.Module):
    def __init__(self, body, body_dim: int = 512, n_pieces: int = 120):
        super().__init__()
        self.body = body
        self.v_head = nn.Linear(body_dim, 1)
        self.piece_head = nn.Linear(body_dim, n_pieces)
        self.output_dim = 1  # for beam compatibility (returns scalar)

    def forward(self, s):
        h = self.body.forward_features(s)
        return self.v_head(h).squeeze(-1)  # beam only sees V

    def forward_with_aux(self, s):
        h = self.body.forward_features(s)
        return self.v_head(h).squeeze(-1), self.piece_head(h)


def per_piece_target(s, solved_state):
    """Per-piece distance: Hamming distance for each position, normalized."""
    return (s != solved_state.unsqueeze(0)).float()  # (B, n_pieces)
```

**Wall.** +5% per epoch.

**Acceptance gate.** Standard.

**Risk.** Low.

---

### C4 — V + cycle-decomposition auxiliary

**Hypothesis.** The cycle structure of a permutation (count of 1-cycles, 2-cycles, etc.)
correlates with distance and is cheap to label. Auxiliary head predicts cycle counts.

```python
def cycle_decomposition_features(s, n_pieces=120):
    """Return (B, max_cycle_len) tensor where bin k counts k-cycles in s.
    Slow CPU path; precompute as labels for the entire training set."""
    out = []
    for row in s.cpu().numpy():
        seen = [False] * n_pieces
        counts = [0] * (n_pieces + 1)
        for i in range(n_pieces):
            if seen[i]: continue
            cur, length = i, 0
            while not seen[cur]:
                seen[cur] = True
                cur = row[cur]
                length += 1
            counts[length] += 1
        out.append(counts)
    return torch.tensor(out, dtype=torch.float32)
```

Aux head outputs cycle-count vector, loss is MSE.

**Wall.** Negligible if labels precomputed once per epoch's data.

**Risk.** Low.

---

## Tier D — Inference-time compute beyond TTT (A3)

### D1 — MCTS within beam nodes (PUCT exploration)

**Hypothesis.** AlphaZero formula: each beam node runs k PUCT simulations using V as the
node-value estimator. Trade beam width × MCTS depth: beam 8k × 32 sims ≈ effective beam 256k
with 1-step lookahead, with potentially better selection on hard nodes.

**Surgery.** New solver class `MCTSBeamSolver` extending `KhoruzhiiSolver`. Per beam node,
maintain `(N, W, Q)` (visit count, total value, mean value). Selection: `argmax_a Q(s,a) + c·sqrt(log N/n_a)`.

```python
import math
import torch


@torch.no_grad()
def puct_simulate(model, puzzle, root_state, n_sims: int, c: float = 1.4,
                  device: str = "cuda") -> dict:
    """Run n_sims PUCT simulations from root_state. Return dict mapping action_idx → mean Q.
    """
    n_gen = len(puzzle.move_names)
    children = [puzzle.apply_move(root_state, puzzle.move_names[a]) for a in range(n_gen)]
    children_t = torch.tensor(children, dtype=torch.int64, device=device)

    # Initialize: V(child) for all children.
    with torch.no_grad():
        v_priors = model(children_t).flatten().to(torch.float32).cpu().numpy()
    visits = [0] * n_gen
    totals = [0.0] * n_gen

    for _ in range(n_sims):
        # PUCT selection.
        n_total = sum(visits) + 1
        scores = [
            -v_priors[a] - (totals[a] / max(visits[a], 1)) +
            c * math.sqrt(math.log(n_total) / (visits[a] + 1))
            for a in range(n_gen)
        ]
        a = max(range(n_gen), key=lambda i: scores[i])
        # Rollout: one step + V(grandchild).
        gc = puzzle.apply_move(children[a], puzzle.move_names[
            min(range(n_gen), key=lambda j: v_priors[j])  # greedy continuation
        ])
        gc_t = torch.tensor([gc], dtype=torch.int64, device=device)
        v_gc = float(model(gc_t).flatten().to(torch.float32).item())
        visits[a] += 1
        totals[a] += -v_gc  # higher = better for backup

    return {a: totals[a] / max(visits[a], 1) for a in range(n_gen)}
```

In beam loop, replace `argmin_a model(child)` with `argmax_a puct_simulate(...)`.

**Wall.** k× single-step cost. With k=32, beam shrinks 32× to compensate.

**Acceptance gate.** Same total wall as current beam-131k baseline. Quality must beat.

**Risk.** Medium-large. Code complexity. Tuning c.

---

### D2 — Looped iterative refinement (V_0 → V_K)

**Hypothesis.** Apply V K times, treating each pass as a denoising step. Score = mean or
max across passes. Same network, no retraining.

**Surgery.** Trivial: `score = sum(model(child) for _ in range(K)) / K`. The trick is making
each call DIFFERENT — needs noise injection or dropout-at-inference (turn dropout on).

```python
def looped_v(model, child_states, K: int = 3, mc_dropout: bool = True):
    """Mean V over K stochastic forward passes (MC-dropout for randomness)."""
    if mc_dropout:
        model.train()  # enables dropout; CAREFUL — also enables BN training mode
    accum = torch.zeros(child_states.size(0), dtype=torch.float32,
                        device=child_states.device)
    for _ in range(K):
        accum = accum + model(child_states).flatten().to(torch.float32)
    if mc_dropout:
        model.eval()
    return accum / K
```

**Wall.** K× per beam step.

**Risk.** Medium — requires dropout in the model or some other randomness source. m05 likely
has minimal dropout.

---

### D3 — Per-puzzle LoRA fine-tune (parameter-efficient TTT)

**Hypothesis.** Like A3 but instead of full-model adaptation, attach LoRA adapters (rank
4-8). ~50K trainable params per puzzle. Faster, lower memory.

**Surgery.** Use `peft` library:

```python
from peft import LoraConfig, get_peft_model
import torch


def attach_lora(model, rank: int = 4):
    """Wrap model with LoRA adapters on linear layers."""
    cfg = LoraConfig(r=rank, lora_alpha=16, target_modules="all-linear",
                     lora_dropout=0.0, bias="none")
    return get_peft_model(model, cfg)


def adapt_with_lora(base_model, puzzle, initial_state, n_steps: int = 30,
                    lr: float = 5e-4, device: str = "cuda"):
    """Per-puzzle LoRA adapt. Returns the adapted PEFT model. Reset adapters between puzzles."""
    peft_model = attach_lora(copy.deepcopy(base_model), rank=4)
    # Identical TTT loop to A3 but optimizer touches only LoRA params (peft handles it).
    optim = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    # ... same data gen + Bellman loss as A3 ...
    return peft_model
```

**Wall.** ~1-1.5s/puzzle (faster than A3).

**Risk.** Medium — LoRA rank tuning, may underfit at rank 4.

---

### D4 — Best-of-N + V verifier

**Hypothesis.** Solve puzzle K times with stochastic policy (e.g. via dropout, Gumbel-top-k
beam, or different seeds), use V to verify each by `verify_path`, pick shortest.

**Subsumed by data-plan T1.B (multi-seed beam ensemble).** Cross-reference there.

---

### D5 — Adaptive computation time / adaptive beam width

**Hypothesis.** Model uncertainty signals difficulty. Allocate more beam width to harder
puzzles and less to easier ones, keeping total wall constant.

**Surgery.** First-pass solve at beam B0. Track (V at start, V at solve). Puzzles with
high V_start and many beam steps get retried at beam 4·B0.

```python
def adaptive_beam_solve(puzzle, solver, state, base_beam: int = 65536,
                         max_beam: int = 524288, threshold_steps: int = 80):
    cfg = KhoruzhiiSearchConfig(beam_width=base_beam, num_steps=150, num_attempts=1)
    found, plen, names = solver.solve(state, cfg)
    if found and plen < threshold_steps:
        return names
    # Hard puzzle: escalate.
    cfg2 = KhoruzhiiSearchConfig(beam_width=max_beam, num_steps=200, num_attempts=1)
    found2, plen2, names2 = solver.solve(state, cfg2)
    if found2 and (not found or plen2 < plen):
        return names2
    return names if found else None
```

**Wall.** ~1.5× on average. Harder puzzles get more compute.

**Risk.** Low. Already partially implemented in `--beams 16384,65536` escalation in
`03_solve.py`.

---

## Tier E — Optimization regime experiments

### E1 — SWA + LR-restart snapshot ensemble

Combine A5 with periodic LR resets during training. Train m27 (or any new model) with cosine
schedule + 5 warm restarts (every 100 epochs). Save snapshot at each restart's nadir. Average
the K snapshots via `swa_average_checkpoints`.

```yaml
training:
  scheduler: cosine_warm_restart
  lr_restart_epochs: [100, 200, 300, 400, 500]
  lr: 5.0e-4
  lr_min: 1.0e-5
```

(Requires `lr_scheduler.CosineAnnealingWarmRestarts` in trainer.)

### E2 — SAM / Lion / Lookahead optimizers

Muon failed. Same plateau probably swallows SAM and Lion. Low priority.

### E3 — Curriculum on k_max (data-curriculum)

Start `k_max=20` walks (cheaper, less label noise) for first 100 epochs, ramp to 80 by
epoch 300. Pre-rejected as part of "C5 1/k curriculum weighting" in IHES (E7: −512 single-solve
hurt). Skip.

---

## Files-to-write summary

| File | Purpose | Tier |
|---|---|---|
| `scripts/19_train_listwise.py` | A1 listwise rank loss training (clones 02_train.py) | A |
| `scripts/c51_head.py` (lib helper) | A2 distributional V head + projection helpers | A |
| `scripts/20_solve_ttt.py` (or `--ttt-steps` arg in 03_solve.py) | A3 TTT integration | A |
| `scripts/21_train_specialist.py` | A4 per-bucket specialist trainer (filters by k_max range) | A |
| `scripts/22_route_specialist.py` | A4 inference-time bucket routing wrapper | A |
| `scripts/swa_avg.py` | A5 weight averaging utility | A |
| `scripts/15_train_transformer.py` + `configs/m29_transformer.yaml` | B1 transformer | B |
| `scripts/16_train_gnn.py` + `configs/m30_gnn.yaml` | B2 GNN | B |
| `scripts/17_train_mamba.py` + `configs/m31_mamba.yaml` | B3 Mamba | B |
| `scripts/23_train_additive.py` | B5 per-piece additive | B |
| `scripts/24_train_bellman_aux.py` | C1 Bellman aux during RW | C |
| `scripts/25_train_v_pi.py` | C2 V + policy (alias for T2.C) | C |
| `scripts/26_train_v_piece.py` | C3 V + per-piece aux | C |
| `scripts/27_train_v_cycle.py` | C4 V + cycle decomp aux | C |
| `scripts/28_solve_mcts.py` | D1 MCTS-augmented beam | D |
| `scripts/29_solve_loop.py` | D2 looped V refinement | D |
| `scripts/30_solve_lora_ttt.py` | D3 LoRA per-puzzle TTT | D |

---

## Recommended sequencing (capacity axis)

This complements the data-axis sequencing in `i-want-you-to-gentle-crayon.md`. Run in
parallel — they share no model artifacts until late.

**Week 1 (capacity probes):**
- A5 SWA average (30 min) → strat-5 immediately. Free shot.
- A3 TTT integration (3h code + full submission eval, ~24h compute). High signal.
- A1 listwise loss training (~6h training + strat-5). Start in parallel.

**Week 1-2:**
- A2 distributional V (C51) — needs body to expose features. ~1d code + ~1d training.
- B1 transformer m29 — ~1d code + ~12h training on 4090.

**Week 2-3 (commit to one Tier-B winner):**
- Pick the winner of A1 / A2 / B1 and start ensembling with m05.
- A4 per-bucket specialists — only if Tier-A levers stall. ~5 days of training across 10 specialists.

**Week 3+ (research / Tier D):**
- D1 MCTS-augmented beam — if all of A1/A2/B1 pass and we want stack-on-top.
- B2 GNN / B3 Mamba — feasibility spikes if main retrains have all settled.

---

## What NOT to do (capacity axis)

- **Scale ResMLPDistance further at the same Bellman recipe.** m26/m26b proved this is dead.
- **Icosahedral-equivariant networks.** Blocked on the same group derivation as sym v3.
- **Set Transformer / DeepSets pure variant.** Position semantics matter; provably-permutation-invariant nets discard signal we need.
- **MoE with soft routing on the SAME loss/recipe.** Each expert hits the same plateau.
- **Energy-based models.** Adds partition-function complexity for unclear capacity gain.
- **SAM / Lion / Lookahead.** Same reason Muon failed.
- **Curriculum on k_max.** Already rejected in IHES.
- **Hyena / NFNets / ViT 1D.** Speculative architectures with no specific reason to beat m05.

---

## Acceptance gates (capacity axis)

Same as data-axis: ≥+3 strat-5 solves AND mean model_avg ≤ 0.95× current best AND no bucket
regresses by more than 1 solve.

Two exceptions:

1. **A3 TTT** — strat-5 understates the value because TTT helps most on hard puzzles in the
tail. Run a full 1001-puzzle submission as the gate.

2. **A4 per-bucket specialists** — gate per bucket, not aggregate. A specialist must beat m05
on ITS bucket's strat-5 subset. Aggregate gate: total over 1001 puzzles must beat 88,195.

---

## Verification end-to-end

For Tier A and B retrains, the canonical proof is the same:

```bash
# Strat-5 eval on the new model:
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint <new_ckpt> --out <out.csv> \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 --bf16
```

For A3 TTT (full submission gate):

```bash
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint models/m05_bellman_warm/epoch_0499.pt \
    --qshort-student models/m23_q_shortlister/epoch_0499.pt \
    --qshort-alpha 2 \
    --ttt-steps 100 \
    --out submissions/_ttt_full.csv \
    --beams 131072 --max-steps 150 --bf16
# Compare submissions/_ttt_full.csv total_moves vs current 88,195.
```

For A5 SWA:

```bash
.venv/Scripts/python.exe scripts/swa_avg.py
.venv/Scripts/python.exe scripts/03_solve.py \
    --checkpoint models/m05_swa_5/epoch_0499.pt \
    --out submissions/_swa_strat5.csv \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 --bf16
```

---

## Final note: orthogonality with data-axis plan

These capacity-axis levers are **stackable** with the data-axis plan in `i-want-you-to-gentle-crayon.md`:

- A5 SWA can be applied to the m27 BFS-d6-mixin model from T1.A → "m27_swa".
- A1 listwise loss can be combined with T1.A's BFS-d6 mixin into a single trainer.
- A3 TTT can be applied at inference on top of any of the data-plan models.
- A4 specialists can be trained with T1.A's BFS-d6 mixin built into each.
- B1/B2/B3 are alternative bodies for any data-axis loss.
- C1 Bellman aux can be combined with T1.A's mixin to give a 3-target loss.

The 70K target probably needs **both axes** working — pick the winner of each, stack them
late.
