"""m38 — Train V with listwise rank loss as auxiliary objective.

Beam search uses ONLY relative ordering of children at each step. m38 trains
the model so V's ordering of children matches the true-distance ordering,
even when absolute scale is wrong. Tests "is the wrong objective the binding
constraint on cluster ceiling?"

Total loss = MSE(parent V, Bellman target) + lambda * ListNet(child Vs, child Bellman targets)

Inputs:
  - cayley-megaminx-snapshot: source code + puzzle data
  - megaminx-m37-extras: m07 warmstart (reused — same checkpoint as m37)
"""
import os, sys, subprocess, time, copy, math
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, x):
        for s in self.streams:
            try: s.write(x); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass


_LOG = open("/kaggle/working/run.log", "w")
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)
t_start = time.time()


def _run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


_run([sys.executable, "-m", "pip", "install", "-q",
      "torch==2.4.1", "torchvision==0.19.1", "cayleypy", "pyyaml"])


# ---- Mount inputs ----
DATA_IN = None
for cand in [Path("/kaggle/input/cayley-megaminx-snapshot"),
             Path("/kaggle/input/datasets/artgor/cayley-megaminx-snapshot")]:
    if cand.exists():
        DATA_IN = cand; break
if DATA_IN is None:
    raise SystemExit("cayley-megaminx-snapshot dataset not mounted")
print(f"data: {DATA_IN}", flush=True)

EXTRAS_IN = None
for cand in [Path("/kaggle/input/megaminx-m37-extras"),
             Path("/kaggle/input/datasets/artgor/megaminx-m37-extras")]:
    if cand.exists():
        EXTRAS_IN = cand; break
if EXTRAS_IN is None:
    raise SystemExit("megaminx-m37-extras dataset not mounted")
print(f"extras: {EXTRAS_IN}", flush=True)

CAY_SRC = DATA_IN / "cayley" / "src"
MEG_SRC = DATA_IN / "megaminx" / "src"
MEG_DATA = DATA_IN / "megaminx" / "data"
sys.path.insert(0, str(CAY_SRC))
sys.path.insert(0, str(MEG_SRC))


import torch
import torch.nn.functional as F
import numpy as np
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
      flush=True)

from cayley.bellman import _apply_all_generators
from cayley.data import GeneratorTable, generate_walks_torch
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


# ---- Inline listwise loss + bellman targets (the snapshot dataset's bellman.py
# is older and doesn't have softmin_temperature; use a local copy to avoid the
# unexpected-keyword-argument error.) ----
def listnet_loss(pred, target, temperature=1.0):
    """ListNet cross-entropy. Smaller V -> closer to solved -> larger weight."""
    p_pred = F.softmax(-pred / temperature, dim=1)
    p_tgt = F.softmax(-target / temperature, dim=1)
    eps = 1e-9
    return -(p_tgt * torch.log(p_pred + eps)).sum(dim=1).mean()


@torch.no_grad()
def _bellman_targets(target_model, states, walk_depths, generators, solved_state,
                     chunk_size, clip_upper, clip_lower):
    """Hard-min Bellman target: y = clip(1 + min_a target(apply(s, a)), 0, walk_depth).
    Inline copy to avoid version mismatch with snapshot dataset's bellman.py."""
    B, S = states.shape
    children = _apply_all_generators(states, generators)  # (B, n_gen, S)
    n_gen = children.shape[1]
    children_flat = children.reshape(B * n_gen, S)
    is_solved = (children_flat == solved_state).all(dim=1)
    child_values = torch.empty(B * n_gen, dtype=torch.float32, device=states.device)
    target_model.eval()
    for i in range(0, B * n_gen, chunk_size):
        vals = target_model(children_flat[i : i + chunk_size]).flatten().to(torch.float32)
        child_values[i : i + chunk_size] = vals
    child_values = torch.where(is_solved, torch.zeros_like(child_values), child_values)
    child_values = child_values.view(B, n_gen)
    reduced = child_values.min(dim=1).values
    target = 1.0 + reduced
    if clip_upper:
        target = torch.minimum(target, walk_depths)
    if clip_lower:
        target = torch.clamp(target, min=0.0)
    return target


# ---- Hyperparams ----
EPOCHS = 200                        # Reduced from 500 to fit Kaggle 9h kernel slot
SAMPLES_PER_EPOCH = 500_000
BATCH_SIZE = 4096                   # Smaller because each batch has 24x more child forwards
K_MAX = 80
LR = 5e-4
N_BACK = 1
SEED = 38
LISTWISE_LAMBDA = 0.5
LISTWISE_TEMPERATURE = 1.0
LISTWISE_WARMUP_EPOCHS = 50
TARGET_NET_CHUNK = 4096
TARGET_UPDATE_EVERY_EPOCHS = 10
WARMSTART = EXTRAS_IN / "m07_epoch_3999.pt"
OUT = Path("/kaggle/working/m38_listwise")
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_EVERY = 25

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model + warmstart ----
puzzle = Megaminx.load(MEG_DATA / "puzzle_info.json")
model = ResMLPDistance(
    state_size=120, num_classes=120,
    hidden_dims=(2048, 512), num_res_blocks=2,
    encoding="embedding", embed_dim=16,
    output_dim=1,
).to(device)
print(f"model params: {model.num_parameters():,}", flush=True)

print(f"warmstart from {WARMSTART}", flush=True)
ckpt = torch.load(WARMSTART, map_location=device, weights_only=False)
sd = ckpt["state_dict"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
model.load_state_dict(sd)

target_model = copy.deepcopy(model).eval()
for p in target_model.parameters():
    p.requires_grad = False

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0,
                           fused=(device == "cuda"))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

gens_table = GeneratorTable.from_puzzle(puzzle)
generators = torch.from_numpy(gens_table.perms).to(device)
solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)

autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.amp.autocast("cuda", enabled=False)

n_walks = max(1, SAMPLES_PER_EPOCH // K_MAX)

print(f"\ntraining: {EPOCHS} ep, batch {BATCH_SIZE}, "
      f"listwise lambda {LISTWISE_LAMBDA} (ListNet T={LISTWISE_TEMPERATURE})", flush=True)

t0 = time.time()
last_avg_mse = float("inf")
last_avg_lw = 0.0
for epoch in range(EPOCHS):
    ep_t0 = time.time()
    lam = LISTWISE_LAMBDA * min(1.0, epoch / max(LISTWISE_WARMUP_EPOCHS, 1))

    states, depths = generate_walks_torch(
        puzzle, n_walks=n_walks, k_max=K_MAX, seed=SEED + epoch,
        device=device, n_back=N_BACK,
    )
    depths_f = depths.to(torch.float32)

    model.train()
    perm = torch.randperm(states.size(0), device=device)
    total_mse = 0.0
    total_lw = 0.0
    n_batches = 0

    for b in range(0, states.size(0), BATCH_SIZE):
        idx = perm[b : b + BATCH_SIZE]
        bs = states[idx]
        bd = depths_f[idx]

        # Bellman target on parent
        bell_target = _bellman_targets(
            target_model, bs, bd, generators, solved_state,
            chunk_size=TARGET_NET_CHUNK,
            clip_upper=True, clip_lower=True,
        )

        # Children for listwise
        B, S = bs.shape
        n_gen = generators.shape[0]
        children = _apply_all_generators(bs, generators)  # (B, n_gen, S)
        children_flat = children.reshape(B * n_gen, S)

        with torch.no_grad():
            target_model.eval()
            child_q = torch.empty(B * n_gen, dtype=torch.float32, device=bs.device)
            for i in range(0, B * n_gen, TARGET_NET_CHUNK):
                vals = target_model(children_flat[i : i + TARGET_NET_CHUNK]).flatten().to(torch.float32)
                child_q[i : i + TARGET_NET_CHUNK] = vals
            child_q = child_q.view(B, n_gen)
            is_solved = (children_flat == solved_state).all(dim=1).view(B, n_gen)
            child_q = torch.where(is_solved, torch.zeros_like(child_q), child_q)

        with autocast_ctx:
            pred = model(bs).flatten()
            mse_loss = F.mse_loss(pred, bell_target)

            if lam > 0:
                pred_children = model(children_flat).view(B, n_gen).float()
                lw_loss = listnet_loss(pred_children, child_q, temperature=LISTWISE_TEMPERATURE)
                total_loss = mse_loss + lam * lw_loss
            else:
                lw_loss = torch.tensor(0.0, device=bs.device)
                total_loss = mse_loss

        optim.zero_grad(set_to_none=True)
        total_loss.backward()
        optim.step()
        total_mse += float(mse_loss.item())
        total_lw += float(lw_loss.item()) if lam > 0 else 0
        n_batches += 1

    scheduler.step()
    last_avg_mse = total_mse / max(n_batches, 1)
    last_avg_lw = total_lw / max(n_batches, 1)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  epoch {epoch:4d} | mse {last_avg_mse:.4f} | listwise {last_avg_lw:.4f} | "
              f"lambda {lam:.3f} | lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{time.time()-ep_t0:.1f}s", flush=True)

    if (epoch + 1) % TARGET_UPDATE_EVERY_EPOCHS == 0:
        src_sd = model.state_dict()
        if any(k.startswith("_orig_mod.") for k in src_sd):
            src_sd = {k.removeprefix("_orig_mod."): v for k, v in src_sd.items()}
        target_model.load_state_dict(src_sd)

    if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt_path = OUT / f"epoch_{epoch:04d}.pt"
        sd = model.state_dict()
        torch.save({
            "epoch": epoch,
            "state_dict": sd,
            "loss": last_avg_mse,
            "listwise_loss": last_avg_lw,
            "model_config": {
                "state_size": 120, "num_classes": 120,
                "hidden_dims": [2048, 512], "num_res_blocks": 2,
                "encoding": "embedding", "embed_dim": 16, "output_dim": 1,
            },
            "training_config": {
                "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                "listwise_mode": "listnet", "listwise_lambda": LISTWISE_LAMBDA,
                "listwise_temperature": LISTWISE_TEMPERATURE,
                "listwise_warmup_epochs": LISTWISE_WARMUP_EPOCHS,
                "warmstart": "m07_epoch_3999",
            },
        }, ckpt_path)

print(f"\nfinal mse: {last_avg_mse:.5f}, listwise: {last_avg_lw:.5f}", flush=True)
print(f"total wall: {time.time() - t_start:.1f}s", flush=True)
