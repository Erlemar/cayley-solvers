"""m37 — Train V on solver-trace as the PRIMARY signal.

Tests whether the training/inference distribution mismatch is the binding
constraint on the cluster ceiling. Pure MSE on the 128K (state, true_remaining_d)
pairs mined from prior verified submissions. Warmstart from m07.

Inputs (Kaggle datasets):
  - cayley-megaminx-snapshot: source code (cayley/, megaminx/) + puzzle data
  - megaminx-m37-extras: m07 checkpoint + solver_trace_train.pt
"""
import os, sys, subprocess, time
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

from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


# ---- Hyperparams ----
EPOCHS = 500
BATCH_SIZE = 8192
LR = 5e-4
SEED = 37
WARMSTART = EXTRAS_IN / "m07_epoch_3999.pt"
DATASET = EXTRAS_IN / "solver_trace_train.pt"
OUT = Path("/kaggle/working/m37_solver_trace")
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_EVERY = 50

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---- Load dataset ----
print(f"loading dataset: {DATASET}", flush=True)
d = torch.load(DATASET, weights_only=False)
states = d["states"]      # (N, 120) int8
distances = d["distances"]  # (N,) int8
n_total = states.size(0)
print(f"  states: {tuple(states.shape)}, distances: {tuple(distances.shape)}", flush=True)
print(f"  distance: mean={distances.float().mean():.2f}, "
      f"min={distances.min().item()}, max={distances.max().item()}", flush=True)

states = states.to(device).long()
distances = distances.to(device).float()


# ---- Build model ----
puzzle = Megaminx.load(MEG_DATA / "puzzle_info.json")
model = ResMLPDistance(
    state_size=120, num_classes=120,
    hidden_dims=(2048, 512), num_res_blocks=2,
    encoding="embedding", embed_dim=16,
    output_dim=1,
).to(device)
print(f"model params: {model.num_parameters():,}", flush=True)

# Warmstart from m07
print(f"warmstart from {WARMSTART}", flush=True)
ckpt = torch.load(WARMSTART, map_location=device, weights_only=False)
sd = ckpt["state_dict"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
model.load_state_dict(sd)


# ---- Training ----
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0,
                           fused=(device == "cuda"))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

# P100 supports bf16 in PyTorch 2.4+; use it
autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.amp.autocast("cuda", enabled=False)

n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\ntraining: {EPOCHS} epochs x {n_batches} batches "
      f"({n_total:,} samples / {BATCH_SIZE} batch)", flush=True)

t0 = time.time()
for epoch in range(EPOCHS):
    ep_t0 = time.time()
    model.train()
    perm = torch.randperm(n_total, device=device)
    total_loss = 0.0
    for b in range(n_batches):
        idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        bs = states[idx]
        bd = distances[idx]
        with autocast_ctx:
            pred = model(bs).flatten()
            loss = F.mse_loss(pred, bd)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += float(loss.item())
    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  epoch {epoch:4d} | loss {avg_loss:.5f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{time.time()-ep_t0:.1f}s", flush=True)

    if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt_path = OUT / f"epoch_{epoch:04d}.pt"
        sd = model.state_dict()
        torch.save({
            "epoch": epoch,
            "state_dict": sd,
            "loss": avg_loss,
            "model_config": {
                "state_size": 120, "num_classes": 120,
                "hidden_dims": [2048, 512], "num_res_blocks": 2,
                "encoding": "embedding", "embed_dim": 16, "output_dim": 1,
            },
            "training_config": {
                "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "seed": SEED,
                "warmstart": "m07_epoch_3999",
                "data": "solver_trace_train (128K pairs)",
                "loss": "mse",
            },
        }, ckpt_path)

print(f"\nfinal loss: {avg_loss:.5f}", flush=True)
print(f"total wall: {time.time() - t_start:.1f}s", flush=True)
