"""m18: Transformer encoder predictor for Megaminx distance regression.

CayleyPy-RL paper says MLPs beat Transformers on permutation Cayley graphs (n>15).
We have n=120, so this is squarely in their "Transformer fails" regime — but the user
wants the empirical confirmation. If m18 hits MSE in the same range as m07 (~64) it's a
viable candidate; if it fails to converge or plateaus much higher, the paper claim is
re-confirmed for our specific setting.

Architecture:
  - Token embed: nn.Embedding(120, d=128) per sticker value
  - Positional embed: nn.Embedding(120, d=128) per position (states are length-120
    sequences with position-dependent meaning)
  - TransformerEncoder: 4 layers, 8 heads, d_model=128, ff=512
  - Pool: mean across positions
  - Head: Linear(128, 1)

Target: walk-depth labels (k_max=80), MSE loss. Same dataset pipeline as m07.
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


DATA_IN = None
for cand in [Path("/kaggle/input/cayley-megaminx-snapshot"),
             Path("/kaggle/input/datasets/artgor/cayley-megaminx-snapshot")]:
    if cand.exists():
        DATA_IN = cand; break
if DATA_IN is None:
    raise SystemExit("dataset not found")
print(f"data: {DATA_IN}", flush=True)

CAY_SRC = DATA_IN / "cayley" / "src"
MEG_SRC = DATA_IN / "megaminx" / "src"
MEG_DATA = DATA_IN / "megaminx" / "data"
sys.path.insert(0, str(CAY_SRC))
sys.path.insert(0, str(MEG_SRC))


import torch
import torch.nn as nn
import torch.nn.functional as F
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

from cayley.data import generate_walks_torch
from megaminx.puzzle import Megaminx


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------
STATE_SIZE = 120
NUM_CLASSES = 120
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 4
D_FF = 512


class TransformerDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(NUM_CLASSES, D_MODEL)
        self.pos_emb = nn.Embedding(STATE_SIZE, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF,
            dropout=0.0, activation="relu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, 1)
        # Cache positional indices
        self.register_buffer("positions", torch.arange(STATE_SIZE).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 120) int64
        B = x.shape[0]
        tok = self.token_emb(x.long())                                         # (B, 120, D)
        pos = self.pos_emb(self.positions.expand(B, STATE_SIZE))               # (B, 120, D)
        h = tok + pos
        h = self.encoder(h)                                                    # (B, 120, D)
        h = h.mean(dim=1)                                                      # (B, D)
        return self.head(h).squeeze(-1)                                        # (B,)


# Hyperparams (mirror m07's "fast recipe" where applicable)
N_EPOCHS = 1000
SAMPLES_PER_EPOCH = 500_000
BATCH_SIZE = 4096               # smaller than MLP because attention is heavier
K_MAX = 80
LR = 5e-4                       # slightly lower than 2e-3 because Transformer is noisier
SEED = 180


device = "cuda" if torch.cuda.is_available() else "cpu"
puzzle = Megaminx.load(MEG_DATA / "puzzle_info.json")

model = TransformerDistance().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"transformer params: {n_params:,}", flush=True)

optim = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=0.01, fused=True if device == "cuda" else False
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS)

OUT = Path("/kaggle/working/m18_transformer")
OUT.mkdir(parents=True, exist_ok=True)
gen = torch.Generator(device=device).manual_seed(SEED)


for epoch in range(N_EPOCHS):
    t_ep = time.time()
    n_walks = max(1, SAMPLES_PER_EPOCH // K_MAX)
    states, depths = generate_walks_torch(
        puzzle, n_walks=n_walks, k_max=K_MAX,
        seed=SEED + epoch, device=device, n_back=1,
    )

    model.train()
    N = states.shape[0]
    perm = torch.randperm(N, generator=gen, device=device)
    total_loss, n_batches = 0.0, 0
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    )
    for i in range(0, N, BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        if autocast_ctx is not None:
            with autocast_ctx:
                pred = model(states[idx])
                loss = F.mse_loss(pred, depths[idx].float())
        else:
            pred = model(states[idx])
            loss = F.mse_loss(pred, depths[idx].float())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += float(loss.item())
        n_batches += 1
    scheduler.step()
    avg = total_loss / max(n_batches, 1)
    print(f"epoch {epoch:4d} | loss {avg:.4f} | "
          f"lr {float(scheduler.get_last_lr()[0]):.2e} | {time.time() - t_ep:.1f}s",
          flush=True)

    if (epoch + 1) % 100 == 0 or epoch == N_EPOCHS - 1:
        ckpt = OUT / f"epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": avg,
            "model_config": {
                "kind": "transformer",
                "state_size": STATE_SIZE, "num_classes": NUM_CLASSES,
                "d_model": D_MODEL, "n_head": N_HEAD,
                "n_layers": N_LAYERS, "d_ff": D_FF,
            },
        }, ckpt)
        print(f"  saved {ckpt.name}", flush=True)


print(f"\ntotal wall: {time.time() - t_start:.1f}s", flush=True)
