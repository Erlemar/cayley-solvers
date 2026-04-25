"""m17: Bellman refinement round 2 — warmstart from m05 (which was Bellman r1 of m07).

Tests Noisy-Student-style iterative refinement. Round 1 (m05) saw a sharper target than
m07 because Bellman targets exceeded walk-depth in many places. Round 2 (m17) sees a
sharper target than m05 for the same reason.

  m07 (RW-trained)  ──Bellman──>  m05  (round 1, MSE-via-Bellman 0.097)
  m05  (round 1)    ──Bellman──>  m17  (round 2, this kernel)

Same arch [2048,512]x2 as m07/m05. 500 epochs. lr 5e-4. Target-net refresh every 10 ep.
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


# Mount: cayley-megaminx-snapshot (code+data) + megaminx-m05-bellman-warm (m05 ckpt).
DATA_IN = None
for cand in [Path("/kaggle/input/cayley-megaminx-snapshot"),
             Path("/kaggle/input/datasets/artgor/cayley-megaminx-snapshot")]:
    if cand.exists():
        DATA_IN = cand; break
if DATA_IN is None:
    raise SystemExit("dataset not found")
print(f"data: {DATA_IN}", flush=True)

M05_IN = None
for cand in [Path("/kaggle/input/megaminx-m05-bellman-warm"),
             Path("/kaggle/input/datasets/artgor/megaminx-m05-bellman-warm")]:
    if cand.exists():
        M05_IN = cand; break
if M05_IN is None:
    raise SystemExit("m05 dataset not found")
print(f"m05: {M05_IN}", flush=True)

CAY_SRC = DATA_IN / "cayley" / "src"
MEG_SRC = DATA_IN / "megaminx" / "src"
MEG_DATA = DATA_IN / "megaminx" / "data"
sys.path.insert(0, str(CAY_SRC))
sys.path.insert(0, str(MEG_SRC))


import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
      flush=True)

from cayley.bellman import BellmanConfig, train_bellman
from cayley.model import ResMLPDistance
from cayley.training import TrainConfig
from megaminx.puzzle import Megaminx


M05_CKPT = M05_IN / "m05_epoch_0499.pt"
print(f"warmstart: {M05_CKPT} ({M05_CKPT.stat().st_size / 1e6:.1f} MB)", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
puzzle = Megaminx.load(MEG_DATA / "puzzle_info.json")

model = ResMLPDistance(
    state_size=120, num_classes=120,
    hidden_dims=(2048, 512), num_res_blocks=2,
    encoding="embedding", embed_dim=16,
)

tc = TrainConfig(
    n_epochs=500,
    samples_per_epoch=500_000,
    batch_size=8192,
    k_max=80,
    lr=5e-4,
    weight_decay=0.0,
    device=device,
    seed=170,
    log_every_batches=200,
    checkpoint_every_epochs=50,
    n_back=1,
    loss="mse",
    curriculum=False,
    amp=True,
    compile_model=False,        # P100 — eager only
    fused_optimizer=True,
)
bcfg = BellmanConfig(
    warmstart_path=str(M05_CKPT),
    target_update_every_epochs=10,
    target_net_chunk=4096,
    clip_upper=True,
    clip_lower=True,
)


def _log(stats):
    print(f"epoch {stats.epoch:4d} | loss {stats.loss:.4f} | "
          f"lr {stats.lr:.2e} | {stats.elapsed_s:.1f}s", flush=True)


OUT = Path("/kaggle/working/m17_bellman_r2_from_m05")
OUT.mkdir(parents=True, exist_ok=True)
result = train_bellman(model, puzzle, tc, bcfg, checkpoint_dir=OUT, on_epoch_end=_log)
print(f"final loss: {result.final_loss:.4f}", flush=True)
print(f"total wall: {time.time() - t_start:.1f}s", flush=True)
