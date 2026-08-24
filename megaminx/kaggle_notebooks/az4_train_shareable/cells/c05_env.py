### Environment: device, precision, compile ###
import contextlib
import time

import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| device:', DEVICE)
if DEVICE == 'cuda':
    print(' ', torch.cuda.get_device_name(0))
    _CAP = torch.cuda.get_device_capability()
    # Recent Kaggle images ship torch builds WITHOUT Pascal (sm_60 / P100) kernels -
    # every CUDA op then fails with "no kernel image is available". Probe once and
    # fail with a clear message instead of crashing mid-training.
    try:
        (torch.ones(2, device='cuda') + 1).sum().item()
    except Exception as e:
        raise RuntimeError(
            f'This torch build cannot run on {torch.cuda.get_device_name(0)} '
            f'(capability {_CAP}). On Kaggle: Settings -> Accelerator -> GPU T4 x2, '
            f'then re-run. Original error: {e}') from None
else:
    _CAP = (0, 0)


def _resolve_precision() -> str:
    p = CFG['precision']
    if p != 'auto':
        return p
    # bf16 only on Ampere+ (P100 has no bf16; T4 emulates it ~40x slower).
    if DEVICE == 'cuda' and _CAP[0] >= 8 and torch.cuda.is_bf16_supported():
        return 'bf16'
    return 'off'


PRECISION = _resolve_precision()
COMPILE = CFG['compile'] if CFG['compile'] != 'auto' else (DEVICE == 'cuda' and _CAP[0] >= 8)
# Fused AdamW: safe on Volta+; skip on Pascal (P100).
FUSED_OK = DEVICE == 'cuda' and _CAP[0] >= 7
print(f'precision: {PRECISION} | compile: {COMPILE} | fused optimizer: {FUSED_OK}')


def make_autocast():
    if PRECISION == 'bf16' and DEVICE == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_optimizer(params, stage_cfg):
    """Name-based optimizer factory (same pattern as the community baselines notebook).

    The learning rate always comes from the stage's 'lr' key; extra constructor kwargs
    from stage_cfg['optimizer']['params'] (e.g. {'weight_decay': 0.01, 'betas': ...}).
    """
    opt_cfg = stage_cfg.get('optimizer') or {'name': 'AdamW', 'params': {}}
    name = opt_cfg.get('name') or 'AdamW'
    cls = getattr(torch.optim, name, None)
    if cls is None:
        raise ValueError(f'unknown optimizer {name!r} (expected a torch.optim name)')
    kw = dict(opt_cfg.get('params') or {})
    kw.setdefault('weight_decay', 0.0)
    if name in ('AdamW', 'Adam') and FUSED_OK:
        kw.setdefault('fused', True)
    return cls(params, lr=stage_cfg['lr'], **kw)


def maybe_compile(model):
    if COMPILE:
        return torch.compile(model, dynamic=False)
    return model


torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])

# Wall-clock budget: training stops gracefully at TRAIN_DEADLINE so the run COMPLETES
# (a Kaggle run killed by the session limit saves no output at all).
NOTEBOOK_T0 = time.time()
TRAIN_DEADLINE = (NOTEBOOK_T0 + CFG['max_wall_hours'] * 3600
                  - CFG['eval_reserve_minutes'] * 60)


def training_time_up():
    return time.time() >= TRAIN_DEADLINE


print(f"wall budget: {CFG['max_wall_hours']}h "
      f"(training stops {CFG['eval_reserve_minutes']}min early for eval + saving)")
