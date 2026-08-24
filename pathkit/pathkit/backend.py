"""Array backend shim so every algorithm here runs unchanged on CPU (numpy) or GPU (torch).

The ball-collision methods are the only expensive ones, and they are expensive in the
same way on both backends: a few very large flat arrays plus one sort. Rather than write
each method twice, everything below is expressed against ~10 primitives whose numpy and
torch spellings differ, and each backend supplies those.

Pick a backend with `get_backend("cpu")` or `get_backend("cuda")`. `cuda` silently falls
back to `cpu` when torch is missing or no GPU is visible -- a missing GPU should slow a
run down, not fail it.
"""
from __future__ import annotations

import numpy as np

try:                                    # torch is optional
    import torch
    _HAS_TORCH = True
except Exception:                       # pragma: no cover - environment dependent
    torch = None
    _HAS_TORCH = False


class NumpyBackend:
    name = "cpu"
    is_torch = False

    def __init__(self, device: str = "cpu"):
        self.device = "cpu"
        self.int8 = np.int8
        self.int32 = np.int32
        self.int64 = np.int64
        self.bool_ = np.bool_

    # --- construction -------------------------------------------------------
    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype=dtype)

    def to_numpy(self, a):
        return np.asarray(a)

    def zeros(self, n, dtype):
        return np.zeros(n, dtype=dtype)

    def full(self, n, value, dtype):
        return np.full(n, value, dtype=dtype)

    def arange(self, n, dtype=np.int64):
        return np.arange(n, dtype=dtype)

    def concat(self, parts):
        return np.concatenate(parts)

    # --- the primitives that actually differ --------------------------------
    def argsort(self, a):
        return np.argsort(a, kind="stable")

    def sort(self, a):
        return np.sort(a)

    def searchsorted(self, sorted_a, keys):
        return np.searchsorted(sorted_a, keys)

    def nonzero(self, mask):
        return np.nonzero(mask)[0]

    def cumsum(self, a):
        return np.cumsum(a)

    def repeat_interleave(self, a, k):
        return np.repeat(a, k)

    def tile(self, a, k):
        return np.tile(a, k)

    # Segment reduces take BOTH a group id per entry and the group start offsets:
    # numpy wants the offsets (ufunc.reduceat, which is fast) and torch wants the ids
    # (scatter_reduce_). Passing both keeps one call site for the two backends.
    # `vals` must already be ordered by group.
    def seg_min(self, vals, gid, starts, ng):
        return np.minimum.reduceat(vals.astype(np.int32), starts)

    def seg_max(self, vals, gid, starts, ng):
        return np.maximum.reduceat(vals.astype(np.int32), starts)

    def gather_rows(self, states, gens):
        """states (n, N) -> (n * n_gen, N): every child of every row."""
        return states[:, gens].reshape(-1, states.shape[1])

    def free(self):
        pass


class TorchBackend:
    name = "torch"
    is_torch = True

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.int8 = torch.int8
        self.int32 = torch.int32
        self.int64 = torch.int64
        self.bool_ = torch.bool

    def asarray(self, a, dtype=None):
        if isinstance(a, torch.Tensor):
            return a.to(self.device) if dtype is None else a.to(self.device, dtype)
        return torch.as_tensor(np.asarray(a), device=self.device) if dtype is None else \
            torch.as_tensor(np.asarray(a), device=self.device).to(dtype)

    def to_numpy(self, a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def zeros(self, n, dtype):
        return torch.zeros(n, dtype=dtype, device=self.device)

    def full(self, n, value, dtype):
        return torch.full((n,), value, dtype=dtype, device=self.device)

    def arange(self, n, dtype=None):
        return torch.arange(n, dtype=dtype or torch.int64, device=self.device)

    def concat(self, parts):
        return torch.cat(parts)

    def argsort(self, a):
        return torch.argsort(a)

    def sort(self, a):
        return torch.sort(a).values

    def searchsorted(self, sorted_a, keys):
        return torch.searchsorted(sorted_a, keys)

    def nonzero(self, mask):
        return torch.nonzero(mask, as_tuple=True)[0]

    def cumsum(self, a):
        return torch.cumsum(a, 0)

    def repeat_interleave(self, a, k):
        return a.repeat_interleave(k)

    def tile(self, a, k):
        return a.repeat(k)

    def seg_min(self, vals, gid, starts, ng):
        out = torch.full((ng,), torch.iinfo(torch.int32).max // 4,
                         dtype=torch.int32, device=self.device)
        out.scatter_reduce_(0, gid, vals.to(torch.int32), reduce="amin")
        return out

    def seg_max(self, vals, gid, starts, ng):
        out = torch.full((ng,), -(torch.iinfo(torch.int32).max // 4),
                         dtype=torch.int32, device=self.device)
        out.scatter_reduce_(0, gid, vals.to(torch.int32), reduce="amax")
        return out

    def gather_rows(self, states, gens):
        return states[:, gens].reshape(-1, states.shape[1])

    def free(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def get_backend(device: str = "cpu"):
    """`cpu` -> numpy. `cuda`/`cuda:N` -> torch, falling back to numpy if unavailable."""
    if device.startswith("cuda"):
        if _HAS_TORCH and torch.cuda.is_available():
            return TorchBackend(device)
        return NumpyBackend("cpu")
    if device == "torch":
        if _HAS_TORCH:
            return TorchBackend("cpu")
        return NumpyBackend("cpu")
    return NumpyBackend("cpu")
