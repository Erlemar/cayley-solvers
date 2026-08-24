"""How much local slack do these paths actually have?

For every window [i,i+L) of a sample of paths, look up the net element w in the exact
B_5 ball and report the joint histogram of (L, d(w)). A window is improvable iff
d(w) < L. Windows with L <= 5 MUST be found (d <= L <= 5) -- that doubles as a
self-test of the lookup, which is why L starts at 2.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HANDOFF = Path("C:/Users/and-l/cayley/cube555_pull/cube555_handoff_2026_08_22/cube555")
sys.path.insert(0, str(HANDOFF / "src"))
from cube555.puzzle import Cube555  # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "C:/Users/and-l/cayley/cube555/postproc/base_109512.csv")
NPID = int(sys.argv[2]) if len(sys.argv) > 2 else 40
MAXW = 14


def inv_perm(p):
    out = np.empty_like(p)
    out[p] = np.arange(p.shape[-1], dtype=p.dtype)
    return out


puz = Cube555.load(HANDOFF / "data" / "puzzle_info.json")
names = list(puz.move_names)
name_idx = {m: i for i, m in enumerate(names)}
G = np.array([puz.generators[m] for m in names], dtype=np.uint8)
tests = {
    int(r["initial_state_id"]): np.array(
        [int(x) for x in r["initial_state"].split(",")], dtype=np.uint8)
    for r in csv.DictReader(open(HANDOFF / "data" / "test.csv", encoding="utf-8"))
}

blob = torch.load(HANDOFF / "data" / "anchors_d5.pt", map_location="cpu",
                  weights_only=False)
states = blob["states"]
depth = blob["depth"].numpy()
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(555)
hv = torch.randint(-(2**62), 2**62, (150,), dtype=torch.int64, device=dev, generator=g)
bh = torch.empty(states.shape[0], dtype=torch.int64, device=dev)
for s in range(0, states.shape[0], 200_000):
    e = min(states.shape[0], s + 200_000)
    bh[s:e] = (states[s:e].to(dev).long() * hv).sum(1)
order = torch.argsort(bh)
bh = bh[order].contiguous()
order_np = order.cpu().numpy()
print(f"ball {states.shape[0]:,}  levels {blob['levels']}", flush=True)
del blob, states

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
step = max(1, len(rows) // NPID)
rows = rows[::step][:NPID]

hist = Counter()
found = Counter()
total = Counter()
for r in rows:
    pid = int(r["initial_state_id"])
    mv = [m for m in r["path"].split(".") if m]
    n = len(mv)
    if n < 6:
        continue
    st = np.empty((n + 1, 150), dtype=np.uint8)
    st[0] = tests[pid]
    for k, m in enumerate(mv):
        st[k + 1] = st[k][G[name_idx[m]]]
    iv = np.stack([inv_perm(s) for s in st])
    wi, wl = [], []
    for i in range(n):
        for L in range(2, min(MAXW, n - i) + 1):
            wi.append(i)
            wl.append(L)
    wi = np.array(wi)
    wl = np.array(wl)
    w = iv[wi][np.arange(wi.size)[:, None], st[wi + wl]]
    H = (torch.as_tensor(w.astype(np.int64), device=dev) * hv).sum(1)
    pos = torch.searchsorted(bh, H).clamp_(max=bh.numel() - 1)
    hit = (bh[pos] == H).cpu().numpy()
    pos_np = pos.cpu().numpy()
    for j in range(wi.size):
        total[int(wl[j])] += 1
        if hit[j]:
            d = int(depth[order_np[pos_np[j]]])
            found[int(wl[j])] += 1
            hist[(int(wl[j]), d)] += 1

print(f"\nsampled {len(rows)} pids, {sum(total.values()):,} windows\n")
print("  L   windows   d<=5 found   d-histogram (d:count)   improvable(d<L)")
for L in range(2, MAXW + 1):
    if not total[L]:
        continue
    ds = {d: c for (LL, d), c in sorted(hist.items()) if LL == L}
    imp = sum(c for d, c in ds.items() if d < L)
    print(f" {L:2d}  {total[L]:8,}  {found[L]:10,}   "
          f"{ {d: c for d, c in sorted(ds.items())} }   {imp}")
