"""Deep MITM window rewriting for cube555 -- fast enough for reach 8 and 9.

Same certificate as shorten.py:

    d(w) <= 5 + A   iff   exists x in B_A with x^-1 o w in B_5

but the probe is restructured so the whole join is a DGEMM instead of an elementwise
broadcast.  For a front element x and a window w,

    h(x^-1 o w) = sum_k XINV[x][k] * hv[w^-1[k]]

so with HVW[j][k] = hv[winv_j[k]] the whole (front x window) hash block is

    H = XINV @ HVW^T

Two independent hash vectors with entries < 2^18 keep every partial sum an exact
integer below 2^53, so float64 DGEMM is EXACT integer arithmetic (reassociation is
harmless -- integer addition is exact in any order).  The two hashes are packed into one
62-bit key, giving a false-positive rate of ~3e-12 per probe; a permutation guard
re-derives every accepted splice anyway, so a phantom can never reach the output.

A bool prefilter over the key's low bits removes ~98% of probes before the binary
search, which is otherwise the bottleneck at these probe counts.

    python shorten_mitm.py in.csv out.csv --front-depth 3 --min-window 9 --max-window 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HANDOFF = Path("C:/Users/and-l/cayley/cube555_pull/cube555_handoff_2026_08_22/cube555")
sys.path.insert(0, str(HANDOFF / "src"))
from cube555.puzzle import Cube555  # noqa: E402

HBITS = 18          # hash entries in [0, 2^18)
SHIFT = 29          # low-half width in the packed key
FLAG_BITS = 29      # prefilter table size = 2^29 bools = 536 MB


def inv_perm(p: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    out[p] = np.arange(p.shape[-1], dtype=p.dtype)
    return out


def build_front(G: np.ndarray, depth: int):
    ident = np.arange(G.shape[1], dtype=np.uint8)
    perms, words = [ident], [[]]
    seen = {ident.tobytes()}
    frontier = [(ident, [])]
    for _ in range(depth):
        nxt = []
        for p, w in frontier:
            for a in range(G.shape[0]):
                c = p[G[a]]
                k = c.tobytes()
                if k in seen:
                    continue
                seen.add(k)
                nxt.append((c, w + [a]))
                perms.append(c)
                words.append(w + [a])
        frontier = nxt
    return np.stack(perms), words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--front-depth", type=int, default=3)
    ap.add_argument("--min-window", type=int, default=9)
    ap.add_argument("--max-window", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pids", default="")
    ap.add_argument("--win-chunk", type=int, default=0, help="0 = auto from VRAM")
    ap.add_argument("--l-parity", type=int, default=-1,
                    help="keep only windows with L %% 2 == this; -1 = all. Every "
                         "generator is an ODD permutation, so d(w) == L (mod 2) and a "
                         "reach-R sweep can only newly fire on L == R (mod 2).")
    args = ap.parse_args()

    dev = args.device
    puz = Cube555.load(HANDOFF / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    name_idx = {m: i for i, m in enumerate(names)}
    G = np.array([puz.generators[m] for m in names], dtype=np.uint8)
    central = np.array(puz.solved_state, dtype=np.uint8)
    inv_name = {m: (m[1:] if m.startswith("-") else "-" + m) for m in names}
    tests = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.uint8)
        for r in csv.DictReader(open(HANDOFF / "data" / "test.csv", encoding="utf-8"))
    }

    t0 = time.time()
    blob = torch.load(HANDOFF / "data" / "anchors_d5.pt", map_location="cpu",
                      weights_only=False)
    ball_states_np = blob["states"].numpy()
    ball_q = blob["q"].numpy()
    ball_depth = blob["depth"].numpy()
    NB = ball_states_np.shape[0]
    print(f"ball d<=5: {NB:,} states ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(20260823)
    hv1 = rng.integers(0, 2 ** HBITS, size=150, dtype=np.int64)
    hv2 = rng.integers(0, 2 ** HBITS, size=150, dtype=np.int64)
    hv1_t = torch.as_tensor(hv1.astype(np.float64), device=dev)
    hv2_t = torch.as_tensor(hv2.astype(np.float64), device=dev)

    def pack(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return (h1.to(torch.int64) << SHIFT) | (h2.to(torch.int64) & ((1 << SHIFT) - 1))

    bkey = torch.empty(NB, dtype=torch.int64, device=dev)
    for s in range(0, NB, 500_000):
        e = min(NB, s + 500_000)
        blk = blob["states"][s:e].to(dev).to(torch.float64)
        bkey[s:e] = pack(blk @ hv1_t, blk @ hv2_t)
    order = torch.argsort(bkey)
    bkey = bkey[order].contiguous()
    order_np = order.cpu().numpy()
    dup = int((bkey[1:] == bkey[:-1]).sum())
    assert dup == 0, f"{dup} ball key collisions -- change the seed"
    bkey_np = bkey.cpu().numpy()
    del blob
    torch.cuda.empty_cache()

    flags = torch.zeros(1 << FLAG_BITS, dtype=torch.bool, device=dev)
    flags[bkey & ((1 << FLAG_BITS) - 1)] = True
    print(f"prefilter density {float(flags.float().mean()):.4f}", flush=True)

    front_perm, front_words = build_front(G, args.front_depth)
    XINV = np.stack([inv_perm(p) for p in front_perm])
    XINV_t = torch.as_tensor(XINV.astype(np.float64), device=dev)
    front_len = np.array([len(w) for w in front_words])
    NA = XINV.shape[0]
    reach = 5 + args.front_depth
    print(f"front B_{args.front_depth}: {NA:,} -> reach {reach}", flush=True)

    win_chunk = args.win_chunk or max(8, int(3.0e8 / NA))

    hv1_np, hv2_np = hv1, hv2

    def lookup_np(w: np.ndarray) -> int:
        wf = w.astype(np.int64)
        k = ((int((wf * hv1_np).sum()) << SHIFT)
             | (int((wf * hv2_np).sum()) & ((1 << SHIFT) - 1)))
        pos = int(np.searchsorted(bkey_np, k))
        if pos >= bkey_np.size or bkey_np[pos] != k:
            return -1
        return int(order_np[pos])

    def optimal_word(idx: int) -> list[str]:
        out: list[int] = []
        cur = ball_states_np[idx]
        while int(ball_depth[idx]) != 0:
            a = int(ball_q[idx].argmin())
            out.append(a)
            cur = cur[G[a]]
            idx = lookup_np(cur)
            if idx < 0:
                raise RuntimeError("descent left the ball")
        return [inv_name[names[a]] for a in reversed(out)]

    rows = list(csv.DictReader(open(args.src, encoding="utf-8")))
    if args.pids:
        if "-" in args.pids and "," not in args.pids:
            lo, hi = (int(x) for x in args.pids.split("-"))
            keep = set(range(lo, hi + 1))
        else:
            keep = {int(x) for x in args.pids.split(",")}
        rows = [r for r in rows if int(r["initial_state_id"]) in keep]

    before = after = 0
    n_splice = n_fail = n_phantom = 0
    band = Counter()
    t0 = time.time()

    for ri, r in enumerate(rows):
        pid = int(r["initial_state_id"])
        mv = [m for m in r["path"].split(".") if m]
        before += len(mv)
        orig = list(mv)

        for _round in range(30):
            n = len(mv)
            st = np.empty((n + 1, 150), dtype=np.uint8)
            st[0] = tests[pid]
            for k, m in enumerate(mv):
                st[k + 1] = st[k][G[name_idx[m]]]
            iv = np.stack([inv_perm(s) for s in st])

            wi_l, wl_l = [], []
            for i in range(n):
                hi = min(args.max_window, n - i)
                for L in range(hi, args.min_window - 1, -1):
                    if args.l_parity >= 0 and L % 2 != args.l_parity:
                        continue
                    wi_l.append(i)
                    wl_l.append(L)
            if not wi_l:
                break
            wi = np.array(wi_l)
            wl = np.array(wl_l)
            winv = iv[wi + wl][np.arange(wi.size)[:, None], st[wi]]
            winv_t = torch.as_tensor(winv.astype(np.int64), device=dev)
            HVW1 = hv1_t[winv_t]
            HVW2 = hv2_t[winv_t]

            best: dict[int, tuple[int, int, int, int]] = {}
            for s in range(0, wi.size, win_chunk):
                e = min(wi.size, s + win_chunk)
                K = pack(XINV_t @ HVW1[s:e].T, XINV_t @ HVW2[s:e].T).reshape(-1)
                cand = torch.nonzero(flags[K & ((1 << FLAG_BITS) - 1)],
                                     as_tuple=False).flatten()
                if cand.numel() == 0:
                    continue
                kc = K[cand]
                pos = torch.searchsorted(bkey, kc).clamp_(max=NB - 1)
                good = bkey[pos] == kc
                if not bool(good.any()):
                    continue
                sel = cand[good].cpu().numpy()
                posc = pos[good].cpu().numpy()
                m = e - s
                for f, pc in zip(sel, posc):
                    xi, wj = int(f // m), int(f % m) + s
                    bidx = int(order_np[pc])
                    tot = int(ball_depth[bidx]) + int(front_len[xi])
                    i, L = int(wi[wj]), int(wl[wj])
                    gain = L - tot
                    if gain > 0 and (i not in best or gain > best[i][0]):
                        best[i] = (gain, L, bidx, xi)
            if not best:
                break

            new: list[str] = []
            i = 0
            changed = False
            while i < len(mv):
                if i in best:
                    _g, L, bidx, xi = best[i]
                    try:
                        word = [names[a] for a in front_words[xi]] + optimal_word(bidx)
                    except RuntimeError:
                        word = None
                        n_phantom += 1
                    if word is not None and len(word) < L:
                        chk = st[i].copy()
                        for mm in word:
                            chk = chk[G[name_idx[mm]]]
                        if np.array_equal(chk, st[i + L]):
                            band[(L, len(word))] += 1
                            new.extend(word)
                            i += L
                            n_splice += 1
                            changed = True
                            continue
                        n_phantom += 1
                new.append(mv[i])
                i += 1
            if not changed:
                break
            mv = new

        cur = tests[pid].copy()
        for m in mv:
            cur = cur[G[name_idx[m]]]
        if not np.array_equal(cur, central):
            n_fail += 1
            mv = orig
        after += len(mv)
        r["path"] = ".".join(mv)
        if (ri + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{ri+1}/{len(rows)}] saved {before-after:,}  {el:.0f}s "
                  f"({el/(ri+1):.2f}s/pid, eta {(len(rows)-ri-1)*el/(ri+1)/60:.0f}m)",
                  flush=True)

    with open(args.dst, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["initial_state_id", "path"])
        for r in sorted(rows, key=lambda r: int(r["initial_state_id"])):
            w_.writerow([r["initial_state_id"], r["path"]])

    print(f"\n{args.src.name} -> reach {reach}: {before:,} -> {after:,} ({after-before:+,})")
    print(f"  splices {n_splice}, phantoms {n_phantom}, replay failures {n_fail}")
    print(f"  band (window_len -> new_len): {sorted(band.items())}")
    print(f"  wall {time.time()-t0:.0f}s   wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
