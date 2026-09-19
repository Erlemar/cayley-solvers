"""Exact window rewriting for cube555 paths -- the tetraminx MITM family, ported.

    python shorten.py <in.csv> <out.csv> --front-depth A --max-window W

For every window [i, i+L) of a path the net group element is

    w = s_i^-1 o s_{i+L}          (array form: w[k] = inv(s_i)[s_{i+L}[k]])

because the solved state is the identity permutation and `apply` right-multiplies.
The exact ball B_5 (10,739,017 states, with per-state Q) is on disk, so d(w) is known
exactly whenever w lands in it. Meet-in-the-middle extends the reach:

    d(w) <= 5 + A   iff   exists x in B_A with x^-1 o w in B_5

and the optimal word is word(x) + word(x^-1 o w). The join is done by hashing:

    h(x^-1 o w) = sum_k XINV[x][k] * hv[w^-1[k]]

which is a dot product of a FIXED per-front-element row with a per-window permuted hash
vector -- no permutation is materialised for the probe, only for the few hits.

Windows are tried longest-first at each position and the pass repeats to a fixed point,
since a splice creates new adjacencies. Every rewritten path is replayed against
test.csv before it is written; a path that fails reverts to its original.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HANDOFF = Path("C:/Users/and-l/cayley/cube555_pull/cube555_handoff_2026_08_22/cube555")
sys.path.insert(0, str(HANDOFF / "src"))
from cube555.puzzle import Cube555  # noqa: E402


def inv_perm(p: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    out[p] = np.arange(p.shape[-1], dtype=p.dtype)
    return out


def hash_rows(arr: torch.Tensor, hv: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
    """Row hashes of an (N,150) uint8 tensor without widening the whole array (rule 30)."""
    n = arr.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=arr.device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out[s:e] = (arr[s:e].long() * hv).sum(1)
    return out


def build_front(G: np.ndarray, depth: int):
    """BFS ball of radius `depth` around the identity. Returns (perms, words)."""
    ident = np.arange(G.shape[1], dtype=np.uint8)
    perms = [ident]
    words: list[list[int]] = [[]]
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
    ap.add_argument("--front-depth", type=int, default=0, help="A; reach is 5 + A")
    ap.add_argument("--max-window", type=int, default=12)
    ap.add_argument("--min-window", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pids", default="", help="comma list or lo-hi range; blank = all")
    ap.add_argument("--probe-budget", type=int, default=1_200_000,
                    help="front_size * windows_per_chunk; caps the (NA,m,150) buffer")
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
            [int(x) for x in r["initial_state"].split(",")], dtype=np.uint8
        )
        for r in csv.DictReader(open(HANDOFF / "data" / "test.csv", encoding="utf-8"))
    }

    t0 = time.time()
    blob = torch.load(HANDOFF / "data" / "anchors_d5.pt", map_location="cpu",
                      weights_only=False)
    ball_states_np = blob["states"].numpy()
    ball_q = blob["q"].numpy()
    ball_depth = blob["depth"].numpy()
    print(f"ball d<=5: {ball_states_np.shape[0]:,} states  ({time.time()-t0:.0f}s)",
          flush=True)

    g = torch.Generator(device=dev).manual_seed(555)
    hv = torch.randint(-(2**62), 2**62, (150,), dtype=torch.int64, device=dev,
                       generator=g)
    bh = hash_rows(blob["states"].to(dev), hv)
    order = torch.argsort(bh)
    bh = bh[order].contiguous()
    order_np = order.cpu().numpy()
    assert int((bh[1:] == bh[:-1]).sum()) == 0, "ball hash collision -- reseed"
    bh_np = bh.cpu().numpy()
    hv_np = hv.cpu().numpy()
    del blob
    torch.cuda.empty_cache()

    front_perm, front_words = build_front(G, args.front_depth)
    XINV = np.stack([inv_perm(p) for p in front_perm])
    XINV_t = torch.as_tensor(XINV.astype(np.int64), device=dev)
    front_len = np.array([len(w) for w in front_words])
    NA = XINV.shape[0]
    print(f"front B_{args.front_depth}: {NA:,} elements -> reach {5+args.front_depth}",
          flush=True)

    def lookup_np(w: np.ndarray) -> int:
        h = int((w.astype(np.int64) * hv_np).sum())
        pos = int(np.searchsorted(bh_np, h))
        if pos >= bh_np.size or bh_np[pos] != h:
            return -1
        return int(order_np[pos])

    def optimal_word(idx: int) -> list[str]:
        """Move-name word for the group element stored at ball position idx.

        Descending by argmin-Q from x yields u with x o u = e, i.e. u = x^-1; the window
        needs x itself, so reverse u and invert each move.
        """
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

    win_chunk = max(1, args.probe_budget // max(1, NA))
    before = after = 0
    n_splice = n_fail = n_phantom = 0
    saved_by_pid: dict[int, int] = {}
    t0 = time.time()

    for ri, r in enumerate(rows):
        pid = int(r["initial_state_id"])
        mv = [m for m in r["path"].split(".") if m]
        before += len(mv)
        orig = list(mv)

        for _round in range(40):
            n = len(mv)
            idxs = [name_idx[m] for m in mv]
            st = np.empty((n + 1, 150), dtype=np.uint8)
            st[0] = tests[pid]
            for k in range(n):
                st[k + 1] = st[k][G[idxs[k]]]
            iv = np.stack([inv_perm(s) for s in st])

            wi_l, wl_l = [], []
            for i in range(n):
                hi = min(args.max_window, n - i)
                for L in range(hi, args.min_window - 1, -1):
                    wi_l.append(i)
                    wl_l.append(L)
            if not wi_l:
                break
            wi = np.array(wi_l)
            wl = np.array(wl_l)
            # w^-1 = inv(s_{i+L}) o s_i  ->  HVW[k] = hv[w^-1[k]]
            winv = iv[wi + wl][np.arange(wi.size)[:, None], st[wi]]
            HVW = hv[torch.as_tensor(winv.astype(np.int64), device=dev)]

            best: dict[int, tuple[int, int, int, int]] = {}  # i -> (gain, L, bidx, xi)
            for s in range(0, wi.size, win_chunk):
                e = min(wi.size, s + win_chunk)
                H = (XINV_t.unsqueeze(1) * HVW[s:e].unsqueeze(0)).sum(-1)  # (NA, m)
                flat = H.reshape(-1)
                pos = torch.searchsorted(bh, flat).clamp_(max=bh.numel() - 1)
                hit = bh[pos] == flat
                if not bool(hit.any()):
                    continue
                sel = torch.nonzero(hit, as_tuple=False).flatten()
                posc = pos[sel].cpu().numpy()
                selc = sel.cpu().numpy()
                m = e - s
                for f, pc in zip(selc, posc):
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
                    _gain, L, bidx, xi = best[i]
                    try:
                        word = [names[a] for a in front_words[xi]] + optimal_word(bidx)
                    except RuntimeError:
                        n_phantom += 1
                        word = None
                    if word is not None and len(word) < L:
                        # exact guard: the spliced word must reproduce the window
                        chk = st[i].copy()
                        for mm in word:
                            chk = chk[G[name_idx[mm]]]
                        if np.array_equal(chk, st[i + L]):
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
        saved_by_pid[pid] = len(orig) - len(mv)
        r["path"] = ".".join(mv)
        if (ri + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{ri+1}/{len(rows)}] saved {before-after:,}  {el:.0f}s "
                  f"({el/(ri+1):.2f}s/pid)", flush=True)

    with open(args.dst, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["initial_state_id", "path"])
        for r in sorted(rows, key=lambda r: int(r["initial_state_id"])):
            w_.writerow([r["initial_state_id"], r["path"]])

    pos_pids = sum(1 for v in saved_by_pid.values() if v > 0)
    print(f"\n{args.src.name}: {before:,} -> {after:,} moves ({after-before:+,})")
    print(f"  splices {n_splice}, pids improved {pos_pids}/{len(rows)}, "
          f"phantoms rejected {n_phantom}, replay failures {n_fail}")
    print(f"  wall {time.time()-t0:.0f}s   wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
