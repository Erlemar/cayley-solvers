"""Balls, hashing, and the two group reduces every exact method here is built from.

THE ONE IDEA. Every exact post-processing method in this package is the same join:
expand a ball of radius r around a set of SOURCE states, label each ball entry with the
source it came from and the depth it was reached at, then look for states reached from
two different sources. If state x is reached from source A at depth d1 and from source B
at depth d2, then A and B are connected by a word of length d1 + d2. What that buys
depends only on what the sources are:

    sources = the states along one path        -> window shortening       (window.py)
    sources = waypoints of every known path    -> pooled bridging + merge (bridge.py)
    sources = one path state and the target    -> tail / suffix rewriting (window.py)

Reach is 2r for EVERY pair at once, and the cost is one BFS per source set plus one
sort -- no pair enumeration, which is what makes a 1000-path sweep affordable.

WHY STATES AND NOT PERMUTATIONS. The collision test is on the state vector, so on a
colour puzzle it accepts any replacement word landing on the same COLOURING. That is
strictly weaker than requiring the same permutation, and on cube444 it is where the only
win came from: a 9-move window collapsed to 7 by a word whose permutation differs at six
centre facelets. On a permutation puzzle the two tests coincide, so nothing is lost by
always using the state test.

WHAT A ZERO PROVES (the band distinction -- read this before quoting a result):
  * window length L <= 2r : the join would have FOUND any shorter word, so no hit means
    the window is exactly geodesic.
  * window length L >  2r : no hit only certifies d(window) > 2r. The window can still be
    non-geodesic with a distance between 2r+1 and L-1.
Reporting the second as "proven optimal" is the most common way to overstate a sweep.
"""
from __future__ import annotations

import numpy as np

from .backend import get_backend


class Zobrist:
    """64-bit XOR-Zobrist hash of a label vector.

    XOR rather than the more common additive form: with numpy int64 there is no defined
    wraparound story for a sum of 96 large values, and XOR has none of that problem while
    hashing equally well. Two states with the same COLOURING hash equal, which is exactly
    what a colour puzzle needs.

    Accumulation runs in column GROUPS. A one-shot cast of a child batch to int64 is an
    N-fold blow-up of the whole batch (7 GB at a 400k frontier on a 96-facelet puzzle) and
    is what OOMs a 16 GB card; grouping caps the temporary at `group` columns.
    """

    GROUP = 8

    def __init__(self, state_size, n_labels, bk, seed=20260820):
        rng = np.random.default_rng(seed)
        tab = rng.integers(-(2 ** 62), 2 ** 62, size=(state_size, n_labels), dtype=np.int64)
        self.bk = bk
        self.state_size = state_size
        self.n_labels = n_labels
        self.flat = bk.asarray(tab.reshape(-1))
        self.col_off = bk.asarray(np.arange(state_size, dtype=np.int64) * n_labels)
        # Per-source salt, so a (source, state) pair can be deduped with one int64 key.
        self.salt = bk.asarray(rng.integers(-(2 ** 62), 2 ** 62, size=8192, dtype=np.int64))

    def hash(self, states):
        bk = self.bk
        n = states.shape[0]
        h = bk.zeros(n, bk.int64)
        for i in range(0, self.state_size, self.GROUP):
            j = min(self.state_size, i + self.GROUP)
            cols = states[:, i:j]
            idx = (cols.astype(np.int64) if not bk.is_torch else cols.long()) + self.col_off[i:j]
            block = self.flat[idx]
            for c in range(block.shape[1]):
                h = h ^ block[:, c]
        return h


class Balls:
    """Flat arrays describing the ball around every source. One entry per (source, state).

    h    int64  hash of the state
    src  int32  which source it was reached from
    dist int8   depth at which it was reached
    par  int32  index of the parent ENTRY (-1 for a source), for word reconstruction
    mv   int8   the move taken from the parent
    """

    __slots__ = ("h", "src", "dist", "par", "mv", "radius", "n_sources", "bk")

    def __init__(self, h, src, dist, par, mv, radius, n_sources, bk):
        self.h, self.src, self.dist, self.par, self.mv = h, src, dist, par, mv
        self.radius, self.n_sources, self.bk = radius, n_sources, bk

    def __len__(self):
        return int(self.h.shape[0])

    def to_numpy(self):
        bk = self.bk
        return (bk.to_numpy(self.h), bk.to_numpy(self.src), bk.to_numpy(self.dist),
                bk.to_numpy(self.par), bk.to_numpy(self.mv))


def build_balls(puzzle, sources, radius, bk=None, zob=None, chunk=150_000):
    """BFS `radius` moves out from every row of `sources` ((M, N) label array).

    Dedup is per (source, state): a state reached twice from the SAME source keeps only
    its first (shallowest) copy, but the same state reached from two different sources is
    kept twice -- that pair is the whole point.

    The last level is streamed out without a global dedup or a `seen` update: nothing
    expands it, so both are pure cost at the widest level, and duplicate entries are
    harmless under the min/max group reduce that consumes them.
    """
    bk = bk or get_backend("cpu")
    zob = zob or Zobrist(puzzle.state_size, puzzle.n_labels, bk)
    src_arr = bk.asarray(np.ascontiguousarray(sources))
    gens = bk.asarray(puzzle.gens)
    M = int(src_arr.shape[0])
    n_gen = puzzle.n_gen

    h0 = zob.hash(src_arr)
    H = [h0]
    SRC = [bk.arange(M).astype(np.int32) if not bk.is_torch else bk.arange(M).to(bk.int32)]
    DIST = [bk.zeros(M, bk.int8)]
    PAR = [bk.full(M, -1, bk.int32)]
    MV = [bk.full(M, -1, bk.int8)]

    cur_states = src_arr
    cur_src = SRC[0]
    cur_gidx = SRC[0]
    seen = bk.sort(h0 ^ zob.salt[_as_index(bk, SRC[0])])
    total = M

    for d in range(1, radius + 1):
        keep_states = d < radius
        parts = []
        stream_last = not keep_states
        for s in range(0, int(cur_states.shape[0]), chunk):
            blk = cur_states[s:s + chunk]
            n = int(blk.shape[0])
            kids = bk.gather_rows(blk, gens)                     # (n * n_gen, N)
            kh = zob.hash(kids)
            ksrc = bk.repeat_interleave(cur_src[s:s + chunk], n_gen)
            kpar = bk.repeat_interleave(cur_gidx[s:s + chunk], n_gen)
            kmv = bk.tile(_arange_i8(bk, n_gen), n)
            key = kh ^ zob.salt[_as_index(bk, ksrc)]
            pos = bk.searchsorted(seen, key)
            pos = _clamp(bk, pos, int(seen.shape[0]) - 1)
            fresh = seen[pos] != key
            sel = bk.nonzero(fresh)
            if int(sel.shape[0]) == 0:
                continue
            if stream_last:
                k_sel = key[sel]
                o = bk.argsort(k_sel)
                ks = k_sel[o]
                uniq = _first_of_group(bk, ks)
                pick = sel[o[uniq]]
                H.append(kh[pick])
                SRC.append(ksrc[pick])
                DIST.append(bk.full(int(pick.shape[0]), d, bk.int8))
                PAR.append(kpar[pick])
                MV.append(kmv[pick])
                total += int(pick.shape[0])
                continue
            parts.append((key[sel], kh[sel], ksrc[sel], kpar[sel], kmv[sel],
                          kids[sel] if keep_states else None))
        if stream_last or not parts:
            break
        key = bk.concat([p[0] for p in parts])
        kh = bk.concat([p[1] for p in parts])
        ksrc = bk.concat([p[2] for p in parts])
        kpar = bk.concat([p[3] for p in parts])
        kmv = bk.concat([p[4] for p in parts])
        kids = bk.concat([p[5] for p in parts])
        del parts
        order = bk.argsort(key)
        key = key[order]
        uniq = _first_of_group(bk, key)
        pick = order[uniq]
        key = key[uniq]
        kh, ksrc, kpar, kmv, kids = kh[pick], ksrc[pick], kpar[pick], kmv[pick], kids[pick]
        n_new = int(kh.shape[0])
        if n_new == 0:
            break
        H.append(kh)
        SRC.append(ksrc)
        DIST.append(bk.full(n_new, d, bk.int8))
        PAR.append(kpar)
        MV.append(kmv)
        cur_gidx = _int32_range(bk, total, total + n_new)
        total += n_new
        seen = bk.sort(bk.concat([seen, key]))
        cur_states, cur_src = kids, ksrc

    return Balls(bk.concat(H), bk.concat(SRC), bk.concat(DIST),
                 bk.concat(PAR), bk.concat(MV), radius, M, bk)


def collide_savings(balls, min_saving=1, bucket_max=16_000_000):
    """Per distinct hash: saving = max(src - dist) - min(src + dist).

    With `src` set to the index along a path, a state reached from position i at depth d1
    and from position j at depth d2 shortens the path by (j - d2) - (i + d1). The best
    pair inside a hash group is therefore one max and one min -- no pair enumeration.

    Returns [(saving, entry_lo, entry_hi)] sorted by saving descending, where entry_lo
    attains min(src + dist) and entry_hi attains max(src - dist).

    Large ball sets are bucketed on the LOW BITS of the hash before sorting: at radius 5 a
    full sweep is ~1e8 entries and a single argsort with its workspace OOMs a 16 GB card.
    Equal hashes share low bits, so bucketing leaves the grouping unchanged.
    """
    bk = balls.bk
    h, src, dist = balls.h, balls.src, balls.dist
    n = int(h.shape[0])
    if n > bucket_max:
        k = 1
        while n // k > bucket_max:
            k *= 2
        out = []
        sel = h & (k - 1)
        for b in range(k):
            idx = bk.nonzero(sel == b)
            if int(idx.shape[0]) < 2:
                continue
            sub = Balls(h[idx], src[idx], dist[idx], balls.par, balls.mv,
                        balls.radius, balls.n_sources, bk)
            idx_np = bk.to_numpy(idx)
            for s, lo, hi in collide_savings(sub, min_saving, bucket_max):
                out.append((s, int(idx_np[lo]), int(idx_np[hi])))
        out.sort(key=lambda t: -t[0])
        return out

    order = bk.argsort(h)
    hs = h[order]
    a = _to_i32(bk, src[order]) + _to_i32(bk, dist[order])        # i + d
    b = _to_i32(bk, src[order]) - _to_i32(bk, dist[order])        # j - d
    gid, starts, ng = _groups(bk, hs)
    if ng == 0:
        return []
    amin = bk.seg_min(a, gid, starts, ng)
    bmax = bk.seg_max(b, gid, starts, ng)
    sav = bmax - amin
    good = bk.nonzero(sav >= min_saving)
    if int(good.shape[0]) == 0:
        return []
    return _extract(bk, order, gid, a, b, amin, bmax, good, sav, want_max_b=True)


def group_min(balls, a_cost, b_cost, limit):
    """Per distinct hash: min(a_cost) + min(b_cost), keeping groups under `limit`.

    The cost-shaped twin of `collide_savings`. That one maximises a saving expressed in
    path indices; this one minimises an explicit cost, which is what pooled bridging needs
    (a = cheapest cost-from-start to reach the entry, b = cheapest cost-to-solved from it).

    Returns [(total, entry_a, entry_b)] sorted by total ascending.
    """
    bk = balls.bk
    order = bk.argsort(balls.h)
    hs = balls.h[order]
    a = _to_i32(bk, a_cost[order])
    b = _to_i32(bk, b_cost[order])
    gid, starts, ng = _groups(bk, hs)
    if ng == 0:
        return []
    amin = bk.seg_min(a, gid, starts, ng)
    bmin = bk.seg_min(b, gid, starts, ng)
    tot = amin + bmin
    good = bk.nonzero(tot < limit)
    if int(good.shape[0]) == 0:
        return []
    res = _extract(bk, order, gid, a, b, amin, bmin, good, tot, want_max_b=False)
    res.sort(key=lambda t: t[0])
    return res


def word_to(par, mv, entry):
    """The word from an entry back to its source, as a list of move indices."""
    w = []
    i = int(entry)
    while par[i] >= 0:
        w.append(int(mv[i]))
        i = int(par[i])
    w.reverse()
    return w


# --------------------------------------------------------------------- helpers
def _groups(bk, sorted_h):
    n = int(sorted_h.shape[0])
    if n == 0:
        return None, None, 0
    bnd = bk.full(n, True, bk.bool_)
    bnd[1:] = sorted_h[1:] != sorted_h[:-1]
    gid = bk.cumsum(bnd.astype(np.int64) if not bk.is_torch else bnd.to(bk.int64)) - 1
    starts = bk.nonzero(bnd)
    ng = int(starts.shape[0])
    return gid, starts, ng


def _extract(bk, order, gid, a, b, agg_a, agg_b, good, score, want_max_b):
    """Pick one representative entry per winning group for each side of the join."""
    ng = int(agg_a.shape[0])
    gset = bk.zeros(ng, bk.bool_)
    gset[good] = True
    ent = bk.nonzero(gset[gid])
    g_np = bk.to_numpy(gid[ent])
    a_np = bk.to_numpy(a[ent])
    b_np = bk.to_numpy(b[ent])
    e_np = bk.to_numpy(ent)
    aa = bk.to_numpy(agg_a)
    bb = bk.to_numpy(agg_b)
    lo, hi = {}, {}
    for e, g, av, bv in zip(e_np.tolist(), g_np.tolist(), a_np.tolist(), b_np.tolist()):
        if g not in lo and av == aa[g]:
            lo[g] = e
        if g not in hi and bv == bb[g]:
            hi[g] = e
    ordr = bk.to_numpy(order)
    sc = bk.to_numpy(score)
    out = []
    for g in bk.to_numpy(good).tolist():
        if g in lo and g in hi:
            out.append((int(sc[g]), int(ordr[lo[g]]), int(ordr[hi[g]])))
    out.sort(key=lambda t: -t[0])
    return out


def _as_index(bk, a):
    return a.astype(np.int64) if not bk.is_torch else a.long()


def _to_i32(bk, a):
    return a.astype(np.int32) if not bk.is_torch else a.to(bk.int32)


def _arange_i8(bk, n):
    return (np.arange(n, dtype=np.int8) if not bk.is_torch
            else bk.arange(n).to(bk.int8))


def _int32_range(bk, lo, hi):
    return (np.arange(lo, hi, dtype=np.int32) if not bk.is_torch
            else bk.arange(hi - lo).to(bk.int32) + lo)


def _clamp(bk, a, hi):
    if bk.is_torch:
        return a.clamp(max=hi)
    return np.clip(a, 0, hi)


def _first_of_group(bk, sorted_keys):
    """Boolean mask selecting the first entry of every run of equal keys."""
    n = int(sorted_keys.shape[0])
    u = bk.full(n, True, bk.bool_)
    u[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return u
