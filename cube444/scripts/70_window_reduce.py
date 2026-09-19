"""Exact local window shortening for the 4x4x4 colour cube (GPU meet-in-the-middle).

PORT OF THE TETRAMINX POST-PROCESSING LADDER (`tetraminx/scripts/59|61|64_*`), with one
structural upgrade that this puzzle allows and tetraminx did not.

WHY STATES AND NOT PERMUTATIONS
-------------------------------
Tetraminx has solved == identity, so a window is replaced only by a word with the SAME
PERMUTATION. Here the state is a 96-sticker COLOURING: the group element is pinned only
up to the stabiliser of the solved colouring (4 same-coloured centres per face, and the
two wings of an edge are interchangeable too). A window w[i:j] may therefore be replaced
by ANY word u with

    s_i[u] == s_j                       (states, not permutations)

because the suffix from j onwards is applied to the same state either way. That is a
STRICTLY weaker condition than perm(u) == perm(w[i:j]), so it finds every permutation
rewrite plus the ones that only differ by an invisible centre/wing permutation. The
price is that the lookup table cannot be shared across windows (it is anchored at s_i,
not at the identity), so this is a per-path meet-in-the-middle rather than one global
ball probe.

THE JOIN
--------
For a path s_0 .. s_L, build the ball of radius r around EVERY s_i. If a state x sits in
ball(s_i) at distance d1 and in ball(s_j) at distance d2, then

    new length = i + d1 + d2 + (L - j)      saving = (j - d2) - (i + d1)

so per colliding state we only need max(i - d) and min(i + d) over the sources that
reached it -- one scatter-reduce, no pair enumeration. Reach is 2r: every window of
length <= 2r is certified geodesic when nothing fires, and windows longer than 2r are
only certified to have d > 2r (the tetraminx band distinction -- see its HANDOFF).

Every splice is replay-verified against the real state before it is allowed into the
path, so a 64-bit hash phantom costs a rejected candidate, never a corrupt row.

    python cube444/scripts/70_window_reduce.py --self-test
    python cube444/scripts/70_window_reduce.py \
        --target cube444/submissions/leader_46718_d7_rot_targeted_3x3.csv \
        --radius 4 --out cube444/submissions/leader_r4.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "cube444" / "data"
STATE_SIZE = 96
NUM_COLORS = 6


class Cube:
    """Move table + hashing, on a torch device."""

    HASH_GROUP = 8          # columns hashed per kernel; see hash()

    def __init__(self, data_dir: Path, device: str = "cuda", seed: int = 20260805):
        info = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.n_gen = len(self.names)
        self.device = torch.device(device)
        gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        assert gen.shape == (self.n_gen, STATE_SIZE)
        self.gen_np = gen
        self.gen = torch.as_tensor(gen, device=self.device)
        self.solved_np = np.array(info["central_state"], dtype=np.uint8)
        self.solved = torch.as_tensor(self.solved_np, device=self.device)
        # inverse move index
        inv = np.zeros(self.n_gen, dtype=np.int64)
        for i, n in enumerate(self.names):
            inv[i] = self.idx[n[1:] if n.startswith("-") else "-" + n]
        ident = np.arange(STATE_SIZE)
        for i in range(self.n_gen):
            assert np.array_equal(gen[i][gen[inv[i]]], ident), f"{self.names[i]} inverse broken"
        self.inv_move_np = inv
        # additive Zobrist: h(state) = sum_i ZT[i, state[i]]  (int64, wraps mod 2^64)
        g = torch.Generator(device="cpu").manual_seed(seed)
        zt = torch.randint(-(2**62), 2**62, (STATE_SIZE, NUM_COLORS), generator=g, dtype=torch.int64)
        self.ZT = zt.to(self.device)
        self.ZTflat = self.ZT.reshape(-1)
        self.col_off = (torch.arange(STATE_SIZE, device=self.device) * NUM_COLORS)
        # per-source salt so that dedup can be keyed on (source, state) in one int64
        self.SALT = torch.randint(-(2**62), 2**62, (4096,), generator=g, dtype=torch.int64).to(self.device)

    def hash(self, states: torch.Tensor) -> torch.Tensor:
        """(N, 96) uint8 -> (N,) int64.

        Accumulated in column GROUPS: a one-shot `states.long()` is a 96x int64 blow-up
        of the whole child batch (7 GB at a 400k frontier chunk) and OOMs the GPU.
        """
        n = states.shape[0]
        h = torch.zeros(n, dtype=torch.int64, device=states.device)
        for i in range(0, STATE_SIZE, self.HASH_GROUP):
            j = min(STATE_SIZE, i + self.HASH_GROUP)
            h += self.ZTflat[states[:, i:j].long() + self.col_off[i:j]].sum(dim=1)
        return h

    def restricted(self, keep: list[int]) -> "Cube":
        """A view of this puzzle using only generators `keep` (indices into self.names).

        The subgroup a subset generates has a much smaller branching factor (the six
        inner slices give ~10 vs 19.2 for the full set), so the same ball budget reaches
        several moves deeper. Any word it returns is still a valid replacement anywhere:
        restricting the SEARCH does not restrict where the answer may be spliced.
        """
        assert keep, "empty generator subset"
        sub = object.__new__(Cube)
        sub.__dict__.update(self.__dict__)
        sub.names = [self.names[i] for i in keep]
        sub.idx = {n: i for i, n in enumerate(sub.names)}
        sub.n_gen = len(keep)
        sub.gen_np = self.gen_np[keep]
        sub.gen = self.gen[keep]
        pos = {g: i for i, g in enumerate(keep)}
        inv = np.zeros(len(keep), dtype=np.int64)
        for i, g in enumerate(keep):
            gi = int(self.inv_move_np[g])
            assert gi in pos, f"subset not closed under inverse: {self.names[g]}"
            inv[i] = pos[gi]
        sub.inv_move_np = inv
        sub.global_idx = list(keep)
        return sub

    def apply_word_np(self, state: np.ndarray, word) -> np.ndarray:
        s = state
        for m in word:
            s = s[self.gen_np[m]]
        return s

    def inv_word(self, word):
        return [int(self.inv_move_np[m]) for m in reversed(word)]

    def path_states_np(self, s0: np.ndarray, word) -> np.ndarray:
        out = np.empty((len(word) + 1, STATE_SIZE), dtype=np.uint8)
        out[0] = s0
        s = s0
        for k, m in enumerate(word):
            s = s[self.gen_np[m]]
            out[k + 1] = s
        return out


def build_balls(cube: Cube, sources: torch.Tensor, radius: int, chunk: int = 150_000):
    """Ball of radius `radius` around each row of `sources` ((M,96) uint8).

    Dedup is per (source, state) -- a state reached twice from the same source keeps only
    its first (shallowest) copy. Returns flat arrays over all entries:
        h (int64) hash, src (int32), dist (int8), par (int32) global index, mv (int8)
    States are not retained past the last expansion (they are only needed to expand).
    """
    dev = cube.device
    M = sources.shape[0]
    h0 = cube.hash(sources)
    H = [h0]
    SRC = [torch.arange(M, dtype=torch.int32, device=dev)]
    DIST = [torch.zeros(M, dtype=torch.int8, device=dev)]
    PAR = [torch.full((M,), -1, dtype=torch.int32, device=dev)]
    MV = [torch.full((M,), -1, dtype=torch.int8, device=dev)]
    cur_states = sources
    cur_src = SRC[0]
    cur_gidx = torch.arange(M, dtype=torch.int32, device=dev)
    seen = torch.sort(h0 ^ cube.SALT[SRC[0].long()]).values
    total = M
    for d in range(1, radius + 1):
        keep_states = d < radius
        parts = []
        # The last level is never expanded, so it needs neither a global cross-chunk
        # dedup nor a `seen` update -- both are pure cost at the widest level (the r=5
        # shell is ~100M entries; the global argsort alone was several GB). Duplicate
        # entries are harmless: the group reduce takes a min/max over them anyway.
        stream_last = not keep_states
        for s in range(0, cur_states.shape[0], chunk):
            blk = cur_states[s:s + chunk]
            n = blk.shape[0]
            kids = blk[:, cube.gen].reshape(-1, STATE_SIZE)          # (n*n_gen, 96)
            kh = cube.hash(kids)
            ksrc = cur_src[s:s + chunk].repeat_interleave(cube.n_gen)
            kpar = cur_gidx[s:s + chunk].repeat_interleave(cube.n_gen)
            kmv = torch.arange(cube.n_gen, dtype=torch.int8, device=dev).repeat(n)
            key = kh ^ cube.SALT[ksrc.long()]
            pos = torch.searchsorted(seen, key).clamp_(max=seen.shape[0] - 1)
            fresh = seen[pos] != key
            sel = torch.nonzero(fresh, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            if stream_last:
                # dedup within the chunk only, then emit straight to the output arrays
                k_sel = key[sel]
                o = torch.argsort(k_sel)
                u = torch.ones_like(o, dtype=torch.bool)
                u[1:] = k_sel[o][1:] != k_sel[o][:-1]
                pick = sel[o[u]]
                H.append(kh[pick])
                SRC.append(ksrc[pick])
                DIST.append(torch.full((pick.numel(),), d, dtype=torch.int8, device=dev))
                PAR.append(kpar[pick])
                MV.append(kmv[pick])
                total += pick.numel()
                continue
            parts.append((key[sel], kh[sel], ksrc[sel], kpar[sel], kmv[sel],
                          kids[sel] if keep_states else None))
        if stream_last:
            break
        if not parts:
            break
        key = torch.cat([p[0] for p in parts])
        kh = torch.cat([p[1] for p in parts])
        ksrc = torch.cat([p[2] for p in parts])
        kpar = torch.cat([p[3] for p in parts])
        kmv = torch.cat([p[4] for p in parts])
        kids = torch.cat([p[5] for p in parts]) if keep_states else None
        del parts
        order = torch.argsort(key)
        key = key[order]
        uniq = torch.ones_like(key, dtype=torch.bool)
        uniq[1:] = key[1:] != key[:-1]
        pick = order[uniq]
        key = key[uniq]
        kh, ksrc, kpar, kmv = kh[pick], ksrc[pick], kpar[pick], kmv[pick]
        if keep_states:
            kids = kids[pick]
        n_new = kh.shape[0]
        H.append(kh)
        SRC.append(ksrc)
        DIST.append(torch.full((n_new,), d, dtype=torch.int8, device=dev))
        PAR.append(kpar)
        MV.append(kmv)
        cur_gidx = torch.arange(total, total + n_new, dtype=torch.int32, device=dev)
        total += n_new
        seen = torch.sort(torch.cat([seen, key])).values
        if keep_states:
            cur_states, cur_src = kids, ksrc
        else:
            break
    return (torch.cat(H), torch.cat(SRC), torch.cat(DIST), torch.cat(PAR), torch.cat(MV))


def find_collisions(h, src, dist, min_saving: int = 1, bucket_max: int = 16_000_000):
    """Per distinct hash: saving = max(src - dist) - min(src + dist).

    Returns (saving, entry_lo, entry_hi) for groups that beat `min_saving`, where
    entry_lo attains min(src + dist) and entry_hi attains max(src - dist).

    Split into hash buckets when the entry count is large: at radius 5 the ball set is
    ~10^8 entries and a single argsort (keys + indices + workspace) OOMs a 16 GB card.
    Bucketing on the LOW BITS keeps every equal hash in one bucket, so the grouping is
    unchanged -- only the peak allocation shrinks.
    """
    n = h.numel()
    if n > bucket_max:
        k = 1
        while n // k > bucket_max:
            k *= 2
        out = []
        sel = h & (k - 1)
        for b in range(k):
            idx = torch.nonzero(sel == b, as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            sub = find_collisions(h[idx], src[idx], dist[idx], min_saving, bucket_max)
            out.extend((s, int(idx[lo]), int(idx[hi])) for s, lo, hi in sub)
        out.sort(key=lambda t: -t[0])
        return out
    dev = h.device
    order = torch.argsort(h)
    hs = h[order]
    a = (src[order].to(torch.int32) + dist[order].to(torch.int32))   # i + d
    b = (src[order].to(torch.int32) - dist[order].to(torch.int32))   # j - d
    bnd = torch.ones_like(hs, dtype=torch.bool)
    bnd[1:] = hs[1:] != hs[:-1]
    gid = torch.cumsum(bnd.to(torch.int64), 0) - 1
    ng = int(gid[-1].item()) + 1 if hs.numel() else 0
    if ng == 0:
        return []
    BIG = torch.iinfo(torch.int32).max // 4
    amin = torch.full((ng,), BIG, dtype=torch.int32, device=dev)
    amin.scatter_reduce_(0, gid, a, reduce="amin")
    bmax = torch.full((ng,), -BIG, dtype=torch.int32, device=dev)
    bmax.scatter_reduce_(0, gid, b, reduce="amax")
    sav = bmax - amin
    good = torch.nonzero(sav >= min_saving, as_tuple=True)[0]
    if good.numel() == 0:
        return []
    out = []
    good_set = torch.zeros(ng, dtype=torch.bool, device=dev)
    good_set[good] = True
    mask = good_set[gid]
    ent = torch.nonzero(mask, as_tuple=True)[0]
    g_ent, a_ent, b_ent = gid[ent], a[ent], b[ent]
    lo_hit = a_ent == amin[g_ent]
    hi_hit = b_ent == bmax[g_ent]
    lo = {}
    hi = {}
    for e, g, l, hh in zip(ent.tolist(), g_ent.tolist(), lo_hit.tolist(), hi_hit.tolist()):
        if l and g not in lo:
            lo[g] = e
        if hh and g not in hi:
            hi[g] = e
    for g in good.tolist():
        out.append((int(sav[g].item()), int(order[lo[g]].item()), int(order[hi[g]].item())))
    out.sort(key=lambda t: -t[0])
    return out


def group_min(h: torch.Tensor, a: torch.Tensor, b: torch.Tensor, limit: int):
    """Per distinct hash: min(a) + min(b). Returns [(total, entry_a, entry_b)] under limit.

    The cost-shaped twin of `find_collisions`: that one maximises a saving over path
    indices, this one minimises an explicit cost. Used for pooled bridging (71) and for
    pairwise endpoint MITM (72).
    """
    order = torch.argsort(h)
    hs, a_s, b_s = h[order], a[order], b[order]
    bnd = torch.ones_like(hs, dtype=torch.bool)
    bnd[1:] = hs[1:] != hs[:-1]
    gid = torch.cumsum(bnd.to(torch.int64), 0) - 1
    ng = int(gid[-1].item()) + 1
    BIG = torch.iinfo(torch.int32).max // 4
    amin = torch.full((ng,), BIG, dtype=torch.int32, device=h.device)
    amin.scatter_reduce_(0, gid, a_s, reduce="amin")
    bmin = torch.full((ng,), BIG, dtype=torch.int32, device=h.device)
    bmin.scatter_reduce_(0, gid, b_s, reduce="amin")
    tot = amin + bmin
    good = torch.nonzero(tot < limit, as_tuple=True)[0]
    if good.numel() == 0:
        return []
    gset = torch.zeros(ng, dtype=torch.bool, device=h.device)
    gset[good] = True
    ent = torch.nonzero(gset[gid], as_tuple=True)[0]
    g_e, a_e, b_e = gid[ent], a_s[ent], b_s[ent]
    lo, hi = {}, {}
    for e, g, av, bv in zip(ent.tolist(), g_e.tolist(), a_e.tolist(), b_e.tolist()):
        if g not in lo and av == int(amin[g]):
            lo[g] = e
        if g not in hi and bv == int(bmin[g]):
            hi[g] = e
    res = [(int(tot[g]), int(order[lo[g]]), int(order[hi[g]])) for g in good.tolist()]
    res.sort()
    return res


def word_to(par, mv, gidx: int):
    """Word from the source state to ball entry `gidx` (list of move indices)."""
    w = []
    i = gidx
    while par[i] >= 0:
        w.append(int(mv[i]))
        i = int(par[i])
    w.reverse()
    return w


def reduce_path(cube: Cube, s0: np.ndarray, word, radius: int, chunk: int,
                stats=None):
    """One pass of ball-collision shortening over a single path. Returns a new word."""
    L = len(word)
    if L < 3:
        return word
    states = cube.path_states_np(s0, word)                      # (L+1, 96)
    src_t = torch.as_tensor(states, device=cube.device)
    h, src, dist, par, mv = build_balls(cube, src_t, radius, chunk)
    cands = find_collisions(h, src, dist, min_saving=1)
    if not cands:
        del h, src, dist, par, mv, src_t
        if cube.device.type == "cuda":
            torch.cuda.empty_cache()
        return word
    par_c = par.cpu().numpy()
    mv_c = mv.cpu().numpy()
    src_c = src.cpu().numpy()
    dist_c = dist.cpu().numpy()
    del h, src, dist, par, mv, src_t
    if cube.device.type == "cuda":
        torch.cuda.empty_cache()
    # candidate splices, best saving first, applied on disjoint intervals
    taken = []
    used = np.zeros(L + 1, dtype=bool)
    for sav, e_lo, e_hi in cands:
        i, d1 = int(src_c[e_lo]), int(dist_c[e_lo])
        j, d2 = int(src_c[e_hi]), int(dist_c[e_hi])
        if j - i < 1 or (j - i) - (d1 + d2) != sav:
            continue
        if used[i:j].any():
            continue
        w1 = word_to(par_c, mv_c, e_lo)
        w2 = word_to(par_c, mv_c, e_hi)
        u = w1 + cube.inv_word(w2)
        if len(u) >= j - i:
            continue
        if not np.array_equal(cube.apply_word_np(states[i], u), states[j]):
            if stats is not None:
                stats["phantom"] = stats.get("phantom", 0) + 1
            continue
        taken.append((i, j, u))
        used[i:j] = True
    if not taken:
        return word
    taken.sort()
    out, prev = [], 0
    for i, j, u in taken:
        out.extend(word[prev:i])
        out.extend(u)
        prev = j
    out.extend(word[prev:])
    if stats is not None:
        stats["splices"] = stats.get("splices", 0) + len(taken)
    return out


def load_tests(data_dir: Path):
    tests = {}
    with io.open(data_dir / "test.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            tests[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.uint8)
    return tests


def load_paths(cube: Cube, path: Path):
    rows = {}
    with io.open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            p = r["path"].strip()
            rows[int(r["initial_state_id"])] = [cube.idx[m] for m in p.split(".")] if p else []
    return rows


def write_paths(cube: Cube, rows, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            wr.writerow([pid, ".".join(cube.names[m] for m in rows[pid])])


def verify_all(cube: Cube, tests, rows) -> int:
    bad = 0
    for pid, w in rows.items():
        if not np.array_equal(cube.apply_word_np(tests[pid], w), cube.solved_np):
            bad += 1
    return bad


def self_test(cube: Cube, radius: int, chunk: int) -> bool:
    """A word with a planted detour must come back at most as long, and still valid."""
    rng = np.random.default_rng(0)
    ok = True
    for t in range(6):
        base = rng.integers(0, cube.n_gen, size=30).tolist()
        # plant a detour: insert a move and its inverse in the middle (a 2-move saving)
        m = int(rng.integers(0, cube.n_gen))
        cut = int(rng.integers(5, 25))
        word = base[:cut] + [m, int(cube.inv_move_np[m])] + base[cut:]
        s0 = cube.solved_np.copy()
        s0 = cube.apply_word_np(s0, cube.inv_word(word))          # scramble so word solves it
        assert np.array_equal(cube.apply_word_np(s0, word), cube.solved_np)
        got = reduce_path(cube, s0, word, radius, chunk)
        valid = np.array_equal(cube.apply_word_np(s0, got), cube.solved_np)
        print(f"  case {t}: {len(word)} -> {len(got)}  valid={valid}")
        if not valid or len(got) > len(word) or len(got) >= len(word) - 1:
            ok = False
    print(f"self-test: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--radius", type=int, default=3, help="ball radius per endpoint; reach is 2x")
    ap.add_argument("--chunk", type=int, default=150_000, help="frontier states expanded per kernel")
    ap.add_argument("--passes", type=int, default=2, help="repeat until no change, at most this many")
    ap.add_argument("--pids", type=str, default="", help="comma list or a:b range; default all")
    ap.add_argument("--min-len", type=int, default=0, help="skip paths shorter than this")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--progress", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="rewrite --out every N pids so a long sweep survives a crash")
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cube = Cube(args.data_dir, dev)
    print(f"device {dev}  radius {args.radius} (reach {2*args.radius})", flush=True)

    if args.self_test:
        return 0 if self_test(cube, args.radius, args.chunk) else 1
    if not args.target:
        print("--target required")
        return 2

    tests = load_tests(args.data_dir)
    rows = load_paths(cube, args.target)
    base_total = sum(len(v) for v in rows.values())
    print(f"target {args.target.name}: {len(rows)} pids, {base_total:,} moves "
          f"(mean {base_total/len(rows):.3f})", flush=True)
    bad = verify_all(cube, tests, rows)
    print(f"input replay check: {bad} unsolved")
    if bad:
        return 3

    if args.pids:
        if ":" in args.pids:
            a, b = args.pids.split(":")
            todo = [p for p in sorted(rows) if int(a) <= p < int(b)]
        else:
            todo = [int(x) for x in args.pids.split(",")]
    else:
        todo = sorted(rows)
    todo = [p for p in todo if len(rows[p]) >= args.min_len]
    print(f"sweeping {len(todo)} pids", flush=True)

    stats = {}
    t0 = time.time()
    saved_total = 0
    for n, pid in enumerate(todo):
        w = rows[pid]
        for _ in range(args.passes):
            nw = reduce_path(cube, tests[pid], w, args.radius, args.chunk, stats)
            if len(nw) == len(w):
                w = nw
                break
            w = nw
        if len(w) < len(rows[pid]):
            d = len(rows[pid]) - len(w)
            saved_total += d
            print(f"  HIT pid {pid}: {len(rows[pid])} -> {len(w)}  (-{d})", flush=True)
            rows[pid] = w
        if args.progress and (n + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)} pids, saved {saved_total}, {el:.0f}s "
                  f"({el/(n+1):.2f}s/pid)", flush=True)
        if args.out and args.checkpoint_every and saved_total and \
                (n + 1) % args.checkpoint_every == 0:
            write_paths(cube, rows, args.out)
            print(f"  checkpoint -> {args.out} ({sum(len(v) for v in rows.values()):,})",
                  flush=True)

    total = sum(len(v) for v in rows.values())
    print(f"\nradius {args.radius}: saved {saved_total} moves  ({base_total:,} -> {total:,})")
    print(f"phantom rejects: {stats.get('phantom', 0)}  splices: {stats.get('splices', 0)}")
    bad = verify_all(cube, tests, rows)
    print(f"output replay check: {bad} unsolved")
    if bad:
        print("REFUSING TO WRITE -- replay failed")
        return 4
    if args.out and saved_total > 0:
        write_paths(cube, rows, args.out)
        print(f"wrote {args.out}")
    elif args.out:
        print("no saving -- not writing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
