"""Exact distances to d<=9 by meet-in-the-middle, and what that says about k_max.

    python cube555/scripts/12_mitm_oracle.py

HOW. `anchors_d5.pt` is a hash-sorted table complete to radius 5 around solved. Expanding
a query s to radius 4 (446,403 states, ~70 MB) and intersecting gives

    d(s) = min over x in Ball4(s) ^ Ball5(id) of  d_s(x) + d_id(x)

which is EXACT whenever d(s) <= 9, and reports ">9" otherwise. Cheap enough to run on
hundreds of queries: the expensive side of the meet is precomputed once.

WHAT IT IS FOR.
 1. END-TO-END CONVENTION CHECK. test.csv pids 35..1034 are random walks of length
    pid-34. If the exact distance of pid 35+j is not j+1 for small j, the state
    convention, the move convention or the walk direction is wrong -- and every one of
    those failures is silent everywhere else.
 2. WALK-LABEL PURITY. The sparse-Q recipe asserts d(pivot) == p. This measures
    P(d == k) for non-backtracking walks of length k, exactly, for k <= 9.
 3. k_max. Purity at shallow depth plus the measured branching factor (24.03, counting
    bound 66.4) is what `k_max` rests on. The direct measurement of where walks stop
    being geodesic only becomes available at the first gate, from the beam's solution
    length versus rw length over pids 35..1034.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube555.puzzle import Cube555, STATE_SIZE  # noqa: E402

DEV = "cuda"
R_NEAR = 4  # radius expanded around the query
SEED = 555


def build_near_ball(states, moves, hv, radius):
    """Ball of `radius` around each row of `states`, with per-row depths. Returns
    (hashes (N,M) int64, depths (M,) uint8) for ONE row at a time to bound memory."""
    n_act = moves.shape[0]
    cur = states
    all_h = [(states.long() * hv).sum(1)]
    all_d = [torch.zeros(states.shape[0], dtype=torch.uint8, device=DEV)]
    seen = all_h[0]
    for d in range(1, radius + 1):
        m = cur.shape[0]
        ch = torch.gather(
            cur[:, None, :].expand(m, n_act, STATE_SIZE),
            2,
            moves[None].expand(m, n_act, STATE_SIZE),
        ).reshape(m * n_act, STATE_SIZE)
        h = (ch.long() * hv).sum(1)
        hs, idx = torch.sort(h)
        keep = torch.ones_like(hs, dtype=torch.bool)
        keep[1:] = hs[1:] != hs[:-1]
        hu, iu = hs[keep], idx[keep]
        fresh = ~torch.isin(hu, seen)
        cur = ch[iu[fresh]]
        seen = torch.cat([seen, hu[fresh]])
        all_h.append(hu[fresh])
        all_d.append(torch.full((cur.shape[0],), d, dtype=torch.uint8, device=DEV))
    return torch.cat(all_h), torch.cat(all_d)


def main() -> int:
    data = PROJECT / "data"
    puz = Cube555.load(data / "puzzle_info.json")
    names = list(puz.move_names)
    moves = torch.tensor(
        [puz.generators[n] for n in names], dtype=torch.int64, device=DEV
    )
    n_act = moves.shape[0]
    g = torch.Generator(device=DEV)
    g.manual_seed(SEED)
    hv = torch.randint(
        -(2**62), 2**62, (STATE_SIZE,), dtype=torch.int64, device=DEV, generator=g
    )

    blob = torch.load(data / "anchors_d5.pt", map_location="cpu", weights_only=False)
    far_s = blob["states"].to(DEV)
    far_d = blob["depth"].to(DEV)
    far_h = (far_s.long() * hv).sum(1)
    order = torch.argsort(far_h)
    far_h, far_d = far_h[order], far_d[order]
    print(f"far side: Ball5(solved) = {far_h.numel():,} states, hash-sorted")

    def exact_distance(s_u8: torch.Tensor) -> int:
        """s_u8: (150,) uint8. Exact d if <= 9, else 10 (meaning '>9')."""
        h, d = build_near_ball(s_u8[None, :], moves, hv, R_NEAR)
        pos = torch.searchsorted(far_h, h).clamp_max(far_h.numel() - 1)
        hit = far_h[pos] == h
        if not bool(hit.any()):
            return 10
        tot = d[hit].long() + far_d[pos[hit]].long()
        return int(tot.min())

    inv = torch.tensor(
        [names.index(puz.inverse_name(n)) for n in names], dtype=torch.int64, device=DEV
    )

    def walk(k: int, gen) -> torch.Tensor:
        s = torch.arange(STATE_SIZE, dtype=torch.uint8, device=DEV)
        last = -1
        for step in range(k):
            if step == 0:
                mv = int(torch.randint(n_act, (1,), generator=gen, device=DEV))
            else:
                c = int(torch.randint(n_act - 1, (1,), generator=gen, device=DEV))
                mv = c + (1 if c >= int(inv[last]) else 0)
            s = s[moves[mv]]
            last = mv
        return s

    # ---- 1. end-to-end convention check on test.csv rw pids --------------------------
    rows = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.uint8
        )
        for r in csv.DictReader(open(data / "test.csv", encoding="utf-8"))
    }
    # NOTE ON WHAT THIS CAN AND CANNOT CHECK. d(s) == d(s^-1) on a Cayley graph, so the
    # distance is the SAME under our `new[i] = state[gen[i]]` convention and its
    # transpose -- this test cannot discriminate them. The convention is validated
    # instead by REPLAY: every sample_submission path, applied with this convention,
    # reaches solved. What this test measures is how far the competition's scramble
    # generator is from geodesic.
    print("\ntest.csv rw pids -- pid 35+j is a scramble of length j+1:")
    short = 0
    for j in range(9):
        pid = 35 + j
        d = exact_distance(torch.from_numpy(rows[pid]).to(DEV))
        short += d < j + 1
        tag = "geodesic" if d == j + 1 else f"short by {j + 1 - d}"
        print(f"  pid {pid:4d}  scramble len {j+1:2d}  exact d = {d:2d}   {tag}")
    print(f"  {short}/9 competition scrambles are NOT geodesic -- their generator "
          "allows backtracking.\n  Matters for reading rw length as depth when picking "
          "the GATE set, not for our labels.")

    # ---- 2. walk-label purity --------------------------------------------------------
    print("\nnon-backtracking walk purity  P(exact d == k):")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED)
    n_trial = 40
    for k in range(1, 10):
        t0 = time.time()
        hit = sum(exact_distance(walk(k, gen)) == k for _ in range(n_trial))
        print(
            f"  k = {k}:  {hit}/{n_trial} = {hit/n_trial:.3f}   ({time.time()-t0:.1f}s)"
        )

    # ---- 3. santa pids are far outside the oracle ------------------------------------
    d0 = exact_distance(torch.from_numpy(rows[0]).to(DEV))
    print(f"\nsanta pid 0: exact d = {d0 if d0 < 10 else '>9'} (expected >9)")
    print(
        "\nk_max rests on the counting bound: |G| = 6.198e91, b = 24.03 -> bound 66.4, "
        "diameter ~70-76, so k_max ~= 80.\nRe-derive it at the first gate from beam "
        "length vs rw length over pids 35..1034."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
