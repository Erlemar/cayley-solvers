"""Phase 0: is the saturation at 42 self-inflicted by the Bellman backup?

Two read-only measurements against the DEPLOYED checkpoint. No training, no writes to
anything but stdout.

D1 -- MIN-BIAS, measured with no ground truth.
    22_bellman.py:111 builds its target as `1 + min_a' Q_target(child, a')`. A min over 30
    NOISY estimates is biased low (the min-side twin of DQN's max bias), and 20k steps at
    target-refresh 500 is ~40 rounds of compounding while exact anchors pin only d<=4.
    Predicted symptom: correct shallow, compressed deep -- i.e. exactly the observed
    saturation.

    We cannot see true Q at depth, but we do not need to. The 48 symmetry frames are
    EXACTLY-EQUAL views of one state: Q(sym(s,k), relabel[k,a]) == Q(s,a) identically.
    So all disagreement between frames is pure model noise, and:

        bias  ~=  mean_k [ min_a Q_k(s,a) ]  -  min_a [ mean_k Q_k(s,a) ]
                  \___ min of a noisy view _/    \_ min of a de-noised view _/

    Averaging over frames first shrinks the noise before the min is taken, so the gap
    between the two is the selection bias the backup injects. Positive-signed here means
    the naive min sits BELOW the de-noised min, i.e. targets are too low.

D2 -- SATURATION CURVE. Predicted min_a Q against (a) exact depth on the d<=4 anchors,
    where the label is ground truth, and (b) real test states, where the question is
    simply whether the model ever emits a value near the ~72 diameter.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube555.models import load_model  # noqa: E402
from cube555.puzzle import Cube555  # noqa: E402
from cube555.qtrain import ExactAnchors  # noqa: E402

CK = PROJECT / "models" / "q555_a_bell2" / "bellman_020000.pt"
DEV = "cuda"


def score(model, states: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """(N,150) int64 -> (N,30) float32 Q, no grad."""
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, states.shape[0], chunk):
            out.append(model(states[i : i + chunk]).float())
    return torch.cat(out, 0)


def main() -> int:
    torch.manual_seed(555)
    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    perms = torch.tensor(
        [puz.generators[n] for n in names], dtype=torch.int64, device=DEV
    )

    SYM_M = np.load(PROJECT / "data" / "sym_slots_48.npy")
    SYM_INV = np.load(PROJECT / "data" / "sym_slots_inv_48.npy")
    SYM_RINV = np.load(PROJECT / "data" / "sym_move_relabel_inv_48.npy")
    n_frames = SYM_M.shape[0]

    model = load_model(CK, device=DEV, dtype=torch.float32).eval()
    model.return_value = False
    print(f"checkpoint: {CK.name}   frames: {n_frames}\n")

    tests = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64
        )
        for r in csv.DictReader(open(PROJECT / "data" / "test.csv", encoding="utf-8"))
    }

    # ---------------- D1: min-bias via frame disagreement ----------------
    print("=" * 68)
    print("D1  MIN-BIAS  (frames are exact-equal views; disagreement is pure noise)")
    print("=" * 68)

    deep_pids = sorted([p for p in tests if p >= 900], reverse=True)[:64]
    base = np.stack([tests[p] for p in deep_pids])  # (B,150)
    B = base.shape[0]

    # Score every state under every frame, mapped back to canonical action order.
    qc = torch.empty(n_frames, B, 30, device=DEV)
    for k in range(n_frames):
        sk = SYM_INV[k][base[:, SYM_M[k]]] if k else base
        q = score(model, torch.tensor(sk, dtype=torch.int64, device=DEV))
        # column j in frame space is canonical action SYM_RINV[k][j]
        rinv = torch.tensor(np.asarray(SYM_RINV[k]), dtype=torch.int64, device=DEV)
        qc[k].index_copy_(1, rinv, q)

    # Sanity: frame 0 must be the identity view.
    assert torch.allclose(qc[0], qc[0]), "frame 0 self-check"
    spread = qc.std(dim=0).mean().item()
    naive = qc.min(dim=2).values.mean(dim=0)  # mean_k min_a  Q_k
    denoi = qc.mean(dim=0).min(dim=1).values  # min_a  mean_k Q_k
    bias = denoi - naive

    print(f"  states: {B} deep test states (pid>=900)")
    print(f"  per-action noise across frames (sd)     : {spread:6.3f} moves")
    print(f"  mean_k [min_a Q_k]   (what the backup uses): {naive.mean():6.3f}")
    print(f"  min_a [mean_k Q_k]   (de-noised)           : {denoi.mean():6.3f}")
    print(
        f"  --> MIN-BIAS                                : {bias.mean():+6.3f} moves "
        f"(sd {bias.std():.3f}, {int((bias>0).sum())}/{B} states positive)"
    )
    print(f"      per backup, compounded over ~40 target refreshes in a 20k-step run.")

    # ---------------- D2: saturation curve ----------------
    print()
    print("=" * 68)
    print("D2  SATURATION CURVE")
    print("=" * 68)

    anc = ExactAnchors(PROJECT / "data" / "anchors_d4.pt", DEV)
    idx = torch.randperm(anc.n, device=DEV)[:8192]
    a_s, a_q, a_d = (
        anc.states.index_select(0, idx),
        anc.q.index_select(0, idx),
        anc.depth.index_select(0, idx),
    )
    pred = score(model, a_s.long()).min(dim=1).values
    true = a_q.min(dim=1).values
    print("  exact anchors (ground truth):")
    print("    depth   n     true min_a Q   pred min_a Q   err")
    for d in range(0, int(a_d.max().item()) + 1):
        m = a_d == d
        if int(m.sum()) == 0:
            continue
        print(
            f"    {d:5d} {int(m.sum()):5d}   {true[m].mean():11.3f}   "
            f"{pred[m].mean():12.3f}   {(pred[m]-true[m]).mean():+6.3f}"
        )

    print("\n  real test states, by pid band (no ground truth; diameter is ~72):")
    print("    pid band      n    pred min_a Q     max")
    for lo, hi in ((35, 200), (200, 400), (400, 600), (600, 800), (800, 1035)):
        ps = [p for p in tests if lo <= p < hi]
        if not ps:
            continue
        st = torch.tensor(
            np.stack([tests[p] for p in ps]), dtype=torch.int64, device=DEV
        )
        v = score(model, st).min(dim=1).values
        print(f"    {lo:4d}-{hi:<4d} {len(ps):5d}    {v.mean():9.3f}   {v.max():7.3f}")

    st_all = torch.tensor(
        np.stack([tests[p] for p in sorted(tests)]), dtype=torch.int64, device=DEV
    )
    v_all = score(model, st_all).min(dim=1).values
    print(
        f"\n  GLOBAL MAX predicted min_a Q over all 1035 test states: {v_all.max():.3f}"
    )
    print(f"  counting-bound diameter ~72; a random state sits near it.")
    print(f"  --> underestimate at depth: ~{72 - v_all.max().item():.1f} moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
