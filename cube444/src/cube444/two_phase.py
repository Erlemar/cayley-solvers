"""The two-phase 4x4x4 solver: neural reduction beam, then a classical 3x3x3 finish.

    phase 1   all 24 generators, beam guided by the masked V, stop on R
    phase 2   the 12 outer generators, solved classically (near-optimal in QTM)

Phase 1 returns MANY reduced endpoints rather than the first one it finds. That
matters: phase-2 cost varies by several moves between endpoints, and 4x4x4 parity
makes some endpoints unsolvable outright, so the driver needs alternatives. The
total is minimised jointly, not greedily per phase.

    total(e) = len(path_to_e) + len(phase2(e))

which is why a slightly longer reduction can win overall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from cube444.phase1 import ReductionTest


@dataclass
class Phase1Result:
    endpoints: list[tuple[list[str], np.ndarray]] = field(default_factory=list)
    steps_run: int = 0
    exhausted: bool = False


class Phase1Beam:
    """Level-synchronous beam over all 24 generators, stopping on the set R."""

    def __init__(self, puzzle, model, device: str = "cuda",
                 beam_width: int = 65536, max_steps: int = 40,
                 score_chunk: int = 262144, seed: int = 0):
        self.names = list(puzzle.move_names)
        self.n_gen = len(self.names)
        self.device = device
        self.model = model.to(device).eval()
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.score_chunk = score_chunk
        self.gens = torch.tensor(
            np.stack([np.asarray(puzzle.generators[n], dtype=np.int64)
                      for n in self.names]),
            dtype=torch.long, device=device,
        )
        self.solved = torch.tensor(np.asarray(puzzle.solved_state, dtype=np.int64),
                                   dtype=torch.long, device=device)
        self.rtest = ReductionTest(device=device)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # random hash vector for state dedup (collisions only ever DROP a
        # duplicate candidate, which a beam tolerates)
        self.hash_vec = torch.randint(
            -(2 ** 62), 2 ** 62, (96,), generator=g, dtype=torch.long, device=device
        )

    def _hash(self, states: torch.Tensor) -> torch.Tensor:
        return (states * self.hash_vec).sum(-1)

    @torch.no_grad()
    def _score(self, states: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, states.shape[0], self.score_chunk):
            outs.append(self.model(states[i:i + self.score_chunk]).float())
        return torch.cat(outs) if outs else torch.zeros(0, device=self.device)

    @torch.no_grad()
    def run(self, start: np.ndarray, want_endpoints: int = 16,
            bf16: bool = True) -> Phase1Result:
        dev = self.device
        s0 = torch.as_tensor(np.asarray(start, dtype=np.int64), dtype=torch.long,
                             device=dev).unsqueeze(0)
        res = Phase1Result()
        if bool(self.rtest.is_reduced(s0)[0]):
            res.endpoints.append(([], np.asarray(start, dtype=np.int64)))
            return res

        beam = s0                                            # (n, 96)
        paths = torch.zeros((1, self.max_steps), dtype=torch.int8, device=dev)
        seen_hash = self._hash(beam)

        ctx = (torch.autocast(dev, dtype=torch.bfloat16)
               if bf16 and dev == "cuda" else torch.no_grad())
        for step in range(self.max_steps):
            n = beam.shape[0]
            kids = torch.gather(
                beam.unsqueeze(1).expand(n, self.n_gen, 96), 2,
                self.gens.unsqueeze(0).expand(n, self.n_gen, 96),
            ).reshape(n * self.n_gen, 96)
            parent = torch.arange(n, device=dev).repeat_interleave(self.n_gen)
            move = torch.arange(self.n_gen, device=dev).repeat(n)

            # dedup within the level, and against everything already expanded
            h = self._hash(kids)
            uniq, inv = torch.unique(h, return_inverse=True)
            first = torch.full((uniq.shape[0],), -1, dtype=torch.long, device=dev)
            first.scatter_reduce_(0, inv, torch.arange(h.shape[0], device=dev),
                                  reduce="amin", include_self=False)
            kids, parent, move, h = kids[first], parent[first], move[first], h[first]
            fresh = ~torch.isin(h, seen_hash)
            kids, parent, move, h = kids[fresh], parent[fresh], move[fresh], h[fresh]
            if kids.shape[0] == 0:
                res.exhausted = True
                break

            # harvest reduced states as endpoints
            red = self.rtest.is_reduced(kids)
            if bool(red.any()):
                idx = torch.nonzero(red, as_tuple=True)[0]
                # prefer the cheapest-to-reach; all are at the same depth here
                for j in idx[: max(0, want_endpoints - len(res.endpoints))].tolist():
                    p = paths[parent[j]].clone()
                    p[step] = move[j]
                    word = [self.names[int(m)] for m in p[: step + 1].tolist()]
                    res.endpoints.append((word, kids[j].cpu().numpy()))
                if len(res.endpoints) >= want_endpoints:
                    res.steps_run = step + 1
                    return res
                keep_mask = ~red
                kids, parent, move, h = (kids[keep_mask], parent[keep_mask],
                                         move[keep_mask], h[keep_mask])
                if kids.shape[0] == 0:
                    res.exhausted = True
                    break

            with ctx:
                scores = self._score(kids)
            k = min(self.beam_width, scores.shape[0])
            top = torch.topk(scores, k, largest=False).indices
            new_paths = paths[parent[top]].clone()
            new_paths[:, step] = move[top]
            beam, paths = kids[top], new_paths
            seen_hash = torch.cat([seen_hash, h[top]])
            res.steps_run = step + 1

        return res


def solve_two_phase(state: np.ndarray, beam: Phase1Beam, phase2, puzzle,
                    want_endpoints: int = 16, verbose: bool = False) -> dict:
    """Full two-phase solve. Returns the best word plus per-phase accounting."""
    gens = {k: np.asarray(v, dtype=np.int64) for k, v in puzzle.generators.items()}
    solved = np.asarray(puzzle.solved_state, dtype=np.int64)

    r = beam.run(state, want_endpoints=want_endpoints)
    out = {
        "found": False, "word": None, "total": None,
        "p1_len": None, "p2_len": None,
        "n_endpoints": len(r.endpoints), "steps_run": r.steps_run,
        "n_parity_fail": 0, "verified": False,
    }
    if not r.endpoints:
        return out

    best = None
    for word1, endpoint in r.endpoints:
        word2 = phase2.solve(endpoint)
        if word2 is None:
            out["n_parity_fail"] += 1
            continue
        tot = len(word1) + len(word2)
        if best is None or tot < best[0]:
            best = (tot, word1, word2)
        if verbose:
            print(f"    endpoint p1={len(word1)} p2={len(word2)} total={tot}")
    if best is None:
        return out

    tot, word1, word2 = best
    word = word1 + word2
    cur = np.asarray(state, dtype=np.int64)
    for m in word:
        cur = cur[gens[m]]
    out.update(found=True, word=word, total=tot, p1_len=len(word1),
               p2_len=len(word2), verified=bool(np.array_equal(cur, solved)))
    return out
