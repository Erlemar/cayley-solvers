"""Bidirectional meet-in-the-middle probe support (big-swings Direction 1).

This is the *frontier-exposing* half of the bidirectional experiment. The core
trick (see `scripts/91_bidir_probe.py` for the full driver and the group-theory
derivation) is:

    The backward beam from solved (identity) toward the scramble `s` is exactly
    the EXISTING forward solver run on `s^-1`.

    Under the relabelling  b_tilde = s^-1 . b  (left-multiply by s^-1):
      * a generator move acts identically:  b.g  <->  b_tilde.g
      * d(b, s) = |b_tilde| = V(b_tilde)        (so the same V guides it)
      * the word the solver APPLIES to s^-1 equals the e->b word w_b exactly.

So we never train or build a second model: we run `beam_frontiers(s, ...)` and
`beam_frontiers(invert(s), ...)`, then look for forward node `f` and backward
node `b = s[b_tilde]` whose residual `z = b^-1 . f` has small V(z) (the residual
lands in V's calibrated, low-variance near-solved band — the whole point of the
direction). We close the residual exactly and splice.

`KhoruzhiiSolver` only returns the final solved path. This subclass adds
`beam_frontiers(...)` which runs a single width-B beam for a fixed number of
steps and returns EVERY layer's frontier states plus the backpointer trees, so
the driver can reconstruct the word to any frontier node at any depth.
"""
from __future__ import annotations

import numpy as np
import torch

from beam_search import KhoruzhiiSolver, Profile, _state_hash


class BidirSolver(KhoruzhiiSolver):
    """KhoruzhiiSolver + a frontier-snapshotting beam for MITM meeting search."""

    def beam_frontiers(self, initial_state, beam_width: int, max_depth: int):
        """Run ONE width-B beam from `initial_state` for `max_depth` steps.

        Single attempt, no stagnation/solved early-break (we want the full
        frontier at every depth). Returns a dict:
            frontier   : list over step j (0-indexed; depth = j+1) of np.int8
                         arrays (n_j, S) = the kept states at that layer.
            tree_move  : np.int16 (D, B)  move index that produced each slot
            tree_idx   : np.int32 (D, B)  parent slot index in the prior layer
            solved_depth: first step j at which solved appeared, else None
            sizes      : list of n_j
        Word to any node (j, pos) is reconstructed with `reconstruct_words`.
        """
        B = beam_width
        state = torch.tensor(list(initial_state), dtype=self.state_dtype, device=self.device)
        states = state.unsqueeze(0).clone()
        states_hashed = _state_hash(states, self.hash_vec, self.internal_batch_size)
        states_bad = torch.empty(0, dtype=torch.int64, device=self.device)

        tree_move = torch.full((max_depth, B), -1, dtype=torch.int32, device=self.device)
        tree_idx = torch.full((max_depth, B), -1, dtype=torch.int32, device=self.device)
        frontier: list[np.ndarray] = []
        prof = Profile()
        solved_depth = None

        for j in range(max_depth):
            states, _vals, moves, parents, chosen_hashes = self._do_greedy_step(
                states, states_hashed, states_bad, B, prof, use_incremental_hash=True,
            )
            if states.numel() == 0:
                break
            leaves = states.size(0)
            tree_move[j, :leaves] = moves.to(torch.int32)
            tree_idx[j, :leaves] = parents.to(torch.int32)
            states_hashed = chosen_hashes
            frontier.append(states.to(torch.int8).cpu().numpy().copy())
            if solved_depth is None and bool((states == self.V0).all(dim=1).any().item()):
                solved_depth = j

        D = len(frontier)
        return {
            "frontier": frontier,
            "tree_move": tree_move[:D].cpu().numpy(),
            "tree_idx": tree_idx[:D].cpu().numpy(),
            "solved_depth": solved_depth,
            "sizes": [f.shape[0] for f in frontier],
        }

    @staticmethod
    def reconstruct_words(tree_move: np.ndarray, tree_idx: np.ndarray, j: int,
                          positions: np.ndarray) -> np.ndarray:
        """Reconstruct the root->node move-index words for `positions` at step `j`.

        Returns int array (len(positions), j+1); row k is the word (generator
        indices, in apply order) from the root to node `positions[k]`.
        """
        positions = np.asarray(positions, dtype=np.int64)
        npos = positions.shape[0]
        words = np.empty((j + 1, npos), dtype=np.int64)
        cur = positions.copy()
        for step in range(j, -1, -1):
            words[step] = tree_move[step, cur]
            cur = tree_idx[step, cur].astype(np.int64)
        return words.T  # (npos, j+1)

    def eval_V(self, states_np: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Batched V on a numpy (N, S) int array. Returns float32 (N,)."""
        from beam_search import _model_predict
        bs = batch_size or self.internal_batch_size
        t = torch.from_numpy(np.ascontiguousarray(states_np)).to(
            device=self.device, dtype=self.state_dtype
        )
        out = _model_predict(self.model, t, bs, pad_to_batch_size=self.pad_to_batch_size)
        return out.float().cpu().numpy()

    def penultimate_embed(self, states_np: np.ndarray, batch_size: int | None = None):
        """Best-effort penultimate-layer embedding phi(state), for the
        front-to-front ANN feasibility check (does ||phi(f)-phi(b)|| track
        d(f,b)?). Registers a forward hook on the last module before the scalar
        head. Returns float32 (N, H) or None if the architecture isn't a
        recognizable ResMLP-with-head.
        """
        base = getattr(self.model, "_orig_mod", self.model)
        head = None
        for attr in ("head", "out", "fc_out", "output", "value_head"):
            if hasattr(base, attr):
                head = getattr(base, attr)
                break
        if head is None:
            return None
        captured = {}

        def hook(_m, inp, _out):
            captured["z"] = inp[0].detach().float().cpu()

        h = head.register_forward_hook(hook)
        try:
            bs = batch_size or self.internal_batch_size
            embs = []
            t = torch.from_numpy(np.ascontiguousarray(states_np)).to(
                device=self.device, dtype=self.state_dtype
            )
            with torch.inference_mode():
                for i in range(0, t.size(0), bs):
                    captured.clear()
                    _ = self.model(t[i:i + bs].long())
                    if "z" not in captured:
                        return None
                    embs.append(captured["z"].numpy().copy())
            return np.concatenate(embs, axis=0)
        finally:
            h.remove()
