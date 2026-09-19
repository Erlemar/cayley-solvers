"""Static-shape beam search that runs entirely on TorchTPU, single or multi chip.

Pure PyTorch -- no JAX, no `jax_op` escape hatch. The algorithm mirrors the JAX
SPMD kernel in `kaggle_notebooks/tpu_beam_tetraminx/`, but every op is one that
torch_tpu registers.

WHY THE EXISTING SOLVER COULD NOT BE PORTED
-------------------------------------------
`cayley.khoruzhii_search.KhoruzhiiSolver` is TPU-hostile three ways:
`torch.unique` (torch_tpu implements it with a `.cpu().item()` to learn the
output size -- host round-trip AND a recompile per distinct count), boolean-mask
indexing (`nonzero` in disguise), and genuinely DYNAMIC shapes
(`cand.index_select(0, unique_pos[:select_B])`). `torch.compile` on TPU requires
`dynamic=False`. So this is a rewrite.

STATIC DEDUP
------------
`unique` is replaced by sort + adjacent-compare + inverse permutation:

    sort_h, sort_idx = h.sort()
    dup_sorted      = cat([False, sort_h[1:] == sort_h[:-1]])
    dup             = dup_sorted[sort_idx.argsort()]

Duplicates are never REMOVED (that would be a dynamic shape) -- they are scored
+BIG so `topk` skips them.

MULTI-CHIP
----------
One code path serves any world size; W=1 degenerates naturally (every child is
owned locally and the exchange is an identity copy), so the single-chip path
cannot rot separately.

Each rank owns `B_local = B_global // W` beam slots. Per step:

  1. expand + score all `24 * B_local` local children
  2. assign each child an OWNER from its hash, so identical states across ranks
     land on the same rank and dedup is global rather than per-rank
  3. per-owner `topk(K)` -> a (W, K) send buffer
  4. `all_to_all_single` so bucket r reaches rank r
  5. dedup the `K*W` received candidates, `topk(B_local)` -> the new local beam

Scores travel with the candidates in one extra float32 exchange. JAX's direct-Q
path also transmits parent-action scores, using bf16. The model forward is the
largest part of the measured step time.

Compiled TPU runs can use TpuBeamConfig.model_scan to keep one model body in the
graph across all fixed-size chunks. This reduces compiled code storage and cold
compilation time; the measured steady step time is unchanged. The CLI enables
scan for compiled TPU runs and provides --model-loop for comparison.

`all_reduce(MIN)` on int64 fails to compile on torch_tpu, so the cross-rank
first-hit latch uses `all_gather` plus a host-side min. That is also what
[[spmd_jax_dead_end]] recommends: aggregate host-side rather than inside the
per-step body.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.distributed as dist
from torch_tpu._internal.sync import sync

# Sentinel score. Anything masked with this is never selected by topk, and any
# survivor still carrying it is padding rather than a real state.
BIG = 1.0e9


@dataclasses.dataclass
class TpuBeamConfig:
    beam_width: int = 1 << 20          # GLOBAL beam, split across ranks
    max_steps: int = 60
    # Rows per model forward. A FIXED chunk count keeps shapes static.
    model_chunk: int = 8192
    # Send-bucket overshoot: each rank receives alpha * B_local candidates.
    alpha: int = 2
    # Steps between host syncs of the hit latch. Every sync ends the deferred
    # graph, so this trades wasted steps against pipeline breaks.
    check_every: int = 4
    bf16: bool = True
    # Keep one model body in the compiled graph instead of one per chunk.
    model_scan: bool = False


def _zobrist(states: torch.Tensor, ztab: torch.Tensor) -> torch.Tensor:
    """Zobrist hash matching EndgameTable.hash (XOR over per-slot tables).

    Loops over the 88 slots rather than gathering an (N, 88) int64 intermediate,
    which at beam scale would be gigabytes; the loop holds one (N,) int64 at a
    time and fuses under compile.
    """
    n, s = states.shape
    h = torch.zeros(n, dtype=torch.int64, device=states.device)
    for i in range(s):
        h = torch.bitwise_xor(h, ztab[i].index_select(0, states[:, i].to(torch.int64)))
    return h


class TpuBeamSearch:
    """One puzzle, one beam. All device state, all fixed shapes."""

    def __init__(
        self,
        model,
        generators: torch.Tensor,      # (n_gen, S) int64 permutation table
        ztab: torch.Tensor,            # (S, num_classes) int64, endgame Zobrist
        table_hashes: torch.Tensor,    # (N,) int64 SORTED endgame hashes
        table_depths: torch.Tensor,    # (N,) int8
        cfg: TpuBeamConfig,
        device: str = "tpu",
        seed: int = 0,
        rank: int = 0,
        world: int = 1,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.rank, self.world = rank, world
        self.gens = generators.to(device)
        self.n_gen, self.state_size = self.gens.shape
        self.gens_flat = self.gens.reshape(-1)
        self.ztab = ztab.to(device)
        self.table_hashes = table_hashes.to(device)
        self.table_depths = table_depths.to(device)

        if cfg.beam_width % world:
            raise ValueError(f"beam_width {cfg.beam_width} must divide world {world}")
        self.b_local = cfg.beam_width // world
        if self.b_local % cfg.model_chunk:
            raise ValueError(
                f"per-rank beam {self.b_local} must be divisible by model_chunk "
                f"{cfg.model_chunk} -- a ragged final chunk is a second shape and "
                "forces a recompile")
        self.n_chunks = self.b_local // cfg.model_chunk
        # Per-peer bucket. Each rank ends up with alpha * B_local candidates.
        self.k_peer = max(1, cfg.alpha * self.b_local // world)
        self.recv_n = self.k_peer * world

        # Same seed on every rank: the owner map must agree globally or a state
        # can survive on two ranks at once.
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        hv = torch.randint(1, 2**62, (self.state_size,), generator=g, dtype=torch.int64)
        self.hash_vec = (hv * 2 + 1).to(device)          # odd multipliers
        # Build in int64 and cast AFTER the divide: arange(recv_n) does not fit
        # in int8 (recv_n is alpha*B_local, e.g. 16384) and would wrap silently.
        # The quotient does fit -- it is a rank id.
        self.sender_rank = torch.arange(
            self.recv_n, device=device, dtype=torch.int64
        ).div(self.k_peer, rounding_mode="floor").to(torch.int8)
        if cfg.model_scan:
            # A scalar created inside the exported scan can remain deferred and
            # trip TorchTPU's compiled-mode assertion. Materialize it once here.
            self._scan_carry = torch.zeros((), dtype=torch.int32, device=device)
            sync.synchronize(None, wait=True)

    # ------------------------------------------------------------------ model
    def _q_scores(self, states: torch.Tensor) -> torch.Tensor:
        """(B_local, S) -> (B_local, n_gen). One forward per PARENT, not per child.

        The all-neighbours Q head scores every child of a state, so the model
        runs on B_local rows rather than B_local * n_gen. Biggest constant factor
        in the whole step.
        """
        if self.cfg.model_scan:
            from torch._higher_order_ops import scan
            chunks = states.reshape(self.n_chunks, self.cfg.model_chunk, self.state_size)

            def apply_batch(carry, chunk):
                out = self.model(chunk.to(torch.int64))
                if isinstance(out, tuple):
                    out = out[0]
                return carry.clone(), out.float()

            _, outs = scan(apply_batch, self._scan_carry, chunks)
            return outs.reshape(self.b_local, self.n_gen)
        outs = []
        for c in range(self.n_chunks):
            lo = c * self.cfg.model_chunk
            # int8 storage, int64 only for the slice the model is about to see.
            chunk = states[lo:lo + self.cfg.model_chunk].to(torch.int64)
            out = self.model(chunk)
            if isinstance(out, tuple):
                out = out[0]
            outs.append(out.float())
        return torch.cat(outs, 0)

    def _hash(self, states: torch.Tensor) -> torch.Tensor:
        """Cheap dot-product hash, accumulated slot by slot.

        `states.to(int64) * hash_vec` would materialise an (N, 88) int64 tensor --
        gigabytes at beam scale, against N*88 bytes for the int8 states. This
        holds one (N,) int64 at a time.
        """
        n = states.shape[0]
        h = torch.zeros(n, dtype=torch.int64, device=states.device)
        for i in range(self.state_size):
            h = h + states[:, i].to(torch.int64) * self.hash_vec[i]
        return h

    # ------------------------------------------------------------------- step
    def step(self, states: torch.Tensor, valid: torch.Tensor):
        """One beam level. Every shape is fixed by (B_local, n_gen, S, K, W)."""
        Bl, W, K = self.b_local, self.world, self.k_peer
        n_gen, S = self.n_gen, self.state_size
        dev = states.device

        # 1. children of the locally owned states
        children = states.index_select(1, self.gens_flat).view(Bl * n_gen, S)
        score = self._q_scores(states).reshape(Bl * n_gen)
        parent_local = torch.arange(Bl * n_gen, device=dev, dtype=torch.int32)
        move = (parent_local % n_gen).to(torch.int8)
        parent_local = torch.div(parent_local, n_gen, rounding_mode="floor")

        h = self._hash(children)
        big = torch.full_like(score, BIG)
        # a child of a padding parent is itself padding
        pvalid = valid.unsqueeze(1).expand(Bl, n_gen).reshape(-1)
        score = torch.where(pvalid, score, big)

        # 2. owner from the hash. xorshift-mix first: the low bits of a plain
        #    dot-product hash are poorly mixed, and an uneven owner split wastes
        #    send-bucket capacity. `& (W-1)` is non-negative even for a negative
        #    h in two's complement.
        mixed = torch.bitwise_xor(h, torch.bitwise_right_shift(h, 29))
        owner = torch.bitwise_and(mixed, W - 1) if W > 1 else None

        # 3. per-owner top-K into the send buffer
        send_states = torch.empty((W * K, S), dtype=torch.int8, device=dev)
        send_parent = torch.empty(W * K, dtype=torch.int32, device=dev)
        send_move = torch.empty(W * K, dtype=torch.int8, device=dev)
        send_score = torch.empty(W * K, dtype=torch.float32, device=dev)
        for r in range(W):
            s_r = score if W == 1 else torch.where(owner == r, score, big)
            top_v, top_i = torch.topk(s_r, K, largest=False, sorted=True)
            sl = slice(r * K, (r + 1) * K)
            send_states[sl] = children.index_select(0, top_i)
            send_parent[sl] = parent_local.index_select(0, top_i)
            send_move[sl] = move.index_select(0, top_i)
            send_score[sl] = top_v

        # 4. exchange: bucket r goes to rank r
        if W > 1:
            recv_states = torch.empty_like(send_states)
            recv_parent = torch.empty_like(send_parent)
            recv_move = torch.empty_like(send_move)
            recv_score = torch.empty_like(send_score)
            dist.all_to_all_single(recv_states, send_states)
            dist.all_to_all_single(recv_parent, send_parent)
            dist.all_to_all_single(recv_move, send_move)
            dist.all_to_all_single(recv_score, send_score)
        else:
            recv_states, recv_parent = send_states, send_parent
            recv_move, recv_score = send_move, send_score
        recv_rank = self.sender_rank

        # 5. dedup the received pool. Because owner() is a pure function of the
        #    state, every copy of a state anywhere in the mesh arrives HERE, so
        #    this local dedup is globally exact.
        rh = self._hash(recv_states)
        sort_h, sort_idx = torch.sort(rh)
        dup_sorted = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=dev),
            sort_h[1:] == sort_h[:-1],
        ])
        dup = dup_sorted.index_select(0, torch.argsort(sort_idx))
        rbig = torch.full_like(recv_score, BIG)
        rscore = torch.where(dup, rbig, recv_score)

        # 6. final selection
        top_score, keep = torch.topk(rscore, Bl, largest=False, sorted=True)
        new_states = recv_states.index_select(0, keep)
        new_valid = top_score < (BIG * 0.5)
        out_parent = recv_parent.index_select(0, keep)
        out_prank = recv_rank.index_select(0, keep)
        out_move = recv_move.index_select(0, keep)

        # 7. goal probe on the SURVIVORS only, with the table's own hash
        hz = _zobrist(new_states, self.ztab)
        pos = torch.searchsorted(self.table_hashes, hz).clamp(
            max=self.table_hashes.numel() - 1)
        in_table = (self.table_hashes.index_select(0, pos) == hz) & new_valid
        depth = self.table_depths.index_select(0, pos).to(torch.int64)
        return new_states, new_valid, out_parent, out_prank, out_move, in_table, depth

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def solve(self, start_state: torch.Tensor, compiled_step=None):
        """Run the beam. Returns (hit_rank, hit_step, hit_pos, hit_depth, trees).

        The tree is kept OUTSIDE the compiled region deliberately: writing
        `tree[j] = ...` inside would make the level index part of the graph and
        recompile every level.
        """
        cfg, dev = self.cfg, self.device
        Bl, S, W = self.b_local, self.state_size, self.world
        stepper = compiled_step if compiled_step is not None else self.step

        # Rank 0 seeds the beam; the other ranks start empty and fill from the
        # exchange. Every rank still runs the identical graph.
        states = torch.zeros((Bl, S), dtype=torch.int8, device=dev)
        valid = torch.zeros(Bl, dtype=torch.bool, device=dev)
        if self.rank == 0:
            states[0] = start_state.to(dev).to(torch.int8)
            valid[0] = True

        t_parent, t_prank, t_move = [], [], []
        neg1 = torch.full((), -1, dtype=torch.int64, device=dev)
        found_step, found_pos = neg1.clone(), torch.zeros((), dtype=torch.int64, device=dev)
        found_depth = torch.zeros((), dtype=torch.int64, device=dev)
        big_i = torch.full((), 1 << 30, dtype=torch.int64, device=dev)
        steps_run = 0

        import os as _os
        import time as _time
        prof = _os.environ.get("BEAM_PROFILE_STEPS") == "1"
        for j in range(cfg.max_steps):
            if prof:
                sync.synchronize(None, wait=True)
                _t0 = _time.perf_counter()
            states, valid, p_loc, p_rank, mv, in_table, depth = stepper(states, valid)
            t_parent.append(p_loc)
            t_prank.append(p_rank)
            t_move.append(mv)
            steps_run = j + 1

            # Latch the FIRST hit on device every step; only the LOOK is periodic.
            # Checking in_table only on a boundary loses solution quality outright
            # -- a goal reached at step 0 gets reported at step check_every-1.
            keyed = torch.where(in_table, depth, big_i.expand_as(depth))
            best = torch.argmin(keyed)
            is_first = (found_step < 0) & in_table.any()
            found_step = torch.where(is_first, torch.full_like(found_step, j), found_step)
            found_pos = torch.where(is_first, best, found_pos)
            found_depth = torch.where(
                is_first, depth.index_select(0, best.view(1))[0], found_depth)

            if prof:
                sync.synchronize(None, wait=True)
                if self.rank == 0:
                    print(f"    [step {j:2d}] {(_time.perf_counter()-_t0)*1000:8.1f} ms",
                          flush=True)

            if (j + 1) % cfg.check_every == 0 or j == cfg.max_steps - 1:
                # all_reduce(MIN) on int64 fails to compile on torch_tpu, so
                # gather and reduce on the host.
                local = torch.stack([found_step, found_pos, found_depth]).to(torch.int64)
                if W > 1:
                    outs = [torch.empty_like(local) for _ in range(W)]
                    dist.all_gather(outs, local)
                    rows = [o.cpu().tolist() for o in outs]
                else:
                    rows = [local.cpu().tolist()]
                hits = [(r, st, po, de) for r, (st, po, de) in enumerate(rows) if st >= 0]
                if hits:
                    # earliest step wins; ties broken by the shallower table entry
                    r, st, po, de = min(hits, key=lambda t: (t[1], t[3]))
                    return r, st, po, de, self._gather_trees(
                        t_parent, t_prank, t_move, steps_run)
                if not bool(valid.any().item()) and W == 1:
                    break

        return None, None, None, None, self._gather_trees(
            t_parent, t_prank, t_move, steps_run)

    def _gather_trees(self, t_parent, t_prank, t_move, steps_run):
        """Stack the per-step tree and make EVERY rank's copy visible to rank 0.

        Walk-back hops between ranks (a state's parent lives wherever it was
        expanded), so one rank's tree alone is not enough to reconstruct a path.
        """
        if not t_parent:
            return None
        par = torch.stack(t_parent)                    # (steps, B_local) int32
        prk = torch.stack(t_prank).to(torch.int32)
        mov = torch.stack(t_move).to(torch.int32)
        if self.world == 1:
            return (par.cpu().numpy()[None], prk.cpu().numpy()[None],
                    mov.cpu().numpy()[None])
        out = []
        for t in (par, prk, mov):
            bufs = [torch.empty_like(t) for _ in range(self.world)]
            dist.all_gather(bufs, t.contiguous())
            out.append(torch.stack(bufs).cpu().numpy())   # (W, steps, B_local)
        return tuple(out)

    # ----------------------------------------------------------- path walkback
    @staticmethod
    def walk_back(trees, hit_rank: int, hit_step: int, hit_pos: int,
                  move_names: list[str]) -> list[str]:
        """Reconstruct the move sequence, hopping ranks as the tree dictates."""
        par, prk, mov = trees
        out: list[str] = []
        r, pos = hit_rank, hit_pos
        for j in range(hit_step, -1, -1):
            out.append(move_names[int(mov[r][j][pos])])
            nr = int(prk[r][j][pos])
            pos = int(par[r][j][pos])
            r = nr
        return list(reversed(out))
