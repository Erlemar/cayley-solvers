"""GPU exact window shortening for IHES using a front-ball/endgame join.

For a window element ``W`` and every ``A`` in the exact front ball, probe
``A^-1 W`` in the exact endgame table.  With front depth 5 and endgame depth 6,
this certifies every element of radius at most 11 and can therefore shorten
windows of length 12 or more.  Hash hits are used only as an index: every word
is reconstructed and checked as a full 72-position permutation, and the final
submission is replay-verified.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bfs_table import BfsTable
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, verify_submission


class ExactJoiner:
    def __init__(
        self,
        puzzle: PictureCube,
        front_table: Path,
        endgame_hash: Path,
        device: str,
    ) -> None:
        self.puzzle = puzzle
        self.names = list(puzzle.move_names)
        self.gen = np.asarray(
            [puzzle.generators[name] for name in self.names], dtype=np.int64
        )
        self.ident = np.arange(self.gen.shape[1], dtype=np.int64)
        self.inv_move = np.asarray(
            [self.names.index(puzzle.inverse_name(name)) for name in self.names],
            dtype=np.int64,
        )
        self.device = torch.device(device)

        t0 = time.time()
        packed = np.load(endgame_hash)
        hashes = packed["hashes"]
        depths = packed["depths"]
        self.ztab = packed["ztab"].astype(np.int64, copy=False)
        self.end_depth = int(packed["max_depth"])
        if int(packed["state_size"]) != self.ident.size:
            raise ValueError("packed endgame state size does not match the puzzle")
        self.hashes_np = hashes
        self.depths_np = depths
        self.hashes_g = torch.from_numpy(hashes).to(self.device)
        self.depths_g = torch.from_numpy(depths).to(self.device)
        self.ztab_g = torch.from_numpy(self.ztab).to(self.device)
        # A direct prefix index is much faster than GPU binary search for the
        # millions of probes in each MITM batch.  Hashes are non-negative 63-bit
        # values, so equal high-bit prefixes form contiguous ranges.
        self.prefix_bits = 24
        self.prefix_shift = 63 - self.prefix_bits
        prefix = hashes >> self.prefix_shift
        counts = np.bincount(prefix, minlength=1 << self.prefix_bits)
        offsets = np.empty(counts.size + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=offsets[1:])
        self.max_bucket = int(counts.max())
        self.offsets_g = torch.from_numpy(offsets).to(self.device)
        print(
            f"endgame: {hashes.size:,} hashes, depth<={self.end_depth}, "
            f"24-bit buckets max={self.max_bucket}, "
            f"{endgame_hash} ({time.time() - t0:.1f}s)",
            flush=True,
        )

        t0 = time.time()
        front_obj = BfsTable.load(front_table)
        n_front = len(front_obj.table)
        front_states = np.empty((n_front, self.ident.size), dtype=np.uint8)
        self.front_words: list[tuple[int, ...]] = []
        self.front_len = np.empty(n_front, dtype=np.int64)
        for i, (state, word) in enumerate(front_obj.table.items()):
            front_states[i] = state
            self.front_words.append(word)
            self.front_len[i] = len(word)
        self.front_depth = int(front_obj.max_depth)
        if int(self.front_len.max()) != self.front_depth:
            raise ValueError("front table depth metadata is inconsistent")
        del front_obj
        # Chunked inverse construction avoids the two >6 GiB int64 temporaries
        # that np.argsort creates for the 10.9M-state depth-6 table.
        front_inv = np.empty_like(front_states)
        positions = np.arange(self.ident.size, dtype=np.uint8)[None, :]
        for lo in range(0, n_front, 100_000):
            hi = min(lo + 100_000, n_front)
            np.put_along_axis(
                front_inv[lo:hi],
                front_states[lo:hi].astype(np.int64, copy=False),
                np.broadcast_to(positions, (hi - lo, self.ident.size)),
                axis=1,
            )
        self.front_inv_g = torch.from_numpy(front_inv).to(self.device)
        self.front_len_g = torch.from_numpy(self.front_len).to(self.device)
        del front_states, front_inv
        self.reach = self.front_depth + self.end_depth
        self.phantoms = 0
        print(
            f"front: {n_front:,} elements, depth<={self.front_depth}; "
            f"certified radius={self.reach} ({time.time() - t0:.1f}s)",
            flush=True,
        )

    def perm_of(self, word: list[int] | tuple[int, ...]) -> np.ndarray:
        state = self.ident.copy()
        for move in word:
            state = state[self.gen[move]]
        return state

    def inv_word(self, word: list[int] | tuple[int, ...]) -> list[int]:
        return [int(self.inv_move[m]) for m in reversed(word)]

    def depth_np(self, states: np.ndarray) -> np.ndarray:
        h = np.zeros(states.shape[0], dtype=np.int64)
        for pos in range(self.ident.size):
            h ^= self.ztab[pos, states[:, pos]]
        idx = np.searchsorted(self.hashes_np, h)
        valid = idx < self.hashes_np.size
        idx.clip(0, self.hashes_np.size - 1, out=idx)
        valid &= self.hashes_np[idx] == h
        return np.where(valid, self.depths_np[idx], -1).astype(np.int64)

    def depth_gpu(self, states: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(states.shape[0], dtype=torch.int64, device=self.device)
        for pos in range(self.ident.size):
            h ^= self.ztab_g[pos, states[:, pos].long()]
        prefix = h >> self.prefix_shift
        lo = self.offsets_g[prefix].long()
        hi = self.offsets_g[prefix + 1].long()
        out = torch.full_like(h, -1)
        for delta in range(self.max_bucket):
            idx = lo + delta
            valid = idx < hi
            idx.clamp_(0, self.hashes_g.numel() - 1)
            match = valid & (self.hashes_g[idx] == h)
            out = torch.where(match, self.depths_g[idx].long(), out)
        return out

    def descend(self, state: np.ndarray) -> list[int]:
        depth = int(self.depth_np(state[None, :])[0])
        if depth < 0:
            raise ValueError("state is not in the endgame table")
        word: list[int] = []
        cur = state.copy()
        while depth:
            children = cur[self.gen]
            child_depths = self.depth_np(children)
            matches = np.flatnonzero(child_depths == depth - 1)
            if matches.size == 0:
                raise ValueError("hash hit has no exact descending child")
            move = int(matches[0])
            word.append(move)
            cur = children[move]
            depth -= 1
        return word

    def solve_batch(
        self,
        windows: np.ndarray,
        best_lengths: np.ndarray,
        n_candidates: int = 4,
    ) -> list[list[int] | None]:
        """Return shortest verified words below each corresponding best length."""
        q = windows.shape[0]
        n_front = self.front_inv_g.shape[0]
        wg = torch.from_numpy(windows.astype(np.int64, copy=False)).to(self.device)
        source = self.front_inv_g.unsqueeze(0).expand(q, -1, -1)
        indices = wg[:, None, :].expand(-1, n_front, -1)
        remainder = torch.gather(source, 2, indices)
        dep = self.depth_gpu(remainder.reshape(-1, self.ident.size)).reshape(q, n_front)
        total = torch.where(
            dep >= 0,
            dep + self.front_len_g[None, :],
            torch.full_like(dep, 1 << 30),
        )
        k = min(n_candidates, n_front)
        vals, idx = torch.topk(total, k, dim=1, largest=False)
        vals_np = vals.cpu().numpy()
        idx_np = idx.cpu().numpy()

        answers: list[list[int] | None] = []
        for qi in range(q):
            answer = None
            for total_len, front_i in zip(vals_np[qi], idx_np[qi]):
                if int(total_len) >= int(best_lengths[qi]):
                    break
                front_i = int(front_i)
                rem = remainder[qi, front_i].cpu().numpy().astype(np.int64)
                try:
                    back_word = self.inv_word(self.descend(rem))
                except ValueError:
                    self.phantoms += 1
                    continue
                candidate = list(self.front_words[front_i]) + back_word
                if len(candidate) != int(total_len):
                    self.phantoms += 1
                    continue
                if not np.array_equal(self.perm_of(candidate), windows[qi]):
                    self.phantoms += 1
                    continue
                answer = candidate
                break
            answers.append(answer)
        return answers


def select_non_overlapping(
    path_len: int,
    candidates: list[tuple[int, int, list[int]]],
) -> list[tuple[int, int, list[int]]]:
    """Maximum-saving non-overlapping interval set via dynamic programming."""
    by_start: dict[int, list[tuple[int, int, list[int]]]] = defaultdict(list)
    for start, end, word in candidates:
        by_start[start].append((start, end, word))
    score = [0] * (path_len + 1)
    choice: list[tuple[int, int, list[int]] | None] = [None] * (path_len + 1)
    for pos in range(path_len - 1, -1, -1):
        score[pos] = score[pos + 1]
        for candidate in by_start.get(pos, ()):
            start, end, word = candidate
            value = (end - start - len(word)) + score[end]
            if value > score[pos]:
                score[pos] = value
                choice[pos] = candidate
    selected = []
    pos = 0
    while pos < path_len:
        candidate = choice[pos]
        if candidate is None:
            pos += 1
        else:
            selected.append(candidate)
            pos = candidate[1]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--front-table", type=Path, default=PROJECT / "data/bfs_table_d5.pkl")
    parser.add_argument(
        "--endgame-hash",
        type=Path,
        default=PROJECT / "data/bfs_table_d6_hash.npz",
    )
    parser.add_argument("--min-window", type=int, default=12)
    parser.add_argument("--max-window", type=int, default=16)
    parser.add_argument(
        "--window-parity",
        choices=("all", "even", "odd"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--min-path-length",
        type=int,
        default=0,
        help="Only scan rows whose baseline path has at least this many moves",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=8,
        help="Verify this many best hash joins per query (guards against collisions)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    joiner = ExactJoiner(
        puzzle, args.front_table, args.endgame_hash, args.device
    )

    if args.self_test:
        rng = np.random.default_rng(0)
        tests = []
        lengths = []
        for _ in range(16):
            length = int(rng.integers(max(1, joiner.reach - 3), joiner.reach + 1))
            word = rng.integers(0, len(joiner.names), size=length).tolist()
            tests.append(joiner.perm_of(word))
            lengths.append(length + 1)
        test_states = np.asarray(tests, dtype=np.uint8)
        test_lengths = np.asarray(lengths, dtype=np.int64)
        got = []
        for lo in range(0, len(tests), args.batch_size):
            hi = min(lo + args.batch_size, len(tests))
            got.extend(
                joiner.solve_batch(
                    test_states[lo:hi],
                    test_lengths[lo:hi],
                    n_candidates=args.n_candidates,
                )
            )
        for i, answer in enumerate(got):
            if answer is None or not np.array_equal(joiner.perm_of(answer), tests[i]):
                raise SystemExit(f"self-test {i} failed")
        print(f"self-test: {len(got)} exact joins PASS", flush=True)
        return 0

    baseline_report = verify_submission(
        puzzle, PROJECT / "data/test.csv", args.baseline
    )
    if not baseline_report.all_valid:
        raise SystemExit(f"invalid baseline: {baseline_report.failures[:5]}")
    named = load_submission(args.baseline)
    name_to_idx = {name: i for i, name in enumerate(joiner.names)}
    rows = {
        pid: [name_to_idx[name] for name in path]
        for pid, path in named.items()
    }
    original_lengths = {pid: len(path) for pid, path in rows.items()}
    target_pids = {
        pid for pid, length in original_lengths.items()
        if length >= args.min_path_length
    }
    print(
        f"target rows: {len(target_pids)}/{len(rows)} with baseline length "
        f">={args.min_path_length}",
        flush=True,
    )

    total_saved = 0
    for pass_no in range(1, args.passes + 1):
        metadata: list[tuple[int, int, int]] = []
        windows: list[np.ndarray] = []
        for pid in sorted(target_pids):
            path = rows[pid]
            for start in range(len(path)):
                state = joiner.ident.copy()
                max_len = min(args.max_window, len(path) - start)
                for length in range(1, max_len + 1):
                    state = state[joiner.gen[path[start + length - 1]]]
                    if length >= args.min_window:
                        if args.window_parity == "even" and length % 2:
                            continue
                        if args.window_parity == "odd" and length % 2 == 0:
                            continue
                        metadata.append((pid, start, length))
                        windows.append(state.copy())
        if not windows:
            break
        states = np.asarray(windows, dtype=np.uint8)
        del windows
        candidates: dict[int, list[tuple[int, int, list[int]]]] = defaultdict(list)
        t0 = time.time()
        for lo in range(0, len(metadata), args.batch_size):
            hi = min(lo + args.batch_size, len(metadata))
            best = np.asarray(
                [metadata[i][2] for i in range(lo, hi)], dtype=np.int64
            )
            answers = joiner.solve_batch(
                states[lo:hi], best, n_candidates=args.n_candidates
            )
            for offset, answer in enumerate(answers):
                if answer is None:
                    continue
                pid, start, length = metadata[lo + offset]
                candidates[pid].append((start, start + length, answer))
            if hi % (args.batch_size * 10) == 0 or hi == len(metadata):
                print(
                    f"pass {pass_no}: {hi:,}/{len(metadata):,} windows, "
                    f"raw hits={sum(map(len, candidates.values())):,}, "
                    f"{time.time() - t0:.1f}s",
                    flush=True,
                )

        pass_saved = 0
        for pid, options in candidates.items():
            selected = select_non_overlapping(len(rows[pid]), options)
            for start, end, word in reversed(selected):
                pass_saved += end - start - len(word)
                rows[pid][start:end] = word
        total_saved += pass_saved
        print(
            f"pass {pass_no}: selected savings={pass_saved}, "
            f"cumulative={total_saved}, phantoms={joiner.phantoms}",
            flush=True,
        )
        if pass_saved == 0:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            writer.writerow([pid, ".".join(joiner.names[m] for m in rows[pid])])

    report = verify_submission(puzzle, PROJECT / "data/test.csv", args.out)
    print("improvements:")
    for pid in sorted(rows):
        if len(rows[pid]) < original_lengths[pid]:
            print(f"  pid {pid}: {original_lengths[pid]} -> {len(rows[pid])}")
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{baseline_report.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - baseline_report.total_moves:+,})",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
