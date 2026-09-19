### Library code: puzzle + random-walk generator (no knobs here) ###
# Vendored (trimmed) from our project's megaminx/src/megaminx/puzzle.py and
# src/cayley/data.py so the notebook is self-contained.
import csv
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

STATE_SIZE = 120
N_GENERATORS = 24
MOVE_SEPARATOR = '.'


@dataclass(frozen=True)
class Megaminx:
    """State = length-120 permutation; solved = identity. 24 generators (12 faces x CW/CCW).

    Convention (matches the competition data and CayleyPy):
        apply(state, gen) -> new_state  where  new_state[i] = state[gen[i]]
    """
    solved_state: tuple
    generators: dict
    move_names: tuple

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            info = json.load(f)
        solved = tuple(info['central_state'])
        gens = {name: tuple(perm) for name, perm in info['generators'].items()}
        assert len(solved) == STATE_SIZE and len(gens) == N_GENERATORS
        return cls(solved_state=solved, generators=gens, move_names=tuple(gens.keys()))

    def inverse_name(self, name: str) -> str:
        return name[1:] if name.startswith('-') else '-' + name

    def apply_move(self, state: Sequence[int], move_name: str) -> tuple:
        gen = self.generators[move_name]
        return tuple(state[g] for g in gen)

    def apply_path(self, state: Sequence[int], path: Iterable[str]) -> tuple:
        cur = tuple(state)
        for m in path:
            cur = self.apply_move(cur, m)
        return cur

    def is_solved(self, state: Sequence[int]) -> bool:
        return tuple(state) == self.solved_state

    def parse_path(self, path_str: str) -> list:
        return path_str.split(MOVE_SEPARATOR) if path_str.strip() else []


@dataclass
class GeneratorTable:
    """All generators as one (n_gen, state_size) int64 array + inverse index map."""
    perms: np.ndarray
    names: tuple
    inverse_idx: np.ndarray

    @classmethod
    def from_puzzle(cls, puzzle) -> 'GeneratorTable':
        names = puzzle.move_names
        perms = np.stack([np.array(puzzle.generators[n], dtype=np.int64) for n in names])
        name_to_idx = {n: i for i, n in enumerate(names)}
        inverse_idx = np.array([name_to_idx[puzzle.inverse_name(n)] for n in names],
                               dtype=np.int64)
        return cls(perms=perms, names=names, inverse_idx=inverse_idx)


def generate_walks_torch(puzzle, n_walks, k_max, seed=0, device='cuda', n_back=1,
                         max_resamples=6):
    """Non-backtracking random walks from solved, all on GPU.

    Emits (state_after_step_i, i) for i = 1..k_max per walk. The depth i is an UPPER
    BOUND on the true distance - the model learns a smoothed version that still guides
    beam search. `n_back` bans the inverses of the last n_back actions.

    Returns: states (n_walks*k_max, state_size) int64, depths (n_walks*k_max,) int64.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(device)
    inv = torch.from_numpy(gens.inverse_idx).to(device)
    n_gen, state_size = perms.shape

    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
    states = solved.unsqueeze(0).expand(n_walks, state_size).clone()
    history = torch.full((n_walks, n_back), -1, dtype=torch.int64, device=device)

    out_states = torch.empty((n_walks * k_max, state_size), dtype=torch.int64, device=device)
    out_depths = torch.empty(n_walks * k_max, dtype=torch.int64, device=device)

    for k in range(1, k_max + 1):
        action = torch.randint(0, n_gen, (n_walks,), generator=g, device=device)
        if n_back > 0:
            hist_safe = torch.where(history >= 0, history, torch.zeros_like(history))
            banned = torch.where(history >= 0, inv[hist_safe], torch.full_like(hist_safe, -1))
            for _ in range(max_resamples):
                bad = (action.unsqueeze(1) == banned).any(dim=1)
                if not bool(bad.any()):
                    break
                n_bad = int(bad.sum().item())
                action = action.clone()
                action[bad] = torch.randint(0, n_gen, (n_bad,), generator=g, device=device)
        states = torch.gather(states, 1, perms[action])
        if n_back > 0:
            history = torch.cat([history[:, 1:], action.unsqueeze(1)], dim=1)
        off = (k - 1) * n_walks
        out_states[off:off + n_walks] = states
        out_depths[off:off + n_walks] = k
    return out_states, out_depths


def apply_all_generators(states, generators):
    """(B, S) x (n_gen, S) -> (B, n_gen, S): children[i, g] = apply(states[i], g)."""
    B, S = states.shape
    n_gen = generators.shape[0]
    return torch.gather(states.unsqueeze(1).expand(B, n_gen, S), 2,
                        generators.unsqueeze(0).expand(B, n_gen, S))


PUZZLE = Megaminx.load(PUZZLE_JSON)
GENS = GeneratorTable.from_puzzle(PUZZLE)
GENERATORS_T = torch.from_numpy(GENS.perms).to(DEVICE)
SOLVED_T = torch.tensor(PUZZLE.solved_state, dtype=torch.int64, device=DEVICE)
with open(TEST_CSV, encoding='utf-8') as f:
    TEST_ROWS = list(csv.DictReader(f))
print(f'puzzle: state_size={STATE_SIZE}, generators={N_GENERATORS}, '
      f'test puzzles={len(TEST_ROWS)}')
