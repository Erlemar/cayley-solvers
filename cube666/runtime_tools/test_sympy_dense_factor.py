import sys
import time

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup

sys.path.insert(0, "src")

from cube666.classical import build_decomposition
from cube666.macro_data import load_macro_action_library
from cube666.puzzle import Cube666Puzzle

puzzle = Cube666Puzzle.load("cayley-py-666-cube/puzzle_info.json")
decomposition = build_decomposition(puzzle.generators)
_, table = load_macro_action_library(
    "cube666/training/kmc_macro_teacher_allruns_sym8_v3/action_library.json",
    puzzle.generators,
    decomposition,
)
identity = np.arange(24, dtype=np.uint8)
active = np.any(table.effects != identity[None, None, :], axis=2)
shortest = {}
for action in np.flatnonzero(active[:, 0]):
    effect = table.effects[action, 0]
    path = table.paths[int(action)]
    old = shortest.get(effect.tobytes())
    if old is None or (len(path), path) < (len(old[1]), old[1]):
        shortest[effect.tobytes()] = (effect.copy(), path)
items = list(shortest.values())
effects = np.asarray([item[0] for item in items], dtype=np.uint8)

started = time.perf_counter()
group = PermutationGroup(
    [Permutation(effect.tolist(), size=24) for effect in effects]
)
print("generators", len(group.generators), "order", group.order(), "build", time.perf_counter() - started)
direct = {tuple(effect): index for index, effect in enumerate(effects.tolist())}
print(
    "strong",
    len(group.strong_gens),
    "direct",
    sum(tuple(generator.array_form) in direct for generator in group.strong_gens),
)
for index, generator in enumerate(group.strong_gens):
    if tuple(generator.array_form) in direct:
        continue
    started_generator = time.perf_counter()
    expanded = group.generator_product(generator, original=True)
    print(
        "derived",
        index,
        "original_actions",
        len(expanded),
        "seconds",
        round(time.perf_counter() - started_generator, 4),
        flush=True,
    )
rng = np.random.default_rng(2)
state = identity.copy()
for action in rng.integers(len(effects), size=100):
    state = state[effects[action]]
inverse = np.empty_like(state)
inverse[state] = identity
started = time.perf_counter()
product = group.generator_product(
    Permutation(inverse.tolist(), size=24), original=True
)
print("factor length", len(product), "seconds", time.perf_counter() - started)
