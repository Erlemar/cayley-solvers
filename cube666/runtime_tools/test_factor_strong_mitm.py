import json
import sys

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup

sys.path.insert(0, "src")

from cube666.classical import build_decomposition
from cube666.macro_data import load_macro_action_library
from cube666.macros import reduce_commuting_quarter_turn_path
from cube666.puzzle import Cube666Puzzle

ladder = json.load(open("cube666/artifacts/stabilizer_ladder_v1.json"))
basis = np.asarray(
    [entry["active_effect"] for entry in ladder["stages"][0]["basis"]],
    dtype=np.uint8,
)
group = PermutationGroup(
    [Permutation(effect.tolist(), size=24) for effect in basis]
)
group.schreier_sims()

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
paths = [item[1] for item in items]
by_effect = {effect.tobytes(): index for index, effect in enumerate(effects)}
inverses = np.empty_like(effects)
rows = np.arange(len(effects))[:, None]
inverses[rows, effects] = np.arange(24, dtype=np.uint8)[None]

found = []
short = [index for index, path in enumerate(paths) if len(path) <= 4]
pairs = {}
for left in short:
    for right in short:
        effect = effects[left][effects[right]]
        candidate = reduce_commuting_quarter_turn_path(paths[left] + paths[right])
        old = pairs.get(effect.tobytes())
        if old is None or (len(candidate), candidate) < (len(old), old):
            pairs[effect.tobytes()] = candidate
print("short", len(short), "pair effects", len(pairs), flush=True)
for strong_index, strong in enumerate(group.strong_gens):
    target = np.asarray(strong.array_form, dtype=np.uint8)
    best = None
    direct = by_effect.get(target.tobytes())
    if direct is not None:
        best = reduce_commuting_quarter_turn_path(paths[direct])
    for left in range(len(effects)):
        needed = inverses[left][target]
        right = by_effect.get(needed.tobytes())
        if right is not None:
            candidate = reduce_commuting_quarter_turn_path(paths[left] + paths[right])
            if best is None or (len(candidate), candidate) < (len(best), best):
                best = candidate
        pair_path = pairs.get(needed.tobytes())
        if pair_path is not None:
            candidate = reduce_commuting_quarter_turn_path(paths[left] + pair_path)
            if best is None or (len(candidate), candidate) < (len(best), best):
                best = candidate
    found.append(best)
    print(strong_index, None if best is None else len(best), flush=True)
print("found", sum(path is not None for path in found), "of", len(found))
