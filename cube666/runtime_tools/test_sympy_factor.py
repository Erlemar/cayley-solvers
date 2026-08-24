import json
import time

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup

ladder = json.load(open("cube666/artifacts/stabilizer_ladder_v1.json"))
effects = np.array(
    [entry["active_effect"] for entry in ladder["stages"][0]["basis"]],
    dtype=np.uint8,
)
group = PermutationGroup(
    [Permutation(effect.tolist(), size=24) for effect in effects]
)
rng = np.random.default_rng(1)
state = np.arange(24, dtype=np.uint8)
for action in rng.integers(len(effects), size=20):
    state = state[effects[action]]
inverse = np.empty_like(state)
inverse[state] = np.arange(24, dtype=np.uint8)
product = group.generator_product(
    Permutation(inverse.tolist(), size=24), original=False
)
print("strong word length", len(product), "strong generators", len(group.strong_gens))
target = Permutation(inverse.tolist(), size=24)
for label, sequence in (("as_is", product), ("reversed", product[::-1])):
    replay = Permutation(list(range(24)), size=24)
    for permutation in sequence:
        replay = replay * permutation
    print(label, replay == target)

for index, strong in enumerate(group.strong_gens):
    started = time.perf_counter()
    original = group.generator_product(strong, original=True)
    print(index, len(original), round(time.perf_counter() - started, 4), flush=True)
