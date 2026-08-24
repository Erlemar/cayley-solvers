import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--max-cost", type=int, required=True)
args = parser.parse_args()

effects = np.load(args.root / "action_effects.npy", allow_pickle=False)
costs = np.load(args.root / "action_costs.npy", allow_pickle=False)
effects = effects[costs <= args.max_cost]
identity = np.arange(24, dtype=np.uint8)
target = 310224200866619719680000


@lru_cache(maxsize=None)
def eligible(locked: tuple[int, ...]) -> np.ndarray:
    if not locked:
        return np.arange(len(effects))
    return np.flatnonzero(
        np.all(np.all(effects[:, locked] == identity, axis=2), axis=1)
    )


@lru_cache(maxsize=None)
def projected_order(locked: tuple[int, ...], cluster: int) -> int:
    rows = eligible(locked)
    if not len(rows):
        return 1
    group = PermutationGroup(
        [Permutation(list(map(int, permutation))) for permutation in effects[rows, cluster]]
    )
    return int(group.order())


def find_order(locked: tuple[int, ...]) -> tuple[int, ...] | None:
    if len(locked) == 6:
        return ()
    for cluster in range(6):
        if cluster in locked:
            continue
        if projected_order(locked, cluster) != target:
            continue
        suffix = find_order(tuple(sorted((*locked, cluster))))
        if suffix is not None:
            return (cluster,) + suffix
    return None


order = find_order(())
stages = []
locked: tuple[int, ...] = ()
if order is not None:
    for cluster in order:
        stages.append(
            {
                "cluster": cluster,
                "eligible_actions": len(eligible(locked)),
                "projected_order": projected_order(locked, cluster),
            }
        )
        locked = tuple(sorted((*locked, cluster)))
print(
    json.dumps(
        {
            "action_count": len(effects),
            "full_ladder": order is not None,
            "max_cost": args.max_cost,
            "order": order,
            "stages": stages,
        },
        indent=2,
    )
)
