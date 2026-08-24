"""Map the verified reusable stabilizer basis into a new macro action table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macros import analyze_corner_fixing_macro  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ladder", type=Path, required=True)
parser.add_argument("--action-effects", type=Path, required=True)
parser.add_argument("--action-costs", type=Path, required=True)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=PROJECT / "cayley-py-666-cube",
)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--out-effects", type=Path, required=True)
parser.add_argument("--out-costs", type=Path, required=True)
args = parser.parse_args()

puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
decomposition = build_decomposition(puzzle.generators)
effects = np.load(args.action_effects, allow_pickle=False)
costs = np.load(args.action_costs, allow_pickle=False)
if costs.shape != (len(effects),):
    raise ValueError("action effects and costs disagree")
extended_effects = [effect.copy() for effect in effects]
extended_costs = [int(cost) for cost in costs]
by_effect = {effect.tobytes(): index for index, effect in enumerate(effects)}
ladder = json.loads(args.ladder.read_text(encoding="utf-8"))

mapped: list[int] = []
rows = []
for stage in ladder["stages"]:
    for entry in stage["basis"]:
        macro = analyze_corner_fixing_macro(
            entry["path"],
            puzzle.generators,
            decomposition,
        )
        key = np.asarray(macro.cluster_permutations, dtype=np.uint8).tobytes()
        action = by_effect.get(key)
        if action is None:
            action = len(extended_effects)
            by_effect[key] = action
            extended_effects.append(
                np.asarray(macro.cluster_permutations, dtype=np.uint8)
            )
            extended_costs.append(len(entry["path"]))
        action = int(action)
        mapped.append(action)
        rows.append(
            {
                "action": action,
                "primitive_length": len(entry["path"]),
                "stage": int(stage["stage_index"]),
            }
        )
if len(set(mapped)) != len(mapped):
    raise ValueError("mapped basis contains duplicate action effects")
effects_output = np.stack(extended_effects)
costs_output = np.asarray(extended_costs, dtype=np.int16)
identity = np.broadcast_to(np.arange(24, dtype=np.uint8), effects_output.shape)
inverse_effects = np.empty_like(effects_output)
np.put_along_axis(inverse_effects, effects_output, identity, axis=-1)
missing_inverses = [
    index
    for index, inverse in enumerate(inverse_effects)
    if inverse.tobytes() not in by_effect
]
if missing_inverses:
    raise ValueError(f"extended action table lacks inverses for {missing_inverses[:10]}")
args.out.parent.mkdir(parents=True, exist_ok=True)
temporary = args.out.with_suffix(args.out.suffix + ".tmp")
with open(temporary, "wb") as handle:
    np.save(handle, np.asarray(mapped, dtype=np.int32))
temporary.replace(args.out)
for path, values in (
    (args.out_effects, effects_output),
    (args.out_costs, costs_output),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, values)
    temporary.replace(path)
report = {
    "actions": len(mapped),
    "action_digest": hashlib.sha256(effects_output.tobytes()).hexdigest(),
    "appended_actions": len(effects_output) - len(effects),
    "extended_action_count": len(effects_output),
    "maximum_primitive_length": max(row["primitive_length"] for row in rows),
    "mean_primitive_length": float(np.mean([row["primitive_length"] for row in rows])),
    "rows": rows,
    "unique_actions": len(set(mapped)),
}
report_path = args.out.with_suffix(".json")
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
