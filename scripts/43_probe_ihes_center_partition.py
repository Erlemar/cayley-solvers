"""Gate a possible stronger cost-partitioned center bound before native changes."""
from pathlib import Path
from collections import deque
import importlib.util
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states
spec = importlib.util.spec_from_file_location("decomp", ROOT / "scripts/build_picture_cube_kpuzzle_v2.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
puzzle = PictureCube.load(ROOT / "data/puzzle_info.json")
names = [sign + axis + str(layer) for axis in "frd" for layer in range(3) for sign in ("", "-")]
moves = [d.derive_move(name, puzzle)["CENTERS"] for name in names]
perms = [tuple(range(6))]
ids = {perms[0]: 0}
middle_distance = [0]
for p in perms:
    for tp, _ in moves:
        q = tuple(p[i] for i in tp)
        if q not in ids:
            ids[q] = len(perms)
            perms.append(q)
            middle_distance.append(middle_distance[ids[p]] + 1)
assert len(perms) == 24
weights = 4 ** np.arange(5, -1, -1)
orientations = (np.arange(4096)[:, None] // weights) % 4
transitions = np.empty((98304, 18), dtype=np.int32)
costs = np.array([int(name[-1] != "1") for name in names])
for m, (tp, to) in enumerate(moves):
    orientation_next = ((orientations[:, tp] + np.array(to)) % 4) @ weights
    for i, p in enumerate(perms):
        j = ids[tuple(p[x] for x in tp)]
        transitions[i*4096:(i+1)*4096, m] = j*4096 + orientation_next
distance = np.full(98304, 100, dtype=np.int16)
distance[0] = 0
queue = deque([0])
while queue:
    s = queue.popleft()
    for m, target in enumerate(transitions[s]):
        cost = int(costs[m])
        value = int(distance[s]) + cost
        if value < distance[target]:
            distance[target] = value
            (queue.append if cost else queue.appendleft)(int(target))
assert distance.max() < 100
for m in range(18):
    assert np.all(np.abs(distance[transitions[:, m]] - distance) <= costs[m])
np.save(ROOT / "data/ihes_pdb/center_outer_cost_probe.npy", distance)
pcs = np.memmap(ROOT / "data/ihes_pdb/corners_center_sum_v1.bin", dtype=np.uint8, mode="r").reshape(-1,2)
paths = load_submission(ROOT / "submission_ihes.csv")
states = load_test_states(ROOT / "data/test.csv")
checked = improved = improved_deep = 0
max_gain = 0
for pid, path in paths.items():
    s = states[pid]
    for offset in range(len(path)+1):
        coordinates = d.derive_state(s)
        cp, co = coordinates["CORNERS"]
        tp, to = coordinates["CENTERS"]
        rank = 0
        for i in range(8): rank = rank*(8-i) + sum(cp[j] < cp[i] for j in range(i+1,8))
        orientation = 0
        for value in co[:7]: orientation = orientation*3 + value
        outer = int(pcs[rank*2187 + orientation, (sum(to)%4)//2])
        mid = middle_distance[ids[tuple(tp)]]
        center = int(distance[ids[tuple(tp)]*4096 + np.dot(to, weights)])
        old = outer + mid
        new = max(outer, center) + mid
        assert new <= len(path)-offset
        checked += 1
        improved += int(new > old)
        improved_deep += int(new > old and len(path)-offset >= 11)
        max_gain = max(max_gain, new-old)
        if offset < len(path): s = puzzle.apply_move(s, path[offset])
report = {"center_states": len(distance), "free_middle_orbit": int(np.sum(distance == 0)),
          "maximum_center_outer_cost": int(distance.max()), "incumbent_states_checked": checked,
          "stronger_bounds": improved, "stronger_bounds_remaining_at_least_11": improved_deep,
          "maximum_gain": max_gain, "accepted_for_benchmark": improved_deep > 0}
(ROOT / "data/ihes_pdb/center_partition_probe.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
