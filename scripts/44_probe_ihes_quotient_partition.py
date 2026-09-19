"""Exact outer-cost PDB for corner orientation and free-middle center orbits."""
import runpy
from collections import deque
import json
import numpy as np

g = runpy.run_path(__file__.replace('44_probe_ihes_quotient_partition.py', '43_probe_ihes_center_partition.py'))
globals().update({k:g[k] for k in ['ROOT','transitions','costs','names','moves','puzzle','d','pcs','paths','states','ids','weights','middle_distance']})
component = np.full(98304, -1, dtype=np.int32)
count = 0
for root in range(len(component)):
    if component[root] >= 0: continue
    component[root] = count
    queue = deque([root])
    while queue:
        s = queue.popleft()
        for t in transitions[s, costs == 0]:
            if component[t] < 0:
                component[t] = count
                queue.append(int(t))
    count += 1
assert count == 128
outer_moves = np.flatnonzero(costs)
adj = {}
for m in outer_moves:
    adj[m] = [np.unique(component[transitions[component == c, m]]) for c in range(count)]
max_degree = max(len(a) for v in adj.values() for a in v)
co_weights = 3 ** np.arange(6, -1, -1)
co = (np.arange(2187)[:, None] // co_weights) % 3
co = np.column_stack([co, (-co.sum(axis=1)) % 3])
co_next = {}
for m in outer_moves:
    cp, delta = d.derive_move(names[m], puzzle)['CORNERS']
    co_next[m] = (((co[:, cp] + delta) % 3)[:, :7] @ co_weights).astype(np.int32)
distance = np.full(2187*count, 255, dtype=np.uint8)
distance[0] = 0
front = np.array([0], dtype=np.int32)
level = 0
while len(front):
    nxt = []
    for m in outer_moves:
        # Union transitions over every representative: an admissible quotient.
        table = np.array([np.pad(a, (0,max_degree-len(a)), mode='edge') for a in adj[m]])
        targets = (co_next[m][front // count, None]*count + table[front % count]).ravel()
        targets = np.unique(targets[distance[targets] == 255])
        distance[targets] = level + 1
        nxt.append(targets)
    front = np.concatenate(nxt)
    level += 1
assert np.all(distance < 255)
# Independently check all quotient edges and all original center representatives.
for m in outer_moves:
    for c in range(count):
        a = adj[m][c]
        source = distance[np.arange(2187)*count+c].astype(int)
        target = distance[co_next[m][:,None]*count+a].astype(int)
        assert np.all(np.abs(source[:,None]-target) <= 1)
for m in np.flatnonzero(costs == 0):
    assert np.all(component == component[transitions[:,m]])
checked = improved = deep = gain = 0
for pid, path in paths.items():
    s = states[pid]
    for offset in range(len(path)+1):
        coords = d.derive_state(s)
        cp, co = coords['CORNERS']; tp, to = coords['CENTERS']
        rank = 0
        for i in range(8): rank = rank*(8-i)+sum(cp[j]<cp[i] for j in range(i+1,8))
        orientation = int(np.dot(co[:7], co_weights))
        outer = int(pcs[rank*2187+orientation, (sum(to)%4)//2])
        center = ids[tuple(tp)]*4096+int(np.dot(to, weights))
        h = int(distance[orientation*count+component[center]])
        mid = middle_distance[ids[tuple(tp)]]
        assert max(outer,h)+mid <= len(path)-offset
        checked += 1; improved += h>outer
        deep += h>outer and len(path)-offset>=11
        gain = max(gain,h-outer)
        if offset<len(path): s=puzzle.apply_move(s,path[offset])
report = dict(states=len(distance), components=count, maximum_degree=max_degree,
              max_distance=int(distance.max()), incumbent_states_checked=checked,
              stronger_bounds=int(improved), stronger_bounds_remaining_at_least_11=int(deep),
              maximum_gain=gain, all_edges_consistent=True)
np.savez(ROOT/'data/ihes_pdb/co_center_quotient_probe.npz', component=component, distance=distance)
(ROOT/'data/ihes_pdb/co_center_quotient_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
