"""Five-million-state corner-permutation / center-orbit outer-cost experiment."""
from pathlib import Path
import runpy
import itertools
import json
import numpy as np

g = runpy.run_path(str(Path(__file__).with_name('44_probe_ihes_quotient_partition.py')))
globals().update({k:g[k] for k in ['ROOT','component','count','adj','outer_moves','max_degree','names','puzzle','d','pcs','paths','states','ids','weights','middle_distance']})
perms = np.array(list(itertools.permutations(range(8))), dtype=np.int32)
perm_next = {}
def rank_many(p):
    rank = np.zeros(len(p), dtype=np.int32)
    for i in range(8): rank = rank*(8-i)+np.sum(p[:,i+1:] < p[:,i,None], axis=1)
    return rank
assert np.array_equal(rank_many(perms), np.arange(40320))
for m in outer_moves:
    cp, _ = d.derive_move(names[m], puzzle)['CORNERS']
    perm_next[m] = rank_many(perms[:,cp])
distance = np.full(40320*count, 255, dtype=np.uint8)
distance[0] = 0
front = np.array([0], dtype=np.int32)
level = 0
tables = {m:np.array([np.pad(a,(0,max_degree-len(a)),mode='edge') for a in adj[m]]) for m in outer_moves}
while len(front):
    print('level',level,'frontier',len(front),flush=True)
    nxt=[]
    for m in outer_moves:
        targets=(perm_next[m][front//count,None]*count+tables[m][front%count]).ravel()
        targets=np.unique(targets[distance[targets]==255])
        distance[targets]=level+1
        nxt.append(targets)
    front=np.concatenate(nxt)
    level+=1
# Center sum parity correlates with corner permutation parity: some pairs may
# be unreachable. They are valid only if no original reachable state uses them.
for m in outer_moves:
    for c in range(count):
        a=adj[m][c]
        source=distance[np.arange(40320)*count+c].astype(int)
        target=distance[perm_next[m][:,None]*count+a].astype(int)
        reachable=source<255
        assert np.all(target[reachable]<255)
        assert np.all(np.abs(source[reachable,None]-target[reachable])<=1)
checked=improved=deep=gain=0
for pid,path in paths.items():
    s=states[pid]
    for offset in range(len(path)+1):
        coords=d.derive_state(s)
        cp,co=coords['CORNERS'];tp,to=coords['CENTERS']
        rank=0
        for i in range(8):rank=rank*(8-i)+sum(cp[j]<cp[i] for j in range(i+1,8))
        ori=0
        for value in co[:7]:ori=ori*3+value
        outer=int(pcs[rank*2187+ori,(sum(to)%4)//2])
        center=ids[tuple(tp)]*4096+int(np.dot(to,weights))
        h=int(distance[rank*count+component[center]])
        assert h<255
        assert max(outer,h)+middle_distance[ids[tuple(tp)]]<=len(path)-offset
        checked+=1;improved+=h>outer;deep+=h>outer and len(path)-offset>=11;gain=max(gain,h-outer)
        if offset<len(path):s=puzzle.apply_move(s,path[offset])
report=dict(states=len(distance),reachable_states=int(np.sum(distance<255)),max_distance=int(distance[distance<255].max()),
            incumbent_states_checked=checked,stronger_bounds=int(improved),stronger_bounds_remaining_at_least_11=int(deep),
            maximum_gain=gain,all_edges_consistent=True)
np.savez(ROOT/'data/ihes_pdb/cp_center_quotient_probe.npz',component=component,distance=distance)
(ROOT/'data/ihes_pdb/cp_center_quotient_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
