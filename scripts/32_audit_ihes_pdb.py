"""Independently audit additive PDBs against true facelet trajectories."""
import importlib.util
import json
import random
import sys
from collections import deque
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,load_test_states
spec=importlib.util.spec_from_file_location("decomp",ROOT/"scripts/build_picture_cube_kpuzzle_v2.py")
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
puzzle=PictureCube.load(ROOT/"data/puzzle_info.json")
folder=ROOT/"data/ihes_pdb"
pc=np.memmap(folder/"corners_v1.bin",dtype=np.uint8,mode="r")
pcs=np.memmap(folder/"corners_center_sum_v1.bin",dtype=np.uint8,mode="r").reshape(-1,2)
assert np.array_equal(pcs.min(axis=1),pc)
middle=[d.derive_move(m,puzzle)["CENTERS"][0] for m in puzzle.move_names if m[-1]=="1"]
identity=tuple(range(6));dist={identity:0};q=deque([identity])
while q:
 s=q.popleft()
 for m in middle:
  t=tuple(s[j] for j in m)
  if t not in dist:dist[t]=dist[s]+1;q.append(t)
assert len(dist)==24

def bound(state):
 trans=d.derive_state(state);cp,co=trans["CORNERS"];tp,to=trans["CENTERS"]
 inv=sum(cp[i]>cp[j] for i in range(8) for j in range(i+1,8))
 assert inv%2==sum(to)%2
 pr=0
 for i in range(8):pr=pr*(8-i)+sum(cp[j]<cp[i] for j in range(i+1,8))
 ori=0
 for v in co[:7]:ori=ori*3+v
 assert sum(co)%3==0
 return int(pcs[pr*2187+ori,(sum(to)%4)//2])+dist[tuple(tp)]

paths=load_submission(ROOT/"submission_ihes.csv");states=load_test_states(ROOT/"data/test.csv")
checked=0;strict=0
for pid,path in paths.items():
 s=states[pid]
 for i in range(len(path)+1):
  h=bound(s);assert h<=len(path)-i,(pid,i,h)
  checked+=1
  if i<len(path):s=puzzle.apply_move(s,path[i])
rng=random.Random(20260906)
edges=0
for _ in range(400):
 s=puzzle.solved_state
 for k in range(30):
  s=puzzle.apply_move(s,rng.choice(puzzle.move_names));h=bound(s)
  assert h<=k+1
  if k==29:
   for move in puzzle.move_names:
    nh=bound(puzzle.apply_move(s,move));assert abs(h-nh)==1,(h,nh)
    edges+=1
result={"status":"pass","incumbent_prefixes_checked":checked,"random_walk_states":12000,
        "consistency_edges":edges,"corner_projection_minimum_matches":len(pc)}
(folder/"audit_verified.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
