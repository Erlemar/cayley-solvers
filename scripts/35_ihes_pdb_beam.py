"""Matched neural-beam probe using the new admissible additive PDB floor."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import deque
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,load_test_states,verify_path,verify_submission
from cayley.search import load_model_checkpoint
from cayley.khoruzhii_search import KhoruzhiiSolver,KhoruzhiiSearchConfig


class AdditivePDB:
 def __init__(self,puzzle,device='cuda'):
  spec=importlib.util.spec_from_file_location('decomp',ROOT/'scripts/build_picture_cube_kpuzzle_v2.py')
  d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
  self.d=d
  ids=np.zeros(72,dtype=np.int64);oris=np.zeros(72,dtype=np.int64)
  for slots in (d.CORNERS,d.CENTERS):
   for pid,ss in enumerate(slots):
    for o,sticker in enumerate(ss):ids[sticker]=pid;oris[sticker]=(-o)%len(ss)
  self.ids=torch.tensor(ids,device=device);self.oris=torch.tensor(oris,device=device)
  self.cpos=torch.tensor([s[0] for s in d.CORNERS],device=device)
  self.tpos=torch.tensor([s[0] for s in d.CENTERS],device=device)
  self.table=torch.from_numpy(np.fromfile(ROOT/'data/ihes_pdb/corners_center_sum_v1.bin',dtype=np.uint8)).to(device)
  middle=[d.derive_move(m,puzzle)['CENTERS'][0] for m in puzzle.move_names if m[-1]=='1']
  identity=tuple(range(6));dist={identity:0};q=deque([identity])
  while q:
   s=q.popleft()
   for m in middle:
    t=tuple(s[j] for j in m)
    if t not in dist:dist[t]=dist[s]+1;q.append(t)
  lookup=np.full(46656,255,dtype=np.uint8)
  for s,value in dist.items():
   key=0
   for v in s:key=key*6+v
   lookup[key]=value
  self.mid=torch.tensor(lookup,device=device)

 def __call__(self,states):
  out=torch.empty(len(states),dtype=torch.float16,device=states.device)
  for start in range(0,len(states),65536):
   x=states[start:start+65536]
   c=x[:,self.cpos].long();t=x[:,self.tpos].long()
   cp=self.ids[c];co=self.oris[c];tp=self.ids[t];to=self.oris[t]
   rank=torch.zeros(len(x),dtype=torch.int64,device=x.device)
   for i in range(8):rank=rank*(8-i)+(cp[:,i+1:]<cp[:,i,None]).sum(dim=1)
   ori=torch.zeros_like(rank)
   for i in range(7):ori=ori*3+co[:,i]
   key=torch.zeros_like(rank)
   for i in range(6):key=key*6+tp[:,i]
   out[start:start+len(x)]=self.table[(rank*2187+ori)*2+(to.sum(dim=1)%4)//2]+self.mid[key]
  return out


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--pids',type=int,nargs='+',default=[30,296,468,918,336,94])
 ap.add_argument('--beam',type=int,default=65536)
 ap.add_argument('--modes',nargs='+',choices=['control','pdb'],default=['control','pdb'])
 ap.add_argument('--checkpoint',type=Path,default=ROOT/'models/e6/epoch_0499.pt')
 ap.add_argument('--out',type=Path,default=ROOT/'submissions/ihes_pdb_beam_probe.csv')
 args=ap.parse_args()
 torch.set_num_threads(2)
 puzzle=PictureCube.load(ROOT/'data/puzzle_info.json');states=load_test_states(ROOT/'data/test.csv')
 paths=load_submission(ROOT/'submission_ihes.csv');pdb=AdditivePDB(puzzle)
 # Independently compare vectorized orientation extraction with the established decomposition.
 sample=[s for pid in args.pids for s in [states[pid],puzzle.apply_path(states[pid],paths[pid][:3])]]
 sample.append(puzzle.solved_state)
 batch=torch.tensor(sample,device='cuda',dtype=torch.int8)
 for s in sample:
  trans=pdb.d.derive_state(s)
  for pos,key in [(pdb.cpos,'CORNERS'),(pdb.tpos,'CENTERS')]:
   stickers=torch.tensor(s,device='cuda')[pos]
   assert pdb.ids[stickers].tolist()==trans[key][0]
   assert pdb.oris[stickers].tolist()==trans[key][1]
 assert pdb(batch)[-1].item()==0
 assert torch.all(pdb(batch[:-1])<=torch.tensor([v for pid in args.pids for v in (len(paths[pid]),len(paths[pid])-3)],device='cuda'))
 print('GPU coordinate extraction and bound controls passed',flush=True)
 model=load_model_checkpoint(args.checkpoint,device='cuda',dtype=torch.bfloat16)
 cfg=KhoruzhiiSearchConfig(beam_width=args.beam,num_steps=30,num_attempts=1)
 results=[]
 for mode in args.modes:
  solver=KhoruzhiiSolver(puzzle,model,device='cuda',random_seed=0,pdb_lookup=pdb if mode=='pdb' else None,pdb_combine_mode='max')
  for pid in args.pids:
   torch.cuda.reset_peak_memory_stats();t=time.monotonic()
   found,length,path=solver.solve(states[pid],cfg)
   if found:
    assert verify_path(puzzle,states[pid],path).ok
    if len(path)<len(paths[pid]):paths[pid]=path
   row={'mode':mode,'pid':pid,'found':found,'length':length if found else None,'path':path if found else None,
        'seconds':round(time.monotonic()-t,3),'peak_gib':torch.cuda.max_memory_allocated()/2**30}
   results.append(row);print(json.dumps(row),flush=True)
   args.out.with_suffix('.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
   with args.out.open('w',encoding='utf-8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['initial_state_id','path'])
    for key in sorted(paths):writer.writerow([key,'.'.join(paths[key])])
 report=verify_submission(puzzle,ROOT/'data/test.csv',args.out);assert report.all_valid;print(report,flush=True)

if __name__=='__main__':main()
