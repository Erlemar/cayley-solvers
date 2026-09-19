"""Score exact-search opening subtrees; ordering changes no coverage or bounds."""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,load_test_states
from cayley.search import load_model_checkpoint

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--pids',type=int,nargs='*')
 ap.add_argument('--out',type=Path,default=ROOT/'data/ihes_pdb/prefix_ranks.txt')
 args=ap.parse_args()
 torch.set_num_threads(2)
 puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
 spec=importlib.util.spec_from_file_location('decomp',ROOT/'scripts/build_picture_cube_kpuzzle_v2.py')
 decomp=importlib.util.module_from_spec(spec);spec.loader.exec_module(decomp)
 paths=load_submission(ROOT/'submission_ihes.csv');states=load_test_states(ROOT/'data/test.csv')
 names=[sign+axis+str(layer) for axis in 'frd' for layer in range(3) for sign in ('','-')]
 combos=np.array(list(itertools.product(range(18),repeat=3)),dtype=np.int64)
 gs=np.array([puzzle.generators[m] for m in names])
 permutations=np.tile(np.arange(72),(len(combos),1))
 for k in range(3):permutations=np.take_along_axis(permutations,gs[combos[:,k]],axis=1)
 native=np.array([(i//2)*3+(2 if i%2 else 0) for i in range(18)])
 codes=27**3+native[combos[:,0]]+27*native[combos[:,1]]+729*native[combos[:,2]]
 checkpoints=[ROOT/'models/e6/epoch_0499.pt',ROOT/'models/az_cube_v2/v_only_0099.pt']
 models=[load_model_checkpoint(p,device='cuda',dtype=torch.bfloat16).eval() for p in checkpoints]
 pids=args.pids or sorted(pid for pid,p in paths.items() if len(p)>=20)
 report=[]
 with args.out.open('w',encoding='utf-8') as f,torch.inference_mode():
  for pid in pids:
   batch=torch.tensor(np.array(states[pid])[permutations],device='cuda',dtype=torch.uint8)
   values=[]
   for model in models:
    value=torch.cat([model(x).flatten().float() for x in batch.split(1024)]).cpu().numpy()
    values.append(value)
   # Mean of standardized within-root values gives each model equal influence.
   ensemble=sum((v-v.mean())/max(float(v.std()),1e-6) for v in values)
   score=values[0]  # E6 won the measured incumbent-opening ranking gate.
   goal=np.array(puzzle.apply_path(states[pid],paths[pid][:3]))
   match=np.all(np.array(states[pid])[permutations]==goal,axis=1)
   assert match.any()
   rank=[int(np.sum(v<v[match].min()))+1 for v in (*values,ensemble)]
   trans=decomp.derive_state(states[pid]);raw=[]
   for key in ('CORNERS','EDGES','CENTERS'):
    for row in trans[key]:raw.extend(row)
   f.write(bytes(raw).hex()+' '+str(len(codes))+'\n')
   for code,value in zip(codes,score):f.write(f'{int(code)} {float(value):.7f}\n')
   report.append({'pid':pid,'length':len(paths[pid]),'known_opening_rank_e6_az_mean':rank,'candidates':len(codes)})
   if len(report)%20==0:print('scored',len(report),'puzzles',flush=True)
 report_path=args.out.with_suffix('.json')
 report_path.write_text(json.dumps({'checkpoints':list(map(str,checkpoints)),'rows':report},indent=2),encoding='utf-8')
 ranks=np.array([r['known_opening_rank_e6_az_mean'] for r in report])
 print('known incumbent prefix ranks: median',np.median(ranks,axis=0).tolist(),'mean',ranks.mean(axis=0).tolist(),'out of',len(codes))

if __name__=='__main__':main()
