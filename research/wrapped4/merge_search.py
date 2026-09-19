"""Test constant-defect sorting of the concatenation of two increasing lists."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from near_rotation_search import build_endgame,solve
from triple_extraction import inv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=12)
    parser.add_argument('--defect',type=int,default=14)
    parser.add_argument('--cap',type=int,default=20000)
    parser.add_argument('--samples',type=int,default=0)
    args=parser.parse_args()
    n=args.n
    assert n<=16
    endgame=build_endgame(n,4,2000000,False)
    rng=random.Random(20260924)
    masks=range(1<<n) if not args.samples else [rng.randrange(1<<n) for _ in range(args.samples)]
    rows=[]
    counts=Counter()
    for k,mask in enumerate(masks):
        p=[i for i in range(n) if (mask>>i)&1]+[i for i in range(n) if not (mask>>i)&1]
        inversions=inv(p)
        budget=(inversions+args.defect)//3
        if (budget-inversions)%2: budget-=1
        row=solve(p,budget,endgame,4,args.cap,wrapped=False)
        if row['status']=='node_limit':
            for seed in (7,13):
                retry=solve(p,budget,endgame,4,args.cap,seed=seed,wrapped=False)
                if retry['status']!='node_limit': row=retry;break
        row['mask']=mask
        rows.append(row)
        counts[row['status']]+=1
        if k%256==0 or row['status']!='found':
            print(json.dumps({'case':k,'counts':counts,'latest':{a:b for a,b in row.items() if a!='word'}}),flush=True)
        if row['status']=='exhausted': break
    result={'n':n,'defect_budget':args.defect,'counts':counts,'rows':rows,
            'scope':'Finite tests of a proposed merge bound, not a general proof.'}
    Path(__file__).with_name(f'merge_n{n}_defect{args.defect}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(counts),flush=True)


if __name__=='__main__': main()
