"""Retry only unresolved candidate searches; preserve original evidence."""
import argparse
import json
from pathlib import Path

from near_rotation_search import build_endgame,solve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=15)
    parser.add_argument('--cap',type=int,default=1000000)
    args=parser.parse_args()
    folder=Path(__file__).parent
    original=json.loads((folder/f'search_mixed_n{args.n}_offset0.json').read_text(encoding='utf-8'))
    pending=[(i,r) for i,r in enumerate(original) if r['status']=='node_limit']
    endgame=build_endgame(args.n,4,2000000)
    rows=[]
    for i,row in pending:
        attempts=[]
        for seed in (0,7,13):
            result=solve(row['permutation'],row['budget'],endgame,4,args.cap,seed)
            result['seed']=seed
            attempts.append(result)
            if result['status']!='node_limit': break
        record={'original_case':i,'attempts':attempts,'status':attempts[-1]['status']}
        rows.append(record)
        print(json.dumps({'case':i,'status':record['status'],'nodes':[a['nodes'] for a in attempts]}),flush=True)
        (folder/f'search_mixed_n{args.n}_retries.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps({s:sum(r['status']==s for r in rows) for s in ['found','exhausted','node_limit']}),flush=True)


if __name__=='__main__': main()
