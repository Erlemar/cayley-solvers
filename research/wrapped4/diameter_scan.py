"""Memory-bounded exact BFS: scan dense distance array instead of storing a queue."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from numba import njit
from diameter_bfs import rank_packed,pack,unpack

@njit(cache=True)
def unrank_packed(rank,n,fact):
    available=np.empty(16,dtype=np.uint64)
    for i in range(n): available[i]=i
    state=np.uint64(0)
    remaining=n
    for i in range(n):
        k=rank//fact[n-1-i]
        rank%=fact[n-1-i]
        state |= available[k]<<np.uint64(4*i)
        for j in range(k,remaining-1): available[j]=available[j+1]
        remaining-=1
    return state

@njit(cache=True)
def scan_layer(dist,begin,end,depth,n,fact,pop,shifts,masks):
    added=0
    processed=0
    for r in range(begin,end):
        if dist[r]!=depth: continue
        processed+=1
        state=unrank_packed(r,n,fact)
        for i in range(n):
            p0,p1,p2,p3=shifts[i]
            a=(state>>p0)&np.uint64(15)
            b=(state>>p1)&np.uint64(15)
            c=(state>>p2)&np.uint64(15)
            d=(state>>p3)&np.uint64(15)
            fixed=state&masks[i]
            for direction in range(2):
                if direction==0: t=fixed|(b<<p0)|(c<<p1)|(d<<p2)|(a<<p3)
                else: t=fixed|(d<<p0)|(a<<p1)|(b<<p2)|(c<<p3)
                rank=rank_packed(t,n,fact,pop)
                if dist[rank]==255:
                    dist[rank]=depth+1
                    added+=1
    return added,processed

def run(n):
    total=math.factorial(n)
    fact=np.array([math.factorial(i) for i in range(n+1)],dtype=np.int64)
    pop=np.array([i.bit_count() for i in range(1<<n)],dtype=np.int64)
    shifts=np.array([[4*((i+j)%n) for j in range(4)] for i in range(n)],dtype=np.uint64)
    masks=np.array([((1<<64)-1)^sum(15<<int(p) for p in row) for row in shifts],dtype=np.uint64)
    dist=np.full(total,255,dtype=np.uint8)
    dist[0]=0
    layers=[1]
    output=Path(__file__).with_name('diameter_exact')
    output.mkdir(exist_ok=True)
    tic=time.monotonic()
    for depth in range(255):
        added=processed=0
        last_report=time.monotonic()
        for begin in range(0,total,2000000):
            a,p=scan_layer(dist,begin,min(begin+2000000,total),depth,n,fact,pop,shifts,masks)
            added+=a;processed+=p
            if time.monotonic()-last_report>=25:
                print(f'n={n} processing_depth={depth} ranks_done={min(begin+2000000,total)}/{total} processed={processed} next_layer={added} seconds={time.monotonic()-tic:.1f}',flush=True)
                last_report=time.monotonic()
        assert processed==layers[-1],(n,depth,processed,layers[-1])
        print(f'n={n} depth={depth} layer={processed} next_layer={added} seconds={time.monotonic()-tic:.2f}',flush=True)
        (output/f'n{n}_scan_progress.json').write_text(json.dumps({'n':n,'completed_depth':depth,'layer_counts':layers,'next_layer':added,'seconds':time.monotonic()-tic}),encoding='utf-8')
        if added==0: break
        layers.append(added)
    assert sum(layers)==total
    assert int(dist.max())==depth
    farthest=[unpack(unrank_packed(int(r),n,fact),n) for r in np.flatnonzero(dist==depth)]
    def get(s): return int(dist[rank_packed(pack(s),n,fact,pop)])
    result={'n':n,'diameter':depth,'layer_counts':layers,'last_layer_size':len(farthest),
            'full_flip':get(list(range(n))[::-1]),'shifted_flip':get([0]+list(range(n-1,0,-1))),
            'rotation_distances':[get([(i+k)%n for i in range(n)]) for k in range(n)],
            'farthest':farthest,'visited':total,'seconds':time.monotonic()-tic,
            'method':'dense distance scan, no search truncation'}
    path=output/f'n{n}.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        assert old['layer_counts']==layers
        assert old['diameter']==depth
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if n<=12: np.save(output/f'n{n}_dist.npy',dist)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='farthest'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('n',type=int)
    args=parser.parse_args()
    run(args.n)
