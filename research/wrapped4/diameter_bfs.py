"""Exact full-group BFS, with packed permutations and dense Lehmer ranks."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from numba import njit


@njit(cache=True)
def rank_packed(state,n,fact,pop):
    rank=0
    used=0
    for i in range(n):
        v=(state >> np.uint64(4*i)) & np.uint64(15)
        smaller=int(v)-pop[used & ((1<<int(v))-1)]
        rank+=smaller*fact[n-1-i]
        used|=1<<int(v)
    return rank


@njit(cache=True)
def advance(queue,dist,begin,end,tail,depth,n,fact,pop,shifts,masks):
    for k in range(begin,end):
        state=queue[k]
        for i in range(n):
            p0,p1,p2,p3=shifts[i]
            a=(state >> p0)&np.uint64(15)
            b=(state >> p1)&np.uint64(15)
            c=(state >> p2)&np.uint64(15)
            d=(state >> p3)&np.uint64(15)
            fixed=state & masks[i]
            for direction in range(2):
                if direction==0:
                    t=fixed | (b<<p0) | (c<<p1) | (d<<p2) | (a<<p3)
                else:
                    t=fixed | (d<<p0) | (a<<p1) | (b<<p2) | (c<<p3)
                rank=rank_packed(t,n,fact,pop)
                if dist[rank]==255:
                    dist[rank]=depth+1
                    queue[tail]=t
                    tail+=1
    return tail


def pack(s):
    return np.uint64(sum(int(v)<<(4*i) for i,v in enumerate(s)))


def unpack(s,n):
    return [(int(s)>>(4*i))&15 for i in range(n)]


def run(n,output):
    tic=time.monotonic()
    total=math.factorial(n)
    fact=np.array([math.factorial(i) for i in range(n+1)],dtype=np.int64)
    pop=np.array([i.bit_count() for i in range(1<<n)],dtype=np.int64)
    shifts=np.array([[4*((i+j)%n) for j in range(4)] for i in range(n)],dtype=np.uint64)
    masks=np.array([((1<<64)-1)^sum(15<<int(p) for p in row) for row in shifts],dtype=np.uint64)
    dist=np.full(total,255,dtype=np.uint8)
    queue=np.empty(total,dtype=np.uint64)
    queue[0]=pack(range(n))
    dist[0]=0
    begin,end,tail,depth=0,1,1,0
    layers=[1]
    while begin<end:
        previous_begin,previous_end=begin,end
        tail=advance(queue,dist,begin,end,tail,depth,n,fact,pop,shifts,masks)
        count=tail-end
        if count==0: break
        begin,end=end,tail
        depth+=1
        layers.append(count)
        print(f'n={n} depth={depth} layer={count} discovered={tail}/{total} seconds={time.monotonic()-tic:.2f}',flush=True)
    assert tail==total,(n,tail,total)
    assert sum(layers)==total
    assert int(dist.max())==depth
    full=pack(list(range(n))[::-1])
    shifted=pack([0]+list(range(n-1,0,-1)))
    rotations=[int(dist[rank_packed(pack([(i+k)%n for i in range(n)]),n,fact,pop)]) for k in range(n)]
    farthest=[unpack(s,n) for s in queue[previous_begin:previous_end]]
    result={'n':n,'diameter':depth,'layer_counts':layers,'last_layer_size':len(farthest),
            'full_flip':int(dist[rank_packed(full,n,fact,pop)]),
            'shifted_flip':int(dist[rank_packed(shifted,n,fact,pop)]),
            'rotation_distances':rotations,'farthest':farthest,
            'seconds':time.monotonic()-tic,'visited':tail}
    output.mkdir(exist_ok=True)
    (output/f'n{n}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if n<=10: np.save(output/f'n{n}_dist.npy',dist)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='farthest'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--min-n',type=int,default=5)
    parser.add_argument('--max-n',type=int,default=10)
    args=parser.parse_args()
    output=Path(__file__).with_name('diameter_exact')
    for n in range(args.min_n,args.max_n+1): run(n,output)
