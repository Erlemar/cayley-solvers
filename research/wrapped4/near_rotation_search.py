"""Bounded exact DFS with an exact cyclic adjacent-length heuristic.

Only an exhausted search certifies nonexistence. Node-limited failures
are explicitly reported as inconclusive. Successful words are replayed.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from affine_insertion import balanced_keys
from diameter_bfs import pack,unpack
from general_sort import apply_to
from rotation_bounds import diameter_lower


@njit(cache=True)
def apply_packed(s,a,n):
    i=a//2
    direction=a%2
    idx=np.empty(4,np.int64)
    vals=np.empty(4,np.uint64)
    mask=np.uint64(0)
    for j in range(4):
        idx[j]=4*((i+j)%n)
        vals[j]=(s>>np.uint64(idx[j]))&np.uint64(15)
        mask|=np.uint64(15)<<np.uint64(idx[j])
    t=s&~mask
    for j in range(4):
        src=(j+1)%4 if direction==0 else (j+3)%4
        t|=vals[src]<<np.uint64(idx[j])
    return t


@njit(cache=True)
def potential(w,n):
    value=0
    for i in range(n):
        for j in range(i+1,n): value+=abs((w[j]-w[i])//n)
    return value


@njit(cache=True)
def advance_lift(w,length,a,n,wrapped=True):
    v=w.copy()
    i=a//2
    forward=a%2==0
    for offset in range(3):
        j=(i+(offset if forward else 2-offset))%n
        k=(j+1)%n
        x=v[j]
        y=v[k]+(n if k==0 else 0)
        length+=-1 if x>y else 1
        v[j]=y
        v[k]=x-(n if k==0 else 0)
    if not wrapped: return v,length
    changed=False
    while True:
        imax=imin=0
        for j in range(1,n):
            if v[j]-j>v[imax]-imax: imax=j
            if v[j]-j<v[imin]-imin: imin=j
        if (v[imax]-imax)-(v[imin]-imin)<=n: break
        v[imax]-=n
        v[imin]+=n
        changed=True
    if changed: length=potential(v,n)
    return v,length


@njit(cache=True)
def build_endgame(n,radius,capacity,wrapped=True):
    dist=Dict.empty(key_type=types.uint64,value_type=types.int16)
    queue=np.empty(capacity,np.uint64)
    identity=np.uint64(0)
    for i in range(n): identity|=np.uint64(i)<<np.uint64(4*i)
    dist[identity]=np.int16(0)
    queue[0]=identity
    begin,end=0,1
    for depth in range(radius):
        tail=end
        for j in range(begin,end):
            s=queue[j]
            for a in range(2*(n if wrapped else n-3)):
                t=apply_packed(s,a,n)
                if t not in dist:
                    assert tail<capacity
                    queue[tail]=t
                    dist[t]=np.int16(depth+1)
                    tail+=1
        begin,end=end,tail
    return dist


@njit(cache=True)
def dfs(s,w,ell,remaining,last,depth,n,radius,endgame,failed,path,stats,cap,commute,seed,wrapped):
    stats[0]+=1
    if stats[0]>cap: return -1
    if ell>3*remaining: return 0
    if s in endgame:
        if endgame[s]<=remaining:
            stats[1]=depth
            stats[2]=int(endgame[s])
            return 1
        return 0
    if remaining<=radius: return 0
    cache_key=(s,last)
    if cache_key in failed and failed[cache_key]>=remaining: return 0
    candidates=np.empty(2*n,np.int64)
    scores=np.empty(2*n,np.int64)
    lengths=np.empty(2*n,np.int64)
    lifts=np.empty((2*n,n),np.int64)
    states=np.empty(2*n,np.uint64)
    count=0
    for a in range(2*(n if wrapped else n-3)):
        if last>=0 and a==(last^1): continue
        if last>=0 and commute[last//2,a//2] and a<last: continue
        v,next_length=advance_lift(w,ell,a,n,wrapped)
        if next_length>3*(remaining-1): continue
        t=apply_packed(s,a,n)
        score=next_length*(2*n)+(a+seed)%(2*n)
        j=count
        while j>0 and scores[j-1]>score:
            scores[j]=scores[j-1]
            candidates[j]=candidates[j-1]
            lengths[j]=lengths[j-1]
            states[j]=states[j-1]
            lifts[j,:]=lifts[j-1,:]
            j-=1
        scores[j]=score
        candidates[j]=a
        lengths[j]=next_length
        states[j]=t
        lifts[j,:]=v
        count+=1
    for j in range(count):
        a=candidates[j]
        path[depth]=a
        result=dfs(states[j],lifts[j],lengths[j],remaining-1,a,depth+1,n,radius,endgame,failed,path,stats,cap,commute,seed,wrapped)
        if result!=0: return result
    # Include the preceding action because the commuting filter depends on it.
    failed[cache_key]=np.int16(remaining)
    return 0


def solve(p,budget,endgame,radius,cap,seed=0,wrapped=True):
    n=len(p)
    w=np.array(balanced_keys(list(p)) if wrapped else p,dtype=np.int64)
    ell=int(potential(w,n))
    assert (budget-ell)%2==0
    failed=Dict.empty(key_type=types.Tuple((types.uint64,types.int64)),value_type=types.int16)
    path=np.zeros(budget+1,np.int64)
    stats=np.zeros(3,np.int64)
    supports=[{(i+j)%n for j in range(4)} for i in range(n)]
    commute=np.array([[not supports[i]&supports[j] for j in range(n)] for i in range(n)],dtype=np.bool_)
    tic=time.monotonic()
    result=dfs(pack(p),w,ell,budget,-1,0,n,radius,endgame,failed,path,stats,cap,commute,seed,wrapped)
    record={'n':n,'permutation':list(map(int,p)),'budget':budget,'adjacent_length':ell,
            'status':{1:'found',0:'exhausted',-1:'node_limit'}[result],
            'nodes':int(stats[0]),'seconds':time.monotonic()-tic}
    if result==1:
        actions=list(map(int,path[:stats[1]]))
        s=pack(p)
        for a in actions: s=np.uint64(apply_packed(s,a,n))
        while int(endgame[s]):
            old=int(endgame[s])
            for a in range(2*(n if wrapped else n-3)):
                t=np.uint64(apply_packed(s,a,n))
                if t in endgame and int(endgame[t])==old-1:
                    actions.append(a);s=t;break
            else: raise AssertionError('missing endgame descent')
        word=[(a//2,1 if a%2==0 else -1) for a in actions]
        assert len(word)<=budget
        assert apply_to(p,word)==list(range(n))
        record['word']=word
        record['length']=len(word)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=15)
    parser.add_argument('--cap',type=int,default=100000)
    parser.add_argument('--cases',type=int,default=12)
    parser.add_argument('--family',choices=['rotation','mixed','known'],default='rotation')
    parser.add_argument('--budget-offset',type=int,default=0)
    parser.add_argument('--report-every',type=int,default=1)
    args=parser.parse_args()
    n=args.n
    assert n<=16
    endgame=build_endgame(n,4,2000000)
    print('endgame',n,len(endgame),flush=True)
    # Verify the incremental heuristic against fresh balancing.
    rng=random.Random(20260922)
    for _ in range(100):
        p=list(range(n));rng.shuffle(p)
        w=np.array(balanced_keys(p),np.int64)
        ell=int(potential(w,n))
        for a in range(2*n):
            v,new_length=advance_lift(w,ell,a,n)
            t=unpack(apply_packed(pack(p),a,n),n)
            assert list(v%n)==t
            fresh=np.array(balanced_keys(t),np.int64)
            assert new_length==int(potential(fresh,n))==int(potential(v,n))
    targets=[]
    if n<=12:
        data=json.loads((Path(__file__).parent/'diameter_exact'/f'n{n}.json').read_text(encoding='utf-8'))
        targets=[(p,data['diameter']+args.budget_offset) for p in data['farthest'][:args.cases]]
    else:
        hard12=json.loads((Path(__file__).parent/'diameter_exact'/'n12.json').read_text(encoding='utf-8'))['farthest']
        for k in range(args.cases):
            family=k%4 if args.family=='mixed' else 0
            if family==0:
                p=[(i+n//2)%n for i in range(n)]
                swaps=[] if k==0 else [0] if k==1 else rng.sample(range(n),3 if k%2 else 5)
                for i in swaps: p[i],p[(i+1)%n]=p[(i+1)%n],p[i]
            elif family==1:
                p=list(range(n))[::-1]
                for i in rng.sample(range(n),rng.randrange(1,7)):
                    p[i],p[(i+1)%n]=p[(i+1)%n],p[i]
            elif family==2:
                small=rng.choice(hard12)
                sizes=[1]*12
                for i in rng.sample(range(12),n-12): sizes[i]+=1
                starts=[sum(sizes[:i]) for i in range(12)]
                p=[v for i in small for v in range(starts[i],starts[i]+sizes[i])]
            else:
                p=list(range(n));rng.shuffle(p)
            ell=int(potential(np.array(balanced_keys(p),np.int64),n))
            budget=diameter_lower(n)+args.budget_offset
            if (budget-ell)%2: budget-=1
            targets.append((p,budget))
    rows=[]
    output=Path(__file__).with_name(f'search_{args.family}_n{n}_offset{args.budget_offset}.json')
    for k,(p,budget) in enumerate(targets):
        record=solve(p,budget,endgame,4,args.cap,seed=k)
        rows.append(record)
        if k%args.report_every==0 or record['status']!='found':
            print(json.dumps({'case':k,**{key:value for key,value in record.items() if key!='word'}}),flush=True)
        output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps({status:sum(r['status']==status for r in rows) for status in ['found','exhausted','node_limit']}),flush=True)


if __name__=='__main__': main()
