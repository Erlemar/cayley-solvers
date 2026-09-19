"""Search for words attaining the inversion/parity lower bound on a line."""
from explore import move
from pathlib import Path
import json
import time


def inv(s):
    return sum(a>b for i,a in enumerate(s) for b in s[i+1:])


def solve(n, time_limit=25):
    start=tuple(range(n-1,-1,-1))
    target=tuple(range(n))
    initial=inv(start)
    lower=(initial+2)//3
    if (lower-initial)%2: lower+=1
    end=time.monotonic()+time_limit
    failed=set()
    calls=0
    def dfs(s,budget,inversions,last):
        nonlocal calls
        calls+=1
        if calls%10000==0 and time.monotonic()>end: raise TimeoutError
        if not inversions: return []
        if not budget or 3*budget<inversions: return None
        key=(s,budget)
        if key in failed: return None
        cand=[]
        for i in range(n-3):
            x=s[i:i+4]
            for d in (1,-1):
                if last==(i,-d): continue
                delta=(2*sum(x[0]>v for v in x[1:])-3) if d==1 else (2*sum(v>x[3] for v in x[:3])-3)
                if inversions-delta>3*(budget-1): continue
                t=move(s,i,d)
                # Prefer decreases and more available three-inversion moves.
                mobility=sum(t[j]>max(t[j+1:j+4]) for j in range(n-3))+sum(t[j+3]<min(t[j:j+3]) for j in range(n-3))
                cand.append((-delta,-mobility,i,d,t))
        cand.sort()
        for negdelta,_,i,d,t in cand:
            path=dfs(t,budget-1,inversions+negdelta,(i,d))
            if path is not None: return [(i,d)]+path
        failed.add(key)
        return None
    try:
        path=dfs(start,lower,initial,None)
        return {'n':n,'lower':lower,'path':path,'calls':calls,'failed':len(failed)}
    except TimeoutError:
        return {'n':n,'lower':lower,'timeout':True,'calls':calls,'failed':len(failed)}


if __name__=='__main__':
    results=[]
    for n in (10,11,12,13,14,15,16,17):
        r=solve(n)
        print(r,flush=True)
        results.append(r)
        Path(__file__).with_name('monotone_results.json').write_text(json.dumps(results,indent=2))
