"""Finite residual patterns for stable extraction of the three least entries."""
from collections import deque
import itertools
import json
from pathlib import Path
import time


def inv(p):
    return sum(a>b for i,a in enumerate(p) for b in p[i+1:])


def residual_patterns(k=3):
    m=3*k
    patterns={}
    for gaps in itertools.product(range(3),repeat=k):
        for order in itertools.permutations(range(k)):
            p=[]
            filler=k
            for gap,x in zip(gaps,order):
                p.extend(range(filler,filler+gap));filler+=gap
                p.append(x)
            p.extend(range(filler,m))
            patterns[bytes(p)]=(gaps,order)
    return patterns


def main():
    m=9
    targets=residual_patterns()
    identity=bytes(range(m))
    seen={identity:(None,None)}
    queue=deque([identity])
    remaining=set(targets)
    words={}
    tic=time.monotonic()
    while remaining:
        s=queue.popleft()
        if s in remaining:
            word=[]
            t=s
            while seen[t][0] is not None:
                parent,move=seen[t]
                word.append(move)
                t=parent
            words[s]=word[::-1]
            remaining.remove(s)
        for i in range(m-3):
            block=s[i:i+4]
            for d in (1,-1):
                rotated=block[1:]+block[:1] if d==1 else block[-1:]+block[:-1]
                t=s[:i]+rotated+s[i+4:]
                if t not in seen:
                    seen[t]=(s,(i,d))
                    queue.append(t)
    certificates=[]
    for p,(gaps,order) in targets.items():
        w=words[p]
        certificates.append({'permutation':list(p),'gaps':gaps,'order':order,'word':w,
                             'length':len(w),'inversions':inv(p),'defect':3*len(w)-inv(p)})
    max_defect=max(x['defect'] for x in certificates)
    result={'patterns':len(certificates),'max_defect':max_defect,
            'worst_cases':[x for x in certificates if x['defect']==max_defect],
            'by_inversions':{i:max(x['length'] for x in certificates if x['inversions']==i) for i in sorted(set(x['inversions'] for x in certificates))},
            'states_seen':len(seen),'seconds':time.monotonic()-tic}
    folder=Path(__file__).parent
    (folder/'triple_extraction_certificates.json').write_text(json.dumps(certificates,indent=2),encoding='utf-8')
    (folder/'triple_extraction_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
