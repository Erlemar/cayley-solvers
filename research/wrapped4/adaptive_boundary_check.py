"""Exhaustive small and padded-boundary verification for the sharper bound."""
import itertools
import json
from pathlib import Path
import time

from adaptive_sort import construct


def replay(p,word):
    p=list(p)
    for i,d in word:
        idx=[(i+j)%len(p) for j in range(4)]
        block=[p[j] for j in idx]
        for j,pos in enumerate(idx): p[pos]=block[(j+d)%4]
    return p


def main():
    rows=[]
    for n,tail in [(n,n) for n in range(5,9)]+[(9,8),(14,8)]:
        tic=time.monotonic()
        count=0
        maximum=0
        for part in itertools.permutations(range(n-tail,n)):
            p=list(range(n-tail))+list(part)
            word,row=construct(p)
            assert replay(p,word)==list(range(n))
            assert len(word)<=(n*n//4+4*n+32)//3
            count+=1;maximum=max(maximum,len(word))
        row={'n':n,'tail_size':tail,'permutations':count,'maximum_word_length':maximum,'seconds':time.monotonic()-tic}
        rows.append(row)
        print(json.dumps(row),flush=True)
    result={'checks':rows,'scope':'Exhaustive small cases and all eight-entry tails with cyclic helper padding; supplements the finite-case proof.'}
    Path(__file__).with_name('adaptive_boundary_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__': main()
