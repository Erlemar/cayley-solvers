"""Constructive D(n) <= n^2/12 + O(n^(3/2)) bound, n>=5.

Uses the cyclic adjacent-transposition sorting theorem of van Zuylen et al.
All generated words are replayed independently in the validation routine.
"""
from math import ceil,sqrt
import json
from pathlib import Path
import random
from verify_theorem import replay

SWAP01=[(0,-1),(1,-1),(0,-1),(1,1),(1,1)]


def circular_adjacent_sort(p):
    """Balanced displacement algorithm from arXiv:1402.4867, Lemma 2."""
    p=list(p)
    n=len(p)
    pos=[p.index(i) for i in range(n)]
    displacement=[i-pos[i] for i in range(n)]
    while max(displacement)-min(displacement)>n:
        a=max(range(n),key=displacement.__getitem__)
        b=min(range(n),key=displacement.__getitem__)
        displacement[a]-=n
        displacement[b]+=n
    original=displacement[:]
    word=[]
    while any(displacement):
        found=False
        for i in range(n):
            j=(i+1)%n
            a,b=p[i],p[j]
            if displacement[a]>displacement[b]:
                displacement[a]-=1
                displacement[b]+=1
                p[i],p[j]=b,a
                word.append(i)
                found=True
        assert found
    assert p==list(range(n))
    assert len(word)<=n*n//4
    return word,original


def construct(p):
    p=list(p)
    n=len(p)
    assert n>=5 and sorted(p)==list(range(n))
    remainder=n%3
    N=n-remainder
    B=min(N,3*ceil(2*sqrt(N)/3))
    word=[]
    adjacent_count=0
    def adjacent(i):
        nonlocal adjacent_count
        p[i],p[i+1]=p[i+1],p[i]
        word.extend(((i+j)%n,d) for j,d in SWAP01)
        adjacent_count+=1
    # Gather at most two leftover labels at their final positions.
    for v in range(n-1,N-1,-1):
        i=p.index(v)
        while i<v:
            adjacent(i)
            i+=1
    assert p[N:]==list(range(N,n))
    classes=[[] for _ in range(ceil(N/B))]
    for i,v in enumerate(p[:N]): classes[v//B].append((i,v))
    triples=[]
    for c,items in enumerate(classes):
        assert len(items)%3==0
        for j in range(0,len(items),3):
            group=items[j:j+3]
            triples.append((group[-1][0],c,[v for _,v in group]))
    triples.sort()
    target_grouped=[v for _,_,group in triples for v in group]
    for i,v in enumerate(target_grouped):
        j=p.index(v)
        while j>i:
            adjacent(j-1)
            j-=1
    grouping_count=adjacent_count
    target_order=sorted(range(len(triples)),key=lambda k:(triples[k][1],k))
    target_index={k:i for i,k in enumerate(target_order)}
    block_ids=[target_index[k] for k in range(len(triples))]
    blocks=[group[:] for _,_,group in triples]
    if remainder:
        block_ids.append(len(blocks))
        blocks.append(list(range(N,n)))
    coarse_word,spins=circular_adjacent_sort(block_ids)
    if remainder: assert spins[-1]==0
    M=len(blocks)
    origin=0
    block_cost=0
    for i in coarse_word:
        j=(i+1)%M
        a,b=len(blocks[i]),len(blocks[j])
        start=(origin+sum(map(len,blocks[:i])))%n
        if a==3:
            word.extend(((start+k)%n,-1) for k in range(b))
            block_cost+=b
        else:
            assert b==3
            word.extend(((start+k)%n,1) for k in range(a-1,-1,-1))
            block_cost+=a
        if i==M-1: origin=(origin+b-a)%n
        blocks[i],blocks[j]=blocks[j],blocks[i]
    assert origin==0
    p[:]=[v for group in blocks for v in group]
    for start in range(0,N,B):
        end=min(start+B,N)
        assert sorted(p[start:end])==list(range(start,end))
        for i in range(start,end):
            j=p.index(i)
            while j>i:
                adjacent(j-1)
                j-=1
    assert p==list(range(n))
    q=ceil(N/B)
    coarse_bound=3*(M*M//4)
    adjacent_bound=remainder*(n-1)+2*N*q+N*(B-1)//2
    bound=coarse_bound+5*adjacent_bound
    assert adjacent_count<=adjacent_bound
    assert block_cost<=coarse_bound
    assert len(word)==5*adjacent_count+block_cost<=bound
    return word,{'n':n,'B':B,'classes':q,'block_count':M,'adjacent_swaps':adjacent_count,
                 'grouping_adjacent_swaps':grouping_count,'coarse_swaps':len(coarse_word),
                 'four_cycle_length':len(word),'explicit_upper_bound':bound}


def apply_to(p,word):
    n=len(p)
    p=list(p)
    for i,d in word:
        idx=[(i+j)%n for j in range(4)]
        old=[p[k] for k in idx]
        for j,k in enumerate(idx): p[k]=old[(j+d)%4]
    return p


if __name__=='__main__':
    assert replay(5,SWAP01)==[1,0,2,3,4]
    rng=random.Random(20260919)
    count=0
    records=[]
    for n in list(range(5,101))+[150,151,152,300,301,302]:
        for _ in range(5):
            p=list(range(n));rng.shuffle(p)
            w,record=construct(p)
            assert apply_to(p,w)==list(range(n)),(n,p)
            count+=1
        records.append(record)
    result={'verified_random_permutations':count,'sizes':'5..100;150..152;300..302','last_sample':records[-1],
            'scope':'Tests validate implementation; all-n upper bound requires the accompanying mathematical proof.'}
    Path(__file__).with_name('general_sort_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
