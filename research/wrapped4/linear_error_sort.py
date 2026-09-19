"""Constructive universal bound D(n) <= floor((floor(n^2/4)+12n-16)/3).

Uses balanced affine lifts and a finite five-position certificate.
All final words are independently replayed.
"""
import json
from pathlib import Path
import random

from affine_insertion import LOCAL, balanced_keys, compile_run
from general_sort import apply_to


def inversions(p):
    return sum(a>b for i,a in enumerate(p) for b in p[i+1:])


def construct(p,interval_sorter=None):
    n=len(p)
    assert n>=5
    keys=balanced_keys(p)
    potential=sum(abs((keys[j]-keys[i])//n) for i in range(n) for j in range(i+1,n))
    word=[]
    def move(i,d):
        i%=n
        idx=list(range(i,i+4))
        old=[keys[j%n]+n*(j//n) for j in idx]
        for offset,j in enumerate(idx):
            keys[j%n]=old[(offset+d)%4]-n*(j//n)
        word.append((i,d))

    def at(i):
        return keys[i%n]+n*(i//n)

    def baseline_linear_sort(start,m):
        before=len(word)
        initial=[at(i) for i in range(start,start+m)]
        target=sorted(initial)
        for offset in range(m-5):
            desired=target[offset]
            first=start+offset
            j=next(j for j in range(first,start+m) if at(j)==desired)
            while j-first>=3:
                move(j-3,-1)
                j-=3
            for _ in range(j-first): move(first,1)
            assert at(first)==desired
        end=start+m-5
        values=[at(i) for i in range(end,end+5)]
        order={v:i for i,v in enumerate(sorted(values))}
        pattern=tuple(order[v] for v in values)
        for i,d in reversed(LOCAL[pattern]): move(end+i,-d)
        assert [at(i) for i in range(start,start+m)]==target
        cost=len(word)-before
        assert 3*cost<=inversions(initial)+6*m-12
        return cost,inversions(initial)

    def linear_sort(start,m):
        if interval_sorter is None: return baseline_linear_sort(start,m)
        return interval_sorter(at,move,start,m,baseline_linear_sort)

    if n==5:
        for i,d in reversed(LOCAL[tuple(p)]): move(i,-d)
        assert apply_to(p,word)==list(range(n))
        return word,{'n':n,'four_cycle_length':len(word),'special_five_case':True}

    first_cost,first_inv=linear_sort(0,n)
    largest=keys[-1]
    travel=largest-(n-1)
    assert 0<=travel<n
    before=len(word)
    for i,d in compile_run(n-1,1,travel,n): move(i,d)
    pivot_cost=len(word)-before
    pivot=largest%n
    assert keys[pivot]==pivot
    assert sorted(at(i) for i in range(pivot+1,pivot+n))==list(range(pivot+1,pivot+n))
    last_cost,last_inv=linear_sort(pivot+1,n-1)
    assert keys==list(range(n)),(p,keys)
    assert first_inv+travel+last_inv==potential<=n*n//4
    assert 3*pivot_cost<=travel+14
    assert 3*len(word)<=potential+12*n-16
    assert apply_to(p,word)==list(range(n))
    return word,{'n':n,'adjacent_length':potential,'four_cycle_length':len(word),
                 'first_inversions':first_inv,'last_inversions':last_inv,
                 'first_cost':first_cost,'pivot_cost':pivot_cost,'last_cost':last_cost,
                 'upper_bound':(n*n//4+12*n-16)//3}


def main():
    certificates=[]
    for p,w in LOCAL.items():
        assert apply_to(list(range(5)),w)==list(p)
        defect=3*len(w)-inversions(p)
        assert defect<=18
        certificates.append({'permutation':p,'word':w,'inversions':inversions(p),'defect':defect})
    output=Path(__file__).parent
    (output/'five_position_certificates.json').write_text(json.dumps(certificates,indent=2),encoding='utf-8')
    rng=random.Random(20260921)
    count=0
    rows=[]
    for n in list(range(5,151))+[250,500,1000]:
        candidates=[list(range(n))[::-1],[(i+n//2)%n for i in range(n)]]
        for _ in range(10):
            p=list(range(n));rng.shuffle(p);candidates.append(p)
        for p in candidates:
            w,row=construct(p)
            count+=1
        rows.append(row)
    result={'verified_permutations':count,'sizes':'all 5..150;250;500;1000',
            'finite_base_permutations':len(certificates),'maximum_base_defect':max(x['defect'] for x in certificates),
            'last_samples':rows[-3:],
            'assertions':'independent full word replay; affine stage inversion decomposition; linear-stage cost bounds; universal upper bound'}
    (output/'linear_error_sort_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
