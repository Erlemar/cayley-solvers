"""Apply the fully covered c=2 extraction certificate to arbitrary inputs."""
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random

from general_sort import apply_to
from linear_error_sort import construct as build_sort,inversions

FOLDER=Path(__file__).parent
DATA=json.loads((FOLDER/'adaptive_extraction_c2.json').read_text(encoding='utf-8'))
TABLE={(row['k'],tuple(row['pattern'])):row for row in DATA['records']}
MIN_ACTIVE=14


def sort_interval(at,move,start,m,baseline):
    original=[at(i) for i in range(start,start+m)]
    initial_inv=inversions(original)
    cost=0
    first=start
    left=m
    while left>=MIN_ACTIVE:
        target=sorted(at(i) for i in range(first,first+left))
        k=3
        while True:
            marked=set(target[:k])
            order=[at(i) for i in range(first,first+left) if at(i) in marked]
            previous=first-1
            for v in order:
                j=next(i for i in range(previous+1,first+left) if at(i)==v)
                while j-previous-1>=3:
                    move(j-3,-1);cost+=1;j-=3
                previous=j
            selected_index={v:i for i,v in enumerate(target[:k])}
            end=max(i for i in range(first,first+left) if at(i) in marked)
            pattern=[]
            filler=k
            for i in range(first,end+1):
                v=at(i)
                if v in marked: pattern.append(selected_index[v])
                else: pattern.append(filler);filler+=1
            row=TABLE[k,tuple(pattern)]
            if row['status']=='found':
                assert row['width']<=left
                for i,d in row['word']:
                    move(first+i,d);cost+=1
                assert [at(i) for i in range(first,first+k)]==target[:k]
                first+=k;left-=k
                break
            k+=1
            assert k<=8
    assert 5<=left<MIN_ACTIVE
    final_cost,_=baseline(first,left)
    cost+=final_cost
    assert [at(i) for i in range(start,start+m)]==sorted(original)
    assert 3*cost<=initial_inv+2*m+40
    return cost,initial_inv


def sort_interval_padded(at,move,start,m,baseline,buffer_size=MIN_ACTIVE):
    """Use fixed surrounding entries as temporary helpers for the final batch."""
    original=[at(i) for i in range(start,start+m)]
    initial_inv=inversions(original)
    first=start
    left=m
    cost=0
    extracted=0
    while left>1:
        active=[at(i) for i in range(first,first+left)]
        if active==sorted(active): break
        helpers=[at(i) for i in range(first+left,first+max(left,buffer_size))]
        target=sorted(active)+helpers
        k=3
        while True:
            marked=set(target[:k])
            order=[at(i) for i in range(first,first+max(left,buffer_size)) if at(i) in marked]
            previous=first-1
            for v in order:
                j=next(i for i in range(previous+1,first+max(left,buffer_size)) if at(i)==v)
                while j-previous-1>=3:
                    move(j-3,-1);cost+=1;j-=3
                previous=j
            selected_index={v:i for i,v in enumerate(target[:k])}
            end=max(i for i in range(first,first+max(left,buffer_size)) if at(i) in marked)
            pattern=[]
            filler=k
            for i in range(first,end+1):
                v=at(i)
                if v in marked: pattern.append(selected_index[v])
                else: pattern.append(filler);filler+=1
            row=TABLE[k,tuple(pattern)]
            if row['status']=='found':
                assert row['width']<=max(left,buffer_size)
                for i,d in row['word']:
                    move(first+i,d);cost+=1
                extracted+=k
                assert [at(i) for i in range(first,first+k)]==target[:k]
                assert [at(i) for i in range(first+left,first+max(left,buffer_size))]==helpers
                if left==2: assert k<=7
                consumed=min(left,k)
                first+=consumed;left-=consumed
                break
            k+=1
            assert k<=8
    assert extracted<=m+5
    assert [at(i) for i in range(start,start+m)]==sorted(original)
    assert 3*cost<=initial_inv+2*m+10
    return cost,initial_inv


def construct(p,padded=True):
    use_padding=padded and len(p)>=9
    buffer_size=min(MIN_ACTIVE,len(p))
    sorter=(lambda at,move,start,m,baseline: sort_interval_padded(at,move,start,m,baseline,buffer_size)) if use_padding else sort_interval
    word,row=build_sort(p,interval_sorter=sorter)
    n=len(p)
    upper=(n*n//4+4*n+92)//3
    assert len(word)<=upper
    assert apply_to(p,word)==list(range(n))
    row['adaptive_upper_bound']=upper
    if use_padding:
        stronger=(n*n//4+4*n+32)//3
        assert len(word)<=stronger
        row['padded_upper_bound']=stronger
    return word,row


def independent_tree_check():
    # Enumerate the abstract states separately from the discovery implementation.
    def pattern(positions,k):
        values=[None]*(max(positions)+1)
        for label,i in enumerate(positions): values[i]=label
        filler=k
        for i,v in enumerate(values):
            if v is None: values[i]=filler;filler+=1
        return tuple(values)
    expected=set()
    for gaps in itertools.product((0,1,2),repeat=3):
        for order in itertools.permutations(range(3)):
            positions=[None]*3
            j=0
            for gap,label in zip(gaps,order):
                j+=gap;positions[label]=j;j+=1
            expected.add(pattern(positions,3))
    summaries=[]
    total=0
    assert len(TABLE)==len(DATA['records'])
    for k in range(3,9):
        current={p:row for (kk,p),row in TABLE.items() if kk==k}
        assert set(current)==expected
        expected=set()
        successes=0
        for p,row in current.items():
            assert sorted(p)==list(range(len(p)))
            positions=[p.index(i) for i in range(k)]
            ordered=sorted(positions)
            assert ordered[0]<=2
            assert all(b-a-1<=2 for a,b in zip(ordered,ordered[1:]))
            if row['status']=='found':
                full=list(p)+list(range(len(p),row['width']))
                assert row['permutation']==full
                assert row['width']==max(9,len(p))
                assert all(0<=i<=row['width']-4 and d in (-1,1) for i,d in row['word'])
                # Direct, elementary replay independent of the search functions.
                a=full[:]
                for i,d in row['word']:
                    b=a[i:i+4]
                    a[i:i+4]=b[1:]+b[:1] if d==1 else b[-1:]+b[:-1]
                assert a==list(range(row['width']))
                assert 3*len(row['word'])-inversions(full)<=2*k
                assert row['width']<=MIN_ACTIVE and k<=MIN_ACTIVE-5
                successes+=1
            else:
                for pos in [i for i in range(len(p)) if p[i]>=k]+list(range(len(p),len(p)+3)):
                    expected.add(pattern(positions+[pos],k+1))
            total+=1
        summaries.append({'k':k,'patterns':len(current),'successful_words':successes})
    assert not expected
    first_swap=next(k for k in range(3,9) if TABLE[k,tuple([1,0]+list(range(2,k)))]['status']=='found')
    assert first_swap<=7
    return {'patterns_checked':total,'stages':summaries,'complete':True}


def main():
    certificate=independent_tree_check()
    rng=random.Random(20260923)
    count=0
    rows=[]
    for n in list(range(5,151))+[250,500,1000]:
        inputs=[list(range(n))[::-1],[(i+n//2)%n for i in range(n)]]
        for _ in range(10):
            p=list(range(n));rng.shuffle(p);inputs.append(p)
        for p in inputs:
            w,row=construct(p);count+=1
        rows.append(row)
    result={'certificate':certificate,'verified_permutations':count,'sizes':'5..150;250;500;1000',
            'last_samples':rows[-3:],'upper_bound':'floor((floor(n^2/4)+4n+32)/3) for n>=9; smaller n checked separately',
            'scope':'Finite exhaustive abstract-case coverage plus all-n induction; random replay checks supplement the proof.'}
    (FOLDER/'adaptive_sort_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
