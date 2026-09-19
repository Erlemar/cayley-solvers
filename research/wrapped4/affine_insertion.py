"""Compress a reduced affine sorting word into O(n) monotone runs.

Discovery/verification code. Every output 4-cycle word is independently replayed.
"""
from collections import deque
import json
from pathlib import Path
import random

from general_sort import apply_to, SWAP01, circular_adjacent_sort


def local_words():
    initial=tuple(range(5))
    words={initial:[]}
    queue=deque([initial])
    while queue:
        p=queue.popleft()
        for i in (0,1):
            for d in (1,-1):
                t=list(p)
                t[i:i+4]=t[i+1:i+4]+t[i:i+1] if d==1 else t[i+3:i+4]+t[i:i+3]
                t=tuple(t)
                if t not in words:
                    words[t]=words[p]+[(i,d)]
                    queue.append(t)
    assert len(words)==120
    return words


LOCAL=local_words()
SWAP_TWO=LOCAL[(1,2,0,3,4)]
MOVE_FOUR=LOCAL[(1,2,3,4,0)]


def balanced_keys(p):
    n=len(p)
    pos=[p.index(v) for v in range(n)]
    spins=[v-pos[v] for v in range(n)]
    while max(spins)-min(spins)>n:
        a=max(range(n),key=spins.__getitem__)
        b=min(range(n),key=spins.__getitem__)
        spins[a]-=n
        spins[b]+=n
    return [i+spins[p[i]] for i in range(n)]


def reduced_runs(p):
    n=len(p)
    keys=balanced_keys(p)
    potential=sum(abs((keys[j]-keys[i])//n) for i in range(n) for j in range(i+1,n))
    initial_keys=keys[:]
    runs=[]

    def swap(i):
        j=(i+1)%n
        if i==n-1:
            assert keys[i]>keys[j]+n
            keys[i],keys[j]=keys[j]+n,keys[i]-n
        else:
            assert keys[i]>keys[j]
            keys[i],keys[j]=keys[j],keys[i]

    # Linear insertion sorting of the lifted target positions.
    for end in range(1,n):
        i=end
        while i>0 and keys[i]<keys[i-1]:
            swap(i-1)
            i-=1
        if i<end: runs.append((end,-1,end-i))
    assert all(keys[i]<keys[i+1] for i in range(n-1))
    largest=keys[-1]
    travel=largest-(n-1)
    assert 0<=travel<n
    count=sum(v+n<largest for v in keys)
    assert count==travel,(initial_keys,keys,travel,count)
    for step in range(travel): swap((n-1+step)%n)
    if travel: runs.append((n-1,1,travel))
    pivot=(n-1+travel)%n
    assert keys[pivot]==pivot

    # Cut after the now completely settled pivot.
    def lifted_at(j):
        i=pivot+1+j
        return keys[i%n]+(i//n)*n

    assert sorted(lifted_at(j) for j in range(n-1))==list(range(pivot+1,pivot+n))
    # The remainder is the concatenation of two increasing lists, of sizes
    # n-1-travel and travel. Move the entries of the smaller list.
    right_size=travel
    left_size=n-1-travel
    if right_size<=left_size:
        for end in range(left_size,n-1):
            i=end
            while i>0 and lifted_at(i)<lifted_at(i-1):
                swap((pivot+i)%n)
                i-=1
            if i<end: runs.append(((pivot+1+end)%n,-1,end-i))
    else:
        for start in range(left_size-1,-1,-1):
            i=start
            while i<n-2 and lifted_at(i)>lifted_at(i+1):
                swap((pivot+1+i)%n)
                i+=1
            if i>start: runs.append(((pivot+1+start)%n,1,i-start))
    assert keys==list(range(n))
    assert sum(length for _,_,length in runs)==potential<=n*n//4
    assert len(runs)<=n+(n-1)//2
    return runs,initial_keys,potential


def compile_run(start,direction,length,n):
    word=[]
    q,r=divmod(length,3)
    if r==1 and q:
        q-=1
        r=4
    for j in range(q):
        if direction==1:
            word.append(((start+3*j)%n,1))
        else:
            word.append(((start-3*j-3)%n,-1))
    pos=start+direction*(3*q)
    if r:
        # On [x,a,b], movement right is the left rotation [a,b,x].
        local=SWAP01 if r==1 else SWAP_TWO if r==2 else MOVE_FOUR
        if direction==1:
            word.extend(((pos+i)%n,d) for i,d in local)
        else:
            # Invert the right-moving word on the same small interval.
            word.extend(((pos-r+i)%n,-d) for i,d in reversed(local))
    return word


def construct(p):
    n=len(p)
    runs,keys,potential=reduced_runs(p)
    word=[move for start,direction,length in runs for move in compile_run(start,direction,length,n)]
    assert apply_to(p,word)==list(range(n)),(p,runs,word)
    assert 3*len(word)<=potential+14*(n+(n-1)//2),(len(word),potential,n,SWAP_TWO)
    return word,{'n':n,'adjacent_length':potential,'run_count':len(runs),
                 'four_cycle_length':len(word),'residual_one':sum(m%3==1 for _,_,m in runs),
                 'residual_two':sum(m%3==2 for _,_,m in runs)}


def main():
    print('adjacent three-cycle word',SWAP_TWO,'length',len(SWAP_TWO),flush=True)
    assert len(SWAP_TWO)<=4 and len(MOVE_FOUR)<=4
    rng=random.Random(20260920)
    count=0
    rows=[]
    for n in list(range(5,151))+[250,500,1000]:
        candidates=[list(range(n))[::-1],[(i+n//2)%n for i in range(n)]]
        for _ in range(10):
            p=list(range(n));rng.shuffle(p);candidates.append(p)
        for p in candidates:
            word,row=construct(p)
            count+=1
            if n<=30:
                old,spins=circular_adjacent_sort(p)
                assert row['adjacent_length']==len(old)
        rows.append(row)
    result={'verified_permutations':count,'sizes':'all 5..150;250;500;1000',
            'last_samples':rows[-3:],'residual_three_cycle_word':SWAP_TWO,
            'five_cycle_word':MOVE_FOUR,
            'assertions':'independent replay; potential equals run length; <=floor(n^2/4) adjacent steps; <=n+floor((n-1)/2) runs'}
    Path(__file__).with_name('affine_insertion_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
