"""Executable certificates for the wrapped consecutive 4-cycle reflection theorem.

No third-party dependencies. Move (i,+1) left-rotates positions i..i+3;
move (i,-1) right-rotates them. All indices are zero based.
"""
from collections import deque
import json
from pathlib import Path
import random

BASE = {
    6: [(0,1),(2,-1),(0,1),(2,-1),(0,1)],
    7: [(0,1),(2,-1),(0,1),(2,-1),(0,1),(3,-1),(0,-1)],
    8: [(0,1),(0,1),(1,-1),(3,1),(2,-1),(0,1),(2,-1),(4,-1),(1,-1),(0,1)],
    9: [(0,1),(1,-1),(0,1),(2,-1),(3,-1),(1,1),(4,-1),(5,-1),(2,-1),(0,1),(2,-1),(0,1)],
    10: [(0,-1),(1,-1),(2,-1),(3,-1),(4,-1),(6,1),(4,-1),(6,1),(4,-1),(0,-1),(2,1),(0,-1),(1,-1),(3,1),(2,-1)],
    11: [(0,-1),(1,-1),(2,-1),(3,-1),(4,-1),(5,-1),(6,-1),(7,1),(0,-1),(3,1),(0,-1),(1,-1),(4,1),(6,1),(5,-1),(1,-1),(4,1),(1,-1),(0,1)],
}
S = (12,1,0,9,4,9,0,1,12,9,16,9)  # shifted flip
R = (0,1,12,9,16,9,12,1,0,9,4,9)  # full flip


def replay(n, word, wrapped=False):
    state = list(range(n))
    for i,d in word:
        assert d in (-1,1)
        assert 0 <= i < (n if wrapped else n-3)
        idx = [(i+j)%n for j in range(4)]
        old = [state[k] for k in idx]
        new = old[1:]+old[:1] if d==1 else old[-1:]+old[:-1]
        for k,v in zip(idx,new): state[k]=v
    return state


def exchange_six_with_m(start, m):
    # [a b c d e f | B] -> [a b c | B | d e f] -> [B | a b c d e f].
    return [(start+3+j,-1) for j in range(m)] + [(start+j,-1) for j in range(m)]


def linear_reversal(m, start=0):
    assert m >= 6
    if m <= 11:
        return [(start+i,d) for i,d in BASE[m]]
    return ([(start+i,d) for i,d in BASE[6]]
            + linear_reversal(m-6,start+6)
            + exchange_six_with_m(start,m-6))


def reflection(n, shifted=False):
    assert n >= (13 if shifted else 12)
    c = 0 if shifted else n-1
    choices = [a for a in range(6,n-5) if n%2 or a%2 == (1 if shifted else 0)]
    a = min(choices,key=lambda a:abs(2*a-n))
    word = linear_reversal(a)+linear_reversal(n-a,a)
    difference = (c-(a-1))%n
    t = (difference*pow(2,-1,n))%n if n%2 else difference//2
    assert (a-1+2*t)%n == c
    return [((i+t)%n,d) for i,d in word], a


def formula(n,shifted=False):
    numerator = n*n-2*n+(S if shifted else R)[n%12]
    assert numerator%12 == 0
    return numerator//12


def lower_bound(n,shifted=False):
    # Universal adjacent-transposition lower bound and sign of reflection.
    crossings=(n-1)**2//4
    bound=(crossings+2)//3
    parity=((n-1)//2 if shifted else n//2)%2
    return bound + ((parity-bound)%2)


def check_lift_potential():
    rng=random.Random(20260919)
    checks=0
    for n in range(5,31):
        # x[label] is its integer lift; residues give distinct occupied positions.
        x=list(range(n))
        def potential():
            return sum(abs((x[j]-x[i])//n) for i in range(n) for j in range(i+1,n))
        for _ in range(500):
            edge=rng.randrange(n)
            first=next(i for i in range(n) if x[i]%n==edge)
            second=next(i for i in range(n) if x[i]%n==(edge+1)%n)
            before=potential()
            x[first]+=1
            x[second]-=1
            assert abs(potential()-before)==1
            assert len({v%n for v in x})==n
            checks+=1
    return checks


def small_bfs(n):
    e=bytes(range(n))
    targets={e[::-1]:'full', bytes([0]+list(range(n-1,0,-1))):'shifted'}
    queue=deque([e])
    dist={e:0}
    answers={}
    while queue:
        s=queue.popleft()
        if s in targets:
            answers[targets[s]]=dist[s]
            if len(answers)==2: return answers,len(dist)
        for i in range(n):
            idx=[(i+j)%n for j in range(4)]
            for d in (1,-1):
                t=bytearray(s)
                for j in range(4): t[idx[j]]=s[idx[(j+d)%4]]
                t=bytes(t)
                if t not in dist:
                    dist[t]=dist[s]+1
                    queue.append(t)
    raise AssertionError('Disconnected graph')


def main():
    for m in range(6,251):
        word=linear_reversal(m)
        assert len(word)==(m*(m-1)+5)//6
        assert replay(m,word)==list(range(m))[::-1]
    count=0
    for n in list(range(12,251))+[503,1000,1001,1012]:
        for shifted in (False,True):
            if shifted and n==12: continue
            word,a=reflection(n,shifted)
            expected=[(-i if shifted else n-1-i)%n for i in range(n)]
            assert replay(n,word,True)==expected,(n,shifted,a)
            assert len(word)==formula(n,shifted)==lower_bound(n,shifted),(n,shifted)
            count+=1
    for n in range(13,10001):
        for shifted in (False,True):
            assert formula(n,shifted)==lower_bound(n,shifted)
    lift_checks=check_lift_potential()
    small={}
    for n in range(5,10):
        answers,states=small_bfs(n)
        small[n]=answers
        print('exact BFS',n,answers,'states',states,flush=True)
    report={
        'linear_reversals_verified':'every m=6..250',
        'reflection_words_verified':count,
        'reflection_sizes':'12..250 and 503,1000,1001,1012 (shifted starts at 13)',
        'formula_equals_parity_rounded_lower_bound':'all n=13..10000, both reflections',
        'random_lift_potential_checks':lift_checks,
        'small_exact_bfs':small,
        's':S,'r':R,
    }
    Path(__file__).with_name('verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
