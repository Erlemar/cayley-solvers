"""Explicit words of exactly H(n) moves for every n>=5.

For n=0,1,5 mod 6 use a half rotation with one adjacent transposition.
For n=2,3,4 mod 6 use the half rotation itself.
"""
import json
from pathlib import Path

from rotation_bounds import exchange,diameter_lower
from verify_theorem import replay


BASE={
    2:[(0,-1),(1,1),(1,1)],
    3:[(1,-1),(2,1),(2,1),(0,1)],
    4:[(2,1),(0,-1),(3,1),(0,-1),(2,-1)],
}


def _b_multiple_three(a,b,start=0):
    assert a>=2 and b>=3 and b%3==0
    amin=3 if a%3==0 else 4 if a%3==1 else 2
    if a>amin:
        return _b_multiple_three(a-3,b,start+3)+[(start+j,-1) for j in range(b)]
    if b>3:
        return _b_multiple_three(a,b-3,start)+[(start+b-3+j,1) for j in range(a-1,-1,-1)]
    return [(start+i,d) for i,d in BASE[a]]


def perturbed_exchange(a,b):
    assert a>=2 and b>=2 and a*b%3==0
    if b%3==0: return _b_multiple_three(a,b)
    n=a+b
    inverse=[(i,-d) for i,d in reversed(_b_multiple_three(b,a))]
    return [((i-b)%n,d) for i,d in inverse]


def witness(n):
    assert n>=5
    a=n//2
    b=n-a
    if n%6 in (0,1,5):
        word=perturbed_exchange(a,b)
        p=list(range(a,n))+list(range(a))
        p[0],p[1]=p[1],p[0]
    else:
        word=exchange(a,b)
        p=list(range(a,n))+list(range(a))
    assert len(word)==diameter_lower(n)
    return p,word


def main():
    blocks=0
    for a in range(2,61):
        for b in range(2,61):
            if a*b%3: continue
            word=perturbed_exchange(a,b)
            p=list(range(a,a+b))+list(range(a))
            p[0],p[1]=p[1],p[0]
            assert len(word)==a*b//3+1
            assert replay(a+b,word,wrapped=True)==p
            blocks+=1
    sizes=list(range(5,251))+[501,1000]
    for n in sizes:
        p,word=witness(n)
        assert replay(n,word,wrapped=True)==p
    examples=[]
    for n in (13,14,15,16,17,18,19,20):
        p,word=witness(n)
        examples.append({'n':n,'permutation':p,'distance':len(word),'word':word})
    result={'perturbed_block_words_checked':blocks,'block_sizes':'2..60 with 3|ab',
            'diameter_lower_witnesses_checked':len(sizes),'sizes':'5..250;501;1000',
            'examples':examples,
            'scope':'These words attain a proved lower bound for the diameter; they do not prove a universal matching upper bound.'}
    Path(__file__).with_name('diameter_witnesses_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'}),flush=True)


if __name__=='__main__': main()
