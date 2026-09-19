"""Construct full reversal words from six-block reversals and block exchanges."""
from explore import move

R6 = [(0, 1), (2, -1), (0, 1), (2, -1), (0, 1)]


def swap_blocks(start, a, b):
    """Exchange consecutive A,B, with a divisible by three; preserve order."""
    assert a % 3 == 0
    word = []
    for triple in range(a//3 - 1, -1, -1):
        p = start + 3*triple
        for j in range(b):
            word.append((p+j, -1))
    return word


def reverse6q(start, size):
    assert size % 6 == 0
    q = size//6
    word = []
    for j in range(q):
        word.extend((start+6*j+i, d) for i,d in R6)
    for end in range(q-1, 0, -1):
        for j in range(end):
            word.extend(swap_blocks(start+6*j, 6, 6))
    return word


def apply(s, word):
    for i,d in word:
        s = move(s, i, d)
    return s


if __name__ == '__main__':
    for a in (3,6,9,12):
        for b in range(1,13):
            s = list(range(a+b))
            assert apply(s, swap_blocks(0,a,b)) == s[a:]+s[:a], (a,b)
    for m in range(6,121,6):
        s=list(range(m))
        word=reverse6q(0,m)
        assert len(word)==m*(m-1)//6
        assert apply(s,word)==s[::-1], m
    print('Verified block swaps and linear reversals through length 120.')
    for n in range(12,241,12):
        m=n//2
        word=reverse6q(0,m)+reverse6q(m,m)
        # Conjugating by rotation t sends i -> m-1-i to i -> n-1-i.
        t=n//4
        word=[((i+t)%n,d) for i,d in word]
        s=list(range(n))
        for i,d in word:
            idx=[(i+j)%n for j in range(4)]
            x=[s[k] for k in idx]
            x=x[1:]+x[:1] if d==1 else x[-1:]+x[:-1]
            for k,v in zip(idx,x): s[k]=v
        assert s==list(range(n))[::-1], n
        assert len(word)==n*(n-2)//12
        print(n,len(word),flush=True)
