"""Exact distances of rotations with both blocks of size >=2, and diameter lower bound."""
from math import ceil
import json
from pathlib import Path
from verify_theorem import replay,formula

BASE={
 (2,2):[(0,1),(0,1)],
 (4,2):[(2,1),(2,1),(0,1),(0,1)],
 (2,4):[(0,1),(0,1),(2,1),(2,1)],
 (4,4):[(1,-1),(2,-1),(0,1),(3,-1),(4,-1),(2,1)],
}

def parity_rounded_third(k):
    return k//3+k%3

def exchange(a,b,start=0):
    assert a>=2 and b>=2
    if a%3==0:
        return [(start+3*t+j,-1) for t in range(a//3-1,-1,-1) for j in range(b)]
    if b%3==0:
        w=exchange(b,a,start)
        return [(i,-d) for i,d in reversed(w)]
    amin=4 if a%3==1 else 2
    bmin=4 if b%3==1 else 2
    if a>amin:
        return exchange(a-3,b,start+3)+[(start+j,-1) for j in range(b)]
    if b>bmin:
        return exchange(a,b-3,start)+[(start+b-3+j,1) for j in range(a-1,-1,-1)]
    return [(start+i,d) for i,d in BASE[(a,b)]]

def diameter_lower(n):
    return (n*n+11)//12+int(n%3==0)

def main():
    count=0
    for a in range(2,41):
        for b in range(2,41):
            w=exchange(a,b)
            assert len(w)==parity_rounded_third(a*b),(a,b,len(w))
            assert replay(a+b,w)==list(range(a,a+b))+list(range(a)),(a,b)
            count+=1
    for n in range(13,10001):
        k=n*n//4
        assert diameter_lower(n)==max(parity_rounded_third(k),parity_rounded_third(k-1))
        assert diameter_lower(n)-max(formula(n),formula(n,True))==n//6+n%2
    known={5:4,6:5,7:6,8:7,9:8,10:10,11:11,12:13,13:15,14:17}
    rows=[{'n':n,'diameter':known.get(n),'proved_lower_bound':diameter_lower(n),
           'candidate_exact_value':diameter_lower(n)} for n in range(5,34)]
    result={'rotation_words_verified':count,'range':'all 2<=a,b<=40',
            'arithmetic_checked':'13<=n<=10000',
            'known_diameter_provenance':{'5..12':'independent full BFS in this task','13..14':'user supplied'},
            'rows':rows}
    Path(__file__).with_name('diameter_bounds.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
