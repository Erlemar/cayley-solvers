"""Independent finite-base checking and exhaustive construction checks."""
import itertools
import json
from pathlib import Path
from time import monotonic

from linear_error_sort import construct


def replay(p,word):
    p=list(p)
    n=len(p)
    for i,d in word:
        idx=[(i+j)%n for j in range(4)]
        old=[p[j] for j in idx]
        for k,j in enumerate(idx): p[j]=old[(k+d)%4]
    return p


def main():
    output=Path(__file__).parent
    certificates=json.loads((output/'five_position_certificates.json').read_text(encoding='utf-8'))
    observed=set()
    max_len_by_inv={}
    for row in certificates:
        p=tuple(row['permutation'])
        assert p not in observed
        observed.add(p)
        assert replay(range(5),row['word'])==list(p)
        inv=sum(p[i]>p[j] for i in range(5) for j in range(i+1,5))
        assert 3*len(row['word'])-inv<=18
        max_len_by_inv[inv]=max(max_len_by_inv.get(inv,0),len(row['word']))
    assert observed==set(itertools.permutations(range(5)))
    rows=[]
    for n in range(5,9):
        tic=monotonic()
        count=0
        max_length=0
        for p in itertools.permutations(range(n)):
            word,record=construct(p)
            assert replay(p,word)==list(range(n))
            assert 3*len(word)<=n*n//4+12*n-16
            max_length=max(max_length,len(word))
            count+=1
        row={'n':n,'permutations_checked':count,'max_constructed_length':max_length,'seconds':monotonic()-tic}
        rows.append(row)
        print(json.dumps(row),flush=True)
    result={'finite_base_count':len(observed),'max_word_length_by_inversion':max_len_by_inv,
            'exhaustive_construction_checks':rows,
            'scope':'Constructed lengths are upper bounds, not distances. Finite checks supplement the all-n proof.'}
    (output/'linear_bound_exhaustive_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
