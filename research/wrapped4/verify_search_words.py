"""Independently replay original and retried sample-search certificates."""
from collections import Counter
import json
from pathlib import Path


def main():
    folder=Path(__file__).parent
    summaries=[]
    for n in (15,16):
        rows=json.loads((folder/f'search_mixed_n{n}_offset0.json').read_text(encoding='utf-8'))
        retries=json.loads((folder/f'search_mixed_n{n}_retries.json').read_text(encoding='utf-8'))
        for retry in retries:
            i=retry['original_case']
            final=retry['attempts'][-1]
            assert rows[i]['permutation']==final['permutation']
            assert rows[i]['budget']==final['budget']
            rows[i]=final
        lengths=Counter()
        unique=set()
        h=(n*n+11)//12+int(n%3==0)
        for row in rows:
            assert row['status']=='found'
            p=row['permutation'][:]
            unique.add(tuple(p))
            assert sorted(p)==list(range(n))
            for i,d in row['word']:
                assert 0<=i<n and d in (-1,1)
                indices=[(i+j)%n for j in range(4)]
                old=[p[j] for j in indices]
                for j,index in enumerate(indices): p[index]=old[(j+d)%4]
            assert p==list(range(n))
            assert len(row['word'])==row['length']<=row['budget']<=h
            lengths[row['length']]+=1
        summary={'n':n,'instances':len(rows),'distinct_permutations':len(unique),
                 'candidate_bound':h,'solution_length_histogram':dict(sorted(lengths.items())),
                 'unresolved':0,'counterexamples':0}
        summaries.append(summary)
    result={'summaries':summaries,'scope':'Verified sample upper bounds only; no exhaustive diameter assertion.'}
    (folder/'candidate_search_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
