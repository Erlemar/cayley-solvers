"""Inspect exact peripheral states without new full-graph enumeration."""
from collections import Counter
import json
from pathlib import Path

from affine_insertion import balanced_keys


def cyclic_adjacent_length(p):
    n=len(p)
    w=balanced_keys(p)
    return sum(abs((w[j]-w[i])//n) for i in range(n) for j in range(i+1,n))


def main():
    folder=Path(__file__).parent
    rows=[]
    for n in range(5,13):
        data=json.loads((folder/'diameter_exact'/f'n{n}.json').read_text(encoding='utf-8'))
        hist=Counter()
        rot=Counter()
        low=None
        for p in data['farthest']:
            a=cyclic_adjacent_length(p)
            hist[a]+=1
            near=min(cyclic_adjacent_length([(v-k)%n for v in p]) for k in range(n))
            rot[near]+=1
            if low is None or a<low[0]: low=(a,p)
        row={'n':n,'diameter':data['diameter'],'peripheral_count':len(data['farthest']),
             'adjacent_length_histogram':dict(sorted(hist.items())),
             'adjacent_distance_to_nearest_rotation_histogram':dict(sorted(rot.items())),
             'lowest_adjacent_length_witness':low}
        rows.append(row)
        print(json.dumps(row),flush=True)
    (folder/'peripheral_analysis.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__': main()
