"""Finite decision tree for improved amortized stable extraction.

At each stage solve a residual pattern with cost defect <= c*k, or
mark the next-smallest token and enumerate every possible refinement.
Only stored, replayable successful words enter a certificate.
"""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path
import time

from near_rotation_search import build_endgame,solve
from triple_extraction import inv
from general_sort import apply_to


def canonical(markers):
    k=len(markers)
    end=max(markers.values())+1
    p=[None]*end
    for label,pos in markers.items(): p[pos]=label
    filler=k
    for i in range(end):
        if p[i] is None: p[i]=filler;filler+=1
    return tuple(p)


def initial_patterns():
    out=set()
    for gaps in itertools.product(range(3),repeat=3):
        for order in itertools.permutations(range(3)):
            markers={}
            pos=0
            for gap,label in zip(gaps,order):
                pos+=gap
                markers[label]=pos
                pos+=1
            out.add(canonical(markers))
    return out


def refinements(p,k):
    marked={label:p.index(label) for label in range(k)}
    end=len(p)
    possible=[i for i in range(end) if p[i]>=k]+[end,end+1,end+2]
    return {canonical({**marked,k:i}) for i in possible}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--c',type=Fraction,default=Fraction(3))
    parser.add_argument('--max-k',type=int,default=7)
    parser.add_argument('--cap',type=int,default=1000000)
    parser.add_argument('--wide',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--inverse-first',action='store_true')
    parser.add_argument('--output-file')
    parser.add_argument('--reuse-known',nargs='*',default=[])
    parser.add_argument('--radius',type=int,default=4)
    parser.add_argument('--beam-on-limit',type=int,default=0)
    args=parser.parse_args()
    if args.wide:
        from wide_search import build_endgame as wide_endgame, solve as wide_solve
    if args.beam_on_limit:
        assert args.wide
        from matching_beam import run as beam_run
    folder=Path(__file__).parent
    pending=initial_patterns()
    stages=[]
    all_records=[]
    endgames={}
    known={}
    def core(p):
        end=max((i+1 for i,v in enumerate(p) if i!=v),default=0)
        return tuple(p[:end])
    for filename in args.reuse_known:
        source=json.loads((folder/filename).read_text())
        for row in source['records']:
            if row['status']!='found': continue
            p=row['pattern']+list(range(len(row['pattern']),row['width']))
            word=row['word']
            key=core(p)
            support=max((i+4 for i,d in word),default=0)
            if key not in known or (len(word),support)<(len(known[key][0]),known[key][1]):
                known[key]=(word,support)
    tic=time.monotonic()
    slug=str(args.c).replace('/','_')+('_wide' if args.wide else '')
    output=folder/f'adaptive_extraction_c{slug}.json'
    if args.output_file:
        output=folder/args.output_file
    start_k=3
    if args.resume:
        previous=json.loads(output.read_text())
        assert previous['defect_per_extracted_token']==str(args.c)
        if previous['complete']:
            print('Certificate is already complete.',flush=True)
            return
        stages=previous['stages']
        all_records=previous['records']
        last=stages[-1]['k']
        pending=set()
        for row in all_records:
            if row['k']==last and row['status']!='found':
                pending.update(refinements(row['pattern'],last))
        start_k=last+1
    for k in range(start_k,args.max_k+1):
        failures=set()
        counts={'found':0,'exhausted':0,'node_limit':0,'too_wide':0}
        for p in sorted(pending):
            m=max(9,len(p))
            if m>(40 if args.wide else 16):
                record={'k':k,'pattern':p,'status':'too_wide','width':m}
            else:
                full=list(p)+list(range(len(p),m))
                inversions=inv(full)
                budget=(inversions*args.c.denominator+args.c.numerator*k)//(3*args.c.denominator)
                if (budget-inversions)%2: budget-=1
                cached=known.get(core(full))
                if cached and len(cached[0])<=budget and cached[1]<=m:
                    assert apply_to(full,cached[0])==list(range(m))
                    record={'k':k,'pattern':p,'width':m,'permutation':full,
                            'status':'found','word':cached[0],'budget':budget,
                            'length':len(cached[0]),'discovery_method':'reused_verified_identity'}
                    counts['found']+=1
                    all_records.append(record)
                    continue
                use_wide=args.wide and (m>16 or args.beam_on_limit)
                radius=args.radius if m<=20 else 4
                if m not in endgames:
                    endgames[m]=(wide_endgame(m,radius,5000000,False) if use_wide
                                  else build_endgame(m,radius,5000000,False))
                solver=wide_solve if use_wide else solve
                if args.inverse_first:
                    inverse=[full.index(i) for i in range(m)]
                    record=solver(inverse,budget,endgames[m],radius,args.cap,seed=7,wrapped=False)
                    if record['status']=='found':
                        record['word']=[(i,-d) for i,d in reversed(record['word'])]
                        record['permutation']=full
                        record['search_orientation']='inverse'
                        assert apply_to(full,record['word'])==list(range(m))
                    elif record['status']=='node_limit':
                        record=solver(full,budget,endgames[m],radius,args.cap,seed=0,wrapped=False)
                else:
                    record=solver(full,budget,endgames[m],radius,args.cap,seed=0,wrapped=False)
                record['k']=k
                record['pattern']=p
                record['width']=m
                if record['status']=='node_limit' and args.beam_on_limit:
                    retry=beam_run(full,budget,args.beam_on_limit,endgames[m],seed=k,
                                   wrapped=False,report=False,endgame_radius=radius)
                    if retry['status']=='found':
                        record={**retry,'k':k,'pattern':p,'width':m,'permutation':full,
                                'budget':budget,'discovery_method':'beam_after_dfs'}
                elif record['status']=='node_limit':
                    for seed in (7,13):
                        retry=solver(full,budget,endgames[m],radius,args.cap,seed=seed,wrapped=False)
                        if retry['status']!='node_limit':
                            record={**retry,'k':k,'pattern':p,'width':m};break
            counts[record['status']]+=1
            all_records.append(record)
            if record['status']=='found':
                word=record['word']
                support=max((i+4 for i,d in word),default=0)
                key=core(list(p)+list(range(len(p),m)))
                if key not in known or (len(word),support)<(len(known[key][0]),known[key][1]):
                    known[key]=(word,support)
            if record['status']!='found': failures.add(p)
        stage={'k':k,'patterns':len(pending),**counts,'seconds':time.monotonic()-tic}
        stages.append(stage)
        print(json.dumps(stage),flush=True)
        result={'defect_per_extracted_token':str(args.c),'stages':stages,'complete':not failures,
                'maximum_width':max(r['width'] for r in all_records),'records':all_records}
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        if not failures: break
        pending=set()
        for p in failures: pending.update(refinements(p,k))


if __name__=='__main__': main()
