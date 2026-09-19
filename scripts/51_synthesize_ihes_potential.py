"""Synthesize and exactly certify a separable piece-position potential for PID 296.

The certificate covers every configuration, not only sampled states: assignment
dual inequalities bound each generator's maximum decrease over a superset of all
legal piece arrangements and orientations. All final checks use integers.
"""
from pathlib import Path
import importlib.util
import json
import sys
import hashlib
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_test_states,load_submission
spec=importlib.util.spec_from_file_location('decomp',ROOT/'scripts/build_picture_cube_kpuzzle_v2.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
start=load_test_states(ROOT/'data/test.csv')[296]
state=d.derive_state(start)
orbits=[('CORNERS',8,3),('EDGES',12,2),('CENTERS',6,4)]
names=list(puzzle.move_names)
moves={name:d.derive_move(name,puzzle) for name in names}
coeff={};nv=0
for orb,n,k in orbits:
    coeff[orb]=np.arange(nv,nv+n*n*k).reshape(n,n,k);nv+=n*n*k
duals={}
for name in names:
    for orb,n,k in orbits:
        duals[name,orb]=(np.arange(nv,nv+n),np.arange(nv+n,nv+2*n));nv+=2*n
rr=[];cc=[];vv=[];rhs=[]
def constraint(columns,values,b):
    r=len(rhs);rr.extend([r]*len(columns));cc.extend(columns);vv.extend(values);rhs.append(b)
for name in names:
    budget=[]
    for orb,n,k in orbits:
        a=coeff[orb];u,v=duals[name,orb];tp,to=moves[name][orb];inv=np.argsort(tp)
        budget.extend(u);budget.extend(v)
        for piece in range(n):
            for slot in range(n):
                target=int(inv[slot])
                for orientation in range(k):
                    next_o=(orientation+to[target])%k
                    constraint([a[piece,slot,orientation],a[piece,target,next_o],u[piece],v[slot]],[1,-1,-1,-1],0)
    constraint(budget,[1]*len(budget),1)
objective=np.zeros(nv)
bounds=[(None,None)]*nv
for orb,n,k in orbits:
    a=coeff[orb];pieces,orientations=state[orb]
    for slot,piece in enumerate(pieces):objective[a[piece,slot,orientations[slot]]]-=1
    for piece in range(n):
        objective[a[piece,piece,0]]+=1
        bounds[a[piece,piece,0]]=(0,0)
A=coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr()
print('LP variables',nv,'inequalities',len(rhs),flush=True)
solution=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'time_limit':120})
assert solution.success,solution.message
print('Floating LP bound',-solution.fun,flush=True)
# Round coefficients to integers, then compute and certify the true move budget.
SCALE=1000000
C={orb:np.rint(solution.x[coeff[orb]]*SCALE).astype(np.int64) for orb,_,_ in orbits}
certificate={};budgets={}
for name in names:
    certificate[name]={};total=0
    for orb,n,k in orbits:
        tp,to=moves[name][orb];inv=np.argsort(tp)
        drops=np.empty((n,n),dtype=np.int64)
        for slot in range(n):
            target=int(inv[slot])
            for piece in range(n):
                drops[piece,slot]=max(int(C[orb][piece,slot,o]-C[orb][piece,target,(o+to[target])%k]) for o in range(k))
        # Assignment dual: u_piece+v_slot >= drops[piece,slot].
        ar=[];ac=[];av=[]
        for piece in range(n):
            for slot in range(n):
                row=piece*n+slot;ar.extend([row,row]);ac.extend([piece,n+slot]);av.extend([-1,-1])
        result=linprog(np.ones(2*n),A_ub=coo_matrix((av,(ar,ac)),shape=(n*n,2*n)).tocsr(),
                       b_ub=-drops.ravel(),bounds=[(None,None)]*(2*n),method='highs')
        assert result.success
        u=np.rint(result.x[:n]).astype(np.int64);v=np.rint(result.x[n:]).astype(np.int64)
        # Repair any rounding deficit exactly; this can only weaken the bound.
        for piece in range(n):u[piece]+=max(0,int(np.max(drops[piece]-u[piece]-v)))
        assert np.all(u[:,None]+v[None,:]>=drops)
        r,c=linear_sum_assignment(drops,maximize=True)
        primal=int(drops[r,c].sum());dual=int(u.sum()+v.sum())
        assert primal<=dual
        total+=dual
        certificate[name][orb]={'u':u.tolist(),'v':v.tolist(),'budget':dual}
    budgets[name]=total
D=max(budgets.values());assert D>0
def score(coordinates):
    value=0
    for orb,n,k in orbits:
        pieces,orientations=coordinates[orb]
        value+=sum(int(C[orb][piece,slot,orientations[slot]]) for slot,piece in enumerate(pieces))
        value-=sum(int(C[orb][piece,piece,0]) for piece in range(n))
    return value
assert score(d.derive_state(tuple(range(72))))==0
start_numerator=score(state)
# Replay the actual solution and check the integer move bound along it as well.
s=start;previous=score(state)
for move in load_submission(ROOT/'submission_ihes.csv')[296]:
    s=puzzle.apply_move(s,move);value=score(d.derive_state(s))
    assert abs(value-previous)<=D
    previous=value
assert previous==0
report={'pid':296,'formula':'F(s)=max(0,(sum piece-position-orientation coefficients minus solved sum)/denominator)',
        'numerator_at_start':start_numerator,'denominator':D,'value_at_start':start_numerator/D,
        'lp_optimum_float':-solution.fun,'reaches_22':start_numerator>=22*D,
        'coefficients':{k:v.tolist() for k,v in C.items()},'move_budgets':budgets,'assignment_duals':certificate,
        'puzzle_info_sha256':hashlib.sha256((ROOT/'data/puzzle_info.json').read_bytes()).hexdigest(),
        'baseline_sha256':hashlib.sha256((ROOT/'submission_ihes.csv').read_bytes()).hexdigest()}
out=ROOT/'data/ihes_pdb/pid296_piece_potential.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['numerator_at_start','denominator','value_at_start','lp_optimum_float','reaches_22']}),flush=True)
print('Exact integer certificate:',out,flush=True)
