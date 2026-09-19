"""Verify the synthesized potential with exact integers, without an LP solver."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_test_states
spec=importlib.util.spec_from_file_location('decomp',ROOT/'scripts/build_picture_cube_kpuzzle_v2.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
report=json.loads((ROOT/'data/ihes_pdb/pid296_piece_potential.json').read_text())
assert hashlib.sha256((ROOT/'data/puzzle_info.json').read_bytes()).hexdigest()==report['puzzle_info_sha256']
denominator=report['denominator'];C=report['coefficients'];checked=0
orbits=[('CORNERS',8,3),('EDGES',12,2),('CENTERS',6,4)]
for move in puzzle.move_names:
    m=d.derive_move(move,puzzle);budget=0
    for orb,n,k in orbits:
        dual=report['assignment_duals'][move][orb];u=dual['u'];v=dual['v']
        tp,to=m[orb]
        for piece in range(n):
            for slot in range(n):
                target=tp.index(slot)
                for o in range(k):
                    change=C[orb][piece][slot][o]-C[orb][piece][target][(o+to[target])%k]
                    assert change<=u[piece]+v[slot]
                    checked+=1
        budget+=sum(u)+sum(v)
    assert budget==report['move_budgets'][move] and budget<=denominator
def potential_numerator(s):
    coordinates=d.derive_state(s);value=0
    for orb,n,k in orbits:
        pieces,orientations=coordinates[orb]
        for slot,piece in enumerate(pieces):value+=C[orb][piece][slot][orientations[slot]]
        for piece in range(n):value-=C[orb][piece][piece][0]
    return value
def F(s):
    """Exact rational, globally valid lower bound; solved state has value zero."""
    return max(Fraction(0),Fraction(potential_numerator(s),denominator))
assert potential_numerator(tuple(range(72)))==0
assert potential_numerator(load_test_states(ROOT/'data/test.csv')[296])==report['numerator_at_start']
print('PASS:',checked,'exact inequalities establish the move bound for every legal state.')
print('F(296) =',report['numerator_at_start'],'/',denominator)
