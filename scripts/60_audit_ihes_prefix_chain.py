"""Prove full prefix coverage across an original journal and one seeded retry."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,verify_submission
ap=argparse.ArgumentParser()
ap.add_argument('--pid',type=int,required=True);ap.add_argument('--seed',type=Path,required=True)
ap.add_argument('--resumed',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefixes=m.canonical_prefixes(2,{n:np.asarray(v,dtype=np.int64) for n,v in puzzle.generators.items()},list(puzzle.move_names))
seed=[json.loads(s) for s in args.seed.read_text().splitlines()]
rows=[json.loads(s) for s in args.resumed.read_text().splitlines()]
assert rows[:len(seed)]==seed
new=rows[len(seed):];settled={};hashes={}
for path,records in [(args.seed,seed),(args.resumed,new)]:
    native=path.with_suffix('.solver.log').read_text();blocks=native.split('\nSolving\n')[1:]
    assert len(blocks)==len(records) and 'Twsearch finished.' in blocks[-1]
    for r,b in zip(records,blocks):
        if r['pid']!=args.pid:continue
        assert r['prefix']==prefixes[r['prefix_id']] and r['max_depth']==18 and r['incumbent_len']==22 and not r['self_test']
        assert r['verdict'] in ('none','timeout')
        if r['verdict']=='none':
            assert 'No solution found in 18' in b.splitlines() and 'Search timed out' not in b
            settled[r['prefix_id']]=r
    hashes[str(path)]={'journal_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_sha256':hashlib.sha256(native.encode()).hexdigest()}
assert set(settled)==set(range(262))
baseline=ROOT/'submission_ihes.csv';digest=hashlib.sha256(baseline.read_bytes()).hexdigest()
assert digest=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
assert len(load_submission(baseline)[args.pid])==22
report=verify_submission(puzzle,ROOT/'data/test.csv',args.candidate)
assert report.n_valid==report.n_total==1003
proof=dict(pid=args.pid,incumbent_moves=22,distinct_two_move_prefixes=262,residual_depth=18,
    verdict='optimal_by_full_prefix_coverage_and_parity',baseline_sha256=digest,sources=hashes,
    verified_moves=report.total_moves)
args.output.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
