"""Verify complete prefix coverage and bind it to the unchanged incumbent."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import verify_submission,load_submission
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
generators={n:np.asarray(v,dtype=np.int64) for n,v in puzzle.generators.items()}
prefixes=m.canonical_prefixes(2,generators,list(puzzle.move_names))
assert len(prefixes)==262
journal=ROOT/'data/ihes_pdb/local_prefix296_20260908.jsonl'
rows=[json.loads(s) for s in journal.read_text().splitlines()]
latest={r['prefix_id']:r for r in rows}
assert set(latest)==set(range(262))
for i,r in latest.items():
    assert r['pid']==296 and r['prefix']==prefixes[i] and r['verdict']=='none'
    assert r['incumbent_len']==22 and r['max_depth']==18 and not r['self_test']
# Bind each settled record to the corresponding explicit native output block.
seed=ROOT/'data/ihes_pdb/prefix_probe_20260907.solver.log'
native=journal.with_suffix('.solver.log')
for path,expected in [(seed,3),(native,259)]:
    blocks=path.read_text().split('\nSolving\n')[1:]
    assert len(blocks)==expected
    assert all('No solution found in 18' in b.splitlines() and 'Search timed out' not in b for b in blocks)
    assert 'Twsearch finished.' in blocks[-1]
digest=hashlib.sha256((ROOT/'submission_ihes.csv').read_bytes()).hexdigest()
assert digest=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
assert len(load_submission(ROOT/'submission_ihes.csv')[296])==22
verified=verify_submission(puzzle,ROOT/'data/test.csv',ROOT/'submissions/ihes_20260908_local_prefix296.csv')
assert verified.n_valid==verified.n_total==1003
report=dict(pid=296,incumbent_moves=22,distinct_two_move_prefixes=262,residual_depth=18,
            exhaustive_total_depth=20,verdict='optimal_by_full_prefix_coverage_and_parity',
            baseline_sha256=digest,journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
            native_log_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            seed_log_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),verified_moves=verified.total_moves)
(ROOT/'data/ihes_pdb/pid296_proof_20260908.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
