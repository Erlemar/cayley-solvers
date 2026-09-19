"""Audit the complete multi-wave PID 336 proof, following every seed edge."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,verify_submission
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefixes=m.canonical_prefixes(2,{n:np.asarray(v,dtype=np.int64) for n,v in puzzle.generators.items()},list(puzzle.move_names))
assert len(prefixes)==262
nodes={};settled={};sources=[]
wave_ids={4:'20260908_wave4_split336',5:'20260908_wave5_resume336',6:'20260908_wave6_resume336_head',7:'20260908_wave7_resume336_tail',8:'20260908_wave8_close336'}
def audit(wave,shard,parent=None):
    directory=ROOT/f'submissions/ihes_20260908_wave{wave}/remote/shard{shard}'
    manifest=json.loads((ROOT/f'data/ihes_pdb/cpu_wave{wave}_manifest.json').read_text())
    job=next(s for s in manifest['shards'] if s['shard']==shard)
    status=json.loads((directory/'run_status.json').read_text())
    assert status['wave_id']==manifest['wave_id']==wave_ids[wave] and status['returncode']==0
    assert status['pids']==['336']
    raw=(directory/'prefix_L22.jsonl').read_bytes();rows=[json.loads(s) for s in raw.decode().splitlines()]
    seed=nodes[parent] if parent else []
    assert rows[:len(seed)]==seed
    added=rows[len(seed):]
    native=(directory/'prefix_L22.solver.log').read_text();blocks=native.split('\nSolving\n')[1:]
    assert len(blocks)==len(added) and 'Twsearch finished.' in blocks[-1]
    lo=job['opening_offset'];hi=lo+job['opening_count']
    seed_done={r['prefix_id'] for r in seed if r['verdict']=='none'}
    expected=[i for i in range(lo,hi) if i not in seed_done]
    assert [r['prefix_id'] for r in added]==expected
    for r,b in zip(added,blocks):
        i=r['prefix_id']
        assert r['pid']==336 and r['prefix']==prefixes[i] and r['max_depth']==18 and r['incumbent_len']==22 and not r['self_test']
        if r['verdict']=='none':
            assert 'No solution found in 18' in b.splitlines() and 'Search timed out' not in b
            settled.setdefault(i,{'wave':wave,'shard':shard})
        else:assert r['verdict']=='timeout' and 'Search timed out' in b
    verification=verify_submission(puzzle,ROOT/'data/test.csv',directory/'submission.csv')
    assert verification.n_valid==verification.n_total==1003 and verification.total_moves==21870
    nodes[wave,shard]=rows
    sources.append(dict(wave=wave,shard=shard,new_records=len(added),journal_sha256=hashlib.sha256(raw).hexdigest(),native_log_sha256=hashlib.sha256(native.encode()).hexdigest()))
for shard in range(1,6):audit(4,shard)
for shard in range(2,6):audit(5,shard,(4,shard))
audit(6,1,(4,1))
for shard in (3,4):audit(7,shard,(4,1))
audit(8,1,(6,1))
assert set(settled)==set(range(262))
baseline=ROOT/'submission_ihes.csv';digest=hashlib.sha256(baseline.read_bytes()).hexdigest()
assert digest=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
assert len(load_submission(baseline)[336])==22
report=dict(pid=336,incumbent_moves=22,distinct_two_move_prefixes=262,residual_depth=18,
            verdict='optimal_by_full_prefix_coverage_and_parity',baseline_sha256=digest,
            sources=sources,settled_prefix_provenance=settled,verified_moves=21870)
(ROOT/'data/ihes_pdb/pid336_proof_20260909.json').write_text(json.dumps(report,indent=2))
print('PID 336 optimal: all 262 prefixes verified across',len(sources),'source runs; score remains 21870')
