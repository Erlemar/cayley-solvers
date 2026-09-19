"""Verify all 262 openings across the capped wave 12 and closing wave 13."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, verify_submission
puzzle = PictureCube.load(ROOT/'data/puzzle_info.json')
spec = importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefixes = m.canonical_prefixes(2,{n:np.asarray(g,dtype=np.int64) for n,g in puzzle.generators.items()},list(puzzle.move_names))
assert len(prefixes)==262
settled=set();sources=[]
for wave,date in [(12,'20260909'),(13,'20260910')]:
    manifest=json.loads((ROOT/f'data/ihes_pdb/cpu_wave{wave}_manifest.json').read_text())
    assert manifest['wave_id']==('20260909_wave12_split46' if wave==12 else '20260910_wave13_close46')
    for job in manifest['shards']:
        directory=ROOT/f'submissions/ihes_{date}_wave{wave}/remote'/f"shard{job['shard']}"
        status=json.loads((directory/'run_status.json').read_text())
        interrupted=wave==12 and job['shard']==1
        assert status['wave_id']==manifest['wave_id'] and status['pids']==['46']
        assert status['returncode']==(-15 if interrupted else 0)
        raw=(directory/'prefix_L22.jsonl').read_bytes()
        rows=[json.loads(s) for s in raw.decode().splitlines()]
        native=(directory/'prefix_L22.solver.log').read_text();blocks=native.split('\nSolving\n')[1:]
        expected=list(range(job['opening_offset'],job['opening_offset']+job['opening_count']))
        if interrupted:
            expected=expected[:51]
            assert len(blocks)==len(rows)+1
        else:
            assert len(blocks)==len(rows) and 'Twsearch finished.' in blocks[-1]
        assert [r['prefix_id'] for r in rows]==expected
        for r,b in zip(rows,blocks):
            i=r['prefix_id'];assert i not in settled and 0<=i<262
            assert r['pid']==46 and r['prefix']==prefixes[i] and r['max_depth']==18
            assert r['incumbent_len']==22 and not r['self_test'] and r['verdict']=='none'
            assert 'No solution found in 18' in b.splitlines() and 'Search timed out' not in b
            settled.add(i)
        verified=verify_submission(puzzle,ROOT/'data/test.csv',directory/'submission.csv')
        assert verified.n_valid==verified.n_total==1003
        sources.append(dict(wave=wave,shard=job['shard'],count=len(rows),journal_sha256=hashlib.sha256(raw).hexdigest(),native_log_sha256=hashlib.sha256(native.encode()).hexdigest(),verified_moves=verified.total_moves))
assert settled==set(range(262))
baseline=ROOT/'submission_ihes.csv';digest=hashlib.sha256(baseline.read_bytes()).hexdigest()
assert digest=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
assert len(load_submission(baseline)[46])==22
verified=verify_submission(puzzle,ROOT/'data/test.csv',baseline)
assert verified.n_valid==verified.n_total==1003
report=dict(pid=46,incumbent_moves=22,distinct_two_move_prefixes=262,residual_depth=18,
            verdict='optimal_by_full_prefix_coverage_and_parity',baseline_sha256=digest,sources=sources)
(ROOT/'data/ihes_pdb/pid46_proof_20260910.json').write_text(json.dumps(report,indent=2))
print('PID 46 optimal at 22: all 262 openings verified across six source runs.')
