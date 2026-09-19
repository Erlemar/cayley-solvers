"""Audit a complete fresh, unseeded split-prefix wave as an optimality proof."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,verify_submission
ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--pid',type=int,required=True)
ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefixes=m.canonical_prefixes(2,{n:np.asarray(v,dtype=np.int64) for n,v in puzzle.generators.items()},list(puzzle.move_names))
manifest=json.loads(args.manifest.read_text());settled=set();sources=[]
for job in manifest['shards']:
    directory=args.directory/'remote'/f"shard{job['shard']}"
    status=json.loads((directory/'run_status.json').read_text())
    assert status['wave_id']==manifest['wave_id'] and status['returncode']==0 and status['pids']==[str(args.pid)]
    raw=(directory/'prefix_L22.jsonl').read_bytes();rows=[json.loads(s) for s in raw.decode().splitlines()]
    native=(directory/'prefix_L22.solver.log').read_text();blocks=native.split('\nSolving\n')[1:]
    expected=list(range(job['opening_offset'],job['opening_offset']+job['opening_count']))
    assert [r['prefix_id'] for r in rows]==expected and len(blocks)==len(rows) and 'Twsearch finished.' in blocks[-1]
    for r,b in zip(rows,blocks):
        i=r['prefix_id'];assert i not in settled
        assert r['pid']==args.pid and r['prefix']==prefixes[i] and r['max_depth']==18 and r['incumbent_len']==22 and not r['self_test']
        assert r['verdict']=='none' and 'No solution found in 18' in b.splitlines() and 'Search timed out' not in b
        settled.add(i)
    verified=verify_submission(puzzle,ROOT/'data/test.csv',directory/'submission.csv')
    assert verified.n_valid==verified.n_total==1003
    sources.append(dict(shard=job['shard'],count=len(rows),journal_sha256=hashlib.sha256(raw).hexdigest(),native_log_sha256=hashlib.sha256(native.encode()).hexdigest(),verified_moves=verified.total_moves))
assert settled==set(range(262))
baseline=ROOT/'submission_ihes.csv';digest=hashlib.sha256(baseline.read_bytes()).hexdigest()
assert digest=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
assert len(load_submission(baseline)[args.pid])==22
report=dict(pid=args.pid,incumbent_moves=22,distinct_two_move_prefixes=262,residual_depth=18,
            verdict='optimal_by_full_prefix_coverage_and_parity',baseline_sha256=digest,sources=sources)
args.output.write_text(json.dumps(report,indent=2));print('PID',args.pid,'optimal at 22; all 262 prefixes checked')
