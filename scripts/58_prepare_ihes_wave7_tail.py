"""Retry a bounded subset of the remaining original shard-1 cases."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import verify_submission
assert not (ROOT/'data/ihes_pdb/cpu_wave7_launch.json').exists(), 'Already uploaded'
WAVE='20260908_wave7_resume336_tail'
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefixes=m.canonical_prefixes(2,{n:np.asarray(v,dtype=np.int64) for n,v in puzzle.generators.items()},list(puzzle.move_names))
remote=ROOT/'submissions/ihes_20260908_wave4/remote/shard1'
status=json.loads((remote/'run_status.json').read_text())
assert status['wave_id']=='20260908_wave4_split336' and status['returncode']==0
seed=(remote/'prefix_L22.jsonl').read_text();rows=[json.loads(s) for s in seed.splitlines()]
assert [r['prefix_id'] for r in rows]==list(range(53))
native=(remote/'prefix_L22.solver.log').read_text();blocks=native.split('\nSolving\n')[1:]
assert len(blocks)==53 and 'Twsearch finished.' in blocks[-1]
for r,b in zip(rows,blocks):
    assert r['pid']==336 and r['prefix']==prefixes[r['prefix_id']] and r['max_depth']==18 and r['incumbent_len']==22 and not r['self_test']
    assert r['verdict'] in ('none','timeout')
    if r['verdict']=='none':assert 'No solution found in 18' in b.splitlines() and 'Search timed out' not in b
verified=verify_submission(puzzle,ROOT/'data/test.csv',ROOT/'submissions/ihes_20260908_wave4_verified.csv')
assert verified.n_valid==verified.n_total==1003 and verified.total_moves==21870
retry=[r['prefix_id'] for r in rows if r['verdict']=='timeout' and r['prefix_id']<28]
pending=[r['prefix_id'] for r in rows if r['verdict']=='timeout' and r['prefix_id']>=28]
assert len(retry)==26 and len(pending)==19
old=json.loads((ROOT/'data/ihes_pdb/cpu_wave4_manifest.json').read_text())['shards'][0]
manifest=dict(wave_id=WAVE,max_concurrent_cpu_notebooks=5,baseline_total=21870,target_total=21838,per_chunk_seconds=1200,wall_limit_seconds=32400,table_mib=8192,shards=[])
covered=[]
for shard,offset,count in [(3,28,13),(4,41,12)]:
    selected=[r['prefix_id'] for r in rows if r['verdict']=='timeout' and offset<=r['prefix_id']<offset+count]
    covered.extend(selected)
    notebook=json.loads((Path(old['folder'])/'search.ipynb').read_text())
    notebook['cells'][0]['source']=[f'PID 336: retry {len(selected)} unresolved opening IDs within {offset}..{offset+count-1}; 1200 seconds each; settled seed records preserved; nine-hour cap.']
    setup=''.join(notebook['cells'][1]['source'])+f"\n(work/'data/ihes_pdb/prefix_L22.jsonl').write_text({seed!r},encoding='utf-8')\n"
    notebook['cells'][1]['source']=setup.splitlines(keepends=True)
    run=''.join(notebook['cells'][4]['source'])
    run=run.replace('20260908_wave4_split336',WAVE).replace("'--time-limit','600'","'--time-limit','1200'").replace("'--opening-offset','0'",f"'--opening-offset','{offset}'").replace("'--max-openings','53'",f"'--max-openings','{count}'")
    notebook['cells'][4]['source']=run.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    folder=ROOT/f'kaggle_notebooks/ihes_wave7_shard{shard}';folder.mkdir(exist_ok=True)
    (folder/'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    metadata=json.loads((ROOT/f'kaggle_notebooks/ihes_wave5_shard{shard}/kernel-metadata.json').read_text())
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    manifest['shards'].append(dict(shard=shard,ref=metadata['id'],pids=[336],folder=str(folder),opening_offset=offset,opening_count=count,retry_prefix_ids=selected,seed_journal_sha256=hashlib.sha256(seed.encode()).hexdigest(),notebook_sha256=hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest()))
assert covered==pending and len(covered)==19
(ROOT/'data/ihes_pdb/cpu_wave7_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Prepared two disjoint retry shards covering the last 19 pending cases')
