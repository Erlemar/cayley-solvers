"""Audit settled wave-4 shards and resume only their unfinished subproblems."""
from pathlib import Path
from collections import Counter
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
assert not (ROOT/'data/ihes_pdb/cpu_wave5_launch.json').exists(), 'Already uploaded'
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefixes=m.canonical_prefixes(2,{n:np.asarray(v,dtype=np.int64) for n,v in puzzle.generators.items()},list(puzzle.move_names))
old=json.loads((ROOT/'data/ihes_pdb/cpu_wave4_manifest.json').read_text())
WAVE='20260908_wave5_resume336'
manifest=dict(wave_id=WAVE,max_concurrent_cpu_notebooks=5,baseline_total=21870,target_total=21838,
              per_chunk_seconds=1200,wall_limit_seconds=32400,table_mib=8192,
              concurrent_other_run='Wave 4 shard 1 retains opening IDs 0..52',shards=[])
audit=[]
for source in old['shards'][1:]:
    i=source['shard'];remote=ROOT/f'submissions/ihes_20260908_wave4/remote/shard{i}'
    status=json.loads((remote/'run_status.json').read_text())
    assert status['wave_id']==old['wave_id'] and status['returncode']==0 and status['pids']==['336']
    seed=(remote/'prefix_L22.jsonl').read_text();rows=[json.loads(s) for s in seed.splitlines()]
    ids=list(range(source['opening_offset'],source['opening_offset']+source['opening_count']))
    assert [r['prefix_id'] for r in rows]==ids
    native=(remote/'prefix_L22.solver.log').read_text();blocks=native.split('\nSolving\n')[1:]
    assert len(blocks)==len(rows) and 'Twsearch finished.' in blocks[-1]
    for r,b in zip(rows,blocks):
        assert r['pid']==336 and r['prefix']==prefixes[r['prefix_id']]
        assert r['max_depth']==18 and r['incumbent_len']==22 and not r['self_test']
        assert r['verdict'] in ('none','timeout')
        if r['verdict']=='none':assert 'No solution found in 18' in b.splitlines() and 'Search timed out' not in b
    verified=verify_submission(puzzle,ROOT/'data/test.csv',remote/'submission.csv')
    assert verified.n_valid==verified.n_total==1003 and verified.total_moves==21870
    counts=dict(Counter(r['verdict'] for r in rows));assert counts.get('timeout',0)>0
    audit.append(dict(shard=i,counts=counts,journal_sha256=hashlib.sha256(seed.encode()).hexdigest(),
                      native_log_sha256=hashlib.sha256(native.encode()).hexdigest(),verified_moves=verified.total_moves))
    notebook=json.loads((Path(source['folder'])/'search.ipynb').read_text())
    notebook['cells'][0]['source']=[f'IHES PID 336 resume shard {i}: preserve {counts.get("none",0)} settled prefixes, retry {counts["timeout"]} unresolved prefixes with 1200 seconds each. Nine-hour cap.']
    setup=''.join(notebook['cells'][1]['source'])
    setup+=f"\n(work/'data/ihes_pdb/prefix_L22.jsonl').write_text({seed!r},encoding='utf-8')\n"
    notebook['cells'][1]['source']=setup.splitlines(keepends=True)
    run=''.join(notebook['cells'][4]['source'])
    assert "'--time-limit','600'," in run and "'--resume'" in run
    run=run.replace(old['wave_id'],WAVE).replace("'--time-limit','600',","'--time-limit','1200',")
    notebook['cells'][4]['source']=run.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    folder=ROOT/f'kaggle_notebooks/ihes_wave5_shard{i}';folder.mkdir(exist_ok=True)
    (folder/'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    metadata=json.loads((Path(source['folder'])/'kernel-metadata.json').read_text())
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    manifest['shards'].append({**source,'folder':str(folder),'notebook_sha256':hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest(),
        'seed_journal_sha256':hashlib.sha256(seed.encode()).hexdigest(),'settled':counts.get('none',0),'retry':counts['timeout']})
(ROOT/'data/ihes_pdb/wave4_partial_audit_20260908_0911.json').write_text(json.dumps(audit,indent=2))
(ROOT/'data/ihes_pdb/cpu_wave5_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Prepared',len(manifest['shards']),'resume shards;',sum(s['settled'] for s in manifest['shards']),'settled;',sum(s['retry'] for s in manifest['shards']),'retries')
