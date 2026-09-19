"""Five disjoint, adequately timed opening shards for unresolved PID 336."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
WAVE='20260908_wave4_split336'
assert not (ROOT/'data/ihes_pdb/cpu_wave4_launch.json').exists(), 'Already uploaded'
gate=ROOT/'submissions/ihes_20260908_prefix_timing_gate/remote/shard1'
passed=json.loads((gate/'prefix_timing_gate_passed.json').read_text())
assert passed['pid']==296 and passed['prefix_id']==0 and passed['max_depth']==18 and passed['verdict']=='none'
assert passed['secs']<600
assert json.loads((gate/'run_status.json').read_text())['wave_id']=='20260908_prefix_timing_gate'
old=json.loads((ROOT/'data/ihes_pdb/cpu_wave3_manifest.json').read_text())
manifest=dict(wave_id=WAVE,max_concurrent_cpu_notebooks=5,baseline_total=21870,target_total=21838,
              strategy='PID 336: 262 two-move openings partitioned across five kernels; exact residual depth <=18',
              per_chunk_seconds=600,wall_limit_seconds=32400,table_mib=8192,gate_seconds=passed['secs'],
              gate_sha256=hashlib.sha256((gate/'prefix_timing_gate_passed.json').read_bytes()).hexdigest(),shards=[])
covered=[]
for i in range(5):
    offset=i*53;size=min(53,262-offset)
    covered.extend(range(offset,offset+size))
    notebook=json.loads((ROOT/'kaggle_notebooks/ihes_wave3_shard3/search.ipynb').read_text())
    notebook['cells'][0]['source']=[f'IHES PID 336 exact search; opening IDs {offset}..{offset+size-1}; residual depth 18; 600 seconds per opening; CPU only; nine-hour cap.']
    run=''.join(notebook['cells'][4]['source'])
    assert "pids=['336']" in run
    assert "'--time-limit','120'," in run
    run=run.replace('20260907_wave3_prefix',WAVE)
    run=run.replace("'--time-limit','120',",f"'--time-limit','600','--opening-offset','{offset}','--max-openings','{size}',")
    notebook['cells'][4]['source']=run.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    folder=ROOT/f'kaggle_notebooks/ihes_wave4_shard{i+1}';folder.mkdir(exist_ok=True)
    (folder/'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    metadata=json.loads((Path(old['shards'][i]['folder'])/'kernel-metadata.json').read_text())
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    manifest['shards'].append(dict(shard=i+1,ref=metadata['id'],pids=[336],opening_offset=offset,opening_count=size,
        folder=str(folder),notebook_sha256=hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest()))
assert covered==list(range(262))
(ROOT/'data/ihes_pdb/cpu_wave4_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Five disjoint shards cover all 262 openings; timing gate',passed['secs'],'seconds')
