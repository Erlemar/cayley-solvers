"""Reuse five CPU kernels for durable per-opening exact search."""
from pathlib import Path
import ast
import base64
import copy
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
WAVE='20260907_wave3_prefix'
assert not (ROOT/'data/ihes_pdb/cpu_wave3_launch.json').exists(), 'Already uploaded'
previous=json.loads((ROOT/'data/ihes_pdb/cpu_wave2_manifest.json').read_text())
ranking=json.loads((ROOT/'data/twsearch_rank_L22_20260805.json').read_text())['ranking']
pids=[x['pid'] for x in ranking if x['pid'] not in {26,28,268,300,868,468}][:5]
manifest=dict(wave_id=WAVE,max_concurrent_cpu_notebooks=5,baseline_total=21870,target_total=21838,
              strategy='262 deduplicated two-move openings per PID; exact residual search through 18',
              per_chunk_seconds=120,wall_limit_seconds=32400,table_mib=8192,
              source_sha256={},shards=[])
extra={name:(ROOT/name).read_text() for name in ['scripts/27_prefix_split_ladder.py','scripts/test_ihes_prefix_failures.py']}
manifest['source_sha256']={k:hashlib.sha256(v.encode()).hexdigest() for k,v in extra.items()}
seed=(ROOT/'data/ihes_pdb/prefix_probe_20260907.jsonl').read_text()
records=[json.loads(line) for line in seed.splitlines()]
assert len(records)==3 and all(r['pid']==296 and r['verdict']=='none' and r['max_depth']==18 for r in records)
manifest['local_seed_sha256']=hashlib.sha256(seed.encode()).hexdigest()
for shard,pid in enumerate(pids,1):
    old=previous['shards'][shard-1]
    notebook=json.loads((Path(old['folder'])/'search.ipynb').read_text())
    metadata=json.loads((Path(old['folder'])/'kernel-metadata.json').read_text())
    notebook['cells'][0]['source']=[f'IHES resumable exact search wave 3, shard {shard}/5, PID {pid}. CPU only. Each completed opening is journalled; nine-hour cap.']
    setup=''.join(notebook['cells'][1]['source'])
    setup+=f"\nfor name,content in json.loads(base64.b64decode({base64.b64encode(json.dumps(extra).encode()).decode()!r})).items():\n (work/name).write_text(content,encoding='utf-8')\n"
    if pid==296:
        setup+=f"(work/'data/ihes_pdb/prefix_L22.jsonl').write_text({seed!r},encoding='utf-8')\n"
    notebook['cells'][1]['source']=setup.splitlines(keepends=True)
    control=''.join(notebook['cells'][3]['source'])
    control+="\nsubprocess.run([sys.executable,'scripts/test_ihes_prefix_failures.py'],check=True,timeout=120)\n"
    control+="subprocess.run([sys.executable,'-u','scripts/27_prefix_split_ladder.py','--baseline','submission_ihes.csv','--journal','data/ihes_pdb/prefix_control.jsonl','--k','2','--self-test','--pids','1','--pop-roots','--exe','data/ihes_pdb/twsearch-pdb','--threads','1','--memory-mib','512','--start-prune-depth','8','--time-limit','30'],check=True,timeout=180)\n"
    notebook['cells'][3]['source']=control.splitlines(keepends=True)
    run=''.join(notebook['cells'][4]['source'])
    run=run[:run.index('rank=json.loads')]+f"pids={[str(pid)]!r}\n"+run[run.index("journal='data/"):]
    run=run.replace('kaggle_L22.jsonl','prefix_L22.jsonl').replace('scripts/24_twsearch_ladder.py','scripts/27_prefix_split_ladder.py')
    run=run.replace("'--window-length','22','--full-path-only',", "'--path-length','22','--k','2','--resume','--pop-roots',")
    run=run.replace("'--pid-order-file','data/twsearch_rank_L22_20260805.json',",'')
    run=run.replace("'--random-start','--max-depth','20','--time-limit','1200',", "'--time-limit','120',")
    run=run.replace('20260907_wave2',WAVE)
    run+="\nsys.path.insert(0,str(work/'src'))\nfrom cayley.puzzle import PictureCube\nfrom cayley.verify import verify_submission\nresult=verify_submission(PictureCube.load('data/puzzle_info.json'),'data/test.csv','submission.csv')\nassert result.n_valid==result.n_total==1003\nprint('Independent replay:',result,flush=True)\n"
    notebook['cells'][4]['source']=run.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    folder=ROOT/f'kaggle_notebooks/ihes_wave3_shard{shard}'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    manifest['shards'].append(dict(shard=shard,ref=old['ref'],pids=[pid],folder=str(folder),notebook_sha256=hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest()))
(ROOT/'data/ihes_pdb/cpu_wave3_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Prepared five resumable shards',pids)
