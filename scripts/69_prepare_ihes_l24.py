"""Prepare bounded, disjoint per-PID length-24 shortening trials on free slots."""
from pathlib import Path
import argparse,ast,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cayley.verify import load_submission
ap=argparse.ArgumentParser()
ap.add_argument('--template',type=Path,required=True)
ap.add_argument('--manifest',type=Path,required=True)
ap.add_argument('--receipt',type=Path,required=True)
ap.add_argument('--wave-id',required=True)
ap.add_argument('--folder-prefix',required=True)
ap.add_argument('--shards',type=int,nargs='+',required=True)
ap.add_argument('--pids',type=int,nargs='+',required=True)
ap.add_argument('--opening-offset',type=int,default=0)
ap.add_argument('--opening-count',type=int,default=24)
args=ap.parse_args()
assert not args.receipt.exists() and not args.manifest.exists()
assert len(args.shards)==len(args.pids) and len(set(args.shards))==len(args.shards)
assert len(set(args.pids))==len(args.pids) and 1<=len(args.shards)<=5
assert 0<=args.opening_offset<262 and 0<args.opening_count<=262-args.opening_offset
baseline=ROOT/'submission_ihes.csv'
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
rows=load_submission(baseline)
assert all(len(rows[pid])==24 for pid in args.pids)
old=json.loads(args.template.read_text())
manifest=dict(wave_id=args.wave_id,max_concurrent_cpu_notebooks=5,baseline_total=21870,
    target_total=21838,path_length=24,target_delta=2,residual_depth=20,
    per_chunk_seconds=1200,wall_limit_seconds=32400,table_mib=8192,
    strategy='Length-24 priority: bounded initial prefix coverage per PID; timeouts remain unresolved',shards=[])
for shard,pid in zip(args.shards,args.pids):
    source=next(s for s in old['shards'] if s['shard']==shard)
    notebook=json.loads((Path(source['folder'])/'search.ipynb').read_text())
    notebook['cells'][0]['source']=[f'IHES PID {pid}: seek <=22 moves from incumbent 24. Initial {args.opening_count}/262 openings; CPU only.']
    run=''.join(notebook['cells'][4]['source'])
    run,n=re.subn(r'^pids=.*$',f'pids={[str(pid)]!r}',run,flags=re.M);assert n==1
    run=run.replace(old['wave_id'],args.wave_id).replace('prefix_L22','prefix_L24')
    assert "'--path-length','22'" in run
    run=run.replace("'--path-length','22'","'--path-length','24'")
    run,n=re.subn(r"'--opening-offset','\d+'",f"'--opening-offset','{args.opening_offset}'",run);assert n==1
    run,n=re.subn(r"'--max-openings','\d+'",f"'--max-openings','{args.opening_count}'",run);assert n==1
    notebook['cells'][4]['source']=run.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    folder=ROOT/f'{args.folder_prefix}{shard}';folder.mkdir(exist_ok=True)
    (folder/'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (folder/'kernel-metadata.json').write_bytes((Path(source['folder'])/'kernel-metadata.json').read_bytes())
    manifest['shards'].append({**source,'pids':[pid],'folder':str(folder),'opening_offset':args.opening_offset,
        'opening_count':args.opening_count,'notebook_sha256':hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest()})
args.manifest.write_text(json.dumps(manifest,indent=2))
print('Prepared length-24 PIDs',args.pids,'on slots',args.shards,'with residual depth 20')
