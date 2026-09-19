"""Prepare a fresh length-22 PID from an immutable, validated split template."""
from pathlib import Path
import argparse,ast,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cayley.verify import load_submission
ap=argparse.ArgumentParser();ap.add_argument('--template',type=Path,required=True)
ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True)
ap.add_argument('--wave-id',required=True);ap.add_argument('--folder-prefix',required=True)
ap.add_argument('--pid',type=int,required=True);ap.add_argument('--time-limit',type=int,default=1200)
args=ap.parse_args();assert not args.receipt.exists(), 'Already uploaded'
assert len(load_submission(ROOT/'submission_ihes.csv')[args.pid])==22
for proof in (ROOT/'data/ihes_pdb').glob('pid*_proof_*.json'):
    data=json.loads(proof.read_text());assert data.get('pid')!=args.pid, 'Target already proved optimal'
old=json.loads(args.template.read_text());manifest={k:v for k,v in old.items() if k!='shards'}
manifest.update(wave_id=args.wave_id,per_chunk_seconds=args.time_limit,strategy=f'Fresh PID {args.pid}: 262 disjoint openings; preserve partial results at wall cap',shards=[])
covered=[]
for source in old['shards']:
    folder=ROOT/f"{args.folder_prefix}{source['shard']}";folder.mkdir(parents=True,exist_ok=True)
    notebook=json.loads((Path(source['folder'])/'search.ipynb').read_text())
    notebook['cells'][0]['source']=[f"IHES PID {args.pid}, prefix shard {source['shard']}. {args.time_limit} seconds per case; nine-hour cap; CPU only."]
    run=''.join(notebook['cells'][4]['source'])
    run,n=re.subn(r'^pids=.*$',f'pids={[str(args.pid)]!r}',run,flags=re.M);assert n==1
    run=run.replace(old['wave_id'],args.wave_id)
    run,n=re.subn(r"'--time-limit','\d+'",f"'--time-limit','{args.time_limit}'",run);assert n==1
    notebook['cells'][4]['source']=run.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    (folder/'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    metadata=json.loads((Path(source['folder'])/'kernel-metadata.json').read_text())
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    manifest['shards'].append({**source,'pids':[args.pid],'folder':str(folder),'notebook_sha256':hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest()})
    covered.extend(range(source['opening_offset'],source['opening_offset']+source['opening_count']))
assert covered==list(range(262))
args.manifest.write_text(json.dumps(manifest,indent=2));print('Prepared PID',args.pid,'in',len(manifest['shards']),'disjoint shards')
