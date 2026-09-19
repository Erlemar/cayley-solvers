"""Build a fixed-parent four-move search using the tested current runner."""
from pathlib import Path
import argparse,ast,base64,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--template',type=Path,required=True)
ap.add_argument('--manifest',type=Path,required=True)
ap.add_argument('--receipt',type=Path,required=True)
ap.add_argument('--wave-id',required=True)
ap.add_argument('--folder-prefix',required=True)
ap.add_argument('--shard',type=int,required=True)
ap.add_argument('--pid',type=int,required=True)
ap.add_argument('--fixed-prefix',required=True)
ap.add_argument('--opening-offset',type=int,default=0)
ap.add_argument('--opening-count',type=int,default=24)
args=ap.parse_args()
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
p=PictureCube.load(ROOT/'data/puzzle_info.json')
parent=args.fixed_prefix.split('.');assert len(parent)==2 and all(m in p.generators for m in parent)
command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/69_prepare_ihes_l24.py'),
 '--template',str(args.template),'--manifest',str(args.manifest),'--receipt',str(args.receipt),
 '--wave-id',args.wave_id,'--folder-prefix',args.folder_prefix,'--shards',str(args.shard),'--pids',str(args.pid),
 '--opening-offset',str(args.opening_offset),'--opening-count',str(args.opening_count)]
subprocess.run(command,cwd=ROOT,check=True)
manifest=json.loads(args.manifest.read_text());job=manifest['shards'][0];folder=Path(job['folder'])
notebook=json.loads((folder/'search.ipynb').read_text())
runner=(ROOT/'scripts/27_prefix_split_ladder.py').read_bytes()
setup=''.join(notebook['cells'][1]['source'])
setup+=f"\nimport base64\n(work/'scripts/27_prefix_split_ladder.py').write_bytes(base64.b64decode({base64.b64encode(runner).decode()!r}))\n"
notebook['cells'][1]['source']=setup.splitlines(keepends=True)
run=''.join(notebook['cells'][4]['source']);assert "'--k','2'" in run
run=run.replace("'--k','2'",f"'--k','4',{'--fixed-prefix='+args.fixed_prefix!r}")
notebook['cells'][4]['source']=run.splitlines(keepends=True)
notebook['cells'][0]['source']=[f'IHES PID {args.pid}: four-move partition of fixed parent {args.fixed_prefix}; residual depth 18; partial suffix coverage only.']
for cell in notebook['cells']:
 if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
(folder/'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
manifest.update(residual_depth=18,k=4,fixed_prefix=parent,strategy='Refine one unresolved two-move parent; suffix IDs are local to that parent, not global proof coverage')
job.update(fixed_prefix=parent,runner_sha256=hashlib.sha256(runner).hexdigest(),notebook_sha256=hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest())
args.manifest.write_text(json.dumps(manifest,indent=2))
print('Prepared fixed parent',parent,'with residual depth 18 and runner',job['runner_sha256'])
