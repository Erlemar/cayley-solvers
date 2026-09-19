"""Launch the five approved CPU slots, recording each upload before continuing."""
from pathlib import Path
import hashlib
import json
import os
import re
import time

ROOT=Path(__file__).resolve().parents[1]
receipt=ROOT/'data/ihes_pdb/cpu_wave8_launch.json'
assert not receipt.exists(), 'Inspect existing receipts before any retry'
manifest_path=ROOT/'data/ihes_pdb/cpu_wave8_manifest.json'
manifest=json.loads(manifest_path.read_text())
assert len(manifest['shards'])==1
os.environ['KAGGLE_API_TOKEN']=re.findall(r'KGAT_[A-Za-z0-9_\-]+',Path('C:/Users/and-l/.claude/projects/C--Users-and-l-cayley/memory/ref_kaggle_credentials.md').read_text(encoding='utf-8'))[-1]
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
def obj(r):return r if isinstance(r,dict) else r.to_dict()
for shard in manifest['shards']:
    assert obj(api.kernels_status(shard['ref']))['status'].upper()=='COMPLETE'
    folder=Path(shard['folder']);metadata=json.loads((folder/'kernel-metadata.json').read_text())
    assert metadata['is_private'] and not metadata['enable_gpu'] and not metadata['enable_tpu']
    assert hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest()==shard['notebook_sha256']
state=dict(wave_id=manifest['wave_id'],started_unix=time.time(),launches=[])
for shard in manifest['shards']:
    row=dict(ref=shard['ref'],stage='UPLOAD_REQUESTED')
    state['launches'].append(row);receipt.write_text(json.dumps(state,indent=2))
    response=obj(api.kernels_push(str(Path(shard['folder']).resolve())))
    row.update(response=response,stage='UPLOAD_RETURNED')
    receipt.write_text(json.dumps(state,indent=2))
    assert not response.get('error'), response
    canonical=response.get('ref',shard['ref']).removeprefix('/code/').lstrip('/')
    shard['ref']=canonical
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(canonical, response,flush=True)
print('One closing retry uploaded; confirm provider state',flush=True)
