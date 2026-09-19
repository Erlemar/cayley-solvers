"""Prepare five independent L24 parent searches after successful split4 pilots."""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
receipt=ROOT/'data/ihes_pdb/cpu_wave20_launch.json'
assert not receipt.exists()
wave='20260911_wave20_L24_parents'
plans=[]
for shard,pid in [(1,106),(2,592),(3,680),(5,764)]:
    directory=ROOT/f'submissions/ihes_20260911_wave18/remote/shard{shard}'
    raw=(directory/'prefix_L24.jsonl').read_bytes()
    rows=[json.loads(s) for s in raw.decode().splitlines()]
    native=(directory/'prefix_L24.solver.log').read_text();blocks=native.split('\nSolving\n')[1:]
    assert len(rows)==len(blocks)==24
    ranked=[]
    for r,b in zip(rows,blocks):
        assert r['pid']==pid and r['verdict']=='timeout' and len(r['prefix'])==2
        match=re.search(r'^Depth 19 in ([0-9.]+) probes',b,re.M)
        if match:ranked.append((float(match[1]),r['prefix_id'],r['prefix']))
    assert ranked,'No completed depth-19 timing; choose another selection rule explicitly'
    secs,parent_id,parent=min(ranked)
    plans.append(dict(shard=shard,pid=pid,parent=parent,parent_id=parent_id,depth19_secs=secs,
        opening_offset=0,opening_count=64,source_journal_sha256=hashlib.sha256(raw).hexdigest(),
        source_native_sha256=hashlib.sha256(native.encode()).hexdigest()))
plans.append(dict(shard=4,pid=936,parent=['-f0','f1'],parent_id=18,
    opening_offset=24,opening_count=238,reason='Continue only remaining suffixes after verified 24-case pilot'))
plans.sort(key=lambda p:p['shard'])
parts=[]
for plan in plans:
    part=ROOT/f"data/ihes_pdb/cpu_wave20_part{plan['shard']}.json"
    command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/72_prepare_ihes_l24_split4.py'),
        '--template','data/ihes_pdb/cpu_wave16_manifest.json','--manifest',str(part),'--receipt',str(receipt),
        '--wave-id',wave,'--folder-prefix','kaggle_notebooks/ihes_wave20_shard',
        '--shard',str(plan['shard']),'--pid',str(plan['pid']),'--fixed-prefix='+'.'.join(plan['parent']),
        '--opening-offset',str(plan['opening_offset']),'--opening-count',str(plan['opening_count'])]
    subprocess.run(command,cwd=ROOT,check=True)
    parts.append(json.loads(part.read_text()))
manifest={k:v for k,v in parts[0].items() if k not in ('shards','fixed_prefix')}
manifest.update(strategy='Five L24 fixed-parent searches; four 64-suffix exploratory slices and one 238-suffix continuation',
    selection='Minimum measured completed depth-19 cost among timed-out cases; not solution-likelihood evidence',
    shards=[part['shards'][0] for part in parts],plans=plans)
assert len({s['ref'] for s in manifest['shards']})==5
(ROOT/'data/ihes_pdb/cpu_wave20_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Prepared wave 20 plans:',json.dumps(plans))
