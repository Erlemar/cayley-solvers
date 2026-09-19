"""Audit a wall-capped shard and prepare only its two unfinished openings."""
from pathlib import Path
import ast, hashlib, importlib.util, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cayley.puzzle import PictureCube
from cayley.verify import verify_submission

assert not (ROOT / 'data/ihes_pdb/cpu_wave13_launch.json').exists()
old = json.loads((ROOT / 'data/ihes_pdb/cpu_wave12_manifest.json').read_text())
puzzle = PictureCube.load(ROOT / 'data/puzzle_info.json')
spec = importlib.util.spec_from_file_location('prefix', ROOT / 'scripts/27_prefix_split_ladder.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
prefixes = m.canonical_prefixes(2, {n: np.asarray(g, dtype=np.int64) for n,g in puzzle.generators.items()}, list(puzzle.move_names))
settled = set(); sources = []
for job in old['shards']:
    directory = ROOT / 'submissions/ihes_20260909_wave12/remote' / f"shard{job['shard']}"
    status = json.loads((directory / 'run_status.json').read_text())
    assert status['wave_id'] == old['wave_id'] and status['pids'] == ['46']
    assert status['returncode'] == (-15 if job['shard'] == 1 else 0)
    raw = (directory / 'prefix_L22.jsonl').read_bytes()
    rows = [json.loads(s) for s in raw.decode().splitlines()]
    native = (directory / 'prefix_L22.solver.log').read_text()
    blocks = native.split('\nSolving\n')[1:]
    expected = list(range(job['opening_offset'], job['opening_offset'] + job['opening_count']))
    if job['shard'] == 1:
        expected = expected[:51]
        assert len(blocks) == len(rows) + 1  # interrupted, unrecorded opening 51
    else:
        assert len(blocks) == len(rows) and 'Twsearch finished.' in blocks[-1]
    assert [r['prefix_id'] for r in rows] == expected
    for r,b in zip(rows, blocks):
        i = r['prefix_id']; assert i not in settled
        assert r['pid'] == 46 and r['prefix'] == prefixes[i]
        assert r['max_depth'] == 18 and r['incumbent_len'] == 22 and not r['self_test']
        assert r['verdict'] == 'none' and 'No solution found in 18' in b.splitlines()
        assert 'Search timed out' not in b
        settled.add(i)
    verified = verify_submission(puzzle, ROOT / 'data/test.csv', directory / 'submission.csv')
    assert verified.n_valid == verified.n_total == 1003
    sources.append(dict(shard=job['shard'],count=len(rows),journal_sha256=hashlib.sha256(raw).hexdigest(),native_log_sha256=hashlib.sha256(native.encode()).hexdigest(),verified_moves=verified.total_moves))
assert set(range(262)) - settled == {51,52}
report = dict(pid=46,known_settled=len(settled),unresolved=[51,52],sources=sources)
(ROOT / 'data/ihes_pdb/wave12_partial_audit_20260910_0434.json').write_text(json.dumps(report,indent=2))

source = old['shards'][0]
notebook = json.loads((Path(source['folder']) / 'search.ipynb').read_text())
wave = '20260910_wave13_close46'
notebook['cells'][0]['source'] = ['IHES PID 46: only unfinished openings 51 and 52. 1200 seconds each; CPU only.']
run = ''.join(notebook['cells'][4]['source']).replace(old['wave_id'],wave)
assert "'--opening-offset','0'" in run and "'--max-openings','53'" in run
run = run.replace("'--opening-offset','0'", "'--opening-offset','51'").replace("'--max-openings','53'", "'--max-openings','2'")
notebook['cells'][4]['source'] = run.splitlines(keepends=True)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
folder = ROOT / 'kaggle_notebooks/ihes_wave13_shard1'; folder.mkdir(exist_ok=True)
(folder / 'search.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
(folder / 'kernel-metadata.json').write_bytes((Path(source['folder']) / 'kernel-metadata.json').read_bytes())
manifest = {**old, 'wave_id':wave, 'strategy':'Only unresolved PID 46 openings 51 and 52; no seeded journal', 'shards':[
    {**source,'folder':str(folder),'opening_offset':51,'opening_count':2,'notebook_sha256':hashlib.sha256((folder/'search.ipynb').read_bytes()).hexdigest()}
]}
(ROOT / 'data/ihes_pdb/cpu_wave13_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Audited 260 exhausted prefixes; prepared ONLY openings 51 and 52.')
