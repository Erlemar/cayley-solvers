"""Retry four L24 cases at 1800 seconds, selected by completed depth-19 cost."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import verify_submission

stem=ROOT/'data/ihes_pdb/local_L24_escalation1_20260911'
journal=stem.with_suffix('.jsonl')
assert not journal.exists(), 'Inspect existing escalation before relaunch'
report=json.loads((ROOT/'data/ihes_pdb/status_20260911_1205.json').read_text())
selected=report['suggested_escalation']
assert {(r['pid'],r['prefix_id']) for r in selected}=={(810,7),(810,10),(936,19),(936,18)}
# Alternate PIDs so both receive the longer budget early.
jobs=[selected[i] for i in (0,2,1,3)]
baseline=ROOT/'submission_ihes.csv'
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
plan=dict(time_limit=1800,jobs=jobs,selection='Two fastest completed depth-19 searches per PID; not a prediction of solution probability',
    source_journal_sha256=hashlib.sha256((ROOT/'data/ihes_pdb/local_L24_810_936_round1_20260911.jsonl').read_bytes()).hexdigest(),
    source_native_sha256=hashlib.sha256((ROOT/'data/ihes_pdb/local_L24_810_936_round1_20260911.solver.log').read_bytes()).hexdigest(),results=[])
planfile=stem.with_suffix('.plan.json');planfile.write_text(json.dumps(plan,indent=2))
os.environ['IHES_CORNER_PDB_DIR']=str(ROOT/'data/ihes_pdb')
out=ROOT/'submissions/ihes_20260911_local_L24_escalation1.csv'
for job in jobs:
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/27_prefix_split_ladder.py'),
        '--baseline',str(baseline),'--journal',str(journal),'--k','2','--pids',str(job['pid']),
        '--path-length','24','--resume','--pop-roots','--exe',str(ROOT/'data/ihes_pdb/twsearch-pdb-ranked.exe'),
        '--threads','16','--memory-mib','8192','--time-limit','1800',
        '--opening-offset',str(job['prefix_id']),'--max-openings','1','--out',str(out)]
    print('Escalating',job,flush=True)
    started=time.time();proc=subprocess.run(command,cwd=ROOT)
    plan['results'].append(dict(**job,returncode=proc.returncode,elapsed=time.time()-started))
    planfile.write_text(json.dumps(plan,indent=2))
    if proc.returncode:raise SystemExit(proc.returncode)
    verified=verify_submission(PictureCube.load(ROOT/'data/puzzle_info.json'),ROOT/'data/test.csv',out)
    assert verified.n_valid==verified.n_total==1003
    print('Independent replay:',verified.total_moves,'moves',flush=True)
print('Four-case escalation complete',flush=True)
