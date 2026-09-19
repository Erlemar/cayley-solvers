"""Check complete suffix partitioning and recover a known native solution."""
from pathlib import Path
import csv,importlib.util,itertools,json,os,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,verify_submission
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=PictureCube.load(ROOT/'data/puzzle_info.json')
g={n:np.asarray(v,dtype=np.int64) for n,v in p.generators.items()}
names=list(p.move_names);suffixes=m.canonical_prefixes(2,g,names)
rows=load_submission(ROOT/'submission_ihes.csv');known=list(rows[1]);assert len(known)==8
parent=known[:2]
for fixed in [parent,['f0','-r0'],['-f0','f1']]:
    full={m.perm_of(fixed+list(t),g).tobytes() for t in itertools.product(names,repeat=2)}
    reduced={m.perm_of(fixed+t,g).tobytes() for t in suffixes}
    assert full==reduced and len(reduced)==262
target=m.perm_of(known[2:4],g)
i=next(i for i,t in enumerate(suffixes) if np.array_equal(m.perm_of(t,g),target))
tag=str(time.time_ns());directory=ROOT/'data/ihes_pdb'/f'fixed_prefix_control_{tag}';directory.mkdir()
baseline=directory/'baseline.csv';rows[1]=known+[names[0],m.invert_official(names[0])]
with baseline.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['initial_state_id','path']);w.writerows((pid,'.'.join(path)) for pid,path in sorted(rows.items()))
journal=directory/'control.jsonl';out=ROOT/'submissions'/f'ihes_fixed_prefix_control_{tag}.csv'
env={**os.environ,'IHES_CORNER_PDB_DIR':str(ROOT/'data/ihes_pdb')}
cmd=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/27_prefix_split_ladder.py'),
     '--baseline',str(baseline),'--journal',str(journal),'--pids','1','--path-length','10',
     '--k','4','--fixed-prefix='+'.'.join(parent),'--opening-offset',str(i),'--max-openings','1',
     '--pop-roots','--exe',str(ROOT/'data/ihes_pdb/twsearch-pdb-ranked.exe'),'--threads','4',
     '--memory-mib','8192','--time-limit','60','--out',str(out)]
result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
(directory/'run.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
records=[json.loads(line) for line in journal.read_text().splitlines()]
assert len(records)==1 and records[0]['verdict']=='hit' and records[0]['max_depth']==4
assert len(load_submission(out)[1])<=8
v=verify_submission(p,ROOT/'data/test.csv',out);assert v.n_valid==v.n_total==1003
print('PASS: all 324 suffix words covered by 262 distinct states for three parents; native known solution recovered and 1003 paths replayed.')
print('Control evidence:',directory)
