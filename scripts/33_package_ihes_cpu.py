"""Package a private, bounded Kaggle CPU exact-search run; no credentials in payload."""
from pathlib import Path
import ast
import base64
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"kaggle_notebooks/ihes_exact_additive_cpu"
OUT.mkdir(parents=True,exist_ok=True)
TWS=ROOT/"third_party/twips/third_party/twsearch_legacy"
files={}
for p in (TWS/"src/cpp").rglob("*"):
 if p.is_file() and p.suffix in (".cpp",".h",".cc"):
  files["native/"+p.relative_to(TWS/"src/cpp").as_posix()]=p.read_text(encoding="utf-8")
for name in ("LICENSE","LICENSE.md"):
 if (TWS/name).exists():files["native/"+name]=(TWS/name).read_text(encoding="utf-8")
for name in ("scripts/ihes_pdb_search.cpp","scripts/ihes_twsearch_pdb_hook.h", "scripts/24_twsearch_ladder.py",
             "scripts/32_audit_ihes_pdb.py","scripts/build_picture_cube_kpuzzle_v2.py",
             "src/cayley/__init__.py","src/cayley/puzzle.py","src/cayley/verify.py",
             "data/puzzle_info.json","data/test.csv","data/picture_cube_pieces_v2_sym.tws",
             "data/ihes_pdb/moves.txt","data/ihes_pdb/manifest.json","data/ihes_pdb/positive_control.csv",
             "data/twsearch_rank_L22_20260805.json","submission_ihes.csv"):
 files[name]=(ROOT/name).read_text(encoding="utf-8")
source=files["native/solve.cpp"]
source=source.replace('#include "cmdlineops.h"','#include "cmdlineops.h"\n#include "ihes_twsearch_pdb_hook.h"')
source=source.replace("int v = pt.lookuphindexed(h);", "int v = pt.lookuphindexed(h);\n  if (!ihes_pdb::distances.empty() && invflag == 0 && v <= togo) v = std::max(v, ihes_pdb::lower(posns[sp]));")
needle="int solve(const puzdef &pd, prunetable &pt, const setval p, generatingset *gs) {"
source=source.replace(needle,needle+"\n  ihes_pdb::init(pd);")
files["native/solve.cpp"]=source
files["native/util.h"]=files["native/util.h"].replace("return __builtin_popcount(v);", "return __builtin_popcountll(v);")
payload=gzip.compress(json.dumps(files).encode("utf-8"),mtime=0)
manifest={"payload_sha256":hashlib.sha256(payload).hexdigest(),"baseline_sha256":hashlib.sha256((ROOT/"submission_ihes.csv").read_bytes()).hexdigest(),
          "baseline_total":21870,"target_total":21838,"hardware":"CPU only; no paid services", "wall_limit_seconds":32400,
          "per_pid_seconds":1200,"table_mib_cap":16384,"source_sha256":{name:hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()}}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
setup=f'''import base64,gzip,json,os,hashlib,subprocess,sys,time,signal
from pathlib import Path
work=Path('/kaggle/working/ihes_run');work.mkdir(parents=True,exist_ok=True)
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={manifest['payload_sha256']!r}
for name,content in json.loads(gzip.decompress(payload)).items():
 p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
os.chdir(work)
deadline=time.monotonic()+32400
os.environ['IHES_CORNER_PDB_DIR']=str(work/'data/ihes_pdb')
os.environ['PYTHONUTF8']='1'
print('Payload verified',len(payload),'bytes',flush=True)
'''
build='''sources=['antipode','canon','cmdlineops','filtermoves','findalgo','generatingset','god','index','parsemoves','prunetable','puzdef','readksolve','solve','test','threads','twsearch','util','workchunks','rotations','orderedgs','coset','descsets','ordertree','unrotate','shorten','cmds','beamsearch','totalvar']
command=['g++','-O3','-march=native','-std=c++20','-pthread','-DUSE_PTHREADS','-DUSE_PPQSORT','-DTWSEARCH_VERSION=ihes_pdb_20260906','-Inative','-Iscripts','-Inative/vendor/cityhash/src',*[f'native/{s}.cpp' for s in sources],'native/vendor/cityhash/src/city.cc','-o','data/ihes_pdb/twsearch-pdb']
subprocess.run(command,check=True,timeout=600)
subprocess.run(['g++','-O3','-march=native','-std=c++17','-pthread','scripts/ihes_pdb_search.cpp','-o','data/ihes_pdb/pdb-builder'],check=True,timeout=120)
subprocess.run(['data/ihes_pdb/pdb-builder','--command','build-additive'],check=True,timeout=1800)
subprocess.run([sys.executable,'scripts/32_audit_ihes_pdb.py'],check=True,timeout=300)
print('Build and independent audit passed',flush=True)
'''
control='''subprocess.run([sys.executable,'-u','scripts/24_twsearch_ladder.py','--baseline','data/ihes_pdb/positive_control.csv','--journal','data/ihes_pdb/positive.jsonl','--window-length','10','--full-path-only','--pids','1','--exe','data/ihes_pdb/twsearch-pdb','--threads','1','--memory-mib','512','--cache-dir','data/twsearch_cache','--start-prune-depth','8','--min-depth','8','--max-depth','8','--time-limit','30','--out','positive_control.csv'],check=True,timeout=600)
records=[json.loads(s) for s in Path('data/ihes_pdb/positive.jsonl').read_text().splitlines()]
assert len(records)==1 and records[0]['verdict']=='hit' and len(records[0]['word'])==8
print('Positive control passed: exactly two inserted moves removed',flush=True)
'''
run='''mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
available=mem['MemAvailable']
cg=Path('/sys/fs/cgroup/memory.max')
if cg.exists() and cg.read_text().strip().isdigit():available=min(available,int(cg.read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text()))
threads=min(16,len(os.sched_getaffinity(0)))
cpu=Path('/sys/fs/cgroup/cpu.max')
if cpu.exists():
 quota,period=cpu.read_text().split()
 if quota.isdigit():threads=min(threads,max(1,int(quota)//int(period)))
memory_mib=16384 if available>25*2**30 else 8192 if available>14*2**30 else 4096
print('CPU threads',threads,'available GiB',round(available/2**30,2),'prune MiB',memory_mib,flush=True)
rank=json.loads(Path('data/twsearch_rank_L22_20260805.json').read_text())['ranking']
skip={26,28,268,300,868,296,468,918,336,94}
pids=[str(x['pid']) for x in rank if x['pid'] not in skip][:24]
journal='data/ihes_pdb/kaggle_L22.jsonl'
command=[sys.executable,'-u','scripts/24_twsearch_ladder.py','--baseline','submission_ihes.csv','--journal',journal,'--window-length','22','--full-path-only','--pids',*pids,'--pid-order-file','data/twsearch_rank_L22_20260805.json','--exe','data/ihes_pdb/twsearch-pdb','--threads',str(threads),'--memory-mib',str(memory_mib),'--cache-dir','data/twsearch_cache','--start-prune-depth','11','--min-depth','20','--max-depth','20','--time-limit','1200','--out','submission.csv']
proc=subprocess.Popen(command,start_new_session=True)
try:
 code=proc.wait(timeout=max(1,deadline-time.monotonic()))
except subprocess.TimeoutExpired:
 os.killpg(proc.pid,signal.SIGTERM);proc.wait();code=-15
Path('run_status.json').write_text(json.dumps({'returncode':code,'pids':pids,'threads':threads,'memory_mib':memory_mib}))
subprocess.run([sys.executable,'scripts/24_twsearch_ladder.py','--baseline','submission_ihes.csv','--journal',journal,'--apply-only','--out','submission.csv'],check=True,timeout=120)
# Keep small durable results at the output root, where they can be downloaded independently.
import shutil
for p in [Path(journal),Path(journal).with_suffix('.solver.log'),Path('submission.csv'),Path('run_status.json'),Path('data/ihes_pdb/audit_verified.json')]:
 if p.exists():shutil.copy2(p,Path('/kaggle/working')/p.name)
print('Verified result saved',flush=True)
'''
cells=[{"cell_type":"markdown","metadata":{},"source":["Private IHES exact search. Baseline 21,870. CPU only; nine-hour wall cap. Every accepted shortening is replay-verified. A timeout is not an optimality proof."]}]
for text in (setup,build,control,run):
 ast.parse(text)
 cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(keepends=True)})
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
(OUT/"ihes-exact-additive-pdb-search.ipynb").write_text(json.dumps(notebook),encoding="utf-8")
metadata={"id":"artgor/ihes-exact-additive-pdb-search","title":"IHES exact additive PDB search","code_file":"ihes-exact-additive-pdb-search.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_tpu":False,"enable_internet":False,"dataset_sources":[],"competition_sources":[],"kernel_sources":[]}
(OUT/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
print('Packaged',len(files),'files;',len(payload),'compressed bytes; AST checks passed')
