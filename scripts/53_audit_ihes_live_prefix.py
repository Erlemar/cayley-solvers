"""Audit a completed PID in an unseeded prefix journal, even during a later PID."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission,verify_submission
ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True)
ap.add_argument('--journal',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
puzzle=PictureCube.load(ROOT/'data/puzzle_info.json')
spec=importlib.util.spec_from_file_location('prefix',ROOT/'scripts/27_prefix_split_ladder.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
generators={n:np.asarray(v,dtype=np.int64) for n,v in puzzle.generators.items()}
prefixes=m.canonical_prefixes(2,generators,list(puzzle.move_names));assert len(prefixes)==262
journal_text=args.journal.read_text();lines=journal_text.splitlines()
# Discard an incomplete tail line while another PID is still being written.
if journal_text and not journal_text.endswith('\n'):lines=lines[:-1]
rows=[json.loads(line) for line in lines]
selected=[(i,r) for i,r in enumerate(rows) if r['pid']==args.pid]
assert len(selected)==262 and {r['prefix_id'] for _,r in selected}==set(range(262))
native=args.journal.with_suffix('.solver.log').read_text()
blocks=native.split('\nSolving\n')[1:]
assert len(blocks)>=len(rows)
for i,r in selected:
    assert r['prefix']==prefixes[r['prefix_id']] and r['verdict']=='none'
    assert r['incumbent_len']==22 and r['max_depth']==18 and not r['self_test']
    assert 'No solution found in 18' in blocks[i].splitlines()
    assert 'Search timed out' not in blocks[i]
last=selected[-1][0]
assert len(blocks)>last+1 or 'Twsearch finished.' in blocks[last]
baseline=ROOT/'submission_ihes.csv';digest=hashlib.sha256(baseline.read_bytes()).hexdigest()
assert digest=='0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab'
assert len(load_submission(baseline)[args.pid])==22
verified=verify_submission(puzzle,ROOT/'data/test.csv',baseline)
assert verified.n_valid==verified.n_total==1003
proof=dict(pid=args.pid,incumbent_moves=22,distinct_two_move_prefixes=262,residual_depth=18,
           verdict='optimal_by_full_prefix_coverage_and_parity',baseline_sha256=digest,
           source_journal=str(args.journal),seconds=sum(r['secs'] for _,r in selected),
           settled_record_sha256=hashlib.sha256(json.dumps([r for _,r in selected],sort_keys=True).encode()).hexdigest(),
           settled_native_blocks_sha256=hashlib.sha256('\nSolving\n'.join(blocks[i] for i,_ in selected).encode()).hexdigest(),
           verified_moves=verified.total_moves)
args.output.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
