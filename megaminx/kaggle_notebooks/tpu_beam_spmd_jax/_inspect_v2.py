import json, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
r = json.load(open(HERE / "_kernel_output_v2/jax_spmd_final.json"))
rows = list(csv.DictReader(open(ROOT / "data/test.csv")))
pi = json.load(open(ROOT / "data/puzzle_info.json"))
GENS = pi["generators"]
MN = list(GENS.keys())
SOLVED = tuple(pi["central_state"])
for rec in r:
    pid = rec["pid"]
    print(f"--- pid {pid} ---")
    print(f"  found={rec['found']}  verify={rec['verify_ok']}  found_step={rec.get('found_step')}  path_len={rec['path_len']}")
    pr = rec.get("path_idx_rotated", [])
    po = rec.get("path_idx_orig", [])
    print(f"  path_idx_rotated={pr}")
    print(f"  path_idx_orig={po}")
    # Apply rotated path to initial state and verify
    s0 = [int(x) for x in rows[pid]["initial_state"].split(",")]
    cur = list(s0)
    for m in pr:
        gen = GENS[MN[m]]
        cur = [cur[g] for g in gen]
    print(f"  applied {len(pr)} rotated moves -> SOLVED? {tuple(cur) == SOLVED}")
    print(f"  applied state first 12: {cur[:12]}")
    mvt = rec.get("min_v_trajectory", [])
    if mvt:
        print(f"  min_v traj first 5: {[round(v,3) for v in mvt[:5]]}  last 3: {[round(v,3) for v in mvt[-3:]]}")
