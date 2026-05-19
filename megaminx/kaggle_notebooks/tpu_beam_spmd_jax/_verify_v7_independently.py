"""Independently verify v7 paths by running the kernel-reported path against
each pid's init state from test.csv on this machine — no JAX, just numpy."""
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

pi = json.loads((ROOT / "data" / "puzzle_info.json").read_text())
GENS = pi["generators"]
MN = list(GENS.keys())
SOLVED = tuple(pi["central_state"])

rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
results = json.loads((HERE / "_kernel_output_v7" / "jax_spmd_final.json").read_text())

print(f"{'pid':>4}  {'path_len':>8}  {'verify_kernel':>12}  {'verify_local':>12}  {'first_5_moves':>30}")
for r in results:
    pid = r["pid"]
    path_orig = r.get("path_idx_orig") or []
    s0 = [int(x) for x in rows[pid]["initial_state"].split(",")]
    cur = list(s0)
    for m in path_orig:
        gen = GENS[MN[m]]
        cur = [cur[g] for g in gen]
    local_verify = tuple(cur) == SOLVED
    print(f"{pid:>4}  {len(path_orig):>8}  {str(r.get('verify_ok')):>12}  {str(local_verify):>12}  {path_orig[:5]}")

# Also print pid 3-7 init states' first/last 12 elements so we can spot
# if they happen to be very close to V0 in some obvious way.
print()
print("Init states (first 12 + last 12) for pids 0-7:")
for pid in range(8):
    s = [int(x) for x in rows[pid]["initial_state"].split(",")]
    print(f"  pid {pid}: {s[:12]} ... {s[-12:]}")
print(f"  SOLVED: {list(SOLVED)[:12]} ... {list(SOLVED)[-12:]}")

# And the current best (merge_v11) for these pids — if there's a shorter
# path than current best, that's a big win.
best_csv = ROOT / "submissions" / "merge_v11_community_plus_m_dd_v0.csv"
if best_csv.exists():
    best = {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(open(best_csv))}
    print()
    print("Current best (76,251) path lengths for these pids:")
    for pid in range(8):
        p = best.get(pid, "")
        moves = p.split(".") if p.strip() else []
        print(f"  pid {pid}: {len(moves)} moves in current best")
