import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
r = json.load(open(HERE / "_kernel_output_v2/jax_spmd_final.json"))
for rec in r:
    pid = rec["pid"]
    fs = rec.get("found_step", -1)
    mvt = rec.get("min_v_trajectory", [])
    print(f"--- pid {pid}  found_step={fs}  path_len={rec['path_len']} ---")
    if mvt:
        # Print min_v around found_step
        if fs >= 0:
            lo = max(0, fs - 2)
            hi = min(len(mvt), fs + 3)
            print(f"  min_v_log[{lo}..{hi}]: {[round(v,3) for v in mvt[lo:hi]]}")
            print(f"  min_v_log[{fs}] = {mvt[fs]}  <-- should be V(V0)~0.957 if V0 was in chosen")
        # Also show the trajectory pattern
        unique_v = sorted(set(round(v, 3) for v in mvt if v < 1e3))
        print(f"  unique min_v values seen: {unique_v[:20]}")
