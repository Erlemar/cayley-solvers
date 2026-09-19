"""Execute the generated 2xT4 notebook's REAL cells against a staged /kaggle/input.

run_multi_gpu.py has its own smoke path; this covers what only lives in the notebook's
cell strings -- the mount discovery, the working-tree staging that the path-relative
anchor builder needs, the subprocess wiring, and the report. A typo there costs a whole
GPU session to discover.

Two devices are simulated as `cuda:0,cuda:0`: it is the orchestration that is under
test, not the second card. `ENDGAME_DEPTH` drops to 4 for the same reason -- two workers
each holding the 1.9 GB d<=5 ball do not fit one 16 GB card (on Kaggle they are on
separate T4s and do). Every substitution asserts its anchor, so a renamed constant fails
here rather than silently running the shipped default.

    .venv/Scripts/python.exe cube555/gpu/test_notebook_local.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
NB = HERE / "cayleypy-cube555-2xt4-beam.ipynb"
SANDBOX = HERE / "_nbtest"
PIDS = [1034, 1020, 900]
DEPTHS = [7, 9, 6]


def stage() -> tuple[Path, Path]:
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    code = SANDBOX / "input" / "cube555-gpu-code"
    comp = SANDBOX / "input" / "cayley-py-555-cube"
    comp.mkdir(parents=True)
    shutil.copytree(HERE / "kaggle_dataset", code)
    assets = PROJECT / "cube555" / "tpu" / "kaggle_dataset"

    info = json.loads((assets / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"])
    G = np.asarray([info["generators"][n] for n in names], dtype=np.int64)
    solved = np.asarray(info["central_state"], dtype=np.int64)
    inv = [names.index(n[1:] if n.startswith("-") else "-" + n) for n in names]

    rows, subs = [], []
    for pid, depth in zip(PIDS, DEPTHS):
        r = np.random.default_rng(3000 + pid)
        s, seq, last = solved.copy(), [], -1
        for _ in range(depth):
            while True:
                m = int(r.integers(0, len(G)))
                if last < 0 or m != inv[last]:
                    break
            s = s[G[m]]
            seq.append(m)
            last = m
        rows.append({"initial_state_id": pid,
                     "initial_state": ",".join(str(int(x)) for x in s)})
        # a valid but deliberately longer baseline, so the "improved" branch is exercised
        undo = [inv[m] for m in reversed(seq)] + [0, inv[0]] * 6
        cur = s.copy()
        for m in undo:
            cur = cur[G[m]]
        assert np.array_equal(cur, solved), "staged baseline does not solve"
        subs.append({"initial_state_id": pid,
                     "path": ".".join(names[m] for m in undo)})
    for path, data, cols in [(comp / "test.csv", rows,
                              ["initial_state_id", "initial_state"]),
                             (comp / "sample_submission.csv", subs,
                              ["initial_state_id", "path"])]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    print(f"staged {len(rows)} synthetic pids {PIDS} at depths {DEPTHS}")
    return code, comp


def patch(cells: list[str], code: Path, comp: Path) -> list[str]:
    assets = PROJECT / "cube555" / "tpu" / "kaggle_dataset"
    work = SANDBOX / "working"
    work.mkdir(parents=True, exist_ok=True)
    subs = [
        ('Path("/kaggle/input/cube555-gpu-code")', f'Path(r"{code}")'),
        ('Path("/kaggle/input/cube555-tpu-artifacts")', f'Path(r"{assets}")'),
        ('Path("/kaggle/input/cayley-py-555-cube")', f'Path(r"{comp}")'),
        ('ROOT = Path("/kaggle/working/cube555_tree")', f'ROOT = Path(r"{work / "tree"}")'),
        ('OUT = "/kaggle/working/submission.csv"', f'OUT = r"{work / "submission.csv"}"'),
        ('WORK = "/kaggle/working/run"', f'WORK = r"{work / "run"}"'),
        ('BEAMS       = "2097152"', 'BEAMS       = "65536"'),
        ('MAX_STEPS   = "300"', 'MAX_STEPS   = "40"'),
        ('INTERNAL_BS = 65536', 'INTERNAL_BS = 16384'),
        ('ENDGAME_DEPTH = 5', 'ENDGAME_DEPTH = 4'),
        ('PIDS = "deepest:8"', f'PIDS = "{",".join(map(str, PIDS))}"'),
        # simulate two devices on the one card this box has
        ('DEVICES = ",".join(f"cuda:{i}" for i in range(len(gpus)))',
         'DEVICES = "cuda:0,cuda:0"'),
    ]
    out = []
    for src in cells:
        for old, new in subs:
            if old in src:
                src = src.replace(old, new)
        out.append(src)
    joined = "\n".join(out)
    for old, _ in subs:
        assert old not in joined, f"substitution anchor survived: {old!r}"
    return out


def main() -> int:
    if not NB.exists():
        raise SystemExit(f"{NB} missing -- run build_notebook.py first")
    code, comp = stage()
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    srcs = patch(["".join(c["source"]) for c in cells], code, comp)
    print(f"notebook: {len(srcs)} code cells")

    ns: dict = {"__name__": "__main__"}
    for label, src in zip(["SETUP", "CONFIG", "ANCHORS", "RUN", "REPORT"], srcs):
        print(f"\n===== {label} " + "=" * (56 - len(label)))
        try:
            exec(compile(src, f"<{label}>", "exec"), ns)
        except SystemExit as e:
            print(f"CELL {label} EXITED: {e}")
            return 1
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"CELL {label} RAISED")
            return 1

    # ---- independent verification of what the notebook wrote -----------------
    print("\n===== independent verification " + "=" * 32)
    sub = Path(ns["OUT"])
    if not sub.exists():
        print("no submission.csv")
        return 1
    assets = PROJECT / "cube555" / "tpu" / "kaggle_dataset"
    info = json.loads((assets / "puzzle_info.json").read_text(encoding="utf-8"))
    G = {n: np.asarray(v, dtype=np.int64) for n, v in info["generators"].items()}
    solved = np.asarray(info["central_state"], dtype=np.int64)
    tests = {int(r["initial_state_id"]):
             np.asarray([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
             for r in csv.DictReader(open(comp / "test.csv", encoding="utf-8"))}
    base = {int(r["initial_state_id"]): len(r["path"].split("."))
            for r in csv.DictReader(open(comp / "sample_submission.csv",
                                         encoding="utf-8"))}
    ok = improved = 0
    rows = list(csv.DictReader(open(sub, encoding="utf-8")))
    for r in rows:
        pid, mv = int(r["initial_state_id"]), [m for m in r["path"].split(".") if m]
        cur = tests[pid].copy()
        for m in mv:
            cur = cur[G[m]]
        good = bool(np.array_equal(cur, solved))
        ok += int(good)
        improved += int(len(mv) < base[pid])
        print(f"  pid {pid}: {len(mv):3d} moves  baseline {base[pid]:3d}  "
              f"solves={good}  improved={len(mv) < base[pid]}")
    good = ok == len(rows) == len(tests) and improved == len(rows)
    print(f"\n  rows {len(rows)}/{len(tests)}  replay-verified {ok}/{len(rows)}  "
          f"improved {improved}/{len(rows)}")
    shutil.rmtree(SANDBOX, ignore_errors=True)
    print("VERDICT :", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
