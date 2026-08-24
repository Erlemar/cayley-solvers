"""Execute the generated notebook's REAL cells on 8 simulated CPU devices.

test_beam_cpu.py exercises the kernel; this exercises the NOTEBOOK -- the config
asserts, the mount/loader cell, the frame transforms, the endgame splice, the
verify-and-discard branch, the per-pid-min submission merge and the report. Those
live only in build_notebook.py's cell strings, so nothing else covers them, and a
typo there costs a whole Kaggle TPU session to discover.

HOW. A throwaway `/kaggle/input` is staged from the real artifact dataset plus a
SYNTHETIC test.csv: three pids whose states are shallow scrambles a tiny CPU beam
can actually reach, under real pid numbers, with a long-but-valid sample_submission
so the baseline-merge path is real too. The cell text is then patched by literal
substitution -- every substitution asserts its anchor, so a renamed constant fails
here rather than silently running the unpatched default.

Deep test.csv states are ~70 moves from solved and no CPU-sized beam touches them,
so using the real ones would exercise only the NOT-FOUND branch.

    .venv/Scripts/python.exe cube555/tpu/test_notebook_cpu.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "kaggle_dataset"
SANDBOX = HERE / "_nbtest"
NB = HERE / "cayleypy-cube555-tpu-beam-q.ipynb"

PIDS = [1034, 1020, 900]
DEPTHS = [7, 8, 6]


def stage() -> tuple[Path, Path]:
    """Synthetic competition data: shallow states + a valid long baseline."""
    comp = SANDBOX / "input" / "cayley-py-555-cube"
    work = SANDBOX / "working"
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    comp.mkdir(parents=True)
    work.mkdir(parents=True)

    info = json.loads((ASSETS / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"])
    G = np.asarray([info["generators"][m] for m in names], dtype=np.int64)
    solved = np.asarray(info["central_state"], dtype=np.int64)
    inv = [names.index(m[1:] if m.startswith("-") else "-" + m) for m in names]

    rows, subs = [], []
    for pid, depth in zip(PIDS, DEPTHS):
        r = np.random.default_rng(1000 + pid)
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
        # Baseline = the inverse scramble, padded with a cancelling pair so it is
        # STRICTLY longer than anything the beam will find -- otherwise the merge's
        # "kept baseline" branch fires and the improvement path goes untested.
        undo = [inv[m] for m in reversed(seq)]
        pad = [0, inv[0]] * 6
        cur = s.copy()
        for m in undo + pad:
            cur = cur[G[m]]
        assert np.array_equal(cur, solved), "staged baseline does not solve"
        subs.append({"initial_state_id": pid,
                     "path": ".".join(names[m] for m in undo + pad)})

    for path, data, cols in [(comp / "test.csv", rows, ["initial_state_id", "initial_state"]),
                             (comp / "sample_submission.csv", subs,
                              ["initial_state_id", "path"])]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    print(f"staged {len(rows)} synthetic pids {PIDS} at depths {DEPTHS}")
    return comp, work


def patch(cells: list[str], comp: Path, work: Path) -> list[str]:
    subs = [
        # mounts -> the real asset dir and the sandbox competition dir
        ('Path("/kaggle/input/cube555-tpu-artifacts")', f'Path(r"{ASSETS}")'),
        ('Path("/kaggle/input/cayley-py-555-cube")', f'Path(r"{comp}")'),
        ('"/kaggle/working/cube555_tpu_results.json"', f'r"{work / "results.json"}"'),
        ('SUB_CSV = "/kaggle/working/submission.csv"', f'SUB_CSV = r"{work / "submission.csv"}"'),
        ('f"/kaggle/working/tree_pid{pid}_k{k}_inv{int(inverted)}.u32"',
         'str(Path(r"' + str(work) + '") / f"tree_pid{pid}_k{k}_inv{int(inverted)}.u32")'),
        # CPU sizing
        ("B_GLOBAL   = 2 * 1024 * 1024", "B_GLOBAL   = 4096"),
        ("NUM_STEPS  = 300", "NUM_STEPS  = 20"),
        ("INTERNAL_BS = 16384", "INTERNAL_BS = 256"),
        ("PIDS       = list(range(1034, 1000, -1))", f"PIDS       = {PIDS}"),
        ("PROGRESS_EVERY = 10", "PROGRESS_EVERY = 0"),
        # bf16 on CPU accumulates in bf16 and is both slow and imprecise here
        ("num_steps=NUM_STEPS, dtype=jnp.bfloat16,", "num_steps=NUM_STEPS, dtype=jnp.float32,"),
    ]
    # FRAMES is deliberately NOT patched: the shipped default [(0, False), (7, True)]
    # already exercises both the identity frame and a real conjugation-plus-inversion,
    # which is exactly what needs covering.
    out = []
    for src in cells:
        for old, new in subs:
            if old in src:
                src = src.replace(old, new)
        out.append(src)
    joined = "\n".join(out)
    for old, _ in subs:
        assert old not in joined, f"substitution anchor survived (not applied?): {old!r}"
    return out


def main() -> int:
    if not NB.exists():
        raise SystemExit(f"{NB} missing -- run build_notebook.py first")
    comp, work = stage()
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"notebook: {len(cells)} code cells")
    cells = patch(cells, comp, work)

    ns: dict = {"__name__": "__main__"}
    labels = ["SETUP", "CONFIG", "LOAD", "SOLVE", "REPORT"]
    for label, src in zip(labels, cells):
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

    # ---- independent check of what the notebook wrote -------------------------
    print("\n===== independent verification " + "=" * 32)
    sub = work / "submission.csv"
    if not sub.exists():
        print("no submission.csv written")
        return 1
    info = json.loads((ASSETS / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"])
    G = {n: np.asarray(v, dtype=np.int64) for n, v in info["generators"].items()}
    solved = np.asarray(info["central_state"], dtype=np.int64)
    tests = {int(r["initial_state_id"]):
             np.asarray([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
             for r in csv.DictReader(open(comp / "test.csv", encoding="utf-8"))}
    base = {int(r["initial_state_id"]): len(r["path"].split("."))
            for r in csv.DictReader(open(comp / "sample_submission.csv", encoding="utf-8"))}

    ok = improved = 0
    rows = list(csv.DictReader(open(sub, encoding="utf-8")))
    for r in rows:
        pid = int(r["initial_state_id"])
        mv = [m for m in r["path"].split(".") if m]
        cur = tests[pid].copy()
        for m in mv:
            cur = cur[G[m]]
        good = np.array_equal(cur, solved)
        ok += int(good)
        better = len(mv) < base[pid]
        improved += int(better)
        print(f"  pid {pid}: {len(mv):3d} moves  baseline {base[pid]:3d}  "
              f"solves={good}  improved={better}")
    complete = len(rows) == len(tests)
    print(f"\n  rows {len(rows)}/{len(tests)}  replay-verified {ok}/{len(rows)}  "
          f"improved on baseline {improved}/{len(rows)}")

    shutil.rmtree(SANDBOX, ignore_errors=True)
    good = ok == len(rows) and complete and improved == len(rows)
    print("VERDICT :", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
