"""Build a private Kaggle CPU script that runs Walton on selected exact 666 states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube666.walton import project_exact_state_to_walton  # noqa: E402


DEFAULT_PIDS = (597, 808, 854, 906)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument("--pids", default=",".join(str(pid) for pid in DEFAULT_PIDS))
    parser.add_argument("--kernel-id", default="artgor/cayley-666-walton-macro-mining")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "kaggle_walton_macro_mining",
    )
    return parser.parse_args()


def render_runner(states: dict[str, str]) -> str:
    state_literal = json.dumps(states, indent=2, sort_keys=True)
    return f'''"""Generated four-PID Walton 6x6 macro-mining job."""
from __future__ import annotations

import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

STATES = {state_literal}
WORK = pathlib.Path("/kaggle/working")
REPO = WORK / "rubiks-cube-NxNxN-solver"
KOCIEMBA = WORK / "kociemba"
OUTPUT = WORK / "walton_runs.json"


def run(command, *, cwd=None):
    print("RUN", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def write_output(payload):
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    temporary.replace(OUTPUT)


payload = {{
    "format_version": 1,
    "pids": sorted(STATES, key=int),
    "runs": {{}},
    "walton_repository": "https://github.com/dwalton76/rubiks-cube-NxNxN-solver",
}}
write_output(payload)

run(["git", "clone", "--depth", "1", "https://github.com/dwalton76/rubiks-cube-NxNxN-solver.git", str(REPO)])
run([
    "gcc", "-O3", "-o", "ida_search_via_graph",
    "rubikscubennnsolver/ida_search_core.c",
    "rubikscubennnsolver/rotate_xxx.c",
    "rubikscubennnsolver/ida_search_666.c",
    "rubikscubennnsolver/ida_search_777.c",
    "rubikscubennnsolver/ida_search_via_graph.c", "-lm",
], cwd=REPO)
run(["git", "clone", "--depth", "1", "https://github.com/dwalton76/kociemba.git", str(KOCIEMBA)])
KOCIEMBA_C = KOCIEMBA / "kociemba" / "ckociemba"
run(["make"], cwd=KOCIEMBA_C)
os.environ["PATH"] = str(KOCIEMBA_C / "bin") + os.pathsep + os.environ["PATH"]

sys.path.insert(0, str(REPO))
os.chdir(REPO)
from rubikscubennnsolver.RubiksCube666 import RubiksCube666

payload["walton_commit"] = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
).strip()
write_output(payload)

for pid in sorted(STATES, key=int):
    started = time.perf_counter()
    print(f"\\n=== PID {{pid}} ===", flush=True)
    try:
        cube = RubiksCube666(STATES[pid], "URFDLB")
        cube.sanity_check()
        cube.solve()
        if not cube.solved():
            raise RuntimeError("Walton solver returned a non-solved colour cube")
        marked = list(cube.solution_with_markers)
        clean = [move for move in cube.solution if not move.startswith("COMMENT")]
        payload["runs"][pid] = {{
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "marked_solution": marked,
            "solution": clean,
            "standard_move_count": len(clean),
            "status": "solved",
        }}
        del cube
        gc.collect()
    except Exception as exc:
        payload["runs"][pid] = {{
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": f"{{type(exc).__name__}}: {{exc}}",
            "status": "error",
            "traceback": traceback.format_exc(),
        }}
    write_output(payload)

# Keep only the compact auditable result as a Kaggle output artifact.
os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(KOCIEMBA, ignore_errors=True)
print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
'''


def main() -> None:
    args = parse_args()
    requested = tuple(int(token) for token in args.pids.split(",") if token.strip())
    rows = {int(state_id): state for state_id, state in Cube666Puzzle.load(
        args.data_dir / "puzzle_info.json"
    ).iter_test_states(args.data_dir / "test.csv")}
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    missing = sorted(set(requested).difference(rows))
    if missing:
        raise KeyError(f"unknown PIDs: {missing}")
    states = {
        str(pid): project_exact_state_to_walton(rows[pid], puzzle.solved_state)
        for pid in requested
    }
    if any(len(state) != 216 for state in states.values()):
        raise AssertionError("projected Walton states must contain 216 face letters")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = args.out_dir / "walton_macro_mining.py"
    metadata = args.out_dir / "kernel-metadata.json"
    runner.write_text(render_runner(states), encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "id": args.kernel_id,
                "title": "Cayley 666 Walton Macro Mining",
                "code_file": runner.name,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"kernel_dir": str(args.out_dir), "pids": requested}, indent=2))


if __name__ == "__main__":
    main()
