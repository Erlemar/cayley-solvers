"""Matched native opening-order probe on deliberately inflated known solutions."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/ihes_pdb"
sys.path.insert(0, str(ROOT / "src"))
from cayley.verify import load_submission

paths = load_submission(ROOT / "submission_ihes.csv")
probe = OUT / "opening_probe_baseline.csv"
with probe.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["initial_state_id", "path"])
    for pid, path in paths.items():
        writer.writerow([pid, ".".join(["f0", "-f0"] + path if pid == 30 else path)])

report = []
for ranked in (False, True):
    tag = "ranked" if ranked else "original"
    journal = OUT / f"opening_probe_{tag}.jsonl"
    if journal.exists():
        raise SystemExit(f"Refusing to overwrite existing probe {journal}")
    env = dict(os.environ, IHES_CORNER_PDB_DIR=str(OUT))
    env.pop("IHES_PREFIX_RANK_FILE", None)
    if ranked:
        env["IHES_PREFIX_RANK_FILE"] = str(OUT / "prefix_ranks.txt")
    command = [sys.executable, "-u", "scripts/24_twsearch_ladder.py",
               "--baseline", str(probe), "--journal", str(journal),
               "--window-length", "22", "--full-path-only", "--pids", "30",
               "--exe", str(OUT / "twsearch-pdb-ranked.exe"), "--threads", "4",
               "--memory-mib", "8192", "--min-depth", "20", "--max-depth", "20",
               "--time-limit", "90", "--out", str(OUT / f"opening_probe_{tag}.csv")]
    print("Starting", tag, flush=True)
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        code = process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        # On Windows the venv launcher can have both a Python child and a native
        # grandchild. Stop this owned tree before recording an unresolved probe.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       check=False, capture_output=True)
        process.wait()
        code = -1
    records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    report.append({"ranked": ranked, "returncode": code, "records": records})
    (OUT / "opening_probe_report.json").write_text(json.dumps(report, indent=2))
    if code:
        raise SystemExit(code)
print("This measures rediscovery of a known solution, not improvement over the incumbent.")
