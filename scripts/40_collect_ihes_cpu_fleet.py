"""Collect at most five approved CPU notebooks into one replay-verified result."""
from pathlib import Path
import importlib.util
import argparse
import json
import os
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, default=ROOT / "data/ihes_pdb/cpu_fleet_manifest.json")
parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions/ihes_20260906_cpu_fleet")
parser.add_argument("--output-file", type=Path, default=ROOT / "submissions/ihes_20260906_fleet_verified.csv")
parser.add_argument("--wave-id")
parser.add_argument("--extra-candidate", action="append", type=Path, default=[])
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("ihes_collector", ROOT / "scripts/37_collect_ihes_kaggle.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
collector.OUT = args.output_dir
fleet = json.loads(args.manifest.read_text())["shards"]
assert len(fleet) <= 5
credentials = Path("C:/Users/and-l/.claude/projects/C--Users-and-l-cayley/memory/ref_kaggle_credentials.md")
os.environ["KAGGLE_API_TOKEN"] = re.findall(r"KGAT_[A-Za-z0-9_\-]+", credentials.read_text(encoding="utf-8"))[-1]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
deadline = time.monotonic() + 10 * 3600
finished = set()
states = {}
result = None
while time.monotonic() < deadline and len(finished) < len(fleet):
    changed = False
    for shard in fleet:
        ref = shard["ref"]
        if ref in finished:
            continue
        try:
            response = api.kernels_status(ref)
            if not isinstance(response, dict):
                response = json.loads(response) if isinstance(response, str) else response.to_dict()
            status = str(response["status"]).upper()
            old = states.get(ref, {}).get("status")
            states[ref] = {"status": status, "failure_message": response.get("failureMessage"),
                           "checked_unix": time.time()}
            if status != old:
                print(ref, status, flush=True)
            if status in ("COMPLETE", "ERROR", "CANCELLED", "CANCELED"):
                collector.REF = ref
                staging = collector.OUT / "staging" / f"shard{shard['shard']}"
                collector.download_results(api, staging)
                if args.wave_id:
                    marker = staging / "run_status.json"
                    if not marker.exists() or json.loads(marker.read_text()).get("wave_id") != args.wave_id:
                        if status == "COMPLETE":
                            states[ref]["status"] = "AWAITING_REQUESTED_WAVE_OUTPUT"
                            continue
                        states[ref]["collection_error"] = "No matching wave output after failure"
                        finished.add(ref)
                        continue
                shutil.copytree(staging, collector.OUT / "remote" / f"shard{shard['shard']}", dirs_exist_ok=True)
                finished.add(ref)
                changed = True
        except Exception as exc:
            states[ref] = {"status": "RETRY", "error_type": type(exc).__name__, "checked_unix": time.time()}
            print(ref, "check/collection retry:", type(exc).__name__, flush=True)
    if changed:
        try:
            result = collector.merge_candidates(args.output_file, args.extra_candidate)
            print("Verified total", result["total_moves"], "saved", result["moves_saved"], flush=True)
        except Exception as exc:
            print("Merge unavailable:", str(exc), flush=True)
            result = {"merge_error": str(exc)}
    collector.save_status({"status": "FINISHED" if len(finished) == len(fleet) else "RUNNING",
                           "max_concurrent_cpu_notebooks": 5, "shards": states,
                           "finished_shards": len(finished), "last_checked_unix": time.time(), "result": result})
    if len(finished) < len(fleet):
        time.sleep(180)
if len(finished) < len(fleet):
    collector.save_status({"status": "COLLECTION_DEADLINE", "shards": states, "result": result})
    raise SystemExit("Ten-hour fleet collection limit reached")
