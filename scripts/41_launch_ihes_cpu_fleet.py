"""Launch four authorized CPU shards only after private cache preflight succeeds."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data/ihes_pdb/cpu_fleet_launch_state.json"
state = {"stage": "WAITING_FOR_CACHE", "launches": {}, "max_concurrent_cpu_notebooks": 5}


def save(stage):
    state["stage"] = stage
    state["updated_unix"] = time.time()
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)
    print(stage, flush=True)


def as_dict(response):
    if isinstance(response, dict):
        return response
    return json.loads(response) if isinstance(response, str) else response.to_dict()


def push(api, folder):
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and not metadata["enable_gpu"] and not metadata["enable_tpu"]
    digest = hashlib.sha256((folder / metadata["code_file"]).read_bytes()).hexdigest()
    response = as_dict(api.kernels_push(str(folder)))
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    for key in ("invalidDatasetSources", "invalid_dataset_sources", "invalidKernelSources", "invalid_kernel_sources"):
        if response.get(key):
            raise RuntimeError(f"Invalid notebook source: {response[key]}")
    state["launches"][metadata["id"]] = {"response": response, "notebook_sha256": digest}
    save("UPLOADED " + metadata["id"])


def main():
    if STATE_PATH.exists():
        previous = json.loads(STATE_PATH.read_text())
        if previous.get("launches"):
            raise RuntimeError("Existing launches must be inspected before retrying to avoid duplicate jobs")
        state["previous_attempt"] = previous
    save("WAITING_FOR_CACHE")
    credentials = Path("C:/Users/and-l/.claude/projects/C--Users-and-l-cayley/memory/ref_kaggle_credentials.md")
    os.environ["KAGGLE_API_TOKEN"] = re.findall(r"KGAT_[A-Za-z0-9_\-]+", credentials.read_text(encoding="utf-8"))[-1]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    deadline = time.monotonic() + 90 * 60
    while time.monotonic() < deadline:
        # The private dataset has no readable metadata until the uploader commits
        # it; Kaggle answers 403 rather than 404 during this expected interval.
        uploaded = ROOT / "data/ihes_pdb/cache_upload_response.txt"
        if not uploaded.exists():
            time.sleep(60)
            continue
        upload_response = json.loads(uploaded.read_text())
        if upload_response.get("error"):
            raise RuntimeError("Private cache upload failed: " + str(upload_response["error"]))
        try:
            status = api.dataset_status("artgor/ihes-additive-exact-search-cache")
            if status == "ready":
                break
            if status == "error":
                raise RuntimeError("Private cache dataset processing failed")
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            if getattr(getattr(exc, "response", None), "status_code", None) not in (403, 404, 429, 500, 502, 503, 504):
                raise
        time.sleep(60)
    else:
        raise RuntimeError("Cache upload/readiness exceeded the 90-minute wait")
    save("CACHE_READY")
    gate_ref = "artgor/ihes-exact-cpu-cache-preflight"
    push(api, ROOT / "kaggle_notebooks/ihes_cpu_cache_preflight")
    deadline = time.monotonic() + 30 * 60
    while time.monotonic() < deadline:
        response = as_dict(api.kernels_status(gate_ref))
        status = str(response["status"]).upper()
        if status in ("COMPLETE", "ERROR", "CANCELLED", "CANCELED"):
            break
        time.sleep(30)
    else:
        raise RuntimeError("Preflight exceeded 30 minutes; additional workers were not launched")
    spec = importlib.util.spec_from_file_location("collector", ROOT / "scripts/37_collect_ihes_kaggle.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.REF = gate_ref
    gate_output = ROOT / "data/ihes_pdb/cpu_gate_output"
    collector.download_results(api, gate_output)
    marker = gate_output / "preflight_passed.json"
    if status != "COMPLETE" or not marker.exists():
        raise RuntimeError("Cache preflight failed; inspect cpu_gate_output/execution.log")
    passed = json.loads(marker.read_text())
    if not all(passed.get(key) is True for key in ("passed", "cache_sha256_verified", "audit_passed", "positive_control_passed")):
        raise RuntimeError("Preflight marker lacks required validation results")
    state["preflight"] = passed
    save("PREFLIGHT_PASSED")
    fleet = json.loads((ROOT / "data/ihes_pdb/cpu_fleet_manifest.json").read_text())["shards"]
    assert len(fleet) == 5 and len({pid for shard in fleet for pid in shard["pids"]}) == 120
    for shard in fleet:
        if not shard.get("existing"):
            push(api, Path(shard["folder"]))
    state["worker_statuses"] = {shard["ref"]: as_dict(api.kernels_status(shard["ref"])) for shard in fleet}
    save("FOUR_ADDITIONAL_WORKERS_UPLOADED")
    log = (ROOT / "data/ihes_pdb/cpu_fleet_collector.log").open("w", encoding="utf-8")
    errors = (ROOT / "data/ihes_pdb/cpu_fleet_collector.err.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/40_collect_ihes_cpu_fleet.py")],
                               cwd=ROOT, stdout=log, stderr=errors, creationflags=subprocess.CREATE_NO_WINDOW)
    state["collector_pid"] = process.pid
    log.close()
    errors.close()
    save("FLEET_RUNNING_AND_COLLECTION_STARTED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state["error"] = str(exc)
        save("LAUNCH_FAILED")
        raise
