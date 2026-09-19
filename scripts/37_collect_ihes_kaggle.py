"""Collect the approved private run and replay-verify a deterministic min merge."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, verify_submission

REF = "artgor/ihes-exact-additive-pdb-search"
BASELINE_HASH = "0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab"
OUT = ROOT / "submissions/ihes_20260906_kaggle"


def replace_with_retry(source, target, attempts=20):
    """Windows readers/antivirus can briefly prevent an otherwise atomic rename."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def save_status(data):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "collection_status.tmp"
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        replace_with_retry(temp, OUT / "collection_status.json")
    except PermissionError:
        # Preserve both the old valid status and the new complete .tmp snapshot.
        # A status reader must not terminate ongoing result collection.
        print("Status file remains locked; preserved new snapshot in collection_status.tmp", flush=True)
        return False
    return True


def merge_candidates(output=None, extra_candidates=()):
    baseline = ROOT / "submission_ihes.csv"
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != BASELINE_HASH:
        raise RuntimeError("Baseline changed; require a fresh comparison before merging")
    puzzle = PictureCube.load(ROOT / "data/puzzle_info.json")
    rows = load_submission(baseline)
    original = {pid: list(path) for pid, path in rows.items()}
    sources = sorted((OUT / "remote").rglob("submission.csv"))
    if not sources:
        raise RuntimeError("Kaggle produced no final submission.csv; inspect its execution log")
    for name in ("ihes_pdb_ranked_L22_trial.csv", "ihes_pdb_ordered_L24_trial.csv"):
        candidate = ROOT / "submissions" / name
        if candidate.exists():
            sources.append(candidate)
    sources.extend(Path(candidate) for candidate in extra_candidates if Path(candidate).exists())
    provenance = []
    for candidate in sources:
        raw = list(csv.DictReader(candidate.open(encoding="utf-8")))
        paths = load_submission(candidate)
        if len(raw) != len(rows) or set(paths) != set(rows):
            raise RuntimeError(f"Invalid coverage or duplicate IDs in {candidate}")
        report = verify_submission(puzzle, ROOT / "data/test.csv", candidate)
        if not report.all_valid:
            raise RuntimeError(f"Candidate replay failed: {candidate}: {report.failures[:3]}")
        gained = 0
        for pid, path in paths.items():
            if len(path) < len(rows[pid]):
                gained += len(rows[pid]) - len(path)
                rows[pid] = path
        provenance.append({"file": str(candidate), "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                           "total_moves": report.total_moves, "incremental_moves_saved": gained})
    output = output or ROOT / "submissions/ihes_20260906_search_verified.csv"
    temp = output.with_suffix(".tmp.csv")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            writer.writerow([pid, ".".join(rows[pid])])
    report = verify_submission(puzzle, ROOT / "data/test.csv", temp)
    if not report.all_valid:
        raise RuntimeError("Merged submission failed replay verification")
    replace_with_retry(temp, output)
    return {"output": str(output), "total_moves": report.total_moves,
            "valid_paths": report.n_valid, "target_met": report.total_moves <= 21839,
            "moves_saved": 21870 - report.total_moves,
            "improvements": [{"pid": pid, "before": len(original[pid]), "after": len(rows[pid])}
                             for pid in sorted(rows) if len(rows[pid]) < len(original[pid])],
            "sources": provenance}


def download_results(api, directory=None):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests
    directory = (directory or OUT / "remote").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r".*(?:submission\.csv|kaggle_L22\.jsonl|prefix_L(?:22|24)\.jsonl|prefix_timing_gate_passed\.json|run_status\.json|audit_verified\.json|preflight_passed\.json|\.log)$")
    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug = REF.split("/")
    request.page_size = 100
    seen_tokens = set()
    with api.build_kaggle_client() as client:
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            if response.log:
                (directory / "execution.log").write_text(response.log, encoding="utf-8")
            for item in response.files or []:
                if not pattern.search(item.file_name):
                    continue
                destination = (directory / item.file_name).resolve()
                if not destination.is_relative_to(directory):
                    raise RuntimeError("Unsafe remote output filename")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with requests.get(item.url, stream=True, timeout=120) as stream:
                    stream.raise_for_status()
                    size = 0
                    with destination.open("wb") as handle:
                        for chunk in stream.iter_content(65536):
                            size += len(chunk)
                            if size > 20 * 2**20:
                                raise RuntimeError("Unexpectedly large result file")
                            handle.write(chunk)
            token = response.next_page_token
            if not token:
                break
            if token in seen_tokens:
                raise RuntimeError("Repeated Kaggle output pagination token")
            seen_tokens.add(token)
            request.page_token = token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        result = merge_candidates()
        save_status({"status": "VERIFIED", **result})
        print(json.dumps(result), flush=True)
        return
    credentials = Path("C:/Users/and-l/.claude/projects/C--Users-and-l-cayley/memory/ref_kaggle_credentials.md")
    tokens = re.findall(r"KGAT_[A-Za-z0-9_\-]+", credentials.read_text(encoding="utf-8"))
    os.environ["KAGGLE_API_TOKEN"] = tokens[-1]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    deadline = time.monotonic() + 10 * 3600
    previous = None
    errors = 0
    while time.monotonic() < deadline:
        try:
            response = api.kernels_status(REF)
            if isinstance(response, str):
                response = json.loads(response)
            elif not isinstance(response, dict):
                response = response.to_dict()
            status = str(response["status"]).upper()
            errors = 0
        except Exception as exc:
            errors += 1
            print("Status read failed", type(exc).__name__, "attempt", errors, flush=True)
            if errors >= 12:
                raise RuntimeError("Repeated Kaggle status failures; collection incomplete") from None
            time.sleep(60)
            continue
        save_status({"status": status, "kernel": REF, "last_checked_unix": time.time(),
                     "failure_message": response.get("failureMessage")})
        if status != previous:
            print("Kaggle", status, flush=True)
            previous = status
        if status in ("COMPLETE", "ERROR", "CANCELLED", "CANCELED"):
            download_results(api)
            if status != "COMPLETE":
                raise RuntimeError(f"Kaggle ended with {status}; execution log downloaded")
            result = merge_candidates()
            save_status({"status": "VERIFIED", "kernel": REF, **result})
            print(json.dumps(result), flush=True)
            return
        time.sleep(180)
    raise RuntimeError("Collection deadline reached; remote status must be checked manually")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save_status({"status": "COLLECTION_FAILED", "error": str(exc)})
        raise
