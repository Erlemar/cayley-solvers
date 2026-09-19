"""Continue the validated CPU campaign on fresh PIDs and shuffled exact subtrees."""
from pathlib import Path
import ast
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
WAVE = "20260907_wave2"
if (ROOT / "data/ihes_pdb/cpu_wave2_launch.json").exists():
    raise SystemExit("Wave 2 was already uploaded; preserve its immutable manifest and prepare a new wave instead")
previous = json.loads((ROOT / "data/ihes_pdb/cpu_fleet_manifest.json").read_text())
used = {pid for shard in previous["shards"] for pid in shard["pids"]}
used |= {26,28,268,300,868,296,468,918,336,94}
ranking = json.loads((ROOT / "data/twsearch_rank_L22_20260805.json").read_text())["ranking"]
pids = [row["pid"] for row in ranking if row["pid"] not in used][:120]
assert len(pids) == len(set(pids)) == 120
template_folder = ROOT / "kaggle_notebooks/ihes_exact_cpu_shard2"
template = json.loads((template_folder / "ihes-exact-additive-cpu-shard-2.ipynb").read_text())
base_metadata = json.loads((template_folder / "kernel-metadata.json").read_text())
manifest = {"wave_id": WAVE, "max_concurrent_cpu_notebooks": 5, "baseline_total": 21870,
            "target_total": 21838, "per_pid_seconds": 1200, "wall_limit_seconds": 32400,
            "table_mib": 8192, "random_start": True, "minimum_depth": "native default",
            "shards": []}
for shard in range(1,6):
    ref = previous["shards"][shard-1]["ref"]
    assigned = pids[(shard-1)*24:shard*24]
    notebook = copy.deepcopy(template)
    notebook["cells"][0]["source"] = [f"Private IHES wave 2, CPU shard {shard}/5. Fresh targets, shuffled exact opening subtrees, all depths through 20; nine-hour cap."]
    control = "".join(notebook["cells"][3]["source"])
    control = control.replace("'--threads','1'", "'--random-start','--threads','1'")
    notebook["cells"][3]["source"] = control.splitlines(keepends=True)
    run = "".join(notebook["cells"][4]["source"])
    lines = run.splitlines()
    assert sum(line.startswith("pids=") for line in lines) == 1
    run = "\n".join(f"pids={list(map(str,assigned))!r}" if line.startswith("pids=") else line for line in lines) + "\n"
    assert "'--min-depth','20'," in run
    run = run.replace("'--min-depth','20',", "'--random-start',")
    run = run.replace("{'returncode':code,", f"{{'wave_id':{WAVE!r},'returncode':code,")
    notebook["cells"][4]["source"] = run.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code": ast.parse("".join(cell["source"]))
    folder = ROOT / f"kaggle_notebooks/ihes_wave2_shard{shard}"
    folder.mkdir(parents=True, exist_ok=True)
    codefile = "search.ipynb"
    (folder / codefile).write_text(json.dumps(notebook), encoding="utf-8")
    metadata = {**base_metadata, "id": ref, "title": f"IHES exact CPU wave 2 shard {shard}", "code_file": codefile}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest["shards"].append({"shard": shard, "ref": ref, "pids": assigned, "folder": str(folder),
                              "notebook_sha256": hashlib.sha256((folder / codefile).read_bytes()).hexdigest()})
(ROOT / "data/ihes_pdb/cpu_wave2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Prepared five private CPU notebooks: 120 fresh length-22 targets, no overlap with wave 1")
