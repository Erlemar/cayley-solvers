"""Prepare private immutable cache data and four disjoint CPU search shards."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle_datasets/ihes_exact_search_cache"
DATA.mkdir(parents=True, exist_ok=True)
sources = [ROOT / "data/twsearch_cache/tws9-picture_cube_pieces_v2_sym-q-8G.dat",
           ROOT / "data/ihes_pdb/corners_v1.bin", ROOT / "data/ihes_pdb/corners_center_sum_v1.bin"]
files = {}
for source in sources:
    destination = DATA / source.name
    if not destination.exists():
        os.link(source, destination)
    with destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    files[source.name] = {"bytes": destination.stat().st_size, "sha256": digest}
    print("Hashed", source.name, flush=True)
cache_manifest = {"files": files, "puzzle": "picture_cube_pieces_v2_sym", "metric": "quarter turn",
                  "source_baseline_sha256": "0073eaa07bd732a88e080d93785ec9f481d09e0d27cbccb5248867158a0d65ab"}
(DATA / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
(DATA / "dataset-metadata.json").write_text(json.dumps({
    "id": "artgor/ihes-additive-exact-search-cache", "title": "IHES additive exact search cache",
    "licenses": [{"name": "CC0-1.0"}],
    "description": "Private machine-generated exact-search cache for the authorized IHES campaign. Includes SHA-256 provenance. No credentials or executable files."
}, indent=2), encoding="utf-8")

base_folder = ROOT / "kaggle_notebooks/ihes_exact_additive_cpu"
base = json.loads((base_folder / "ihes-exact-additive-pdb-search.ipynb").read_text())
manifest = json.loads((base_folder / "manifest.json").read_text())
base_run = "".join(base["cells"][4]["source"])
setup_cache = f'''
import shutil
cache_candidates=list(Path('/kaggle/input').rglob('cache_manifest.json'))
cache_root=None
for candidate in cache_candidates:
 if json.loads(candidate.read_text()).get('files')=={files!r}:
  cache_root=candidate.parent;break
assert cache_root is not None, 'Expected immutable search cache is missing'
for name,record in {files!r}.items():
 source=cache_root/name
 assert source.stat().st_size==record['bytes']
 with source.open('rb') as handle: assert hashlib.file_digest(handle,'sha256').hexdigest()==record['sha256'], name
 target=work/('data/twsearch_cache' if name.endswith('.dat') else 'data/ihes_pdb')/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.symlink_to(source)
print('All uploaded cache hashes verified',flush=True)
'''
build = "".join(base["cells"][2]["source"])
build_lines = [line for line in build.splitlines() if "scripts/ihes_pdb_search.cpp" not in line and "--command','build-additive" not in line]
build = "\n".join(build_lines) + "\n"
control = "".join(base["cells"][3]["source"]).replace("'--memory-mib','512'", "'--memory-mib','8192'").replace("'--start-prune-depth','8'", "'--start-prune-depth','11'")
ranking = json.loads((ROOT / "data/twsearch_rank_L22_20260805.json").read_text())["ranking"]
skip = {26,28,268,300,868,296,468,918,336,94}
ordered = [row["pid"] for row in ranking if row["pid"] not in skip]
fleet = [{"shard": 1, "ref": "artgor/ihes-exact-additive-pdb-search", "pids": ordered[:24], "existing": True}]
for shard in range(2,6):
    pids = ordered[(shard-1)*24:shard*24]
    folder = ROOT / f"kaggle_notebooks/ihes_exact_cpu_shard{shard}"
    folder.mkdir(parents=True, exist_ok=True)
    slug = f"ihes-exact-additive-cpu-shard-{shard}"
    notebook = copy.deepcopy(base)
    notebook["cells"][0]["source"] = [f"Private IHES CPU shard {shard}/5. Disjoint targets, reused verified cache, nine-hour wall cap."]
    notebook["cells"][1]["source"] = ("".join(base["cells"][1]["source"]) + setup_cache).splitlines(keepends=True)
    notebook["cells"][2]["source"] = build.splitlines(keepends=True)
    notebook["cells"][3]["source"] = control.splitlines(keepends=True)
    run = base_run.replace("memory_mib=16384 if available>25*2**30 else 8192 if available>14*2**30 else 4096",
                           "assert available>11*2**30, 'Insufficient RAM for the verified 8 GiB cache'\nmemory_mib=8192")
    needle = "pids=[str(x['pid']) for x in rank if x['pid'] not in skip][:24]"
    assert run.count(needle) == 1
    run = run.replace(needle, f"pids={list(map(str,pids))!r}")
    run += "\nfor name in " + repr(list(files)) + ":\n p=work/('data/twsearch_cache' if name.endswith('.dat') else 'data/ihes_pdb')/name\n if p.is_symlink():p.unlink()\n"
    notebook["cells"][4]["source"] = run.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    (folder / f"{slug}.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    metadata = {"id": f"artgor/{slug}", "title": f"IHES exact additive CPU shard {shard}", "code_file": f"{slug}.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["artgor/ihes-additive-exact-search-cache"], "competition_sources": [], "kernel_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({**manifest, "shard": shard, "pids": pids,
        "table_mib_cap": 8192, "cache": cache_manifest}, indent=2), encoding="utf-8")
    if shard == 2:
        gate = copy.deepcopy(notebook)
        gate["cells"] = gate["cells"][:4]
        gate["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": ["Path('/kaggle/working/preflight_passed.json').write_text(json.dumps({'passed':True,'cache_sha256_verified':True,'audit_passed':True,'positive_control_passed':True}))\n", "print('PREFLIGHT PASSED',flush=True)\n",
                       "for name in " + repr(list(files)) + ":\n p=work/('data/twsearch_cache' if name.endswith('.dat') else 'data/ihes_pdb')/name\n if p.is_symlink():p.unlink()\n"]})
        (folder / "preflight.ipynb").write_text(json.dumps(gate), encoding="utf-8")
        gate_folder = ROOT / "kaggle_notebooks/ihes_cpu_cache_preflight"
        gate_folder.mkdir(parents=True, exist_ok=True)
        (gate_folder / "preflight.ipynb").write_text(json.dumps(gate), encoding="utf-8")
        (gate_folder / "kernel-metadata.json").write_text(json.dumps({**metadata,
            "id": "artgor/ihes-exact-cpu-cache-preflight", "title": "IHES exact CPU cache preflight",
            "code_file": "preflight.ipynb"}, indent=2), encoding="utf-8")
        for cell in gate["cells"]:
            if cell["cell_type"] == "code": ast.parse("".join(cell["source"]))
    fleet.append({"shard": shard, "ref": metadata["id"], "pids": pids, "folder": str(folder)})
assert len({pid for shard in fleet for pid in shard["pids"]}) == 120
fleet_path = ROOT / "data/ihes_pdb/cpu_fleet_manifest.json"
fleet_path.write_text(json.dumps({"max_concurrent_cpu_notebooks": 5, "shards": fleet}, indent=2), encoding="utf-8")
print("Prepared four additional disjoint shards; 120 unique targets including the existing worker", flush=True)
