from pathlib import Path
from shutil import copy2


source = Path("/marimo/storage/cube666_verified_d6_import_patch_v1/macro_policy.py")
target = Path("/marimo/storage/cube666_verified_d6_v1/macro_policy.py")
copy2(source, target)
print({"source": str(source), "target": str(target), "bytes": target.stat().st_size})
