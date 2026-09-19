import json
from pathlib import Path

import numpy as np


root = Path("/marimo/storage/cube666_fullpath_primmoves_clean_v1")
with np.load(root / "teacher.npz", allow_pickle=False) as payload:
    arrays = {
        key: {"shape": payload[key].shape, "dtype": str(payload[key].dtype)}
        for key in payload.files
    }
print({"arrays": arrays, "report": json.loads((root / "report.json").read_text())})
