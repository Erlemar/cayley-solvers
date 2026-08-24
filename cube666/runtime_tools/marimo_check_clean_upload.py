import base64
import hashlib
import json
from pathlib import Path


_path = Path("/marimo/storage/cube666_fullpath_primmoves_clean_v1/payload.b64")
_text = _path.read_text() if _path.exists() else ""
_raw = base64.b64decode(_text) if _text else b""
print(
    json.dumps(
        {
            "characters": len(_text),
            "decoded": len(_raw),
            "sha256": hashlib.sha256(_raw).hexdigest(),
        }
    )
)
