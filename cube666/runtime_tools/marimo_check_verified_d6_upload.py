import base64
import hashlib
from pathlib import Path


root = Path("/marimo/storage/cube666_verified_d6_v1")
payload = root / "payload.b64"
text = payload.read_text() if payload.exists() else ""
raw = base64.b64decode(text) if text else b""
print(
    {
        "characters": len(text),
        "decoded": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "files": sorted(
            (str(path.relative_to(root)), path.stat().st_size)
            for path in root.rglob("*")
            if path.is_file()
        ),
    }
)
