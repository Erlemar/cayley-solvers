import json
import os
import platform

info = {
    "cwd": os.getcwd(),
    "python": platform.python_version(),
    "hostname": platform.node(),
}

try:
    import torch

    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        info["cuda_devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gib": round(
                    torch.cuda.get_device_properties(i).total_memory / 2**30, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
except Exception as exc:
    info["torch_error"] = repr(exc)

print(json.dumps(info, indent=2))
