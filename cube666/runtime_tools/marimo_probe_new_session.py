import os
import subprocess
import sys
import contextlib
import io

import marimo._code_mode as cm

_help_buffer = io.StringIO()
with contextlib.redirect_stdout(_help_buffer):
    help(cm)
print(_help_buffer.getvalue()[:8000])
print({"cwd": os.getcwd(), "executable": sys.executable})
print(
    subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader",
        ],
        text=True,
    )
)
async with cm.get_context() as ctx:
    print(
        {
            "cells": len(ctx.cells),
            "cell_names": [cell.name for cell in ctx.cells][-16:],
        }
    )
