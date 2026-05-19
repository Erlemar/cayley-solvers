"""Static-validate every code cell in the built .ipynb via ast.parse."""
import ast
import json
import sys
from pathlib import Path

nb_path = Path(__file__).parent / "cayleypy-tpu-beam-spmd-jax.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"cells: {len(cells)}")
errors = 0
for i, cell in enumerate(cells):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] != "code":
        print(f"  cell {i}: {cell['cell_type']} ({len(src)} chars)")
        continue
    try:
        ast.parse(src)
        n_lines = src.count("\n")
        print(f"  cell {i}: code OK ({len(src)} chars, {n_lines} lines)")
    except SyntaxError as e:
        errors += 1
        print(f"  cell {i}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
        for j, line in enumerate(src.splitlines(), 1):
            if e.lineno and abs(j - e.lineno) <= 2:
                print(f"    {j}: {line}")
sys.exit(1 if errors else 0)
