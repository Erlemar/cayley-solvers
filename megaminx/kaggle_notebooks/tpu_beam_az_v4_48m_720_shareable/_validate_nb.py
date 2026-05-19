"""Static-validate every code cell in the built .ipynb via ast.parse."""
import ast, json, sys
from pathlib import Path

nb_path = Path(__file__).parent / "cayleypy-megaminx-beam-az-v4-48m-720-shareable-jax.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
errors = 0
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] != "code":
        print(f"  cell {i}: {cell['cell_type']} ({len(src)} chars)")
        continue
    try:
        ast.parse(src)
        print(f"  cell {i}: code OK ({len(src)} chars, {src.count(chr(10))} lines)")
    except SyntaxError as e:
        errors += 1
        print(f"  cell {i}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
sys.exit(1 if errors else 0)
