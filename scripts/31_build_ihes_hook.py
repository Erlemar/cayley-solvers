"""Build an isolated twsearch binary with an optional IHES additive PDB."""
from pathlib import Path
import subprocess
import os
import argparse

ROOT = Path(__file__).resolve().parents[1]
TWS = ROOT / "third_party/twips/third_party/twsearch_legacy"
OUT = ROOT / "data/ihes_pdb"
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='twsearch-pdb.exe')
args=parser.parse_args()
CXX = Path("C:/Users/and-l/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/g++.exe")
source = (TWS / "src/cpp/solve.cpp").read_text(encoding="utf-8")
source = source.replace('#include "cmdlineops.h"', '#include "cmdlineops.h"\n#include "ihes_twsearch_pdb_hook.h"')
needle = "int v = pt.lookuphindexed(h);"
assert source.count(needle) == 1
source = source.replace(needle, needle + "\n  if (!ihes_pdb::distances.empty() && invflag == 0 && v <= togo)\n    v = std::max(v, ihes_pdb::lower(posns[sp]));")
needle = "int solve(const puzdef &pd, prunetable &pt, const setval p, generatingset *gs) {"
assert source.count(needle) == 1
source = source.replace(needle, needle + "\n  ihes_pdb::init(pd);")
assert source.count("workat = 0;")==1
source=source.replace("workat = 0;", "ihes_pdb::order_chunks(p, workchunks);\n    workat = 0;")
(OUT / "solve-hook.cpp").write_text(source, encoding="utf-8")
env = dict(os.environ, PATH=str(CXX.parent)+os.pathsep+os.environ.get("PATH", ""))
subprocess.run([str(CXX), "-O3", "-march=native", "-std=c++20", "-DUSE_PTHREADS", "-DUSE_PPQSORT",
                "-I"+str(TWS/"src/cpp"), "-I"+str(ROOT/"scripts"), "-c", str(OUT/"solve-hook.cpp"), "-o", str(OUT/"solve-hook.o")], check=True, env=env)
objects = sorted(p for p in (TWS/"build/cpp").glob("*.o") if not p.name.startswith("solve"))
objects += [TWS/"build/cpp/vendor/cityhash/city.o", OUT/"solve-hook.o"]
subprocess.run([str(CXX), "-O3", "-pthread", "-static", "-o", str(OUT/args.output), *map(str,objects)], check=True, env=env)
print("Built", OUT/args.output)
