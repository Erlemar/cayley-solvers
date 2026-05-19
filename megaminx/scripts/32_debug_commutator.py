"""Debug: trace specific commutators on megaminx to verify parser correctness."""
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))
from megaminx.puzzle import Megaminx

p = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
solved = p.solved_state


def apply_word(word):
    state = solved
    for m in word:
        state = p.apply_move(state, m)
    return state, sum(1 for a, b in zip(state, solved) if a != b)


print("=== Test 1: single commutator [R, D] = R D R' D' ===")
_, h = apply_word(["R", "D", "-R", "-D"])
print(f"  hamming = {h}/120 (NON-zero = commutator is non-identity, expected)")

print()
print("=== Test 2: 4x of [R', D'] = (R' D' R D)^4 ===")
_, h = apply_word(["-R", "-D", "R", "D"] * 4)
print(f"  hamming = {h}/120")

print()
print("=== Test 3: 5x of [R', D'] = (R' D' R D)^5 ===")
_, h = apply_word(["-R", "-D", "R", "D"] * 5)
print(f"  hamming = {h}/120")

print()
print("=== Test 4: cp1 = R' BR' R BR R' F' R BR' R' BR F R ===")
cp1_word = ["-R", "-BR", "R", "BR", "-R", "-F", "R", "-BR", "-R", "BR", "F", "R"]
_, h = apply_word(cp1_word)
print(f"  hamming = {h}/120")

print()
print("=== Test 5: cp1 with face mapping check ===")
# Maybe speedcuber's "BR" = our "FR"? Or some other mapping?
for br_alt in ["BR", "FR", "BL", "DR", "DL", "FL"]:
    word = ["-R", f"-{br_alt}", "R", br_alt, "-R", "-F", "R", f"-{br_alt}", "-R", br_alt, "F", "R"]
    _, h = apply_word(word)
    print(f"  cp1 with BR -> {br_alt}: hamming = {h}/120")

print()
print("=== Test 6: sledgehammer-3x ===")
_, h = apply_word(["R", "U", "-R", "-U"] * 3)
print(f"  (R U R' U')^3 = hamming {h}/120 (KEPT in macros)")

# Period of (R U R' U') on megaminx
print("=== Period check: (R U R' U')^k for k=1..10 ===")
state = solved
for k in range(1, 11):
    state = p.apply_path(state, ["R", "U", "-R", "-U"])
    h = sum(1 for a, b in zip(state, solved) if a != b)
    print(f"  k={k}: hamming = {h}/120{' (returned to identity!)' if h == 0 else ''}")
