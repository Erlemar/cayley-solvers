"""Compare gap-coordinate symmetry reduction to independent full permutations."""
import json
from pathlib import Path
import random

from sparse_gap_roots import canonical, generate
from verify_two_defect_certificate import orbit, build


def main():
    folder = Path(__file__).parent
    rows, stats = generate(2)
    complete = json.loads((folder / 'two_defect_inflation_complete.json').read_text())
    actual = {min(orbit(tuple(r['permutation']))) for r in rows}
    assert len(actual) == len(rows) == 942
    assert actual == set(map(tuple, complete['roots']))
    d3 = json.loads((folder / 'gap_roots_d3.json').read_text())
    rng = random.Random(3241)
    samples = rng.sample(d3['rows'], 300)
    for row in samples:
        a, alpha, gaps = row['a'], row['core'], row['gaps']
        n, t = row['n'], len(alpha)
        target = min(orbit(build(alpha, gaps, a)))
        inverse = [alpha.index(i) for i in range(t)]
        for inv, source in ((False, alpha), (True, inverse)):
            aa = n - a if inv else a
            for reflect in (False, True):
                bb = n - aa if reflect else aa
                core = [(-source[-i % t]) % t for i in range(t)] if reflect else source
                spaces = [gaps[(-i - 1) % t] for i in range(t)] if reflect else gaps
                for shift in range(t):
                    cp = tuple((core[(i + shift) % t] - shift) % t for i in range(t))
                    gp = tuple(spaces[(i + shift) % t] for i in range(t))
                    assert canonical(bb, cp, gp) == canonical(a, alpha, gaps)
                    assert min(orbit(build(cp, gp, bb))) == target
    report = {'deficit_two': stats, 'matches_complete_independent_root_set': True,
              'deficit_three_sample_orbits': len(samples),
              'scope': 'Complete d=2 comparison; sampled d=3 geometric identities, not full d=3 coverage.'}
    (folder / 'sparse_gap_geometry_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
