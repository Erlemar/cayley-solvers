"""Exact eccentricities in small colour quotients; not Cayley diameters."""
import json
import math
from pathlib import Path
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict


@njit(cache=True)
def scan(counts, capacity):
    n = np.sum(counts)
    start = np.uint64(0)
    pos = 0
    for color in range(len(counts)):
        for _ in range(counts[color]):
            start |= np.uint64(color) << np.uint64(2 * pos)
            pos += 1
    distances = Dict.empty(types.uint64, types.int16)
    distances[start] = np.int16(0)
    queue = np.empty(capacity, np.uint64)
    queue[0] = start
    head, tail = 0, 1
    histogram = np.zeros(1000, np.int64)
    histogram[0] = 1
    maximum = 0
    farthest = start
    while head < tail:
        state = queue[head]
        distance = distances[state]
        head += 1
        for begin in range(n):
            for direction in (-1, 1):
                new = state
                for offset in range(4):
                    source = (begin + (offset + direction) % 4) % n
                    target = (begin + offset) % n
                    value = (state >> np.uint64(2 * source)) & np.uint64(3)
                    mask = np.uint64(3) << np.uint64(2 * target)
                    new = (new & ~mask) | (value << np.uint64(2 * target))
                if new not in distances:
                    assert tail < capacity
                    distances[new] = np.int16(distance + 1)
                    queue[tail] = new
                    tail += 1
                    histogram[distance + 1] += 1
                    if distance + 1 > maximum:
                        maximum = distance + 1
                        farthest = new
    assert tail == capacity
    return maximum, histogram[:maximum + 1], farthest


def main():
    rows = []
    for counts in ([7, 8], [5, 5, 5], [4, 4, 7]):
        n = sum(counts)
        capacity = math.factorial(n) // math.prod(math.factorial(c) for c in counts)
        assert capacity <= 1000000
        begin = time.monotonic()
        radius, histogram, farthest = scan(np.array(counts, np.int64), capacity)
        example = [(int(farthest) >> (2 * i)) & 3 for i in range(n)]
        row = {'counts': counts, 'states': capacity, 'base_eccentricity': int(radius),
               'histogram': list(map(int, histogram)), 'farthest_example': example,
               'seconds': time.monotonic() - begin,
               'scope': 'Exact quotient eccentricity only; not the diameter of S_n.'}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
