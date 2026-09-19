"""Small exact computations; moves rotate four consecutive positions left/right."""
from collections import deque
import json
from pathlib import Path


def move(s, i, direction=1):
    x = s[i:i+4]
    x = x[1:] + x[:1] if direction == 1 else x[-1:] + x[:-1]
    return s[:i] + x + s[i+4:]


def bfs(n):
    e = bytes(range(n))
    target = e[::-1]
    seen = {e: (None, None)}
    queue = deque([e])
    while queue:
        s = queue.popleft()
        if s == target:
            path = []
            while seen[s][0] is not None:
                prev, a = seen[s]
                path.append(a)
                s = prev
            return path[::-1], len(seen)
        for i in range(n-3):
            for d in (1, -1):
                t = move(s, i, d)
                if t not in seen:
                    seen[t] = (s, (i, d))
                    queue.append(t)
    return None, len(seen)


if __name__ == '__main__':
    results = {}
    for n in range(5, 10):
        path, count = bfs(n)
        print(n, len(path), path, 'states', count, flush=True)
        results[n] = {'length': len(path), 'path': path, 'states': count}
    Path(__file__).with_name('linear_bfs.json').write_text(json.dumps(results, indent=2))
