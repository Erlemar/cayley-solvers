"""Mechanical check of lrx_diameter_bound_answer.md.

Conventions (from the file): words act left-to-right on a list state.
  L: [x0,x1,...,x_{n-1}] -> [x1,...,x_{n-1},x0]
  R: [x0,...,x_{n-1}]    -> [x_{n-1},x0,...,x_{n-2}]
  X: swap positions 0,1
State of pi is [pi(0),...,pi(n-1)]; sorted = [0..n-1].
Positive arc sign = increasing index = direction the XL window moves.
"""
import itertools, random, sys
from collections import deque

def apply_word(state, word):
    s = list(state)
    for g in word:
        if g == 'L':
            s = s[1:] + s[:1]
        elif g == 'R':
            s = s[-1:] + s[:-1]
        elif g == 'X':
            s[0], s[1] = s[1], s[0]
        else:
            raise ValueError(g)
    return s

INV = {'X': 'X', 'L': 'R', 'R': 'L'}
def reduce_word(word):
    st = []
    for g in word:
        if st and st[-1] == INV[g]:
            st.pop()
        else:
            st.append(g)
    return ''.join(st)

def delta(n, i, j):
    d = abs(i - j) % n
    return min(d, n - d)

def cycles_of(f):
    n = len(f); seen = [False] * n; out = []
    for i in range(n):
        if not seen[i] and f[i] != i:
            c = []; j = i
            while not seen[j]:
                seen[j] = True; c.append(j); j = f[j]
            out.append(c)
    return out

def signed_arcs(n, cyc, tie):
    k = len(cyc); d = []
    for j in range(k):
        p, q = cyc[j], cyc[(j + 1) % k]
        fwd = (q - p) % n; back = n - fwd
        if fwd < back or (fwd == back and tie == '+'):
            d.append(fwd)
        else:
            d.append(-back)
    return d

def build_PC(n, cyc, tie):
    """Lemma construction. Returns dict with b, unreduced P, reduced P, F, k, t, both, A, claimed cancellations."""
    k = len(cyc)
    d = signed_arcs(n, cyc, tie)
    F = sum(abs(x) for x in d)
    both = any(x > 0 for x in d) and any(x < 0 for x in d)
    if both:
        jb = next(j for j in range(k) if d[j - 1] < 0 < d[j])
        t = sum(1 for j in range(k) if d[j - 1] < 0 < d[j])
    else:
        jb = 0; t = 0
    b = cyc[jb]
    cyc2 = [(cyc[(jb + i) % k] - b) % n for i in range(k)]
    d2 = [d[(jb + i) % k] for i in range(k)]
    assert cyc2[0] == 0
    ptr = 1; word = []
    def move(target, sign):
        nonlocal ptr
        steps = 0
        while ptr != target:
            if sign > 0:
                ptr = ptr + 1
                if ptr == n: ptr = 1
                word.extend('XL')
            else:
                ptr = ptr - 1
                if ptr == 0: ptr = n - 1
                word.extend('RX')
            steps += 1
        return steps
    A = 0
    for j in range(k - 1):
        A += move(cyc2[j + 1], d2[j]); word.append('X')
    A += move(1, d2[k - 1])
    P = ''.join(word)
    claimed = (k - 1 - t) if both else (k - 1)
    return dict(b=b, P=P, Pred=reduce_word(P), F=F, k=k, t=t, both=both, A=A, d=d, claimed=claimed)

def H(n, c):
    return n if c == 0 else n + delta(n, 0, c) - 2

def rot_word(e, n):
    e %= n
    return 'L' * e if e <= n - e else 'R' * (n - e)

def construct(n, pi, c, tie):
    f = [(pi[i] + c) % n for i in range(n)]
    cyc = cycles_of(f)
    K = len(cyc)
    Fc = sum(delta(n, i, f[i]) for i in range(n))
    parts = [build_PC(n, cy, tie) for cy in cyc]
    sumP = sum(len(p['Pred']) for p in parts)
    best = None
    inc = sorted(parts, key=lambda p: p['b'])
    dec = sorted(parts, key=lambda p: (p['b'] != 0, -p['b']))
    for order in (inc, dec):
        T = 0; word = ''; prev = 0
        for p in order:
            word += rot_word(p['b'] - prev, n); T += delta(n, prev, p['b'])
            word += p['Pred']; prev = p['b']
        word += rot_word(c - prev, n); T += delta(n, prev, c)
        if best is None or len(word) < len(best[0]):
            best = (word, T)
    word, T = best
    return dict(word=word, T=T, K=K, Fc=Fc, sumP=sumP, parts=parts, f=f, cyc=cyc)

def check_perm(n, pi, tie):
    """Run construction for all c; return min length. Raise on any violated claim."""
    P_ = (n * n) // 4
    lengths = []
    for c in range(n):
        r = construct(n, pi, c, tie)
        for p, cy in zip(r['parts'], r['cyc']):
            F, k, t, A = p['F'], p['k'], p['t'], p['A']
            assert len(p['P']) == 2 * A + k - 1
            if p['both']:
                assert A <= F - 2, (n, pi, c, cy, p)
                assert 1 <= t <= k // 2
                assert len(p['Pred']) <= 2 * A + k - 1 - 2 * p['claimed'], (n, pi, c, cy, p)
                assert len(p['Pred']) <= 2 * F - k - 3 + 2 * t   # (5)
            else:
                assert A <= F - 1, (n, pi, c, cy, p)
                assert len(p['Pred']) <= 2 * A + k - 1 - 2 * (k - 1)
                assert len(p['Pred']) <= 2 * F - k - 1            # (5)
            assert len(p['Pred']) <= 2 * F - 3                    # (2)
            fstate = list(r['f'])
            want = list(fstate)
            for pos in cy: want[pos] = pos
            got = apply_word(fstate, 'L' * p['b'] + p['P'] + 'R' * p['b'])
            assert got == want, ("cycle not sorted", n, pi, c, cy, p, got, want)
            got2 = apply_word(fstate, 'L' * p['b'] + p['Pred'] + 'R' * p['b'])
            assert got2 == want
        assert r['T'] <= H(n, c), (n, pi, c, r['T'], H(n, c))
        assert len(r['word']) <= 2 * r['Fc'] - 3 * r['K'] + H(n, c), (n, pi, c)
        assert apply_word(pi, r['word']) == list(range(n)), ("full word fails", n, pi, c, r['word'])
        lengths.append(len(r['word']))
    Fs = [sum(delta(n, i, (pi[i] + c) % n) for i in range(n)) for c in range(n)]
    assert sum(Fs) == n * P_, (n, pi, Fs)
    assert sum(H(n, c) for c in range(n)) == n * n - 2 * n + 2 + P_
    is_shift = all((pi[i] - pi[0]) % n == i for i in range(n))
    bound12 = 2 * P_ + n - 5 + (P_ + 2) // n
    if not is_shift:
        assert min(lengths) <= bound12, (n, pi, min(lengths), bound12)
    return min(lengths)

def bfs_ecc(n):
    start = tuple(range(n)); dist = {start: 0}; q = deque([start])
    while q:
        s = q.popleft(); d = dist[s]
        for g in 'LRX':
            t = tuple(apply_word(s, g))
            if t not in dist:
                dist[t] = d + 1; q.append(t)
    return dist

def main():
    for n in range(4, 401):
        P_ = n * n // 4
        lhs = 2 * P_ + n - 5 + (P_ + 2) // n
        rhs = n * (n - 1) // 2 + (7 * n) // 4 - 5
        assert lhs <= rhs, (n, lhs, rhs)
    print("inequality (1) RHS chain holds for n=4..400")

    for n in range(4, 8):
        dist = bfs_ecc(n)
        diam = max(dist.values())
        P_ = n * n // 4
        bound12 = 2 * P_ + n - 5 + (P_ + 2) // n
        worst = 0
        for pi in itertools.permutations(range(n)):
            for tie in '+-':
                m = check_perm(n, list(pi), tie)
                worst = max(worst, m)
        print(f"n={n}: true diam={diam}, bound (12)={bound12}, max constructed len={worst}, ok={diam <= bound12}")
    rng = random.Random(0)
    for n in range(8, 21):
        P_ = n * n // 4
        bound12 = 2 * P_ + n - 5 + (P_ + 2) // n
        worst = 0
        for _ in range(300):
            pi = list(range(n)); rng.shuffle(pi)
            for tie in '+-':
                worst = max(worst, check_perm(n, pi, tie))
        print(f"n={n}: bound (12)={bound12}, C(n,2)={n*(n-1)//2}, max constructed len over 300 random={worst}")
    n = 8; pi = [0, 2, 4, 6, 1, 3, 5, 7]
    r = construct(n, pi, 0, '+')
    print("section 5 example parts:", [(p['b'], p['Pred'], len(p['Pred'])) for p in r['parts']], "full len", len(r['word']), r['word'])
    w13 = 'L' + 'LXLXRXRX' + 'LL' + 'XLXLXRRX' + 'RRR'
    print("word (13) sorts:", apply_word(pi, w13) == list(range(8)), "len", len(w13))
    n = 13; pi = [(2 * i) % 13 for i in range(13)]
    for c in range(13):
        f = [(pi[i] + c) % n for i in range(n)]
        cyc = cycles_of(f); Fc = sum(delta(n, i, f[i]) for i in range(n))
        assert len(cyc) == 1 and len(cyc[0]) == 12 and Fc == 42, (c, cyc, Fc)
    print("section 6: K_c=1, F_c=42 for all c; 2F-3K =", 2 * 42 - 3)
    lens = [len(construct(13, pi, c, '+')['word']) for c in range(13)]
    print("section 6: actual constructed lengths per c:", lens, "min", min(lens), "C(13,2)=78")

main()
