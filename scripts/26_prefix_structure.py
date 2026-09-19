"""Do short paths open differently from long ones?  (prefix-structure analysis)

Tests a community observation: *"longer paths often share common prefixes, while
optimal paths start from unique or rare prefixes."*

Everything is measured on the population of distinct, replay-verified solutions we
already own -- 151 CSVs x 1003 pids -- with matched controls, because the naive
version of every test here is confounded:

* comparing "best vs the rest" is asymmetric (the rest is a crowd, the best is one
  path), so the primary statistic is per-path `agree(x) = mean LCP(x, y != x)` and
  we ask where the SHORTEST path ranks on it -- a rank that is uniform under H0;
* many of our paths are descendants of each other (window rewriting, bridges,
  ladder), which inflates agreement within lineages -- hence the same-family /
  cross-family split;
* adjacent same-axis moves commute, so two "different" openings can be the same
  move set reordered -- hence the canonicalised variant;
* suffix agreement is measured as a mirror control: if long paths also agree on
  their ENDINGS, the effect is "similar solvers" and not "opening choice".

    python scripts/26_prefix_structure.py
    python scripts/26_prefix_structure.py --roots submissions data/ladder --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube  # noqa: E402
from cayley.verify import load_test_states  # noqa: E402

csv.field_size_limit(10_000_000)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages",
             ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}


# ---------------------------------------------------------------- loading


def discover(roots: list[Path]) -> list[Path]:
    out, seen = [], set()
    for root in roots:
        if root.is_file():
            cand = [root]
        else:
            cand = [p for p in root.rglob("*.csv") if p.is_file()]
        for p in cand:
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return sorted(out)


def load_population(files: list[Path], states: dict[int, tuple[int, ...]],
                    puzzle: PictureCube) -> tuple[dict[int, dict[tuple[str, ...], set[str]]], dict[str, int]]:
    """{pid: {path_tuple: {source_file, ...}}} for verified paths only."""
    gens = {name: np.asarray(perm, dtype=np.int64) for name, perm in puzzle.generators.items()}
    solved = np.asarray(puzzle.solved_state, dtype=np.int64)

    raw: dict[int, dict[tuple[str, ...], set[str]]] = defaultdict(dict)
    file_totals: dict[str, int] = {}
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                rd = csv.reader(f)
                head = next(rd, None)
                if not head or head[:2] != ["initial_state_id", "path"]:
                    continue
                tag = path.name
                rows, total = 0, 0
                for row in rd:
                    if len(row) < 2:
                        continue
                    try:
                        pid = int(row[0])
                    except ValueError:
                        continue
                    txt = row[1].strip()
                    if not txt or pid not in states:
                        continue
                    mv = tuple(txt.split("."))
                    if any(m not in gens for m in mv):
                        break  # different puzzle's alphabet -> drop the whole file
                    raw[pid].setdefault(mv, set()).add(tag)
                    rows += 1
                    total += len(mv)
                if rows:
                    file_totals[tag] = total
        except (OSError, UnicodeDecodeError):
            continue

    # replay-verify every DISTINCT path once
    pop: dict[int, dict[tuple[str, ...], set[str]]] = {}
    n_bad = 0
    for pid, cand in raw.items():
        st0 = np.asarray(states[pid], dtype=np.int64)
        keep: dict[tuple[str, ...], set[str]] = {}
        for mv, srcs in cand.items():
            cur = st0
            for m in mv:
                cur = cur[gens[m]]
            if np.array_equal(cur, solved):
                keep[mv] = srcs
            else:
                n_bad += 1
        if keep:
            pop[pid] = keep
    if n_bad:
        print(f"[warn] dropped {n_bad} distinct paths that do not replay to solved")
    return pop, file_totals


# ---------------------------------------------------------------- path ops


def axis(move: str) -> str:
    return move[1] if move.startswith("-") else move[0]


def canonicalise(mv: tuple[str, ...]) -> tuple[str, ...]:
    """Sort each maximal same-axis run: same-axis layer turns commute."""
    out: list[str] = []
    i = 0
    while i < len(mv):
        j = i
        a = axis(mv[i])
        while j < len(mv) and axis(mv[j]) == a:
            j += 1
        out.extend(sorted(mv[i:j]))
        i = j
    return tuple(out)


def lcp(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def lcs_suffix(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    return lcp(a[::-1], b[::-1])


def family(tag: str) -> str:
    """Coarse lineage bucket from a result-file name."""
    stem = tag[:-4] if tag.endswith(".csv") else tag
    for key in ("twsearch", "ladder", "mitm", "bridge", "crossover", "cross_bridges",
                "plateau", "relation", "sym", "beam", "khoruzhii", "public", "community",
                "kociemba", "koc", "gfn", "niss", "smoke", "medium", "combined", "e5", "e6"):
        if key in stem:
            return key
    return stem.split("_")[0]


# ---------------------------------------------------------------- stats


def spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return float("nan") if dx == 0 or dy == 0 else num / (dx * dy)


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def mean(v) -> float:
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


# ---------------------------------------------------------------- analyses


def analyse_within_pid(pop, *, canon: bool, min_distinct: int = 6):
    """Primary test: does opening-agreement predict path length, within a pid?"""
    rows = []
    for pid, cand in pop.items():
        keys = list(cand)
        if canon:
            merged: dict[tuple[str, ...], set[str]] = {}
            for mv, srcs in cand.items():
                c = canonicalise(mv)
                merged.setdefault(c, set()).update(srcs)
            keys = list(merged)
            cand = merged
        if len(keys) < min_distinct:
            continue
        lens = [len(k) for k in keys]
        if len(set(lens)) < 2:
            continue
        n = len(keys)
        pre = [0.0] * n
        suf = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                p = lcp(keys[i], keys[j])
                s = lcs_suffix(keys[i], keys[j])
                pre[i] += p
                pre[j] += p
                suf[i] += s
                suf[j] += s
        pre = [v / (n - 1) for v in pre]
        suf = [v / (n - 1) for v in suf]

        best_i = min(range(n), key=lambda i: lens[i])
        best_len = lens[best_i]
        tied = [i for i in range(n) if lens[i] == best_len]
        worse = [i for i in range(n) if lens[i] > best_len]

        # rank of the shortest path's agreement among all paths (0 = least agreeing)
        order = sorted(range(n), key=lambda i: pre[i])
        rank_pre = mean(order.index(i) for i in tied) / (n - 1)
        order_s = sorted(range(n), key=lambda i: suf[i])
        rank_suf = mean(order_s.index(i) for i in tied) / (n - 1)

        rows.append(dict(
            pid=pid, n=n, best_len=best_len, max_len=max(lens),
            rho_pre=spearman(lens, pre), rho_suf=spearman(lens, suf),
            rank_pre=rank_pre, rank_suf=rank_suf,
            agree_best=mean(pre[i] for i in tied),
            agree_worse=mean(pre[i] for i in worse) if worse else float("nan"),
            suff_best=mean(suf[i] for i in tied),
            suff_worse=mean(suf[i] for i in worse) if worse else float("nan"),
            first_move_unique=all(
                keys[i][0] not in {keys[j][0] for j in range(n) if lens[j] > best_len}
                for i in tied[:1]),
        ))
    return rows


def analyse_first_move(pop, only: set[int] | None = None):
    """Is the shortest path's opening move used by the longer paths at all?"""
    share_k = {k: [0, 0] for k in (1, 2, 3)}          # [n_shared, n_total] best vs worse
    ctrl_k = {k: [0, 0] for k in (1, 2, 3)}           # matched control: worse vs worse
    for pid, cand in pop.items():
        if only is not None and pid not in only:
            continue
        keys = list(cand)
        if len(keys) < 6:
            continue
        lens = [len(k) for k in keys]
        best_len = min(lens)
        if max(lens) == best_len:
            continue
        best = [k for k in keys if len(k) == best_len]
        worse = [k for k in keys if len(k) > best_len]
        for k in share_k:
            wpref = Counter(w[:k] for w in worse)
            for b in best:
                share_k[k][0] += 1 if b[:k] in wpref else 0
                share_k[k][1] += 1
            # control: each worse path against the OTHER worse paths
            for w in worse:
                rest = wpref.copy()
                rest[w[:k]] -= 1
                ctrl_k[k][0] += 1 if rest[w[:k]] > 0 else 0
                ctrl_k[k][1] += 1
    return share_k, ctrl_k


def analyse_lineage(pop):
    """Same-family vs cross-family opening agreement, for worse paths only."""
    same, cross = [], []
    best_same, best_cross = [], []
    for pid, cand in pop.items():
        keys = list(cand)
        if len(keys) < 6:
            continue
        lens = [len(k) for k in keys]
        best_len = min(lens)
        fam = {k: {family(t) for t in cand[k]} for k in keys}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                p = lcp(keys[i], keys[j])
                shared_fam = bool(fam[keys[i]] & fam[keys[j]])
                involves_best = lens[i] == best_len or lens[j] == best_len
                if lens[i] > best_len and lens[j] > best_len:
                    (same if shared_fam else cross).append(p)
                elif involves_best and not (lens[i] == best_len and lens[j] == best_len):
                    (best_same if shared_fam else best_cross).append(p)
    return dict(worse_same=mean(same), worse_same_n=len(same),
                worse_cross=mean(cross), worse_cross_n=len(cross),
                best_same=mean(best_same), best_same_n=len(best_same),
                best_cross=mean(best_cross), best_cross_n=len(best_cross))


def analyse_cross_pid_prefix(pop, kmax: int = 5, max_slack: int | None = None):
    """THE community claim: a prefix shared by many *different scrambles* is a
    non-optimal opening; a rare/unique prefix signals an optimal path.

    Unit of analysis is one path; the label is its SLACK (len - best known len for
    that pid), which is the non-optimality measure -- absolute length is not, since
    it is dominated by how hard the scramble is.  Popularity counts DISTINCT pids
    (so a pid with 27 stored paths cannot inflate its own prefix), minus self.
    """
    floor = {pid: min(len(p) for p in cand) for pid, cand in pop.items()}
    if max_slack is not None:
        pop = {pid: {p: s for p, s in cand.items() if len(p) - floor[pid] <= max_slack}
               for pid, cand in pop.items()}
        pop = {pid: cand for pid, cand in pop.items() if cand}
    out = {}
    for k in range(1, kmax + 1):
        cnt: Counter = Counter()
        for pid, cand in pop.items():
            for q in {p[:k] for p in cand if len(p) >= k}:
                cnt[q] += 1
        pops, slacks = [], []
        by_bucket: dict[int, list[int]] = defaultdict(list)
        for pid, cand in pop.items():
            for p in cand:
                if len(p) < k:
                    continue
                pop_k = cnt[p[:k]] - 1
                sl = len(p) - floor[pid]
                pops.append(pop_k)
                slacks.append(sl)
                by_bucket[_pop_bucket(pop_k)].append(sl)
        # inverse view: mean popularity of optimal-so-far vs suboptimal paths
        opt_pop = mean(pp for pp, sl in zip(pops, slacks) if sl == 0)
        sub_pop = mean(pp for pp, sl in zip(pops, slacks) if sl > 0)
        # the rule stated as a classifier: P(not optimal | prefix shared with another scramble)
        sh = [sl for pp, sl in zip(pops, slacks) if pp > 0]
        un = [sl for pp, sl in zip(pops, slacks) if pp == 0]
        out[k] = dict(rho=spearman([float(x) for x in pops], [float(x) for x in slacks]),
                      n=len(pops), opt_pop=opt_pop, sub_pop=sub_pop,
                      p_nonopt_shared=mean(1.0 if s > 0 else 0.0 for s in sh), n_shared=len(sh),
                      p_nonopt_unique=mean(1.0 if s > 0 else 0.0 for s in un), n_unique=len(un),
                      buckets={b: (mean(v), len(v)) for b, v in sorted(by_bucket.items())})
    return out


def analyse_paired_opt_vs_plus2(pop, kmax: int = 4, plus: int = 2):
    """Cleanest form of the claim, paired within a scramble.

    Restricted to the near-optimal band, a path with slack >= 2 is *provably* not
    optimal (we hold a shorter one for the same pid).  For every pid that has both a
    floor path and a +`plus` path, ask which of the two has the more popular opening
    across the OTHER scrambles.  Pairing inside a pid removes every difficulty,
    length and pid-level confound; only the opening differs.
    """
    floor = {pid: min(len(p) for p in cand) for pid, cand in pop.items()}
    band = {pid: [p for p in cand if len(p) - floor[pid] <= plus] for pid, cand in pop.items()}
    out = {}
    for k in range(1, kmax + 1):
        cnt: Counter = Counter()
        for pid, paths in band.items():
            for q in {p[:k] for p in paths if len(p) >= k}:
                cnt[q] += 1
        wins = losses = ties = 0
        d_pop = []
        for pid, paths in band.items():
            best = [p for p in paths if len(p) == floor[pid]]
            worse = [p for p in paths if len(p) > floor[pid]]
            if not best or not worse:
                continue
            pb = mean(cnt[p[:k]] - 1 for p in best)
            pw = mean(cnt[p[:k]] - 1 for p in worse)
            d_pop.append(pw - pb)
            if pw > pb:
                wins += 1
            elif pw < pb:
                losses += 1
            else:
                ties += 1
        n = wins + losses
        out[k] = dict(wins=wins, losses=losses, ties=ties,
                      frac=wins / n if n else float("nan"),
                      mean_delta=mean(d_pop),
                      z=(wins - n / 2) / ((n / 4) ** 0.5) if n else float("nan"))
    return out


def _pop_bucket(p: int) -> int:
    for edge in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256):
        if p <= edge:
            return edge
    return 999


def analyse_cross_pid_per_file(files: list[Path], pop, kmax: int = 4, min_rows: int = 500):
    """Same test inside ONE csv at a time -- one path per scramble, as stated."""
    floor = {pid: min(len(p) for p in cand) for pid, cand in pop.items()}
    rows = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                rd = csv.DictReader(f)
                if rd.fieldnames is None or rd.fieldnames[:2] != ["initial_state_id", "path"]:
                    continue
                sub = {}
                for r in rd:
                    txt = r["path"].strip()
                    pid = int(r["initial_state_id"])
                    if txt and pid in floor:
                        sub[pid] = tuple(txt.split("."))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if len(sub) < min_rows:
            continue
        entry = dict(file=path.name, n=len(sub),
                     total=sum(len(v) for v in sub.values()),
                     slack=sum(len(v) - floor[p] for p, v in sub.items()))
        for k in range(1, kmax + 1):
            cnt = Counter(v[:k] for v in sub.values())
            pops = [cnt[v[:k]] - 1 for v in sub.values()]
            slacks = [float(len(v) - floor[p]) for p, v in sub.items()]
            entry[f"rho{k}"] = spearman([float(x) for x in pops], slacks)
            uniq = [s for pp, s in zip(pops, slacks) if pp == 0]
            shared = [s for pp, s in zip(pops, slacks) if pp > 0]
            entry[f"slack_uniq{k}"] = mean(uniq)
            entry[f"slack_shared{k}"] = mean(shared)
            entry[f"n_shared{k}"] = len(shared)
        rows.append(entry)
    return rows


def analyse_per_move_slack(pop, max_slack: int | None = None):
    """Is cross-pid popularity just 'some moves are used more'?  Slack per opening move."""
    floor = {pid: min(len(p) for p in cand) for pid, cand in pop.items()}
    by_move: dict[str, list[int]] = defaultdict(list)
    for pid, cand in pop.items():
        for p in cand:
            sl = len(p) - floor[pid]
            if max_slack is not None and sl > max_slack:
                continue
            by_move[p[0]].append(sl)
    return {m: (mean(v), len(v)) for m, v in sorted(by_move.items(), key=lambda kv: -mean(kv[1]))}


def analyse_slack_profile(pop, min_distinct: int = 6):
    """Mean opening-agreement as a function of slack (len - best_len).

    Guards against reading a monotone law into a non-monotone effect: if agreement
    simply fell with length, "the best path is unique" would be a restatement of
    "the best path is short".
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    buckets_suf: dict[int, list[float]] = defaultdict(list)
    for pid, cand in pop.items():
        keys = list(cand)
        if len(keys) < min_distinct:
            continue
        n = len(keys)
        lens = [len(k) for k in keys]
        best_len = min(lens)
        pre = [0.0] * n
        suf = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                p = lcp(keys[i], keys[j])
                s = lcs_suffix(keys[i], keys[j])
                pre[i] += p
                pre[j] += p
                suf[i] += s
                suf[j] += s
        for i in range(n):
            slack = min(lens[i] - best_len, 12)
            buckets[slack].append(pre[i] / (n - 1))
            buckets_suf[slack].append(suf[i] / (n - 1))
    return {k: (mean(v), len(v), mean(buckets_suf[k])) for k, v in sorted(buckets.items())}


def analyse_opening_value(pop, min_distinct: int = 6, n_draw: int = 40, seed: int = 0):
    """What does the population's FAVOURITE opening cost, vs the best opening?

    MATCHED CONTROL (rule 28): restricting to the modal opening also shrinks the
    sample, and a min over fewer paths is worse for free.  So the modal group is
    scored against random subsets of the SAME SIZE drawn from the same pid, and the
    singleton rate against the pid's own base rate of singleton openings.
    """
    rng = np.random.default_rng(seed)
    tot_best = tot_modal = 0
    tot_rand = 0.0
    n_pid = n_singleton = n_modal_is_best = 0
    base_singleton = []
    for pid, cand in pop.items():
        keys = list(cand)
        if len(keys) < min_distinct:
            continue
        lens = {k: len(k) for k in keys}
        best_len = min(lens.values())
        by_open: dict[str, list[int]] = defaultdict(list)
        for k in keys:
            by_open[k[0]].append(lens[k])
        modal = max(by_open, key=lambda m: len(by_open[m]))
        size = len(by_open[modal])
        tot_best += best_len
        tot_modal += min(by_open[modal])
        all_lens = np.array([lens[k] for k in keys])
        draws = [all_lens[rng.choice(len(all_lens), size=size, replace=False)].min()
                 for _ in range(n_draw)]
        tot_rand += float(np.mean(draws))
        n_pid += 1
        winners = {k[0] for k in keys if lens[k] == best_len}
        if any(len(by_open[w]) == 1 for w in winners):
            n_singleton += 1
        base_singleton.append(sum(1 for g in by_open.values() if len(g) == 1) / len(by_open))
        if modal in winners:
            n_modal_is_best += 1
    return dict(n_pid=n_pid, tot_best=tot_best, tot_modal=tot_modal,
                tot_random_same_size=tot_rand,
                cost_of_modal=tot_modal - tot_best,
                cost_of_random_same_size=tot_rand - tot_best,
                frac_singleton_opening=n_singleton / n_pid,
                frac_singleton_baseline=mean(base_singleton),
                frac_modal_is_best=n_modal_is_best / n_pid)


def analyse_opening_mass(pop, min_distinct: int = 6):
    """Test B as a MASS share, not a hit/miss flag.

    "Shares with >=1 longer path" rewards a concentrated crowd automatically.  Here
    the statistic is: what FRACTION of the longer paths share this path's k-prefix --
    computed for the shortest path, and for each longer path against its peers.
    """
    out = {}
    for k in (1, 2, 3):
        best_share, ctrl_share = [], []
        for pid, cand in pop.items():
            keys = list(cand)
            if len(keys) < min_distinct:
                continue
            lens = {p: len(p) for p in keys}
            bl = min(lens.values())
            best = [p for p in keys if lens[p] == bl]
            worse = [p for p in keys if lens[p] > bl]
            if len(worse) < 3:
                continue
            c = Counter(w[:k] for w in worse)
            for b in best:
                best_share.append(c[b[:k]] / len(worse))
            for w in worse:
                ctrl_share.append((c[w[:k]] - 1) / (len(worse) - 1))
        out[k] = (mean(best_share), len(best_share), mean(ctrl_share), len(ctrl_share))
    return out


def load_proven_optimal(pop) -> set[int]:
    """pids whose incumbent path twsearch proved optimal (full path, no shorter word)."""
    proven: set[int] = set()
    for d in ("data/ladder", "data/ladder_fleet"):
        root = PROJECT / d
        if not root.exists():
            continue
        for jf in root.glob("*.jsonl"):
            for line in open(jf, "r", encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("start") != 0 or r.get("verdict") != "none":
                    continue
                pid = r.get("pid")
                if pid in pop and min(len(k) for k in pop[pid]) == r.get("window_len"):
                    proven.add(pid)
    return proven


def analyse_across_pid(files: list[Path], states, puzzle, names: list[str]):
    """Within ONE csv: do long paths share openings ACROSS pids more than short ones?"""
    out = []
    by_name = {p.name: p for p in files}
    for name in names:
        p = by_name.get(name)
        if p is None:
            continue
        rows = {}
        with open(p, "r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                pid = int(row["initial_state_id"])
                txt = row["path"].strip()
                if txt:
                    rows[pid] = tuple(txt.split("."))
        if not rows:
            continue
        lens = sorted(len(v) for v in rows.values())
        med = lens[len(lens) // 2]
        short = [v for v in rows.values() if len(v) <= med]
        long_ = [v for v in rows.values() if len(v) > med]
        entry = dict(file=name, total=sum(len(v) for v in rows.values()), n=len(rows),
                     median=med, n_short=len(short), n_long=len(long_))
        for k in (1, 2, 3):
            for label, grp in (("short", short), ("long", long_)):
                if len(grp) < 20:
                    entry[f"coll{k}_{label}"] = float("nan")
                    continue
                c = Counter(v[:k] for v in grp)
                n = len(grp)
                # collision probability: chance two random pids share the k-prefix
                coll = sum(m * (m - 1) for m in c.values()) / (n * (n - 1))
                entry[f"coll{k}_{label}"] = coll
                entry[f"top1_{k}_{label}"] = max(c.values()) / n
        out.append(entry)
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["submissions"])
    ap.add_argument("--min-distinct", type=int, default=6)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    states = load_test_states(PROJECT / "data/test.csv")
    roots = [PROJECT / r for r in args.roots]
    files = discover(roots)
    print(f"scanning {len(files)} csv files under {', '.join(args.roots)}")

    # sanity: same-axis layer turns really do commute (the canonical form depends on it)
    for a, b in (("f0", "f1"), ("f1", "f2"), ("r0", "-r2"), ("d0", "d1")):
        s1 = puzzle.apply_path(puzzle.solved_state, [a, b])
        s2 = puzzle.apply_path(puzzle.solved_state, [b, a])
        assert s1 == s2, f"{a} and {b} do not commute"

    pop, file_totals = load_population(files, states, puzzle)
    n_dist = [len(v) for v in pop.values()]
    print(f"pids: {len(pop)}   distinct verified solutions: {sum(n_dist)} "
          f"(median {sorted(n_dist)[len(n_dist)//2]}/pid)")
    best_total = sum(min(len(k) for k in v) for v in pop.values())
    print(f"n-way per-pid min (verified): {best_total}")
    parity_ok = all(len({len(k) % 2 for k in v}) == 1 for v in pop.values())
    print(f"length parity constant within every pid: {parity_ok}")

    report: dict = {"n_pids": len(pop), "best_total": best_total}

    # ---- the community claim, as stated: prefixes shared ACROSS scrambles ----
    print("\n=== 0. ACROSS-SCRAMBLE prefix popularity vs slack (slack = len - best known) ===")
    print("   popularity = how many OTHER pids have a stored path with the same k-prefix")
    for ms in (None, 8, 2):
        cross = analyse_cross_pid_prefix(pop, max_slack=ms)
        lab = "all paths" if ms is None else f"slack <= {ms}"
        print(f"  -- {lab} --")
        for k, e in cross.items():
            print(f"    k={k}: spearman(popularity, slack) = {e['rho']:+.4f}   n={e['n']:6d}"
                  f"   mean pop  optimal {e['opt_pop']:8.2f}  vs non-optimal {e['sub_pop']:8.2f}")
        report[f"cross_pid_slack_{ms}"] = {
            str(k): {kk: vv for kk, vv in e.items() if kk != "buckets"} for k, e in cross.items()}
        if ms == 8:
            print("    mean slack by prefix-popularity bucket (k=3):")
            for b, (m, n) in cross[3]["buckets"].items():
                print(f"      pop <= {b:4d}: mean slack {m:5.2f}   (n={n})")
            print("    the rule as a CLASSIFIER -- P(path is NOT optimal | its k-prefix):")
            for k, e in cross.items():
                print(f"      k={k}: shared with another scramble {e['p_nonopt_shared']*100:5.1f}%"
                      f" (n={e['n_shared']:5d})   unique to this scramble "
                      f"{e['p_nonopt_unique']*100:5.1f}% (n={e['n_unique']:5d})")

    paired = analyse_paired_opt_vs_plus2(pop)
    print("\n=== 0a. PAIRED inside a scramble: optimal path vs a +2 path of the SAME pid ===")
    print("   (a +2 path is provably non-optimal -- we hold a shorter one for that pid)")
    for k, e in paired.items():
        print(f"  k={k}: the non-optimal path has the MORE popular opening in "
              f"{e['wins']}/{e['wins']+e['losses']} pids = {e['frac']*100:5.1f}%"
              f"   (ties {e['ties']}, z={e['z']:+.2f}, mean pop delta {e['mean_delta']:+.3f})")
    report["paired_opt_vs_plus2"] = {str(k): v for k, v in paired.items()}

    per_file = analyse_cross_pid_per_file(files, pop)
    per_file.sort(key=lambda e: e["total"])
    near = [e for e in per_file if e["total"] <= 26000 and e["slack"] > 20]
    print("\n=== 0b. same test inside ONE csv (one path per scramble) ===")
    print("   only near-optimal csvs are informative: the 21870 files have zero slack "
          "variance,\n   and the smoke files (>100k moves) are not solutions in any "
          "meaningful sense.")
    print(f"  {'file':40s} {'total':>7s} {'slack':>6s} {'rho1':>6s} {'rho2':>6s} {'rho3':>6s} "
          f"{'slack|uniq3':>11s} {'slack|shared3':>13s}")
    for e in near[:10]:
        print(f"  {e['file'][:40]:40s} {e['total']:7d} {e['slack']:6d} {e['rho1']:+6.3f} "
              f"{e['rho2']:+6.3f} {e['rho3']:+6.3f} {e['slack_uniq3']:11.2f} "
              f"{e['slack_shared3']:13.2f}")
    for label, grp in (("near-optimal (<=26k)", near), ("all csvs", per_file)):
        for k in (1, 2, 3):
            vals = [e[f"rho{k}"] for e in grp if e[f"rho{k}"] == e[f"rho{k}"]]
            if not vals:
                continue
            pos = sum(1 for v in vals if v > 0)
            print(f"  {label:22s} k={k}: mean rho {mean(vals):+.4f}, "
                  f"positive in {pos}/{len(vals)} files")
    report["cross_pid_per_file"] = per_file

    pm = analyse_per_move_slack(pop, max_slack=8)
    items = list(pm.items())
    print("\n=== 0c. is it just 'some moves are commoner'?  mean slack per opening move "
          "(slack <= 8) ===")
    print("  worst 4: " + "  ".join(f"{m}:{v[0]:.2f}(n={v[1]})" for m, v in items[:4]))
    print("  best  4: " + "  ".join(f"{m}:{v[0]:.2f}(n={v[1]})" for m, v in items[-4:]))
    report["per_move_slack"] = {m: v for m, v in pm.items()}

    for canon in (False, True):
        rows = analyse_within_pid(pop, canon=canon, min_distinct=args.min_distinct)
        tag = "canonical" if canon else "raw"
        rho = [r["rho_pre"] for r in rows if r["rho_pre"] == r["rho_pre"]]
        rho_s = [r["rho_suf"] for r in rows if r["rho_suf"] == r["rho_suf"]]
        pos = sum(1 for v in rho if v > 0)
        print(f"\n=== A. within-pid: agreement-with-others vs path length [{tag}] "
              f"({len(rows)} pids, >= {args.min_distinct} distinct) ===")
        print(f"  spearman(len, PREFIX agreement)  mean {mean(rho):+.3f}   "
              f"positive on {pos}/{len(rho)} pids ({pos/len(rho)*100:.0f}%)")
        print(f"  spearman(len, SUFFIX agreement)  mean {mean(rho_s):+.3f}   [mirror control]")
        print(f"  mean prefix agreement:  shortest {mean(r['agree_best'] for r in rows):.3f} moves"
              f"   vs longer {mean(r['agree_worse'] for r in rows if r['agree_worse'] == r['agree_worse']):.3f}")
        print(f"  mean suffix agreement:  shortest {mean(r['suff_best'] for r in rows):.3f} moves"
              f"   vs longer {mean(r['suff_worse'] for r in rows if r['suff_worse'] == r['suff_worse']):.3f}")
        print(f"  normalised rank of the SHORTEST path on prefix agreement: "
              f"{mean(r['rank_pre'] for r in rows):.3f}   (0.5 = no effect, <0.5 = more unique)")
        print(f"  same rank on suffix agreement:                            "
              f"{mean(r['rank_suf'] for r in rows):.3f}   [mirror control]")
        report[f"within_{tag}"] = dict(
            n_pids=len(rows), rho_prefix=mean(rho), rho_suffix=mean(rho_s),
            frac_positive=pos / len(rho) if rho else float("nan"),
            agree_best=mean(r["agree_best"] for r in rows),
            rank_pre=mean(r["rank_pre"] for r in rows),
            rank_suf=mean(r["rank_suf"] for r in rows))

    share_k, ctrl_k = analyse_first_move(pop)
    print("\n=== B. does the shortest path's opening appear among the longer ones? ===")
    for k in sorted(share_k):
        s, t = share_k[k]
        cs, ct = ctrl_k[k]
        print(f"  first {k} move(s): shortest shares with >=1 longer path "
              f"{s}/{t} = {s/t*100:5.1f}%      matched control (longer vs longer) "
              f"{cs}/{ct} = {cs/ct*100:5.1f}%")
    report["first_move"] = {str(k): dict(best=share_k[k], control=ctrl_k[k]) for k in share_k}

    proven = load_proven_optimal(pop)
    if proven:
        ps, pc = analyse_first_move(pop, only=proven)
        print(f"\n=== B2. same test on the {len(proven)} pids whose incumbent twsearch "
              f"PROVED optimal ===")
        for k in sorted(ps):
            s, t = ps[k]
            cs, ct = pc[k]
            if t and ct:
                print(f"  first {k} move(s): optimal shares {s}/{t} = {s/t*100:5.1f}%"
                      f"      control (longer vs longer) {cs}/{ct} = {cs/ct*100:5.1f}%")
        cross_p = analyse_cross_pid_prefix({p: pop[p] for p in proven})
        report["proven_optimal"] = dict(n=len(proven),
                                        share={str(k): ps[k] for k in ps},
                                        control={str(k): pc[k] for k in pc})

    prof = analyse_slack_profile(pop)
    print("\n=== A2. opening agreement vs slack (is the effect monotone in length?) ===")
    print(f"  {'slack':>6s} {'n paths':>8s} {'prefix agree':>13s} {'suffix agree':>13s}")
    for s, (m, n, ms) in prof.items():
        print(f"  {s:6d} {n:8d} {m:13.3f} {ms:13.3f}")
    report["slack_profile"] = {str(k): v for k, v in prof.items()}

    om = analyse_opening_mass(pop)
    print("\n=== B3. same test as MASS share (removes the concentrated-crowd artefact) ===")
    for k, (bs, nb, cs, nc) in om.items():
        print(f"  k={k}: share of longer paths that open the same way -- "
              f"shortest {bs*100:5.1f}%   vs a longer path against its peers {cs*100:5.1f}%"
              f"   (n={nb}/{nc})")
    report["opening_mass"] = {str(k): v for k, v in om.items()}

    ov = analyse_opening_value(pop)
    print("\n=== A3. what the population's FAVOURITE opening costs ===")
    print(f"  best achievable total over {ov['n_pid']} pids                 : {ov['tot_best']}")
    print(f"  best total if restricted to the modal opening       : {ov['tot_modal']}"
          f"   (+{ov['cost_of_modal']})")
    print(f"  CONTROL: random opening subset of the same size      : "
          f"{ov['tot_random_same_size']:.0f}   (+{ov['cost_of_random_same_size']:.0f})")
    print(f"  pids where a winning opening is used by exactly ONE stored path: "
          f"{ov['frac_singleton_opening']*100:.1f}%"
          f"   (base rate of singleton openings {ov['frac_singleton_baseline']*100:.1f}%)")
    print(f"  pids where the modal opening also achieves the best length     : "
          f"{ov['frac_modal_is_best']*100:.1f}%")
    report["opening_value"] = ov

    lin = analyse_lineage(pop)
    print("\n=== C. lineage control: mean LCP by pair type ===")
    print(f"  longer vs longer, SAME solver family : {lin['worse_same']:.3f}  (n={lin['worse_same_n']})")
    print(f"  longer vs longer, CROSS family       : {lin['worse_cross']:.3f}  (n={lin['worse_cross_n']})")
    print(f"  shortest vs longer, SAME family      : {lin['best_same']:.3f}  (n={lin['best_same_n']})")
    print(f"  shortest vs longer, CROSS family     : {lin['best_cross']:.3f}  (n={lin['best_cross_n']})")
    report["lineage"] = lin

    picks = sorted(file_totals.items(), key=lambda kv: kv[1])
    names = [picks[0][0], picks[len(picks) // 4][0], picks[len(picks) // 2][0], picks[-3][0]]
    acr = analyse_across_pid(files, states, puzzle, names)
    print("\n=== D. across-pid opening concentration within ONE csv "
          "(collision prob. of the k-prefix; random baseline 1/18^k) ===")
    print(f"  {'file':46s} {'total':>7s} {'k=1 short':>10s} {'k=1 long':>9s} "
          f"{'k=2 short':>10s} {'k=2 long':>9s}")
    for e in acr:
        print(f"  {e['file'][:46]:46s} {e['total']:7d} {e['coll1_short']:10.4f} "
              f"{e['coll1_long']:9.4f} {e['coll2_short']:10.4f} {e['coll2_long']:9.4f}")
    print(f"  {'random baseline':46s} {'-':>7s} {1/18:10.4f} {1/18:9.4f} "
          f"{1/324:10.4f} {1/324:9.4f}")
    report["across_pid"] = acr

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
