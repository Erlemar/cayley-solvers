#!/bin/bash
# Publish one compact status object for both monolithic and prefix-split journals.
BUCKET=${1:-ihes-tws-gen0977634337}
while true; do
  : > /tmp/all.jsonl
  for u in $(gcloud storage ls "gs://$BUCKET/journals/" 2>/dev/null | grep '\.jsonl$'); do
    gcloud storage cat "$u" 2>/dev/null >> /tmp/all.jsonl
  done

  prev=$(gcloud storage cat "gs://$BUCKET/status.txt" 2>/dev/null \
           | awk '/^records/{print $2}')
  prev=${prev:-0}

  python3 - "$prev" <<'PY' > /tmp/status.txt 2>/dev/null
import json, sys
from collections import Counter, defaultdict

prev = int(sys.argv[1] or 0)
tot = Counter()
by = defaultdict(Counter)
hits = []
n = 0
for line in open('/tmp/all.jsonl', encoding='utf-8', errors='replace'):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    n += 1
    verdict = r.get('verdict')
    tot[verdict] += 1
    if r.get('window_len') is not None:
        group = f"len {r['window_len']}"
    elif r.get('prefix_id') is not None:
        group = f"prefix k{len(r.get('prefix') or [])}"
    else:
        group = "other"
    by[group][verdict] += 1
    if verdict == 'hit':
        hit = {'pid': r.get('pid'), 'word': r.get('word')}
        if r.get('prefix_id') is not None:
            hit.update(prefix_id=r.get('prefix_id'), prefix=r.get('prefix'),
                       incumbent_len=r.get('incumbent_len'))
        else:
            hit.update(start=r.get('start'), window_len=r.get('window_len'))
        hits.append(hit)

# A short object read must never clobber a larger published aggregate.
if n < prev:
    sys.exit(1)
print('records', n)
print('verdicts', dict(tot))
for group in sorted(by):
    print(' ', group, dict(by[group]))
print('HITS', len(hits))
for hit in hits:
    print('HIT', json.dumps(hit))
PY
  if [ -s /tmp/status.txt ]; then
    gcloud storage cp /tmp/status.txt "gs://$BUCKET/status.txt" >/dev/null 2>&1
  fi
  sleep 180
done
