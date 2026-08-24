#!/bin/bash
# Resumable IHES exact-search worker for replacement fleet nodes.
#
# The bundle is immutable and checksum-verified.  Each worker owns two stable
# global shard ids, pulls their JSONL journals from GCS, and resumes only
# unfinished work.  A preempted VM therefore loses at most one sync interval.

exec > /var/log/ihes-startup.log 2>&1
set -euxo pipefail

md() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.compute.internal/computeMetadata/v1/instance/attributes/$1"
}

BUCKET=$(md bucket)
BUNDLE=$(md bundle)
BUNDLE_SHA256=$(md bundle-sha256)
VM_INDEX=$(md vm-index)
N_VMS=$(md n-vms)
SHARDS=$(md shards-per-vm)
THREADS=$(md threads)
CAMPAIGNS=$(md campaigns)

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential python3-numpy

WORK=/opt/work
EXE=$WORK/tws/build/bin/twsearch
mkdir -p "$WORK" "$WORK/ladder" "$WORK/logs" "$WORK/cache"
cd "$WORK"

usable() {
  [ -x "$1" ] || return 1
  [ "$(stat -c %s "$1" 2>/dev/null || echo 0)" -ge 1000000 ] || return 1
  "$1" --help >/dev/null 2>&1
  local rc=$?
  [ "$rc" -ne 126 ] && [ "$rc" -ne 127 ]
}

# A mid-build preemption can leave a truncated file with its executable bit
# intact, so validate both size and execution before reusing the binary.
if ! usable "$EXE"; then
  rm -rf "$WORK/tws"
  gcloud storage cp "gs://$BUCKET/$BUNDLE" "$WORK/ihes_work.tgz"
  echo "$BUNDLE_SHA256  $WORK/ihes_work.tgz" | sha256sum -c -
  tar xzf "$WORK/ihes_work.tgz" -C "$WORK"
  mkdir -p "$WORK/tws"
  tar xzf "$WORK/tws_src.tgz" -C "$WORK/tws"
  grep -q "sequence of shorter chunks cannot postpone" \
    "$WORK/tws/src/cpp/solve.cpp"
  (cd "$WORK/tws" && make build-cpp TWSEARCH_VERSION=fleet-deadline-fixed -j"$(nproc)")
fi
usable "$EXE"

# Recover this logical worker's durable records before starting any search.
gcloud storage cp "gs://$BUCKET/journals/vm${VM_INDEX}_*.jsonl" \
  "$WORK/ladder/" 2>/dev/null || true

cat > /usr/local/bin/ihes-sync <<SYNC
#!/bin/bash
while true; do
  for f in $WORK/ladder/*.jsonl; do
    [ -e "\$f" ] || continue
    gcloud storage cp "\$f" "gs://$BUCKET/journals/\$(basename "\$f")" \
      >/dev/null 2>&1
  done
  sleep 60
done
SYNC
chmod +x /usr/local/bin/ihes-sync
nohup /usr/local/bin/ihes-sync >/dev/null 2>&1 &

# Keep the small fleet-wide reducer alive on every node.  Concurrent reducers
# write equivalent status objects; last-writer-wins is harmless.
if gcloud storage cp "gs://$BUCKET/aggregator.sh" /usr/local/bin/ihes-agg; then
  chmod +x /usr/local/bin/ihes-agg
  nohup /usr/local/bin/ihes-agg "$BUCKET" >/dev/null 2>&1 &
fi

TOTAL_SHARDS=$((N_VMS * SHARDS))

# Build the depth-10/11-attempt pruning cache once, before concurrent shards.
if ! ls "$WORK"/cache/*.dat >/dev/null 2>&1; then
  echo "f0 r1 d2" | "$EXE" -q -si -M 8192 -t "$(nproc)" \
    --maxdepth 4 --startprunedepth 11 --cachedir "$WORK/cache" \
    "$WORK/data/picture_cube_pieces_v2_sym.tws"
fi
ls -la "$WORK/cache"

# Campaign format (pipe-separated because gcloud metadata treats commas as
# separators): LEN:DEPTH:SECONDS or WLEN:DEPTH:SECONDS for sliding windows.
run_campaign() {
  local spec=$1 depth=$2 limit=$3 tag
  local -a extra
  if [[ $spec == W* ]]; then
    local win=${spec#W}
    tag="W${win}"
    extra=(--window-length "$win" --min-path-length $((win + 2)))
  else
    tag="L${spec}"
    local parity=even
    [ $((spec % 2)) -eq 1 ] && parity=odd
    extra=(--full-path-any --parity "$parity"
           --min-path-length "$spec" --max-path-length "$spec")
  fi

  local -a children=()
  for s in $(seq 0 $((SHARDS - 1))); do
    local shard=$((VM_INDEX * SHARDS + s))
    nohup python3 -u "$WORK/scripts/24_twsearch_ladder.py" \
      --baseline "$WORK/data/base_21870.csv" \
      --journal "$WORK/ladder/vm${VM_INDEX}_${tag}_s${shard}.jsonl" \
      --out "/tmp/${tag}_s${shard}.csv" \
      "${extra[@]}" \
      --min-depth "$depth" --max-depth "$depth" \
      --n-shards "$TOTAL_SHARDS" --shard "$shard" \
      --exe "$EXE" --tws "$WORK/data/picture_cube_pieces_v2_sym.tws" \
      --cache-dir "$WORK/cache" \
      --memory-mib 8192 --threads "$THREADS" --start-prune-depth 11 \
      --time-limit "$limit" --resume \
      > "$WORK/logs/${tag}_s${shard}.log" 2>&1 &
    children+=($!)
    sleep 2
  done
  wait "${children[@]}"
}

IFS='|' read -ra campaign_list <<< "$CAMPAIGNS"
for campaign in "${campaign_list[@]}"; do
  IFS=':' read -r spec depth limit <<< "$campaign"
  echo "=== campaign $spec depth=$depth limit=$limit ==="
  run_campaign "$spec" "$depth" "$limit"
done

for f in "$WORK"/ladder/*.jsonl; do
  [ -e "$f" ] || continue
  gcloud storage cp "$f" "gs://$BUCKET/journals/$(basename "$f")"
done
echo ALL_CAMPAIGNS_DONE
gcloud storage cp /var/log/ihes-startup.log \
  "gs://$BUCKET/logs/vm${VM_INDEX}_startup.log"
