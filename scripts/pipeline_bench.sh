#!/usr/bin/env bash
# End-to-end index-pipeline benchmark — boots the working-tree binary on an
# ISOLATED RocksDB data dir but the SAME home-anchored settings.json (so API
# keys + embedding cache are shared; phase-1 re-embeds are cache hits and cheap).
#
# Generalizes phase2_bench.sh into a WHOLE-pipeline benchmark: it triggers a
# fresh full rebuild of one repo, measures total trigger->idle wall-clock, prints
# the "PERF SUMMARY full_rebuild" stage breakdown, issues a real first query and
# records its latency, then prints a deterministic call-graph digest (sorted
# node ids + sorted edge endpoint pairs, hashed) for output-invariance diffing.
#
# Repo-agnostic: arguments stay (repo_path, port) so it runs UNCHANGED for
# notepad-ade and the Linux kernel.
#
# Usage:  scripts/pipeline_bench.sh <repo_path> [port]
set -euo pipefail

REPO="${1:?usage: pipeline_bench.sh <repo_path> [port]}"
PORT="${2:-7902}"
URL="http://127.0.0.1:${PORT}"

export LIBCLANG_PATH="${LIBCLANG_PATH:-/c/Program Files/LLVM/bin}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# BIN may be overridden to point at a saved baseline binary; defaults to the
# freshly-built working-tree release binary.
BIN="${BIN:-${DIR}/target/release/context-engine-rs.exe}"
TMP_ROOT="${TMP_ROOT:-D:/projects/Python/ce_pipeline_tmp}"
DATA="${TMP_ROOT}/data"
LOG="${TMP_ROOT}/server.log"

# PERSISTED results: written to a FIXED dir that cleanup never removes, so the
# numbers survive even though TMP_ROOT (the isolated RocksDB data dir) is wiped.
# Override with RESULTS_DIR. One timestamped .txt per run + a copy of the log.
RESULTS_DIR="${RESULTS_DIR:-D:/projects/Python/ce_bench_results}"
RUN_TAG="${RUN_TAG:-$(printf '%s' "$REPO" | tr '/:\\' '___' | tr 'A-Z' 'a-z')_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${RESULTS_DIR}/${RUN_TAG}.txt"
mkdir -p "$RESULTS_DIR"

# Append a line to BOTH stdout and the persisted results file.
emit() { echo "$@"; echo "$@" >>"$RESULTS"; }

# repo_id = urlsafe-base64(no-pad) of normalized (lowercased, backslash) path.
REPO_NORM="$(printf '%s' "$REPO" | tr '/' '\\' | tr 'A-Z' 'a-z')"
REPO_ID="$(printf '%s' "$REPO_NORM" | base64 | tr '+/' '-_' | tr -d '=')"

PID=""
cleanup() {
  local code=$?
  [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
  sleep 2
  # Persist the server log alongside the results BEFORE wiping the temp data dir,
  # so PERF SUMMARY lines and any error trail survive cleanup.
  [ -f "$LOG" ] && cp "$LOG" "${RESULTS_DIR}/${RUN_TAG}.log" 2>/dev/null || true
  rm -rf "$TMP_ROOT" 2>/dev/null || true
  echo "[bench] results persisted: ${RESULTS}"
  exit $code
}
trap cleanup EXIT

rm -rf "$TMP_ROOT" 2>/dev/null || true
mkdir -p "$DATA"

[ -x "$BIN" ] || { echo "ERROR: binary missing: $BIN" >&2; exit 1; }
echo "[bench] repo=${REPO}  repo_id=${REPO_ID}  port=${PORT}  bin=${BIN}"
# Self-describing header in the persisted results file.
{
  echo "# pipeline_bench run"
  echo "repo=${REPO}"
  echo "bin=${BIN}"
  echo "port=${PORT}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "max_ram_symbols_override=${CONTEXT_ENGINE_MAX_RAM_SYMBOLS:-default}"
} >"$RESULTS"

RUST_LOG="context_engine_rs=info,warn" "$BIN" \
  --port "$PORT" --bind 127.0.0.1 --data-dir "$DATA" >"$LOG" 2>&1 &
PID=$!

# Wait for server up.
for i in $(seq 1 60); do
  curl -fsS "${URL}/api/index-status" >/dev/null 2>&1 && break
  sleep 1
done

echo "[bench] triggering full rebuild ..."
# Total wall-clock starts the moment we trigger the rebuild.
T_START=$(date +%s.%N)
curl -fsS -X POST "${URL}/api/repos/${REPO_ID}/rebuild" >/dev/null
sleep 3

# Poll per-repo status until idle with a fresh last_indexed_at.
for i in $(seq 1 7200); do
  body="$(curl -fsS "${URL}/api/repos/${REPO_ID}/status" 2>/dev/null || true)"
  state="$(printf '%s' "$body" | grep -o '"state":"[a-z]*"' | head -1 | sed 's/.*:"//;s/"//' || true)"
  indexed_at="$(printf '%s' "$body" | grep -o '"last_indexed_at":"[^"]*"' | head -1 || true)"
  if [ "$state" = "error" ]; then echo "[bench] ERROR: $body" >&2; exit 1; fi
  if [ "$state" = "idle" ] && [ -n "$indexed_at" ]; then
    echo "[bench] done (${indexed_at})"; break
  fi
  if [ $((i % 10)) -eq 0 ]; then echo "[bench] indexing... (${i}s, state=${state:-?})"; fi
  sleep 1
done
T_IDLE=$(date +%s.%N)
WALL=$(python -c "print('%.1f' % (${T_IDLE} - ${T_START}))" 2>/dev/null || echo "?")

emit "===== TOTAL WALL-CLOCK (trigger -> idle) ====="
emit "[bench] total_wall_clock_s=${WALL}"

emit "===== PERF SUMMARY full_rebuild (stage breakdown) ====="
{ grep -E "PERF SUMMARY full_rebuild" "$LOG" | tail -1 || echo "(no full_rebuild PERF SUMMARY found)"; } | tee -a "$RESULTS"
emit "===== PERF SUMMARY phase2 / stage3 ====="
{ grep -E "PERF SUMMARY (phase2|streaming_index stage3)" "$LOG" | tail -2 || true; } | tee -a "$RESULTS"

# First-query latency: issue a real query against the repo after idle. If
# embedding/rerank keys are absent the call may degrade to an empty result;
# we measure the call latency regardless and DO NOT fail the bench on a
# non-2xx body (the latency itself is the datum).
emit "===== FIRST-QUERY LATENCY (warm shard, same process) ====="
Q_BODY="$(python -c "import json,sys; print(json.dumps({'query':'initialize the main entry point','repo':sys.argv[1],'top_k':10,'rerank':False}))" "$REPO")"
Q_START=$(date +%s.%N)
QRESP="$(curl -sS -X POST "${URL}/api/query" -H 'content-type: application/json' -d "$Q_BODY" 2>/dev/null || true)"
Q_END=$(date +%s.%N)
QLAT=$(python -c "print('%.0f' % ((${Q_END} - ${Q_START})*1000))" 2>/dev/null || echo "?")
emit "[bench] warm_same_process_query_latency_ms=${QLAT}"
emit "[bench] first_query_response_head=${QRESP:0:160}"

# Output-invariance digest: the call-graph endpoint reads calls.in_name/out_name.
# A stable sorted digest (sorted node ids + sorted edge endpoint pairs, hashed)
# lets us prove resolved-edge byte-identity across builds.
emit "===== GRAPH DIGEST (sorted nodes+edges, sha) ====="
GRAPH_JSON="$(curl -fsS "${URL}/api/repos/${REPO_ID}/graph" 2>/dev/null || true)"
{ printf '%s' "$GRAPH_JSON" \
  | python -c "import sys,json,hashlib; d=json.load(sys.stdin); n=sorted(x.get('id','') for x in d.get('nodes',[])); e=sorted((x.get('source',''),x.get('target','')) for x in d.get('edges',[])); blob=repr((n,e)).encode(); print('nodes=%d edges=%d sha=%s'%(len(n),len(e),hashlib.sha256(blob).hexdigest()))" 2>/dev/null \
  || echo "graph digest unavailable: ${GRAPH_JSON:0:120}"; } | tee -a "$RESULTS"

# ── COLD first-query latency (the real "time to query") ──────────────────────
# The query above ran against a WARM shard: full_rebuild populated the vector
# index in-RAM (replace_repo), so it never exercised load_from_db. The metric a
# user actually feels is the FIRST query after opening the tool fresh — a cold
# process that must warm the shard from the DB on demand. We measure that here:
# kill the server, reboot it on the SAME data-dir (no rebuild — the index is
# already durable), and time the first query. This includes shard load_from_db
# warm latency, which is a first-class part of "start indexing -> user can query".
echo "===== COLD-RESTART FIRST-QUERY LATENCY ====="
kill "$PID" 2>/dev/null || true
# Wait for the RocksDB exclusive lock to release before reopening the data-dir.
for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
sleep 2
RUST_LOG="context_engine_rs=info,warn" "$BIN" \
  --port "$PORT" --bind 127.0.0.1 --data-dir "$DATA" >>"$LOG" 2>&1 &
PID=$!
for i in $(seq 1 60); do
  curl -fsS "${URL}/api/index-status" >/dev/null 2>&1 && break
  sleep 1
done
# First query on the freshly-booted process — triggers lazy shard warm.
CQ_START=$(date +%s.%N)
CQRESP="$(curl -sS -X POST "${URL}/api/query" -H 'content-type: application/json' -d "$Q_BODY" 2>/dev/null || true)"
CQ_END=$(date +%s.%N)
CQLAT=$(python -c "print('%.0f' % ((${CQ_END} - ${CQ_START})*1000))" 2>/dev/null || echo "?")
emit "[bench] cold_first_query_latency_ms=${CQLAT}  (includes shard load_from_db warm)"
emit "[bench] cold_first_query_response_head=${CQRESP:0:160}"
# A second query on the now-warm shard, for the warm/cold delta.
WQ_START=$(date +%s.%N)
curl -sS -X POST "${URL}/api/query" -H 'content-type: application/json' -d "$Q_BODY" >/dev/null 2>&1 || true
WQ_END=$(date +%s.%N)
WQLAT=$(python -c "print('%.0f' % ((${WQ_END} - ${WQ_START})*1000))" 2>/dev/null || echo "?")
emit "[bench] cold_restart_warm_second_query_latency_ms=${WQLAT}  (shard already resident)"
# Surface the shard load_from_db warm time straight from the log (authoritative).
SHARD_WARM="$(grep -E "loaded embeddings into VectorIndex" "$LOG" | tail -1 || true)"
emit "[bench] shard_warm_log=${SHARD_WARM:-'(no load_from_db marker — shard may have warmed lazily mid-query)'}"
emit "===== TIME-TO-QUERY SUMMARY ====="
emit "[bench] index_wall_clock_s=${WALL}  cold_first_query_ms=${CQLAT}  cold_warm_second_query_ms=${WQLAT}"
emit "[bench] results_file=${RESULTS}"

