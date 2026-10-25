#!/usr/bin/env bash
# ce-tune.sh — live Context Engine APG tuning. No restart needed:
# PUT /api/config persists settings.json; workers pick it up without restart.
# Benchmarking lives in scripts/ce-benchmark.sh (see docs/AGENTIC-RAG.md).
set -euo pipefail
CE="${CE_URL:-http://127.0.0.1:6699}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

get_cfg() { curl -s -m 10 "$CE/api/config" -o "$TMP/cfg.json"; }
show() {
  get_cfg
  python3 -c "import json;c=json.load(open('$TMP/cfg.json'))['llm'];
for k in ['agentic_rag','agentic_rag_max_turns','agentic_rag_max_chunk_chars','agentic_rag_grep_read','rerank_model','provider']: print(k+'='+json.dumps(c.get(k)))"
}
patch() { # patch <python-expr on dict c>
  python3 -c "import json;p='$TMP/cfg.json';c=json.load(open(p));$1;json.dump(c,open(p,'w'))"
}
put_cfg() { curl -s -m 10 -X PUT "$CE/api/config" -H 'Content-Type: application/json' -d @"$TMP/cfg.json" | head -c 300; echo; }

usage() {
  echo "usage: $0 show | apg on [turns] [chars] | apg off | preset lookup|explore|deep"
  exit 1
}

cmd="${1:-}"; case "$cmd" in
  show) show ;;
  apg)
    onoff="${2:-}"; get_cfg
    if [ "$onoff" = "on" ]; then
      patch "c['agentic_rag']=True;c['agentic_rag_max_turns']=int('${3:-9}');c['agentic_rag_max_chunk_chars']=int('${4:-50000}')"
    elif [ "$onoff" = "off" ]; then
      patch "c['agentic_rag']=False"
    else usage; fi
    put_cfg; show ;;
  preset)
    p="${2:-}"; get_cfg
    case "$p" in
      lookup)  patch "c['agentic_rag']=False" ;;   # single-shot suffices
      explore) patch "c['agentic_rag']=True;c['agentic_rag_max_turns']=5;c['agentic_rag_max_chunk_chars']=50000" ;;
      deep)    patch "c['agentic_rag']=True;c['agentic_rag_max_turns']=9;c['agentic_rag_max_chunk_chars']=50000" ;;
      *) usage ;;
    esac
    put_cfg; echo "preset $p applied:"; show ;;
  *) usage ;;
esac
