#!/usr/bin/env bash
# ce-benchmark.sh — APG off/turns matrix with server timing. See docs/AGENTIC-RAG.md.
# Auto-restores config. Usage: ce-benchmark.sh "question" REPO [off 1 5 9]
set -euo pipefail
CE="${CE_URL:-http://127.0.0.1:6699}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

[ $# -ge 2 ] || { echo "usage: $0 \"question\" REPO [off 1 5 9]"; exit 1; }
q="$1"; repo="$2"; shift 2
cfgs=("$@"); [ ${#cfgs[@]} -eq 0 ] && cfgs=(off 1 5 9)

curl -s -m 10 "$CE/api/config" -o "$TMP/cfg.json"; cp "$TMP/cfg.json" "$TMP/orig.json"
python3 -c "import json;print(json.dumps({'query':'''$q''','repo':'''$repo''','top_k':10,'rerank':True}))" > "$TMP/query.json"
for c in "${cfgs[@]}"; do
  if [ "$c" = "off" ]; then
    python3 -c "import json;p='$TMP/cfg.json';s=json.load(open(p));s['llm']['agentic_rag']=False;json.dump(s,open(p,'w'))"
  else
    python3 -c "import json;p='$TMP/cfg.json';s=json.load(open(p));s['llm']['agentic_rag']=True;s['llm']['agentic_rag_max_turns']=$c;json.dump(s,open(p,'w'))"
  fi
  curl -s -m 10 -X PUT "$CE/api/config" -H 'Content-Type: application/json' -d @"$TMP/cfg.json" -o /dev/null
  totals=""
  for rep in 1 2 3 4; do
    curl -s -m 180 -X POST "$CE/api/query" -H 'Content-Type: application/json' -d @"$TMP/query.json" -o "$TMP/out.json"
    line=$(python3 -c "import json;r=json.load(open('$TMP/out.json'));t=r.get('timing',{});print(','.join(map(str,[t.get('total_ms'),t.get('embed_ms'),t.get('search_ms'),t.get('graph_ms'),t.get('rerank_ms'),r.get('warming'),len(r.get('results',[]))])))")
    if [ "$rep" -eq 1 ]; then warm="warmup[$line]"; else totals="$totals $line"; fi
  done
  echo "cfg[$c] $warm"
  echo "cfg[$c] reps (total,embed,search,graph,rerank,warming,n):$totals"
  grep -o '[A-Za-z_/.-]*\.[a-z]*#L[0-9]*-[0-9]*' "$TMP/out.json" | head -n 8 | tr '\n' ' '; echo
  echo "---"
done
curl -s -m 10 -X PUT "$CE/api/config" -H 'Content-Type: application/json' -d @"$TMP/orig.json" -o /dev/null
echo "config restored to original."
