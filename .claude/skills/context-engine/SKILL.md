# Context Engine skill

Semantic code search via a Context Engine MCP server (`POST /mcp`,
Streamable HTTP). Full API reference: `docs/MCP-API.md`.
APG tuning method: `docs/AGENTIC-RAG.md`.

## 1. Query SOP (every codebase question)

1. `workspace_full_path` = absolute repo root, copy from cwd — never guess.
2. Location unknown / overview / before-edit → `codebase-retrieval` (ONE call,
   gather all related symbols). Known identifier + all occurrences → grep.
   Known file + unknown section → `file-retrieval`. Known file + line → read.
3. `scope`: implementation/where/calls → omit (default `"code"`);
   why/convention/setup/spec → `scope:"docs"`.
4. After retrieval: read the returned ranges (expanded) → then edit.
5. Respect prefixes: `(resolving the call graph…)` = no callers/callees yet;
   `(being updated…)` = stale snapshot, verify before editing;
   `[scope fallback]` = pass the served scope explicitly next time.
6. On `Codebase is indexing, use grep instead`: grep now, retry later —
   never spam MCP calls (each burns the `mcp_index_wait_secs` budget).
7. Pre-check a repo: `GET /api/repos`, match the (lowercased) `repo`,
   expect `state` `active`/`indexed` + a fresh `last_indexed_at`.

## 2. Agentic RAG — when and how much

Off by default. When `llm.agentic_rag=true`, reranking runs an agent loop with
`query` (extra searches, consumes the turn budget) + `add_chunks` (commits)
+ optional `grep`/`read` (exact tools, do NOT consume the turn budget).
Every MCP query then costs extra LLM calls — only enable when single-shot
rerank misses cross-file context.

Knobs (`llm.*`, all live via `PUT /api/config` — see §3):

| Knob | Default | Meaning |
|---|---|---|
| `agentic_rag` | false | on/off |
| `agentic_rag_max_turns` | 9 | budget = number of `query` calls (UI caps 1–20) |
| `agentic_rag_max_chunk_chars` | 50000 | char budget on committed chunks; 0 = off |
| `agentic_rag_grep_read` | true | give the agent grep/read alongside query |

Presets (evidence in `docs/AGENTIC-RAG.md`):

| Purpose | Setting |
|---|---|
| single symbol lookup ("where is X") | **OFF** — same answer, zero extra LLM calls |
| multi-file explore ("how does flow work") | **ON, turns 5**, chars 50000 (~1 min, best recall) |
| deep-dive unknown codebase | **ON, turns 9** (drop to 5 if results look thin) |
| exact-name hunt | OFF, use grep |
| docs/why questions | OFF + `scope:"docs"` |

Decision rule (default OFF, escalate, de-escalate):

1. Start every question OFF. It answers lookups fully and costs 1 rerank call.
2. Escalate ONLY when OFF comes back thin: few files for a question that
   should span files, a rerank-rejected/empty answer, or missing
   caller/callee coverage.
3. After the explore, turn APG off again — never leave it ON across unrelated
   questions.

## 3. Live config (no restart)

`PUT /api/config` persists `settings.json`; workers pick up changes without a
restart. APG knobs apply on the next query. Scripts:

```bash
scripts/ce-tune.sh show                       # current llm.* knobs
scripts/ce-tune.sh apg on [turns] [chars]     # enable APG, defaults 9 / 50000
scripts/ce-tune.sh apg off                    # back to single-shot rerank
scripts/ce-tune.sh preset lookup|explore|deep # presets from §2
scripts/ce-benchmark.sh "question" REPO       # off/turns matrix, see docs/AGENTIC-RAG.md
```

PUT replaces the whole settings object — the scripts GET first, patch `llm.*`,
PUT back, so unrelated settings survive. Invalid values are rejected with 400
and leave disk untouched.
