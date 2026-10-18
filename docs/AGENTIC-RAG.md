# Agentic RAG: evidence, benchmark protocol, cut diagnosis

Read this only when tuning APG knobs or diagnosing a cut/abort.
Daily usage lives in `.claude/skills/context-engine/SKILL.md`.

## 1. How the loop works

When `llm.agentic_rag=true`, the rerank step becomes a tool-calling agent loop
(`src/query/reranker.rs`). Tools: `query` (extra vector searches — consumes the
turn budget `agentic_rag_max_turns`), `add_chunks` (commits chunks to the
result), plus optional `grep`/`read` (exact tools — do NOT consume the turn
budget, but DO consume the shared char budget `agentic_rag_max_chunk_chars`).

Guardrails (code, not config): total iterations ≤ `turns*4` (min 12);
history byte cap 1MB (`AGENTIC_HISTORY_BYTE_CAP`); forced
`add_chunks → query → add_chunks` cadence. In practice the char budget usually
bites before high turn counts do.

## 2. Evidence (measured on a mid-size Rust repo, rerank via hosted LLM)

Lookup (single answer): OFF vs ON-1 vs ON-9 — ON adds nothing over OFF for a
question with one answer; ON-9 output was byte-identical to ON-1. Leave OFF.

Multi-file explore ("query flow MCP→readiness→vector→graph→rerank"):
OFF (7 snippets, misses pipeline entry + gate mapping) vs ON-5 (13 snippets,
full coverage, ~1 min) vs ON-9 (12 snippets — drops one chunk ON-5 kept).
Higher budget ≠ better: the agent stops early when satisfied.

Lesson: a turns-only comparison without the OFF baseline is meaningless — it
shows where a query converges, not whether APG is worth enabling.

## 3. Benchmark protocol (kill warm-up + LLM noise)

1. **Use server timing, not wall.** `POST /api/query` returns
   `timing: {embed_ms, search_ms, graph_ms, merge_ms, rerank_ms, total_ms}`.
   `warming:true` = cold shard, discard that run.
2. **Warm-up run first, then measure.** Same question once (loads the shard
   into RAM), discard, then N measured reps. Compare `total_ms`.
3. **LLM variance dominates.** Same question + config can vary ~8x in
   `rerank_ms` across runs. So: ≥3 reps per config, compare **medians**, never
   single runs. `rerank_ms` holds the APG cost; `embed/search/graph` are
   stable — if THOSE differ, it's warm-up, not APG.
4. **Quality is deterministic-ish, latency is not.** Compare result SETS
   (file#line headers) for recall; timing medians for cost.
5. Always include the OFF baseline on the same warmed question.

Run: `scripts/ce-benchmark.sh "question" REPO [off 1 5 9]` — off/turns matrix,
warm-up discarded, 3 reps, timing breakdown, auto-restores config.

Cost model per query ≈ 1 embed + top_k vector + (rerank 1 LLM call)
+ APG ? (turns × LLM calls, each with growing history).

## 4. "Agent got cut" — which layer cut it?

- **The engine cuts itself (by design, not a bug).** 6 exit reasons
  (`LoopExit`): `AgentDone | LlmError | QueryBudget | ChunkCharBudget |
  IterationCap | ByteCap`. Decisive rule: **if ≥1 chunk was committed via
  `add_chunks`, results survive ANY exit** (`UseAccumulated`). Only a cut with
  zero commits falls back to original order. "Cut but results fine" = budget
  tripped after committing — working as designed.
- **MCP 300s abort (real failure).** Clients typically abort a tool after 300s
  with no progress; the server counters with a progress heartbeat every 15s
  (`src/mcp/progress.rs`). Modest turn counts never reach 300s. If an abort
  happens: check the server log for "heartbeat send failed" (worker died
  first), or lower turns.
- **To know for sure:** the exit reason is logged explicitly
  (`RUST_LOG`, "agentic rerank: … stopping"). Never guess — read the log.
