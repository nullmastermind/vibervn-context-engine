# MCP API reference (agents + scripts)

Companion to the README. Covers the MCP transport, the two search tools, the
REST API, and every degrade message an agent may receive.

## Transports

| Surface | Endpoint | Protocol |
|---|---|---|
| MCP (global) | `POST /mcp` | Streamable HTTP: JSON-RPC request, SSE response (`data:` lines) |
| MCP (per-repo) | `ANY /mcp-repo/:sanitized_name` | same; `workspace_full_path` pre-bound, tools take no path arg |
| REST | `/api/*` | plain JSON, no session |
| Web UI | `GET /` | serves the dashboard |

### MCP handshake

1. `POST /mcp` with `{"method":"initialize",...}` + header
   `Accept: application/json, text/event-stream`.
2. Server replies SSE + assigns header `mcp-session-id`.
3. Every later call (`tools/list`, `tools/call`) must carry that
   `mcp-session-id` header. The response is always SSE — parse `data:` lines,
   not raw JSON.
4. Sessions self-heal: idle-dropped workers are transparently restored from a
   disk-persisted session store, so a stale id does not 404.

`:sanitized_name` for `/mcp-repo/` = repo path with non-alphanumerics → `_`,
trimmed, last 64 chars (NOT base64 — that encoding is only for REST `:repo_id`).

## MCP tools (exactly 2)

### codebase-retrieval — semantic code search

| Param | Required | Notes |
|---|---|---|
| `information_request` | yes | natural-language question |
| `workspace_full_path` | yes | **absolute repo root**, must exist as a directory |
| `filter_kind` | no | e.g. `["function","class"]` |
| `filter_lang` | no | e.g. `["rust","typescript"]` |
| `filter_path` | no | path substring, e.g. `"src/mcp"` |
| `scope` | no | `"code"` (default: source+config) or `"docs"` (markdown-style docs). One scope per call, no "both" |

Returns snippets as `path#Lx-y`. **Snippets may be incomplete** — read the line
range (expanded) before editing. The index reflects the current disk only, no
version-control history.

Use when: the file location is unknown, an overview is needed, or before
editing (one call gathering all related symbols, not many small calls).
Do NOT use when: an exact identifier and all its occurrences are needed (use
grep); file+line are already known (read the file directly).

### file-retrieval — snippets inside a known file

| Param | Required | Notes |
|---|---|---|
| `workspace_full_path` | yes | absolute repo root |
| `file_path` | yes | **relative** to the repo root, e.g. `"src/mcp.rs"` |
| `information_request` | yes | what to find in the file |
| `top_k` | no | default 5 |

Use instead of reading the whole file when the relevant section of a large
file is unknown. Then read the returned range before editing.

## workspace_full_path rules (error source #1)

- Must be an absolute, existing directory. Otherwise:
  `Error: workspace '...' does not exist or is not a directory.`
- Normalized on Windows: lowercase + backslashes (`C:/x` ≡ `c:\x`); trailing
  separators stripped.
- An unregistered repo is **auto-registered and its index triggered**, but the
  first call waits up to `mcp_index_wait_secs` (default **50s**) then degrades.
  Check first: `GET /api/repos`, match the (lowercased) `repo`, expect `state`
  `active`/`indexed` (not `not_indexed`) and a fresh `last_indexed_at`.

## MCP degrade messages (prose, not codes)

| Output | Meaning | Action |
|---|---|---|
| `Codebase is indexing, use grep instead.` | wait budget expired, nothing servable | grep now, retry later; don't spam calls (each burns the wait budget) |
| `Error: indexing failed (…). Use grep…` | index run failed (e.g. embedding key) | grep now, fix the cause in `…`, don't blind-retry |
| `(index is resolving the call graph; …)` prefix | phase-2 edge resolution: vectors final, **no callers/callees yet** | use results but don't conclude about the call graph; re-query later if graph needed |
| `(index is being updated; …)` prefix | answer from the previous snapshot (stale while re-indexing) | fine for reading; verify before editing from it |
| `[scope fallback] …` prefix | requested scope empty, other scope searched once | pass the served scope explicitly to search it directly |
| `Error: workspace_full_path is required…` | param missing/empty | pass the absolute repo root |
| `Error: no embedding API keys configured…` | missing embedding key | add a key in the Context Engine UI |

REST equivalents: indexing-in-progress → **503**
`{"status":"warming","retry":true}`; failed run → **502**
`{"status":"error","retry":false}`.

## REST endpoints

### Query (the index over HTTP)

- `POST /api/query` — body `{query, repo, top_k=30, rerank=true}`.
  Same pipeline as MCP but exposes **`top_k`** and **`rerank`** knobs MCP lacks.
  The response JSON includes a `graph_pending` flag.
- `POST /api/mcp-tool` — body `{information_request, workspace_full_path,
  scope}`. Response `{"result":"..."}` is **byte-identical** to the MCP tool
  output for the same inputs.
- `POST /api/mcp-tool/file-retrieval` — body `{workspace_full_path,
  file_path, information_request, top_k}`. Same for file-retrieval.

### Repo management & diagnose

`:repo_id` = **base64url (no padding)** of the full repo path.

- `GET /api/repos` — all repos: `repo`, `state`, file counts,
  `last_indexed_at`, `worker_active`.
- `GET /api/index-status` — compact per-repo state + file counts.
- `GET /api/repos/:repo_id/status` — live run state (`idle`/`indexing`),
  `indexed_files`/`total_files`, `phase`, `error`.
- `GET /api/repos/:repo_id/index-stats | files | chunks | graph | index-events | ignored-files`
- `POST /api/repos/:repo_id/index | rebuild | cancel-index | ignore-file(s) | chat | mcp-setup`
- `POST /api/index-all`

### Misc

- `GET/PUT /api/config` — server settings (incl. `mcp_index_wait_secs`,
  `mcp_stale_after_days`). PUT persists `settings.json`; workers pick up
  changes without a restart.
- `DELETE /api/embedding-cache`
- `GET /api/defender-status`, `POST /api/defender-exclude` (Windows)
- Liveness: the HTTP root serves the UI, so check liveness with an MCP
  `initialize` POST to `/mcp`, not a GET health path.
