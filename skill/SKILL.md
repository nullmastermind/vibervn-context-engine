---
name: vibervn-context-engine-search
description: Use vibervn-context-engine for semantic search, find code by meaning, codebase retrieval, and reading relevant code snippets from a running context-engine server.
---

# Vibervn Context Engine Semantic Search

Use this skill when you need semantic code search or targeted code reading from a repository using the `vibervn-context-engine` CLI.

## Prerequisites

- Bash or shell execution capability.
- A context engine server must already be running:
  - `vibervn-context-engine`
  - `npx vibervn-context-engine@latest`
- Repository paths passed to `--repo` must be absolute paths.
- File paths passed to `--file` must be relative to the repository root.

## Commands Covered

Only use these subcommands:

- `search`
- `read`

## Server URL

By default, the CLI uses its default server URL.

For a non-default server URL, use either:

```bash
vibervn-context-engine search --query "authentication middleware" --repo "/absolute/path/to/repo" --server "http://127.0.0.1:3000"
```

or:

```bash
CONTEXT_ENGINE_URL="http://127.0.0.1:3000" vibervn-context-engine search --query "authentication middleware" --repo "/absolute/path/to/repo"
```

## Search

Find files or code regions by meaning.

Syntax:

```bash
vibervn-context-engine search --query <text> --repo <absolute-path> [--server <url>]
```

Flags:

- `--query <text>`: Natural-language search query.
- `--repo <absolute-path>`: Absolute path to the repository root.
- `--server <url>`: Optional server URL override.

Examples:

```bash
vibervn-context-engine search --query "where semantic search requests are handled" --repo "/home/PNguyen/development/vibervn-context-engine"
```

```bash
vibervn-context-engine search --query "CLI argument parsing for search and read commands" --repo "/home/PNguyen/development/vibervn-context-engine"
```

```bash
vibervn-context-engine search --query "error handling when the server is unreachable" --repo "/home/PNguyen/development/vibervn-context-engine" --server "http://127.0.0.1:3000"
```

## Read

Read the most relevant snippets from a specific file for a query.

Syntax:

```bash
vibervn-context-engine read --repo <absolute-path> --file <relative-path> --query <text> [--top-k N] [--server <url>]
```

Flags:

- `--repo <absolute-path>`: Absolute path to the repository root.
- `--file <relative-path>`: File path relative to the repository root.
- `--query <text>`: Natural-language question or topic to retrieve from the file.
- `--top-k N`: Optional number of snippets to return.
- `--server <url>`: Optional server URL override.

Examples:

```bash
vibervn-context-engine read --repo "/home/PNguyen/development/vibervn-context-engine" --file "src/main.rs" --query "CLI subcommand definitions"
```

```bash
vibervn-context-engine read --repo "/home/PNguyen/development/vibervn-context-engine" --file "src/main.rs" --query "exit code mapping for server failures" --top-k 5
```

```bash
CONTEXT_ENGINE_URL="http://127.0.0.1:3000" vibervn-context-engine read --repo "/home/PNguyen/development/vibervn-context-engine" --file "README.md" --query "usage examples" --top-k 3
```

## Output

- Output is plain text written to stdout.
- Results are intended to be read directly by agents or piped into other shell commands.
- Errors are written by the CLI in plain terminal format.

## Exit Codes

- `0`: Success.
- `2`: Server unreachable. Start the server or check `--server` / `CONTEXT_ENGINE_URL`.
- `3`: Server returned an error. Inspect the message and server logs.

## Agent Workflow

1. Run `search` with a meaning-based query to identify relevant files or regions.
2. Run `read` on specific files returned by search for focused snippets.
3. Use exact repository absolute paths and relative file paths.
4. If exit code is `2`, verify the server is running and the URL is correct.
5. If exit code is `3`, report the server error instead of guessing results.
