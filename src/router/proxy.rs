//! Router → worker REVERSE PROXY with streaming passthrough (SSE-safe).
//!
//! Forwards an incoming request to a repo's worker on `127.0.0.1:<port>` and
//! streams the response back BODY-AND-ALL without buffering — so Server-Sent
//! Events (`/api/repos/:id/index-events`) and any chunked response flow through
//! live. Uses reqwest (already a dep, `stream` feature) as the upstream client
//! and converts its streaming body into an axum body via `Body::from_stream`.
//!
//! Single-flight + spawn-on-miss live in [`acquire_and_proxy`]: it asks the
//! [`Registry`] for a ready worker; on a miss it either performs the spawn (if
//! elected) or awaits the in-flight spawn, then proxies. A spawn that exceeds
//! its budget degrades to a `503` "warming, retry" — NEVER a connection-refused
//! or a hang (window-A is owned here, see `spawn::SPAWN_READY_TIMEOUT`).

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use tracing::warn;

use super::jobobject::JobObject;
use super::registry::{Acquire, ReadyWorker, Registry};
use super::spawn::spawn_worker;

/// Shared proxy context: the upstream HTTP client + the worker registry + spawn
/// inputs. Cloned into the router's per-repo handlers.
#[derive(Clone)]
pub struct ProxyCtx {
    pub client: reqwest::Client,
    pub registry: Registry,
    pub job: Arc<JobObject>,
    /// Path to the current executable (workers run the same binary).
    pub exe: std::path::PathBuf,
    /// Passthrough args so workers resolve the same dirs/bind as the router
    /// (e.g. `--data-dir X --bind 127.0.0.1`).
    pub worker_args: Vec<String>,
}

impl ProxyCtx {
    pub fn new(job: Arc<JobObject>, exe: std::path::PathBuf, worker_args: Vec<String>) -> Self {
        Self {
            // No global timeout on the client: long index/query requests stream
            // for a while. Per-attempt budgets are enforced upstream (MCP wait /
            // spawn timeout), not here.
            client: reqwest::Client::builder()
                .pool_idle_timeout(Duration::from_secs(90))
                .build()
                .expect("reqwest client"),
            registry: Registry::new(),
            job,
            exe,
            worker_args,
        }
    }
}

/// Acquire (or spawn) the worker for `repo`, then proxy `req` to it. Degrades to
/// a 503 on spawn timeout/failure. The degrade signal distinguishes two cases so
/// the UI can react correctly (see `warming_response` / `error_response`):
/// - the spawn is still in flight / coalescing → `warming` (`retry:true`),
/// - the spawn genuinely failed (timeout, process error) → `error`
///   (`retry:false`).
///
/// `obtain_worker` returns `Err(msg)` for BOTH today; we map a spawn that never
/// converged to `error` and everything else to `warming`. The distinction is
/// carried in the message prefix `spawn failed:` set by `obtain_worker`.
pub async fn acquire_and_proxy(ctx: &ProxyCtx, repo: &str, req: Request) -> Response {
    // Single-flight acquire loop: a SpawnElected caller spawns; AwaitSpawn waits
    // then re-acquires; Ready proxies. Bounded retries so a thrashing spawn can't
    // loop forever.
    let worker = match obtain_worker(ctx, repo).await {
        Ok(w) => w,
        Err(msg) => {
            // A genuine spawn failure (process couldn't start / readiness
            // timeout) is an ERROR the user must see + manually retry; a spawn
            // that simply hasn't converged yet is WARMING (auto-retry). Map by
            // the `spawn failed:` prefix obtain_worker sets on a hard failure.
            if msg.starts_with("spawn failed:") {
                warn!(repo = %repo, error = %msg, "worker spawn failed; surfacing error");
                return error_response(&msg);
            }
            warn!(repo = %repo, error = %msg, "worker warming; signalling retry");
            return warming_response(&msg);
        }
    };
    proxy_to(ctx, &worker, req).await
}

/// SPAWN-BLOCKED proxy: proxy `req` to the repo's worker ONLY IF one is already
/// live. Returns `None` (without spawning) when no worker is resident, so the
/// caller can serve a cold/native response instead. This is the mechanism that
/// keeps "view" routes (status/graph/files/index-events/cancel/chat-DELETE) from
/// ever starting a worker — they call THIS, never `acquire_and_proxy` (which is
/// the spawn path). A worker present but mid-request still proxies (fresh data);
/// absent → cold.
pub async fn proxy_if_live(ctx: &ProxyCtx, repo: &str, req: Request) -> Option<Response> {
    // Cheap residency check — does NOT spawn. Mirrors the registry's ready set.
    if !ctx.registry.ready_repos().await.iter().any(|r| r == repo) {
        return None;
    }
    // A worker is live: fetch its handle (Ready path of acquire is spawn-free
    // when the entry is already Ready) and proxy. If it raced to death between
    // the check and here, obtain_worker would re-spawn — so guard against that
    // by only proceeding on the Ready fast-path; on any miss, fall back to cold.
    match ctx.registry.acquire(repo).await {
        super::registry::Acquire::Ready(w) => Some(proxy_to(ctx, &w, req).await),
        // The worker died in the race window; do NOT spawn for a view route —
        // serve cold instead (the caller's None branch).
        _ => None,
    }
}

/// Acquire (or spawn) the worker for `repo`, POST `body` as JSON to `path` on
/// it, and return the worker's `{"result": "..."}` string unwrapped. Used by the
/// global `/mcp` proxy handler (which can't stream — it needs the funnel's text
/// back to wrap in a `CallToolResult`) and by any router-native handler that
/// must call a single worker JSON endpoint and read the body.
///
/// Unlike `acquire_and_proxy` (which streams an arbitrary HTTP response), this
/// buffers the worker's JSON response and extracts the `result` field, mirroring
/// what the monolith's in-process MCP tool returned. On spawn/connection failure
/// it returns an `Err(String)` so the caller can surface a tool-level error
/// instead of a transport panic.
pub async fn forward_json_to_worker(
    ctx: &ProxyCtx,
    repo: &str,
    path: &str,
    body: serde_json::Value,
) -> Result<String, String> {
    let worker = obtain_worker(ctx, repo).await?;
    let url = format!("http://127.0.0.1:{}{}", worker.port, path);
    let resp = ctx
        .client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("worker request failed: {e}"))?;
    let raw = resp
        .text()
        .await
        .map_err(|e| format!("worker response read failed: {e}"))?;
    Ok(super::mcp_proxy::unwrap_mcp_tool_result(&raw))
}

/// GET `path` on an ALREADY-LIVE worker for `repo` and return the parsed JSON
/// body verbatim (NO MCP `.result` unwrap, NO spawn). Returns `None` if no
/// worker is live or the request/parse fails — the caller then serves a cold
/// view. This is the read-only sibling of `forward_json_to_worker`: that one
/// POSTs to action/MCP endpoints and unwraps `{result}`, which is WRONG for a
/// plain GET status endpoint (a POST would 405, and `/status` returns the status
/// object directly, not an MCP `{result}` envelope). Used by `get_index_status`
/// to read a live worker's authoritative status without spawning.
pub async fn get_json_if_live(ctx: &ProxyCtx, repo: &str, path: &str) -> Option<serde_json::Value> {
    // Spawn-free: only proceed if a worker is already Ready.
    if !ctx.registry.ready_repos().await.iter().any(|r| r == repo) {
        return None;
    }
    let worker = match ctx.registry.acquire(repo).await {
        super::registry::Acquire::Ready(w) => w,
        _ => return None, // raced to death — serve cold, don't spawn
    };
    let url = format!("http://127.0.0.1:{}{}", worker.port, path);
    let resp = ctx.client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<serde_json::Value>().await.ok()
}

/// Resolve a ready worker for `repo`, spawning under single-flight if needed.
pub(crate) async fn obtain_worker(ctx: &ProxyCtx, repo: &str) -> Result<Arc<ReadyWorker>, String> {
    // At most a couple of acquire rounds: Ready, or elect→spawn, or await→re-acquire.
    for _ in 0..3 {
        match ctx.registry.acquire(repo).await {
            Acquire::Ready(w) => return Ok(w),
            Acquire::SpawnElected { ready } => {
                // We own the spawn for this repo. On success publish + wake
                // awaiters; on failure abandon so another request can retry.
                match spawn_worker(&ctx.exe, repo, &ctx.worker_args, &ctx.job).await {
                    Ok(worker) => {
                        ctx.registry
                            .publish_ready(repo, worker.clone(), &ready)
                            .await;
                        return Ok(worker);
                    }
                    Err(e) => {
                        ctx.registry.abandon_spawn(repo, &ready).await;
                        return Err(format!("spawn failed: {e:#}"));
                    }
                }
            }
            Acquire::AwaitSpawn { ready } => {
                // Another request is spawning. Wait for it (bounded), then loop
                // to re-acquire (it should now be Ready, or absent on failure).
                let _ = tokio::time::timeout(
                    super::spawn::SPAWN_READY_TIMEOUT + Duration::from_secs(2),
                    ready.notified(),
                )
                .await;
                continue;
            }
        }
    }
    Err("worker spawn did not converge after retries".to_string())
}

/// Forward `req` to the worker and stream the response back.
async fn proxy_to(ctx: &ProxyCtx, worker: &ReadyWorker, req: Request) -> Response {
    let (parts, body) = req.into_parts();

    // Rebuild the upstream URL: same path+query, worker's loopback port.
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    let upstream_uri = format!("http://127.0.0.1:{}{}", worker.port, path_and_query);
    let _ = upstream_uri.parse::<Uri>(); // validate shape; reqwest re-parses

    // Collect the request body. Most proxied requests are small JSON; we buffer
    // the request body (not the response) — streaming the request would add
    // complexity for no benefit here (queries/index triggers are tiny POSTs).
    let body_bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("could not read request body: {e}"),
            )
                .into_response();
        }
    };

    // Translate method + headers.
    let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);
    let mut upstream_req = ctx
        .client
        .request(method, &upstream_uri)
        .body(body_bytes.to_vec());
    for (name, value) in parts.headers.iter() {
        // Skip hop-by-hop / host headers the upstream sets itself.
        let n = name.as_str().to_ascii_lowercase();
        if matches!(n.as_str(), "host" | "content-length" | "connection") {
            continue;
        }
        if let Ok(v) = value.to_str() {
            upstream_req = upstream_req.header(name.as_str(), v);
        }
    }

    // Send. A connection error here means the worker died between acquire and
    // proxy (rare — try_wait guards it); degrade to warming/retry.
    let upstream_resp = match upstream_req.send().await {
        Ok(r) => r,
        Err(e) => {
            warn!(port = worker.port, error = %e, "proxy upstream connect failed; degrading");
            return warming_response("worker connection lost; retry");
        }
    };

    // Build the downstream response, STREAMING the body (SSE-safe).
    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream_resp.headers().iter() {
        let n = name.as_str().to_ascii_lowercase();
        if matches!(
            n.as_str(),
            "connection" | "transfer-encoding" | "content-length"
        ) {
            continue;
        }
        if let (Ok(hn), Ok(hv)) = (
            axum::http::HeaderName::from_bytes(name.as_str().as_bytes()),
            axum::http::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(hn, hv);
        }
    }

    // Stream the upstream body bytes through without buffering — this is what
    // keeps SSE (`/index-events`) live end-to-end.
    let stream = upstream_resp.bytes_stream();
    let body = Body::from_stream(stream);
    builder
        .body(body)
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

/// Max proxied REQUEST body size (queries / index triggers / config are small).
/// 16 MiB is generous; responses are streamed and not bounded here.
const MAX_REQUEST_BODY: usize = 16 * 1024 * 1024;

/// A uniform "worker warming, retry shortly" degrade response. 503 with a short
/// JSON body + Retry-After. Mirrors the query-warm-path "warming, retry" signal:
/// the client retries and the worker will be ready, instead of a refused
/// connection or a hang.
fn warming_response(detail: &str) -> Response {
    let body = serde_json::json!({
        "status": "warming",
        "detail": detail,
        "retry": true,
    });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("retry-after", "2")],
        axum::Json(body),
    )
        .into_response()
}

/// A "worker failed to start" degrade response. 503 with `retry:false` so the UI
/// shows a manual-retry error banner (NOT the auto-retry warming skeleton). The
/// distinction lets the frontend render the three states correctly: warming
/// (auto-retry skeleton) vs error (red banner + Retry button).
fn error_response(detail: &str) -> Response {
    let body = serde_json::json!({
        "status": "error",
        "detail": detail,
        "retry": false,
    });
    (StatusCode::SERVICE_UNAVAILABLE, axum::Json(body)).into_response()
}
