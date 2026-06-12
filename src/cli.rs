use std::io::Write;
use std::process::ExitCode;
use std::time::Duration;

use serde::{Deserialize, Serialize};

const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:6699";
const CONTEXT_ENGINE_URL_ENV: &str = "CONTEXT_ENGINE_URL";

/// Arguments for the `search` subcommand.
#[derive(Debug, Clone)]
pub struct SearchArgs {
    pub query: String,
    pub repo: String,
    pub server: Option<String>,
}

/// Arguments for the `read` subcommand.
#[derive(Debug, Clone)]
pub struct ReadArgs {
    pub repo: String,
    pub file: String,
    pub query: String,
    pub top_k: Option<usize>,
    pub server: Option<String>,
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    information_request: &'a str,
    workspace_full_path: &'a str,
}

#[derive(Serialize)]
struct ReadRequest<'a> {
    workspace_full_path: &'a str,
    file_path: &'a str,
    information_request: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
}

#[derive(Deserialize)]
struct ToolResponse {
    result: String,
}

/// Runs the `search` subcommand and writes only the tool result to stdout.
pub async fn run_search(args: SearchArgs) -> ExitCode {
    run_search_with_writer(args, &mut std::io::stdout()).await
}

/// Runs the `read` subcommand and writes only the tool result to stdout.
pub async fn run_read(args: ReadArgs) -> ExitCode {
    run_read_with_writer(args, &mut std::io::stdout()).await
}

/// Runs the `search` subcommand with an injectable writer for tests.
pub async fn run_search_with_writer<W>(args: SearchArgs, writer: &mut W) -> ExitCode
where
    W: Write,
{
    let base_url = resolve_base_url(args.server.as_deref());
    let endpoint = endpoint_url(&base_url, "/api/mcp-tool");
    let body = SearchRequest {
        information_request: &args.query,
        workspace_full_path: &args.repo,
    };

    run_request(&endpoint, &body, writer).await
}

/// Runs the `read` subcommand with an injectable writer for tests.
pub async fn run_read_with_writer<W>(args: ReadArgs, writer: &mut W) -> ExitCode
where
    W: Write,
{
    let base_url = resolve_base_url(args.server.as_deref());
    let endpoint = endpoint_url(&base_url, "/api/mcp-tool/file-retrieval");
    let body = ReadRequest {
        workspace_full_path: &args.repo,
        file_path: &args.file,
        information_request: &args.query,
        top_k: args.top_k,
    };

    run_request(&endpoint, &body, writer).await
}

fn resolve_base_url(flag: Option<&str>) -> String {
    let selected = flag
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .or_else(|| {
            std::env::var(CONTEXT_ENGINE_URL_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| DEFAULT_SERVER_URL.to_owned());

    selected.trim().trim_end_matches('/').to_owned()
}

fn build_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(60))
        .build()
}

fn endpoint_url(base_url: &str, path: &str) -> String {
    format!("{base_url}{path}")
}

async fn run_request<T, W>(endpoint: &str, body: &T, writer: &mut W) -> ExitCode
where
    T: Serialize + ?Sized,
    W: Write,
{
    let client = match build_client() {
        Ok(client) => client,
        Err(error) => {
            eprintln!("error: could not create HTTP client for {endpoint}: {error}");
            return ExitCode::from(2);
        }
    };

    let response = match client.post(endpoint).json(body).send().await {
        Ok(response) => response,
        Err(error) => {
            eprintln!(
                "error: could not reach {endpoint}: {error}. Start context-engine-rs server and retry."
            );
            return ExitCode::from(2);
        }
    };

    let status = response.status();
    let response_body = match response.text().await {
        Ok(body) => body,
        Err(error) => {
            eprintln!("error: failed to read response from {endpoint}: {error}");
            return ExitCode::from(3);
        }
    };

    if !status.is_success() {
        eprintln!("error: server returned {status} from {endpoint}: {response_body}");
        return ExitCode::from(3);
    }

    let parsed: ToolResponse = match serde_json::from_str(&response_body) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("error: invalid response from {endpoint}: {error}; body: {response_body}");
            return ExitCode::from(3);
        }
    };

    if let Err(error) = writer.write_all(parsed.result.as_bytes()) {
        eprintln!("error: could not write result to stdout: {error}");
        return ExitCode::from(2);
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvGuard {
        original: Option<String>,
    }

    impl EnvGuard {
        fn set(value: Option<&str>) -> Self {
            let original = std::env::var(CONTEXT_ENGINE_URL_ENV).ok();
            unsafe {
                match value {
                    Some(value) => std::env::set_var(CONTEXT_ENGINE_URL_ENV, value),
                    None => std::env::remove_var(CONTEXT_ENGINE_URL_ENV),
                }
            }
            Self { original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(CONTEXT_ENGINE_URL_ENV, value),
                    None => std::env::remove_var(CONTEXT_ENGINE_URL_ENV),
                }
            }
        }
    }

    #[test]
    fn resolves_base_url_with_precedence_and_trimming() {
        let _guard = EnvGuard::set(None);
        assert_eq!(resolve_base_url(None), DEFAULT_SERVER_URL);
        assert_eq!(
            resolve_base_url(Some("http://flag.example///")),
            "http://flag.example"
        );

        let _guard = EnvGuard::set(Some("http://env.example/"));
        assert_eq!(resolve_base_url(None), "http://env.example");
        assert_eq!(
            resolve_base_url(Some("http://flag.example/")),
            "http://flag.example"
        );
    }

    #[test]
    fn serializes_read_request_top_k_only_when_present() {
        let without_top_k = serde_json::to_value(ReadRequest {
            workspace_full_path: "/R",
            file_path: "src/x.rs",
            information_request: "Q",
            top_k: None,
        })
        .expect("serialize read request without top_k");
        assert!(without_top_k.get("top_k").is_none());

        let with_top_k = serde_json::to_value(ReadRequest {
            workspace_full_path: "/R",
            file_path: "src/x.rs",
            information_request: "Q",
            top_k: Some(10),
        })
        .expect("serialize read request with top_k");
        assert_eq!(with_top_k["top_k"], 10);
    }
}
