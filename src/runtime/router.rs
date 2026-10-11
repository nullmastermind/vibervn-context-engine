use std::future::IntoFuture;

use tracing::info;

use context_engine_rs::router;

use crate::Cli;

pub async fn run(cli: &Cli, bind: &str) {
    let port = cli.port.unwrap_or(6699);
    let addr: std::net::SocketAddr = format!("{bind}:{port}").parse().unwrap_or_else(|error| {
        exit_with_error(&format!("invalid bind address '{bind}:{port}': {error}"), 2)
    });
    let (app, proxy) = router::build_router_app(router::RouterBootOptions {
        data_dir: cli.data_dir.clone(),
        embeddings_dir: cli.embeddings_dir.clone(),
        bind: bind.to_owned(),
        home_dir: cli.home_dir.clone(),
        worker_exe: None,
        listen_port: Some(port),
    })
    .await
    .unwrap_or_else(|error| exit_with_error(&format!("{error:#}"), 2));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|error| exit_with_error(&format!("could not bind to {addr}: {error}"), 2));
    info!("Context Engine router listening on http://{addr}");

    let server = axum::serve(listener, app).into_future();
    tokio::select! {
        result = server => {
            result.unwrap_or_else(|error| {
                exit_with_error(&format!("router server error: {error}"), 1)
            });
        }
        _ = shutdown_signal() => {
            info!("shutdown signal received; killing live workers");
            proxy.registry.kill_all().await;
            info!("workers killed; router exiting");
            std::process::exit(0);
        }
    }
}

async fn shutdown_signal() {
    #[cfg(windows)]
    {
        use tokio::signal::windows;
        let mut close = match windows::ctrl_close() {
            Ok(signal) => signal,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        let mut shutdown = windows::ctrl_shutdown().ok();
        let mut logoff = windows::ctrl_logoff().ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = close.recv() => {}
            _ = async { match shutdown.as_mut() { Some(s) => { s.recv().await; }, None => std::future::pending().await } } => {}
            _ = async { match logoff.as_mut() { Some(s) => { s.recv().await; }, None => std::future::pending().await } } => {}
        }
    }
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = match signal(SignalKind::terminate()) {
            Ok(signal) => signal,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(any(windows, unix)))]
    let _ = tokio::signal::ctrl_c().await;
}

fn exit_with_error(message: &str, code: i32) -> ! {
    eprintln!("error: {message}");
    std::process::exit(code);
}
