use std::path::Path;

use super::RouterBootOptions;

pub(super) fn build(opts: &RouterBootOptions, home_dir: &Path) -> Vec<String> {
    let mut args = vec![
        "--bind".to_string(),
        opts.bind.clone(),
        "--home-dir".to_string(),
        home_dir.to_string_lossy().to_string(),
    ];
    if let Some(data_dir) = &opts.data_dir {
        args.push("--data-dir".to_string());
        args.push(data_dir.to_string_lossy().to_string());
    }
    if let Some(embeddings_dir) = &opts.embeddings_dir {
        args.push("--embeddings-dir".to_string());
        args.push(embeddings_dir.to_string_lossy().to_string());
    }
    args
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn propagates_router_home_and_directory_overrides() {
        let opts = RouterBootOptions {
            data_dir: Some(PathBuf::from("data")),
            embeddings_dir: Some(PathBuf::from("embeddings")),
            bind: "127.0.0.1".to_string(),
            home_dir: None,
            worker_exe: None,
            listen_port: None,
        };

        assert_eq!(
            build(&opts, Path::new("home")),
            [
                "--bind",
                "127.0.0.1",
                "--home-dir",
                "home",
                "--data-dir",
                "data",
                "--embeddings-dir",
                "embeddings",
            ]
        );
    }
}
