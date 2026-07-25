use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoadDatasetError {
    #[error("Failed to read dataset file {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse dataset file {path}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Error, Debug)]
pub enum GanshotError {
    #[error("Failed to load dataset")]
    LoadDataset(#[from] LoadDatasetError),
}
