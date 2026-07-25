pub mod backend;
pub mod data;
pub mod error;
pub mod models;
pub mod training;

pub type Result<T> = std::result::Result<T, error::GanshotError>;
