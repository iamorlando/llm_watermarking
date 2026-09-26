#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod error;
pub mod synthid;

pub use error::WatermarkError;

/// Uncalibrated evidence over eligible tokens, not a probability of authorship.
#[derive(Clone, Debug, PartialEq)]
pub struct WatermarkDetection {
    pub tokens_scored: usize,
    pub mean_g_value: Option<f64>,
}
