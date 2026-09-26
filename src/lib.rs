#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod common;
mod error;
pub mod exponential;
pub mod inverse_transform;
pub mod kgw;
pub mod mpac;
pub mod sampling;
pub mod semstamp;
pub mod synthid;
#[cfg(feature = "candle")]
pub mod tensor;
pub mod unigram;

#[cfg(feature = "candle")]
pub use candle_core;

pub use error::WatermarkError;

/// Count-based evidence. The nominal z-score is not a calibrated probability.
/// Repetitions, dependence, and selecting a payload from the same observations
/// can invalidate a binomial interpretation; calibrate on held-out data.
#[derive(Clone, Debug, PartialEq)]
pub struct CountDetection {
    pub trials: usize,
    pub successes: usize,
    pub expected_rate: f64,
    pub observed_rate: Option<f64>,
    pub z_score: Option<f64>,
}

/// Uncalibrated evidence over eligible tokens, not a probability of authorship.
#[derive(Clone, Debug, PartialEq)]
pub struct WatermarkDetection {
    pub tokens_scored: usize,
    pub mean_g_value: Option<f64>,
}
