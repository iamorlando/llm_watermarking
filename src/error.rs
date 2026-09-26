use std::fmt;

use crate::synthid::{MAX_DEPTH, MAX_NGRAM_LEN};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WatermarkError {
    InvalidNgramLen,
    InvalidDepth,
    PromptLengthExceedsContext,
    EmptyDistribution,
    InvalidProbability { index: usize },
    ZeroProbabilityMass,
}

impl fmt::Display for WatermarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNgramLen => write!(
                f,
                "watermark ngram_len must be between 2 and {MAX_NGRAM_LEN}"
            ),
            Self::InvalidDepth => write!(f, "watermark depth must be between 1 and {MAX_DEPTH}"),
            Self::PromptLengthExceedsContext => f.write_str("prompt_len exceeds token count"),
            Self::EmptyDistribution => f.write_str("probability distribution must not be empty"),
            Self::InvalidProbability { index } => write!(
                f,
                "probability at index {index} must be finite and nonnegative"
            ),
            Self::ZeroProbabilityMass => {
                f.write_str("probability distribution must have positive mass")
            }
        }
    }
}

impl std::error::Error for WatermarkError {}
