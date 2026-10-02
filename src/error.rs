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
    InvalidVocabSize,
    VocabularySizeMismatch { expected: usize, actual: usize },
    TokenOutOfRange { token: u32, vocab_size: usize },
    InvalidContextWidth,
    InsufficientContext,
    InvalidGreenFraction,
    InvalidDelta,
    InvalidSequenceLength,
    InvalidAlignment,
    AlignmentTooLarge,
    InvalidRadix,
    InvalidPayloadLength,
    InvalidPayloadSymbol { index: usize },
    InvalidEmbeddingDimension,
    InvalidHyperplaneCount,
    InvalidMargin,
    InvalidMaxAttempts,
    EmbeddingDimensionMismatch { expected: usize, actual: usize },
    InvalidEmbedding { index: usize },
    ZeroEmbedding,
    InvalidTraceOptions,
    TraceTooLarge,
    InvalidTournamentRounds,
    UnsupportedTournamentDepth { configured: usize, maximum: usize },
    InvalidGenerationTournamentOptions,
    InvalidGenerationRngInfo,
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
            Self::InvalidVocabSize => f.write_str("vocab_size must be between 2 and u32::MAX"),
            Self::VocabularySizeMismatch { expected, actual } => {
                write!(f, "expected {expected} vocabulary weights, got {actual}")
            }
            Self::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token {token} is outside vocabulary of size {vocab_size}"
            ),
            Self::InvalidContextWidth => f.write_str("context_width must be between 1 and 32"),
            Self::InsufficientContext => f.write_str("not enough tokens to seed the watermark"),
            Self::InvalidGreenFraction => f.write_str(
                "green_fraction must be finite and select at least one but not all items",
            ),
            Self::InvalidDelta => f.write_str("delta must be finite and nonnegative"),
            Self::InvalidSequenceLength => f.write_str("sequence_len must be between 1 and 65536"),
            Self::InvalidAlignment => f.write_str(
                "block_size must be positive and edit_penalty, if supplied, finite and nonnegative",
            ),
            Self::AlignmentTooLarge => f.write_str(
                "alignment cost matrix exceeds max_cells; shorten the input or increase max_cells",
            ),
            Self::InvalidRadix => {
                f.write_str("radix must be between 2 and 256 and no greater than vocab_size")
            }
            Self::InvalidPayloadLength => f.write_str(
                "payload_len must be between 1 and 65536 and match the supplied payload",
            ),
            Self::InvalidPayloadSymbol { index } => {
                write!(f, "payload symbol at index {index} is outside the radix")
            }
            Self::InvalidEmbeddingDimension => {
                f.write_str("embedding_dim must be between 1 and 65536")
            }
            Self::InvalidHyperplaneCount => f.write_str("num_hyperplanes must be between 1 and 16"),
            Self::InvalidMargin => f.write_str("margin must be finite and in [0, 1)"),
            Self::InvalidMaxAttempts => f.write_str("max_attempts must be positive"),
            Self::EmbeddingDimensionMismatch { expected, actual } => {
                write!(f, "expected embedding dimension {expected}, got {actual}")
            }
            Self::InvalidEmbedding { index } => {
                write!(f, "embedding component at index {index} must be finite")
            }
            Self::ZeroEmbedding => f.write_str("embedding must have nonzero norm"),
            Self::InvalidTraceOptions => f.write_str("trace max_layers must be between 0 and 256"),
            Self::TraceTooLarge => {
                f.write_str("trace snapshot exceeds row or element bounds (or is empty)")
            }
            Self::InvalidTournamentRounds => {
                f.write_str("demonstration tournament rounds must be between 1 and 4 and not exceed SynthID depth")
            }
            Self::UnsupportedTournamentDepth { configured, maximum } => write!(
                f, "explicit tournament depth {configured} exceeds supported maximum {maximum}; generation depth is never reduced"
            ),
            Self::InvalidGenerationTournamentOptions => f.write_str(
                "generation tournament capture exceeds maximum matches or layers"
            ),
            Self::InvalidGenerationRngInfo => f.write_str(
                "generation RNG version must be nonempty and effective_seed must be a decimal integer string"
            ),
        }
    }
}

impl std::error::Error for WatermarkError {}
