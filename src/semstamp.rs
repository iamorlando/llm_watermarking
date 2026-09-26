//! Sentence-level semantic rejection sampling (Hou et al., 2024).
//! The host supplies sentence generation, segmentation, and embeddings from the
//! same model at generation and detection. No embedding model is bundled.
use std::{collections::HashSet, fmt};

use sha2::Sha256;

use crate::{common, WatermarkError};

pub const HASH_DOMAIN: &[u8] = b"llm-watermarking-semstamp-v1\0";

#[derive(Clone)]
pub struct SemStampConfig {
    pub key: [u8; 32],
    pub embedding_dim: usize,
    /// 1..=16, giving 2^num_hyperplanes semantic regions.
    pub num_hyperplanes: usize,
    pub green_fraction: f64,
    /// Minimum absolute cosine to every hyperplane normal, in [0, 1).
    pub margin: f64,
    pub max_attempts: usize,
    /// Detection only: count a (previous region, current region) pair once.
    pub ignore_repeated_transitions: bool,
}

impl SemStampConfig {
    pub fn new(key: [u8; 32], embedding_dim: usize) -> Self {
        Self {
            key,
            embedding_dim,
            num_hyperplanes: 8,
            green_fraction: 0.25,
            margin: 0.02,
            max_attempts: 100,
            ignore_repeated_transitions: true,
        }
    }

    pub fn validate(&self) -> Result<(), WatermarkError> {
        if !(1..=65536).contains(&self.embedding_dim) {
            return Err(WatermarkError::InvalidEmbeddingDimension);
        }
        if !(1..=16).contains(&self.num_hyperplanes) {
            return Err(WatermarkError::InvalidHyperplaneCount);
        }
        common::green_count(1 << self.num_hyperplanes, self.green_fraction)?;
        if !self.margin.is_finite() || !(0.0..1.0).contains(&self.margin) {
            return Err(WatermarkError::InvalidMargin);
        }
        if self.max_attempts == 0 {
            return Err(WatermarkError::InvalidMaxAttempts);
        }
        Ok(())
    }
}

impl fmt::Debug for SemStampConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemStampConfig")
            .field("key", &"[redacted]")
            .field("embedding_dim", &self.embedding_dim)
            .field("num_hyperplanes", &self.num_hyperplanes)
            .field("green_fraction", &self.green_fraction)
            .field("margin", &self.margin)
            .field("max_attempts", &self.max_attempts)
            .field(
                "ignore_repeated_transitions",
                &self.ignore_repeated_transitions,
            )
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemStampDetection {
    pub sentences_scored: usize,
    pub valid_sentences: usize,
    pub expected_rate: f64,
    pub valid_fraction: Option<f64>,
    /// Nominal count statistic; semantic region occupancy need not be uniform.
    pub z_score: Option<f64>,
}

/// A bounded generation attempt. On exhaustion the final candidate is returned
/// with `accepted = false`, matching the paper's maxout fallback explicitly.
#[derive(Clone, Debug)]
pub struct SentenceSample<T> {
    pub sentence: T,
    pub embedding: Vec<f32>,
    pub attempts: usize,
    pub accepted: bool,
}

#[derive(Debug)]
pub enum SentenceSamplingError<E> {
    Watermark(WatermarkError),
    Generator(E),
}

impl<E: fmt::Display> fmt::Display for SentenceSamplingError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Watermark(error) => error.fmt(f),
            Self::Generator(error) => write!(f, "sentence generation failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SentenceSamplingError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Watermark(error) => Some(error),
            Self::Generator(error) => Some(error),
        }
    }
}

#[derive(Clone)]
pub struct SemStamp {
    prefix: Sha256,
    normals: Vec<Vec<f64>>,
    embedding_dim: usize,
    region_count: usize,
    green_count: usize,
    margin: f64,
    max_attempts: usize,
    deduplicate: bool,
}

impl SemStamp {
    pub fn new(config: &SemStampConfig) -> Result<Self, WatermarkError> {
        config.validate()?;
        let prefix = common::prefix(
            HASH_DOMAIN,
            &config.key,
            &[config.embedding_dim, config.num_hyperplanes],
        );
        let mut rng = common::HashRng::new(&prefix, 3);
        let normals = (0..config.num_hyperplanes)
            .map(|_| {
                // Independent standard Gaussian coordinates via Box-Muller.
                let mut normal: Vec<_> = (0..config.embedding_dim)
                    .map(|_| {
                        (-2.0 * rng.uniform().ln()).sqrt()
                            * (std::f64::consts::TAU * rng.uniform()).cos()
                    })
                    .collect();
                let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
                for value in &mut normal {
                    *value /= norm;
                }
                normal
            })
            .collect();
        let region_count = 1 << config.num_hyperplanes;
        Ok(Self {
            prefix,
            normals,
            embedding_dim: config.embedding_dim,
            region_count,
            green_count: common::green_count(region_count, config.green_fraction)?,
            margin: config.margin,
            max_attempts: config.max_attempts,
            deduplicate: config.ignore_repeated_transitions,
        })
    }

    fn signature_and_margin(&self, embedding: &[f32]) -> Result<(u32, f64), WatermarkError> {
        if embedding.len() != self.embedding_dim {
            return Err(WatermarkError::EmbeddingDimensionMismatch {
                expected: self.embedding_dim,
                actual: embedding.len(),
            });
        }
        let mut norm_squared = 0.0;
        for (index, &value) in embedding.iter().enumerate() {
            if !value.is_finite() {
                return Err(WatermarkError::InvalidEmbedding { index });
            }
            norm_squared += f64::from(value).powi(2);
        }
        if norm_squared == 0.0 {
            return Err(WatermarkError::ZeroEmbedding);
        }
        let norm = norm_squared.sqrt();
        let mut signature = 0;
        let mut margin: f64 = 1.0;
        for (bit, normal) in self.normals.iter().enumerate() {
            let cosine = normal
                .iter()
                .zip(embedding)
                .map(|(&a, &b)| a * f64::from(b) / norm)
                .sum::<f64>();
            if cosine > 0.0 {
                signature |= 1 << bit;
            }
            margin = margin.min(cosine.abs());
        }
        Ok((signature, margin))
    }

    pub fn signature(&self, embedding: &[f32]) -> Result<u32, WatermarkError> {
        Ok(self.signature_and_margin(embedding)?.0)
    }

    fn mask(&self, previous: u32) -> Vec<bool> {
        common::green_mask(
            &common::context_hash(&self.prefix, &[previous]),
            self.region_count,
            self.green_count,
        )
    }

    /// Candidate must be in a valid region AND satisfy every cosine margin.
    pub fn accepts(
        &self,
        previous_embedding: &[f32],
        candidate_embedding: &[f32],
    ) -> Result<bool, WatermarkError> {
        let previous = self.signature(previous_embedding)?;
        let (candidate, margin) = self.signature_and_margin(candidate_embedding)?;
        Ok(margin >= self.margin && self.mask(previous)[candidate as usize])
    }

    /// The callback generates another complete sentence from the SAME context
    /// and embeds it. Rejected candidates must not be committed to host history.
    /// Generator and embedding errors propagate immediately, without retries.
    pub fn sample_sentence<T, E>(
        &self,
        previous_embedding: &[f32],
        mut generate: impl FnMut() -> Result<(T, Vec<f32>), E>,
    ) -> Result<SentenceSample<T>, SentenceSamplingError<E>> {
        let previous = self
            .signature(previous_embedding)
            .map_err(SentenceSamplingError::Watermark)?;
        let mask = self.mask(previous);
        for attempts in 1..=self.max_attempts {
            let (sentence, embedding) = generate().map_err(SentenceSamplingError::Generator)?;
            let (signature, margin) = self
                .signature_and_margin(&embedding)
                .map_err(SentenceSamplingError::Watermark)?;
            let accepted = margin >= self.margin && mask[signature as usize];
            if accepted || attempts == self.max_attempts {
                return Ok(SentenceSample {
                    sentence,
                    embedding,
                    attempts,
                    accepted,
                });
            }
        }
        unreachable!("max_attempts is validated as positive")
    }

    /// `prompt_len` is in SENTENCES. With completion-only embeddings, the first
    /// sentence seeds the transition and is not scored. Detection counts valid
    /// regions regardless of margin; the margin is a generation constraint only.
    pub fn detect<E: AsRef<[f32]>>(
        &self,
        embeddings: &[E],
        prompt_len: usize,
    ) -> Result<SemStampDetection, WatermarkError> {
        if prompt_len > embeddings.len() {
            return Err(WatermarkError::PromptLengthExceedsContext);
        }
        let signatures: Vec<_> = embeddings
            .iter()
            .map(|e| self.signature(e.as_ref()))
            .collect::<Result<_, _>>()?;
        let mut seen = HashSet::new();
        let (mut trials, mut successes) = (0, 0);
        for i in prompt_len.max(1)..signatures.len() {
            if self.deduplicate && !seen.insert((signatures[i - 1], signatures[i])) {
                continue;
            }
            trials += 1;
            successes += usize::from(self.mask(signatures[i - 1])[signatures[i] as usize]);
        }
        let evidence = common::count_detection(
            trials,
            successes,
            self.green_count as f64 / self.region_count as f64,
        );
        Ok(SemStampDetection {
            sentences_scored: evidence.trials,
            valid_sentences: evidence.successes,
            expected_rate: evidence.expected_rate,
            valid_fraction: evidence.observed_rate,
            z_score: evidence.z_score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsh_signs_cosine_margin_and_detection_have_distinct_roles() {
        let mut config = SemStampConfig::new([42; 32], 2);
        config.num_hyperplanes = 2;
        config.green_fraction = 0.5;
        config.margin = 0.3;
        let mut watermark = SemStamp::new(&config).unwrap();
        // Known orthogonal normals give independently calculable dot products.
        watermark.normals = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(watermark.signature(&[1.0, 1.0]).unwrap(), 3);
        assert_eq!(watermark.signature(&[-1.0, 1.0]).unwrap(), 2);
        assert_eq!(watermark.signature(&[1.0, -1.0]).unwrap(), 1);
        assert_eq!(watermark.signature(&[-1.0, -1.0]).unwrap(), 0);
        let (_, margin) = watermark.signature_and_margin(&[3.0, 4.0]).unwrap();
        assert!((margin - 0.6).abs() < 1e-12);
        let previous = [1.0, 1.0];
        let mask = watermark.mask(3);
        let valid = mask.iter().position(|&value| value).unwrap();
        let x = if valid & 1 == 0 { -1.0 } else { 1.0 };
        let y = if valid & 2 == 0 { -1.0 } else { 1.0 };
        assert!(watermark.accepts(&previous, &[x, y]).unwrap());
        let near_boundary = [x, y * 0.01];
        assert!(!watermark.accepts(&previous, &near_boundary).unwrap());
        assert_eq!(
            watermark
                .detect(&[previous, near_boundary], 1)
                .unwrap()
                .valid_sentences,
            1
        );
        assert!(!watermark.accepts(&previous, &[1.0, 0.0]).unwrap());
    }
}
