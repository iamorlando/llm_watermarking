//! Shared configuration and evidence for position-keyed sampling watermarks.
//! Scores are alignment costs (smaller is stronger), not p-values. A calibrated
//! randomization test must repeat the entire search with independent null keys.
use std::fmt;

use sha2::{Digest, Sha256};

use crate::{common, WatermarkError};

#[derive(Clone)]
pub struct SamplingConfig {
    pub key: [u8; 32],
    pub vocab_size: usize,
    /// Period of the pseudorandom key sequence, in tokens (1..=65536).
    /// Use a period at least as long as a generation to avoid reusing its rows.
    pub sequence_len: usize,
}

impl SamplingConfig {
    pub fn new(key: [u8; 32], vocab_size: usize) -> Self {
        Self {
            key,
            vocab_size,
            sequence_len: 1024,
        }
    }

    pub fn validate(&self) -> Result<(), WatermarkError> {
        common::validate_vocab(self.vocab_size)?;
        if !(1..=65536).contains(&self.sequence_len) {
            return Err(WatermarkError::InvalidSequenceLength);
        }
        Ok(())
    }
}

impl fmt::Debug for SamplingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SamplingConfig")
            .field("key", &"[redacted]")
            .field("vocab_size", &self.vocab_size)
            .field("sequence_len", &self.sequence_len)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingDetection {
    pub tokens_scored: usize,
    pub mean_cost: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct AlignmentConfig {
    /// Search every contiguous text window of this length, and every key offset.
    pub block_size: usize,
    /// None uses direct alignment; Some enables insertion/deletion dynamic
    /// programming, with this nonnegative cost for each gap.
    pub edit_penalty: Option<f64>,
    /// Upper bound on cached token/key costs (input length * sequence_len).
    /// This bounds memory, not runtime; edit search can still be expensive.
    pub max_cells: usize,
}

impl AlignmentConfig {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            edit_penalty: None,
            max_cells: 4_194_304,
        }
    }

    pub fn validate(&self) -> Result<(), WatermarkError> {
        if self.block_size == 0 || self.edit_penalty.is_some_and(|p| !p.is_finite() || p < 0.0) {
            return Err(WatermarkError::InvalidAlignment);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlignmentDetection {
    /// Window length, including tokens aligned to gaps; zero if input is too short.
    pub tokens_scored: usize,
    /// Minimum total cost divided by block_size, including gap penalties.
    pub mean_cost: Option<f64>,
    /// Absolute index in the supplied token slice, never inside the prompt.
    pub text_offset: Option<usize>,
    pub key_offset: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct SamplingCore {
    pub(crate) prefix: Sha256,
    #[cfg(feature = "candle")]
    pub(crate) tensor_seed: crate::device_metadata::Seed,
    pub(crate) vocab_size: usize,
    pub(crate) sequence_len: usize,
}

impl SamplingCore {
    pub(crate) fn new(config: &SamplingConfig, domain: &[u8]) -> Result<Self, WatermarkError> {
        config.validate()?;
        Ok(Self {
            #[cfg(feature = "candle")]
            tensor_seed: crate::device_metadata::Seed::prefix(
                domain,
                &config.key,
                &[config.vocab_size, config.sequence_len],
            ),
            prefix: common::prefix(
                domain,
                &config.key,
                &[config.vocab_size, config.sequence_len],
            ),
            vocab_size: config.vocab_size,
            sequence_len: config.sequence_len,
        })
    }

    pub(crate) fn position_hash(&self, position: usize) -> Sha256 {
        let mut hash = self.prefix.clone();
        hash.update(((position % self.sequence_len) as u64).to_le_bytes());
        hash
    }

    pub(crate) fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos: &[u32],
        start_position: usize,
        cost: impl Fn(u32, usize) -> f64,
    ) -> Result<SamplingDetection, WatermarkError> {
        let end = common::completion_end(tokens, prompt_len, eos, self.vocab_size)?;
        let tokens_scored = end - prompt_len;
        let sum: f64 = tokens[prompt_len..end]
            .iter()
            .enumerate()
            .map(|(i, &token)| {
                cost(
                    token,
                    (start_position % self.sequence_len + i % self.sequence_len)
                        % self.sequence_len,
                )
            })
            .sum();
        Ok(SamplingDetection {
            tokens_scored,
            mean_cost: (tokens_scored > 0).then(|| sum / tokens_scored as f64),
        })
    }

    pub(crate) fn align(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos: &[u32],
        config: &AlignmentConfig,
        cost: impl Fn(u32, usize) -> f64,
    ) -> Result<AlignmentDetection, WatermarkError> {
        config.validate()?;
        let end = common::completion_end(tokens, prompt_len, eos, self.vocab_size)?;
        let mut result = AlignmentDetection {
            tokens_scored: 0,
            mean_cost: None,
            text_offset: None,
            key_offset: None,
        };
        let text = &tokens[prompt_len..end];
        let k = config.block_size;
        if text.len() < k {
            return Ok(result);
        }
        let period = self.sequence_len;
        let cells = text
            .len()
            .checked_mul(period)
            .filter(|&n| n <= config.max_cells)
            .ok_or(WatermarkError::AlignmentTooLarge)?;
        let mut costs = Vec::new();
        costs
            .try_reserve_exact(cells)
            .map_err(|_| WatermarkError::AlignmentTooLarge)?;
        for &token in text {
            for phase in 0..period {
                costs.push(cost(token, phase));
            }
        }
        let mut previous = vec![0.0; k + 1];
        let mut current = vec![0.0; k + 1];
        let mut best = f64::INFINITY;
        for start in 0..=text.len() - k {
            for phase in 0..period {
                let total = if let Some(gap) = config.edit_penalty {
                    // Global Levenshtein alignment within this pair of windows.
                    // Nonnegative costs prevent unbounded rewards from gaps.
                    for (j, value) in previous.iter_mut().enumerate() {
                        *value = j as f64 * gap;
                    }
                    for i in 1..=k {
                        current[0] = i as f64 * gap;
                        for j in 1..=k {
                            let substitution = costs
                                [(start + i - 1) * period + (phase + (j - 1) % period) % period];
                            current[j] = (previous[j - 1] + substitution)
                                .min(previous[j] + gap)
                                .min(current[j - 1] + gap);
                        }
                        std::mem::swap(&mut previous, &mut current);
                    }
                    previous[k]
                } else {
                    (0..k)
                        .map(|i| costs[(start + i) * period + (phase + i % period) % period])
                        .sum()
                };
                if total < best {
                    best = total;
                    result = AlignmentDetection {
                        tokens_scored: k,
                        mean_cost: Some(total / k as f64),
                        text_offset: Some(prompt_len + start),
                        key_offset: Some(phase),
                    };
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_alignment_matches_a_hand_computed_insertion_and_deletion() {
        let mut config = SamplingConfig::new([0; 32], 4);
        config.sequence_len = 4;
        let core = SamplingCore::new(&config, b"test").unwrap();
        let cost = |token: u32, position| if token as usize == position { 0.0 } else { 1.0 };
        let mut alignment = AlignmentConfig::new(4);
        let straight = core.align(&[0, 3, 1, 2], 0, &[], &alignment, cost).unwrap();
        assert_eq!(straight.mean_cost, Some(0.5));
        alignment.edit_penalty = Some(0.1);
        let edited = core.align(&[0, 3, 1, 2], 0, &[], &alignment, cost).unwrap();
        assert!((edited.mean_cost.unwrap() - 0.05).abs() < 1e-12);
    }
}
