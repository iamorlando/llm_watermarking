//! Fixed keyed vocabulary partition (Zhao et al., 2023).
use std::{collections::HashSet, fmt};

use crate::{common, CountDetection, WatermarkError};

pub const HASH_DOMAIN: &[u8] = b"llm-watermarking-unigram-v1\0";

#[derive(Clone)]
pub struct UnigramConfig {
    pub key: [u8; 32],
    pub vocab_size: usize,
    pub green_fraction: f64,
    pub delta: f64,
    /// Optional distinct-token scoring; false follows the paper's token count.
    pub ignore_repeated_tokens: bool,
}

impl UnigramConfig {
    pub fn new(key: [u8; 32], vocab_size: usize) -> Self {
        Self {
            key,
            vocab_size,
            green_fraction: 0.5,
            delta: 2.0,
            ignore_repeated_tokens: false,
        }
    }

    pub fn validate(&self) -> Result<(), WatermarkError> {
        common::validate_vocab(self.vocab_size)?;
        common::green_count(self.vocab_size, self.green_fraction)?;
        common::validate_delta(self.delta)
    }
}

impl fmt::Debug for UnigramConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnigramConfig")
            .field("key", &"[redacted]")
            .field("vocab_size", &self.vocab_size)
            .field("green_fraction", &self.green_fraction)
            .field("delta", &self.delta)
            .field("ignore_repeated_tokens", &self.ignore_repeated_tokens)
            .finish()
    }
}

#[derive(Clone)]
pub struct Unigram {
    mask: Vec<bool>,
    green_count: usize,
    delta: f64,
    deduplicate: bool,
}

impl Unigram {
    pub fn new(config: &UnigramConfig) -> Result<Self, WatermarkError> {
        config.validate()?;
        let green_count = common::green_count(config.vocab_size, config.green_fraction)?;
        let prefix = common::prefix(HASH_DOMAIN, &config.key, &[config.vocab_size]);
        Ok(Self {
            mask: common::green_mask(&prefix, config.vocab_size, green_count),
            green_count,
            delta: config.delta,
            deduplicate: config.ignore_repeated_tokens,
        })
    }

    pub fn green_list(&self) -> Vec<u32> {
        self.mask
            .iter()
            .enumerate()
            .filter_map(|(i, &green)| green.then_some(i as u32))
            .collect()
    }

    pub fn is_green(&self, token: u32) -> Result<bool, WatermarkError> {
        common::token(token, self.mask.len())?;
        Ok(self.mask[token as usize])
    }

    /// Boost a fixed partition, preserving excluded tokens. Errors are atomic.
    pub fn apply(&self, probs: &mut [f32]) -> Result<(), WatermarkError> {
        common::weights(probs, self.mask.len())?;
        common::boost(probs, |i| self.mask[i], self.delta);
        Ok(())
    }

    /// Count favored tokens. Repeated token identities are correlated under the
    /// key; the nominal z-score requires empirical calibration, even for long text.
    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
    ) -> Result<CountDetection, WatermarkError> {
        let end = common::completion_end(tokens, prompt_len, eos_token_ids, self.mask.len())?;
        let mut seen = HashSet::new();
        let (mut trials, mut successes) = (0, 0);
        for &token in &tokens[prompt_len..end] {
            if self.deduplicate && !seen.insert(token) {
                continue;
            }
            trials += 1;
            successes += usize::from(self.mask[token as usize]);
        }
        Ok(common::count_detection(
            trials,
            successes,
            self.green_count as f64 / self.mask.len() as f64,
        ))
    }
}
