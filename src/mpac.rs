//! Multi-bit watermarking via position allocation (Yoo et al., 2024).
use std::{collections::HashSet, fmt};

use sha2::Sha256;

use crate::{common, CountDetection, WatermarkError};

pub const HASH_DOMAIN: &[u8] = b"llm-watermarking-mpac-v1\0";

#[derive(Clone)]
pub struct MpacConfig {
    pub key: [u8; 32],
    pub vocab_size: usize,
    /// Number of radix-r symbols, not bytes or bits (except when radix is 2).
    pub payload_len: usize,
    pub radix: usize,
    pub context_width: usize,
    pub delta: f64,
    pub ignore_repeated_ngrams: bool,
}

impl MpacConfig {
    pub fn new(key: [u8; 32], vocab_size: usize, payload_len: usize) -> Self {
        Self {
            key,
            vocab_size,
            payload_len,
            radix: 2,
            context_width: 1,
            delta: 2.0,
            ignore_repeated_ngrams: true,
        }
    }

    pub fn validate(&self) -> Result<(), WatermarkError> {
        common::validate_vocab(self.vocab_size)?;
        common::validate_width(self.context_width)?;
        common::validate_delta(self.delta)?;
        if !(2..=256).contains(&self.radix) || self.radix > self.vocab_size {
            return Err(WatermarkError::InvalidRadix);
        }
        if !(1..=65536).contains(&self.payload_len) {
            return Err(WatermarkError::InvalidPayloadLength);
        }
        Ok(())
    }
}

impl fmt::Debug for MpacConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MpacConfig")
            .field("key", &"[redacted]")
            .field("vocab_size", &self.vocab_size)
            .field("payload_len", &self.payload_len)
            .field("radix", &self.radix)
            .field("context_width", &self.context_width)
            .field("delta", &self.delta)
            .field("ignore_repeated_ngrams", &self.ignore_repeated_ngrams)
            .finish()
    }
}

/// Blind payload recovery. A tied or unobserved position is an erasure (`None`).
/// The winning fraction is selection-biased and is NOT a binomial test statistic.
#[derive(Clone, Debug, PartialEq)]
pub struct MpacDetection {
    pub tokens_scored: usize,
    pub payload: Vec<Option<u8>>,
    pub votes: Vec<Vec<usize>>,
    pub winning_fraction: Option<f64>,
}

#[derive(Clone)]
pub struct Mpac {
    prefix: Sha256,
    vocab_size: usize,
    payload_len: usize,
    radix: usize,
    context_width: usize,
    delta: f64,
    deduplicate: bool,
}

impl Mpac {
    pub fn new(config: &MpacConfig) -> Result<Self, WatermarkError> {
        config.validate()?;
        Ok(Self {
            prefix: common::prefix(
                HASH_DOMAIN,
                &config.key,
                &[
                    config.vocab_size,
                    config.context_width,
                    config.payload_len,
                    config.radix,
                ],
            ),
            vocab_size: config.vocab_size,
            payload_len: config.payload_len,
            radix: config.radix,
            context_width: config.context_width,
            delta: config.delta,
            deduplicate: config.ignore_repeated_ngrams,
        })
    }

    fn validate_payload(&self, payload: &[u8]) -> Result<(), WatermarkError> {
        if payload.len() != self.payload_len {
            return Err(WatermarkError::InvalidPayloadLength);
        }
        for (index, &symbol) in payload.iter().enumerate() {
            if symbol as usize >= self.radix {
                return Err(WatermarkError::InvalidPayloadSymbol { index });
            }
        }
        Ok(())
    }

    fn allocation(&self, context: &[u32]) -> (usize, Vec<usize>) {
        let hash =
            common::context_hash(&self.prefix, &context[context.len() - self.context_width..]);
        let position = common::HashRng::new(&hash, 1).below(self.payload_len);
        let group_size = self.vocab_size / self.radix;
        // The remainder belongs to no colorlist and receives no bias or vote.
        let mut groups = vec![self.radix; self.vocab_size];
        for (rank, token) in common::permutation(&hash, self.vocab_size)
            .into_iter()
            .enumerate()
            .take(group_size * self.radix)
        {
            groups[token] = rank / group_size;
        }
        (position, groups)
    }

    /// Encode a caller-owned payload. Short contexts are unchanged; errors are atomic.
    pub fn apply(
        &self,
        probs: &mut [f32],
        context: &[u32],
        prompt_len: usize,
        payload: &[u8],
    ) -> Result<(), WatermarkError> {
        self.validate_payload(payload)?;
        common::context(context, prompt_len, self.vocab_size)?;
        common::weights(probs, self.vocab_size)?;
        if context.len() < self.context_width {
            return Ok(());
        }
        let (position, groups) = self.allocation(context);
        common::boost(
            probs,
            |i| groups[i] == payload[position] as usize,
            self.delta,
        );
        Ok(())
    }

    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
    ) -> Result<MpacDetection, WatermarkError> {
        let end = common::completion_end(tokens, prompt_len, eos_token_ids, self.vocab_size)?;
        let mut seen = HashSet::new();
        let mut votes = vec![vec![0; self.radix]; self.payload_len];
        let mut tokens_scored = 0;
        for i in prompt_len.max(self.context_width)..end {
            if self.deduplicate && !seen.insert(&tokens[i - self.context_width..=i]) {
                continue;
            }
            let (position, groups) = self.allocation(&tokens[i - self.context_width..i]);
            let symbol = groups[tokens[i] as usize];
            if symbol < self.radix {
                votes[position][symbol] += 1;
            }
            tokens_scored += 1;
        }
        let mut winners = 0;
        let payload = votes
            .iter()
            .map(|row| {
                let max = *row.iter().max().unwrap();
                winners += max;
                let mut leaders = row.iter().enumerate().filter(|(_, &n)| n == max);
                let first = leaders.next().unwrap().0;
                (max > 0 && leaders.next().is_none()).then_some(first as u8)
            })
            .collect();
        Ok(MpacDetection {
            tokens_scored,
            payload,
            votes,
            winning_fraction: (tokens_scored > 0).then(|| winners as f64 / tokens_scored as f64),
        })
    }

    /// Score a payload fixed BEFORE inspecting the text. Do not pass a payload
    /// recovered from this same sample and interpret the z-score as significance.
    pub fn detect_payload(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
        payload: &[u8],
    ) -> Result<CountDetection, WatermarkError> {
        self.validate_payload(payload)?;
        let detection = self.detect(tokens, prompt_len, eos_token_ids)?;
        let successes = detection
            .votes
            .iter()
            .zip(payload)
            .map(|(row, &symbol)| row[symbol as usize])
            .sum();
        Ok(common::count_detection(
            detection.tokens_scored,
            successes,
            (self.vocab_size / self.radix) as f64 / self.vocab_size as f64,
        ))
    }
}
