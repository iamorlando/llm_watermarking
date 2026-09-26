//! Exponential-race sampling, equivalent to Gumbel-max sampling.
//! Implements the position-keyed EXP construction from Kuditipudi et al. (2023),
//! using the paper's nonnegative theoretical cost -ln(U) for detection.
use sha2::Digest;

use crate::{
    common,
    sampling::{AlignmentConfig, AlignmentDetection, SamplingCore, SamplingDetection},
    WatermarkError,
};

pub use crate::sampling::SamplingConfig as ExponentialRaceConfig;
pub const HASH_DOMAIN: &[u8] = b"llm-watermarking-exponential-v1\0";

#[derive(Clone)]
pub struct ExponentialRace {
    core: SamplingCore,
}

impl ExponentialRace {
    pub fn new(config: &ExponentialRaceConfig) -> Result<Self, WatermarkError> {
        Ok(Self {
            core: SamplingCore::new(config, HASH_DOMAIN)?,
        })
    }

    fn uniform(&self, position: usize, token: u32) -> f64 {
        let mut hash = self.core.position_hash(position);
        hash.update(token.to_le_bytes());
        common::HashRng::new(&hash, 2).uniform()
    }

    /// Return `argmin(-ln(U[token]) / weight[token])`; zero weights never win.
    /// `position` is the key-stream index, not an index including prompt tokens.
    /// Replaying the same position and weights returns the same token.
    pub fn sample(&self, probs: &[f32], position: usize) -> Result<u32, WatermarkError> {
        common::weights(probs, self.core.vocab_size)?;
        let mut best = f64::INFINITY;
        let mut winner = 0;
        for (token, &prob) in probs.iter().enumerate() {
            if prob > 0.0 {
                let race = -self.uniform(position, token as u32).ln() / f64::from(prob);
                if race < best {
                    best = race;
                    winner = token as u32;
                }
            }
        }
        Ok(winner)
    }

    /// Direct, known-offset score. Independent unmarked tokens have mean cost
    /// near one under fresh uniform key rows. Smaller costs are stronger evidence.
    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
        start_position: usize,
    ) -> Result<SamplingDetection, WatermarkError> {
        self.core.detect(
            tokens,
            prompt_len,
            eos_token_ids,
            start_position,
            |token, position| -self.uniform(position, token).ln(),
        )
    }

    /// Search all cyclic key offsets and text windows, optionally allowing edits.
    pub fn detect_aligned(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
        config: &AlignmentConfig,
    ) -> Result<AlignmentDetection, WatermarkError> {
        self.core.align(
            tokens,
            prompt_len,
            eos_token_ids,
            config,
            |token, position| -self.uniform(position, token).ln(),
        )
    }
}

#[cfg(feature = "candle")]
impl ExponentialRace {
    pub fn prepare_tensor(
        &self,
        position: usize,
        device: &candle_core::Device,
    ) -> candle_core::Result<crate::tensor::PreparedSampler> {
        let position = ((position % self.core.sequence_len) as u64).to_le_bytes();
        let seed = self.core.tensor_seed.tensor(&position, device)?;
        let gumbels = crate::device_metadata::exponential(&seed, self.core.vocab_size)?;
        crate::tensor::PreparedSampler::exponential(gumbels)
    }

    /// F32 Gumbel-max scores. The host selects argmax, not a categorical draw.
    pub fn selection_scores_tensor(
        &self,
        probabilities: &candle_core::Tensor,
        position: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.prepare_tensor(position, probabilities.device())?
            .apply(probabilities)
    }
}

#[cfg(feature = "candle")]
impl ExponentialRace {
    /// K-only hashing using actual vocabulary IDs and the full vocabulary seed.
    pub fn prepare_indexed(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        position: usize,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedSampler> {
        candidates.check_vocab(self.core.vocab_size)?;
        let position = ((position % self.core.sequence_len) as u64).to_le_bytes();
        let seed = self
            .core
            .tensor_seed
            .tensor(&position, candidates.device())?;
        self.indexed_seed(candidates, &seed)
    }

    /// `position` is a U32 tensor of shape [1], independent of prompt length.
    /// Wrapping to the configured key-stream period executes on the device.
    pub fn prepare_indexed_device(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        position: &candle_core::Tensor,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedSampler> {
        candidates.check_vocab(self.core.vocab_size)?;
        let prefix = self.core.tensor_seed.tensor(&[], candidates.device())?;
        let seed =
            crate::device_metadata::position_seed(&prefix, position, self.core.sequence_len)?;
        self.indexed_seed(candidates, &seed)
    }

    fn indexed_seed(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        seed: &candle_core::Tensor,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedSampler> {
        let gumbels = crate::device_metadata::exponential_indexed(seed, candidates.token_ids())?;
        Ok(crate::tensor::PreparedIndexedSampler::new(
            candidates,
            crate::tensor::PreparedSampler::exponential(gumbels)?,
        ))
    }
}
