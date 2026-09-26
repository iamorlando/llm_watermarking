//! Inverse-transform watermark with a fixed keyed vocabulary permutation and a
//! position-keyed uniform sequence (ITS, Kuditipudi et al., 2023).
use crate::{
    common,
    sampling::{AlignmentConfig, AlignmentDetection, SamplingCore, SamplingDetection},
    WatermarkError,
};

pub use crate::sampling::SamplingConfig as InverseTransformConfig;
pub const HASH_DOMAIN: &[u8] = b"llm-watermarking-inverse-transform-v1\0";

#[derive(Clone)]
pub struct InverseTransform {
    core: SamplingCore,
    order: Vec<usize>,
    ranks: Vec<usize>,
    #[cfg(feature = "candle")]
    tensor_permutation:
        crate::device_metadata::DeviceCache<(candle_core::Tensor, candle_core::Tensor)>,
}

impl InverseTransform {
    pub fn new(config: &InverseTransformConfig) -> Result<Self, WatermarkError> {
        let core = SamplingCore::new(config, HASH_DOMAIN)?;
        let order = common::permutation(&core.prefix, core.vocab_size);
        let mut ranks = vec![0; core.vocab_size];
        for (rank, &token) in order.iter().enumerate() {
            ranks[token] = rank;
        }
        Ok(Self {
            core,
            order,
            ranks,
            #[cfg(feature = "candle")]
            tensor_permutation: crate::device_metadata::DeviceCache::default(),
        })
    }

    fn uniform(&self, position: usize) -> f64 {
        common::HashRng::new(&self.core.position_hash(position), 1).uniform()
    }

    fn cost(&self, token: u32, position: usize) -> f64 {
        (self.uniform(position)
            - self.ranks[token as usize] as f64 / (self.core.vocab_size - 1) as f64)
            .abs()
    }

    /// Sample the CDF in the keyed vocabulary order using `U[position]`. Input
    /// weights need not sum to one. Zero-weight intervals are always skipped.
    pub fn sample(&self, probs: &[f32], position: usize) -> Result<u32, WatermarkError> {
        let total = common::weights(probs, self.core.vocab_size)?;
        let target = self.uniform(position) * total;
        let mut cumulative = 0.0;
        let mut last_positive = 0;
        for &token in &self.order {
            if probs[token] > 0.0 {
                last_positive = token as u32;
                cumulative += f64::from(probs[token]);
                if target < cumulative {
                    return Ok(token as u32);
                }
            }
        }
        // Different summation orders can differ by a few ulps near the endpoint.
        Ok(last_positive)
    }

    /// Model-free rank-distance evidence, not the probability of a token. Lower
    /// is stronger. Prompt tokens do not advance the supplied start_position.
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
            |token, position| self.cost(token, position),
        )
    }

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
            |token, position| self.cost(token, position),
        )
    }
}

#[cfg(feature = "candle")]
impl InverseTransform {
    pub fn prepare_tensor(
        &self,
        position: usize,
        device: &candle_core::Device,
    ) -> candle_core::Result<crate::tensor::PreparedSampler> {
        if self.core.vocab_size > crate::tensor::MAX_INVERSE_TENSOR_VOCAB {
            candle_core::bail!("inverse-transform tensor vocabulary exceeds the F32 rank limit");
        }
        let (order, ranks) = self.tensor_permutation.get(device, || {
            let seed = self.core.tensor_seed.tensor(&[], device)?;
            crate::device_metadata::permutation(&seed, self.core.vocab_size)
        })?;
        crate::tensor::PreparedSampler::inverse(order, ranks, self.uniform(position))
    }

    /// F32 scores selecting the first occupied CDF interval above the keyed
    /// uniform. The host selects argmax; these are not categorical weights.
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
impl InverseTransform {
    /// Reuse the full-vocabulary permutation, sort only candidate ranks and scan
    /// only K weights. Output scores are in the caller's original candidate order.
    pub fn prepare_indexed(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        position: usize,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedSampler> {
        candidates.check_vocab(self.core.vocab_size)?;
        self.prepare_tensor(position, candidates.device())?
            .indexed(candidates)
    }

    /// Position [1] U32 and the resulting keyed uniform stay on the device.
    pub fn prepare_indexed_device(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        position: &candle_core::Tensor,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedSampler> {
        candidates.check_vocab(self.core.vocab_size)?;
        if self.core.vocab_size > crate::tensor::MAX_INVERSE_TENSOR_VOCAB {
            candle_core::bail!("inverse-transform tensor vocabulary exceeds the F32 rank limit");
        }
        let prefix = self.core.tensor_seed.tensor(&[], candidates.device())?;
        let (_, ranks) = self.tensor_permutation.get(candidates.device(), || {
            crate::device_metadata::permutation(&prefix, self.core.vocab_size)
        })?;
        let seed =
            crate::device_metadata::position_seed(&prefix, position, self.core.sequence_len)?;
        crate::tensor::PreparedIndexedSampler::inverse(
            candidates,
            &ranks,
            crate::device_metadata::uniform(&seed)?,
        )
    }
}
