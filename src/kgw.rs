//! Context-dependent soft green-list watermark (Kirchenbauer et al., 2023).
use std::{collections::HashSet, fmt};

use sha2::Sha256;

use crate::{common, CountDetection, WatermarkError};

pub const HASH_DOMAIN: &[u8] = b"llm-watermarking-kgw-v1\0";

#[derive(Clone)]
pub struct KgwConfig {
    pub key: [u8; 32],
    pub vocab_size: usize,
    pub context_width: usize,
    pub green_fraction: f64,
    /// Additive logit bias, applied equivalently to probability weights.
    pub delta: f64,
    /// Detection only: count each (context, token) n-gram once.
    pub ignore_repeated_ngrams: bool,
}

impl KgwConfig {
    pub fn new(key: [u8; 32], vocab_size: usize) -> Self {
        Self {
            key,
            vocab_size,
            context_width: 1,
            green_fraction: 0.5,
            delta: 2.0,
            ignore_repeated_ngrams: true,
        }
    }

    pub fn validate(&self) -> Result<(), WatermarkError> {
        common::validate_vocab(self.vocab_size)?;
        common::validate_width(self.context_width)?;
        common::green_count(self.vocab_size, self.green_fraction)?;
        common::validate_delta(self.delta)
    }
}

impl fmt::Debug for KgwConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KgwConfig")
            .field("key", &"[redacted]")
            .field("vocab_size", &self.vocab_size)
            .field("context_width", &self.context_width)
            .field("green_fraction", &self.green_fraction)
            .field("delta", &self.delta)
            .field("ignore_repeated_ngrams", &self.ignore_repeated_ngrams)
            .finish()
    }
}

#[derive(Clone)]
pub struct Kgw {
    prefix: Sha256,
    #[cfg(feature = "candle")]
    tensor_seed: crate::device_metadata::Seed,
    vocab_size: usize,
    context_width: usize,
    green_count: usize,
    delta: f64,
    deduplicate: bool,
}

impl Kgw {
    pub fn new(config: &KgwConfig) -> Result<Self, WatermarkError> {
        config.validate()?;
        Ok(Self {
            #[cfg(feature = "candle")]
            tensor_seed: crate::device_metadata::Seed::prefix(
                HASH_DOMAIN,
                &config.key,
                &[config.vocab_size, config.context_width],
            ),
            prefix: common::prefix(
                HASH_DOMAIN,
                &config.key,
                &[config.vocab_size, config.context_width],
            ),
            vocab_size: config.vocab_size,
            context_width: config.context_width,
            green_count: common::green_count(config.vocab_size, config.green_fraction)?,
            delta: config.delta,
            deduplicate: config.ignore_repeated_ngrams,
        })
    }

    fn mask(&self, context: &[u32]) -> Vec<bool> {
        common::green_mask(
            &common::context_hash(&self.prefix, &context[context.len() - self.context_width..]),
            self.vocab_size,
            self.green_count,
        )
    }

    /// Return the exact floor(green_fraction * vocab_size) favored token IDs.
    pub fn green_list(&self, context: &[u32]) -> Result<Vec<u32>, WatermarkError> {
        common::context(context, 0, self.vocab_size)?;
        if context.len() < self.context_width {
            return Err(WatermarkError::InsufficientContext);
        }
        Ok(self
            .mask(context)
            .iter()
            .enumerate()
            .filter_map(|(i, &green)| green.then_some(i as u32))
            .collect())
    }

    /// Normalize and boost favored tokens. Short contexts leave weights unchanged.
    /// Validation errors never modify the input slice.
    pub fn apply(
        &self,
        probs: &mut [f32],
        context: &[u32],
        prompt_len: usize,
    ) -> Result<(), WatermarkError> {
        common::context(context, prompt_len, self.vocab_size)?;
        common::weights(probs, self.vocab_size)?;
        if context.len() < self.context_width {
            return Ok(());
        }
        let mask = self.mask(context);
        common::boost(probs, |i| mask[i], self.delta);
        Ok(())
    }

    /// Exclude prompt, warmup, generated EOS and its suffix, and (by default)
    /// duplicate n-grams. The null rate uses the actual rounded partition size.
    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
    ) -> Result<CountDetection, WatermarkError> {
        let end = common::completion_end(tokens, prompt_len, eos_token_ids, self.vocab_size)?;
        let mut seen = HashSet::new();
        let (mut trials, mut successes) = (0, 0);
        for i in prompt_len.max(self.context_width)..end {
            if self.deduplicate && !seen.insert(&tokens[i - self.context_width..=i]) {
                continue;
            }
            trials += 1;
            successes +=
                usize::from(self.mask(&tokens[i - self.context_width..i])[tokens[i] as usize]);
        }
        Ok(common::count_detection(
            trials,
            successes,
            self.green_count as f64 / self.vocab_size as f64,
        ))
    }
}

#[cfg(feature = "candle")]
impl Kgw {
    pub fn prepare_tensor(
        &self,
        context: &[u32],
        prompt_len: usize,
        device: &candle_core::Device,
    ) -> candle_core::Result<crate::tensor::PreparedWatermark> {
        common::context(context, prompt_len, self.vocab_size).map_err(candle_core::Error::wrap)?;
        if context.len() < self.context_width {
            return Ok(crate::tensor::PreparedWatermark::identity(
                self.vocab_size,
                device,
            ));
        }
        let seed = self
            .tensor_seed
            .context(&context[context.len() - self.context_width..], device)?;
        let mask = crate::device_metadata::green_mask(&seed, self.vocab_size, self.green_count)?;
        crate::tensor::PreparedWatermark::bias(mask, self.delta)
    }

    pub fn apply_tensor(
        &self,
        probabilities: &candle_core::Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.prepare_tensor(context, prompt_len, probabilities.device())?
            .apply(probabilities)
    }
}

#[cfg(feature = "candle")]
impl Kgw {
    /// Full-vocabulary partition with probability work restricted to K candidates.
    pub fn prepare_indexed(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        context: &[u32],
        prompt_len: usize,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedWatermark> {
        candidates.check_vocab(self.vocab_size)?;
        self.prepare_tensor(context, prompt_len, candidates.device())?
            .indexed(candidates)
    }

    /// Device-resident context preparation. No history readback or host seed construction.
    pub fn prepare_indexed_device(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        history: &crate::tensor::DeviceHistory,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedWatermark> {
        use crate::tensor::{PreparedIndexedWatermark, PreparedWatermark};
        candidates.check_vocab(self.vocab_size)?;
        let (seed, active) =
            history.seed(&self.tensor_seed, candidates, self.context_width, false)?;
        let (_, ranks) = crate::device_metadata::permutation(&seed, self.vocab_size)?;
        let mask = ranks
            .index_select(&candidates.safe_ids()?, 0)?
            .lt(self.green_count as u32)?;
        Ok(
            PreparedIndexedWatermark::new(candidates, PreparedWatermark::bias(mask, self.delta)?)
                .with_active(active),
        )
    }
}
