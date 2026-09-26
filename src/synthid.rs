use std::{collections::HashSet, fmt};

use sha2::{Digest, Sha256};

use crate::{WatermarkDetection, WatermarkError};

pub const KEY_BYTES: usize = 32;
pub const DEFAULT_NGRAM_LEN: usize = 5;
pub const MAX_NGRAM_LEN: usize = 32;
pub const DEFAULT_DEPTH: usize = 30;
pub const MAX_DEPTH: usize = 256;
pub const CONTEXT_HISTORY_SIZE: usize = 1024;
pub const HASH_DOMAIN: &[u8] = b"llm-watermarking-synthid-text-v1\0";

/// Algorithm parameters; the key should contain 32 independently random bytes.
#[derive(Clone)]
pub struct SynthIdConfig {
    pub key: [u8; KEY_BYTES],
    pub ngram_len: usize,
    pub depth: usize,
}

impl SynthIdConfig {
    pub fn new(key: [u8; KEY_BYTES]) -> Self {
        Self {
            key,
            ngram_len: DEFAULT_NGRAM_LEN,
            depth: DEFAULT_DEPTH,
        }
    }

    pub fn validate(&self) -> Result<(), WatermarkError> {
        if !(2..=MAX_NGRAM_LEN).contains(&self.ngram_len) {
            return Err(WatermarkError::InvalidNgramLen);
        }
        if !(1..=MAX_DEPTH).contains(&self.depth) {
            return Err(WatermarkError::InvalidDepth);
        }
        Ok(())
    }
}

impl fmt::Debug for SynthIdConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SynthIdConfig")
            .field("key", &"[redacted]")
            .field("ngram_len", &self.ngram_len)
            .field("depth", &self.depth)
            .finish()
    }
}

/// Two-candidate SynthID-Text tournaments with context-keyed SHA-256 binary scores.
#[derive(Clone)]
pub struct SynthIdText {
    hash_prefix: Sha256,
    #[cfg(feature = "candle")]
    tensor_seed: crate::device_metadata::Seed,
    context_len: usize,
    depth: usize,
}

impl SynthIdText {
    pub fn new(config: &SynthIdConfig) -> Result<Self, WatermarkError> {
        Self::with_domain(config, HASH_DOMAIN)
    }

    /// Generation and detection must use the same byte-for-byte domain separator.
    pub fn with_domain(config: &SynthIdConfig, domain: &[u8]) -> Result<Self, WatermarkError> {
        config.validate()?;
        let mut hash_prefix = Sha256::new();
        hash_prefix.update(domain);
        hash_prefix.update(config.key);
        hash_prefix.update((config.ngram_len as u32).to_le_bytes());
        Ok(Self {
            hash_prefix,
            #[cfg(feature = "candle")]
            tensor_seed: {
                let mut bytes = domain.to_vec();
                bytes.extend_from_slice(&config.key);
                bytes.extend_from_slice(&(config.ngram_len as u32).to_le_bytes());
                crate::device_metadata::Seed::new(bytes)
            },
            context_len: config.ngram_len - 1,
            depth: config.depth,
        })
    }

    fn context_hash(&self, context: &[u32]) -> Sha256 {
        let mut hash = self.hash_prefix.clone();
        for token in context {
            hash.update(token.to_le_bytes());
        }
        hash
    }

    fn g_values(hash: &Sha256, token: u32) -> [u8; KEY_BYTES] {
        let mut hash = hash.clone();
        hash.update(token.to_le_bytes());
        hash.finalize().into()
    }

    fn g_value(values: &[u8; KEY_BYTES], layer: usize) -> f64 {
        f64::from((values[layer / 8] >> (layer % 8)) & 1)
    }

    fn repeated_context(&self, context: &[u32], prompt_len: usize) -> bool {
        let current = &context[context.len() - self.context_len..];
        let start = prompt_len
            .max(self.context_len)
            .max(context.len().saturating_sub(CONTEXT_HISTORY_SIZE));
        (start..context.len()).any(|end| &context[end - self.context_len..end] == current)
    }

    /// Reweight token-indexed probabilities after the caller's filters; skipped contexts leave weights unchanged.
    pub fn apply(
        &self,
        probs: &mut [f32],
        context: &[u32],
        prompt_len: usize,
    ) -> Result<(), WatermarkError> {
        if prompt_len > context.len() {
            return Err(WatermarkError::PromptLengthExceedsContext);
        }
        if probs.is_empty() {
            return Err(WatermarkError::EmptyDistribution);
        }
        let mut positive_mass = false;
        for (index, prob) in probs.iter().enumerate() {
            if !prob.is_finite() || *prob < 0.0 {
                return Err(WatermarkError::InvalidProbability { index });
            }
            positive_mass |= *prob > 0.0;
        }
        if !positive_mass {
            return Err(WatermarkError::ZeroProbabilityMass);
        }
        if context.len() < self.context_len || self.repeated_context(context, prompt_len) {
            return Ok(());
        }
        let hash = self.context_hash(&context[context.len() - self.context_len..]);
        let mut candidates: Vec<_> = probs
            .iter()
            .enumerate()
            .filter(|(_, prob)| **prob > 0.0)
            .map(|(token, prob)| (token, f64::from(*prob), Self::g_values(&hash, token as u32)))
            .collect();
        for layer in 0..self.depth {
            let total: f64 = candidates.iter().map(|(_, prob, _)| prob).sum();
            let g_mass = candidates
                .iter()
                .map(|(_, prob, values)| prob * Self::g_value(values, layer))
                .sum::<f64>()
                / total;
            for (_, prob, values) in &mut candidates {
                // The exact two-candidate tournament distribution avoids drawing 2^depth tokens.
                *prob = (*prob / total) * (1.0 + Self::g_value(values, layer) - g_mass);
            }
        }
        for (token, prob, _) in candidates {
            probs[token] = prob as f32;
        }
        Ok(())
    }

    /// Excludes prompt tokens, tokens at or after the first generated EOS, and every repeated context.
    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
    ) -> Result<WatermarkDetection, WatermarkError> {
        if prompt_len > tokens.len() {
            return Err(WatermarkError::PromptLengthExceedsContext);
        }
        let mut seen = HashSet::new();
        let mut tokens_scored = 0;
        let mut sum = 0.0;
        for position in prompt_len..tokens.len() {
            if eos_token_ids.contains(&tokens[position]) {
                break;
            }
            if position < self.context_len {
                continue;
            }
            let context = &tokens[position - self.context_len..position];
            if !seen.insert(context) {
                continue;
            }
            let values = Self::g_values(&self.context_hash(context), tokens[position]);
            sum += (0..self.depth)
                .map(|layer| Self::g_value(&values, layer))
                .sum::<f64>();
            tokens_scored += 1;
        }
        Ok(WatermarkDetection {
            tokens_scored,
            mean_g_value: (tokens_scored > 0)
                .then(|| sum / (tokens_scored as f64 * self.depth as f64)),
        })
    }
}

#[cfg(feature = "candle")]
impl SynthIdText {
    /// Prepare key/context metadata on the requested device. Does not read model weights.
    pub fn prepare_tensor(
        &self,
        vocab_size: usize,
        context: &[u32],
        prompt_len: usize,
        device: &candle_core::Device,
    ) -> candle_core::Result<crate::tensor::PreparedWatermark> {
        use crate::tensor::PreparedWatermark;
        if prompt_len > context.len() {
            return Err(candle_core::Error::wrap(
                WatermarkError::PromptLengthExceedsContext,
            ));
        }
        if vocab_size == 0 || vocab_size > u32::MAX as usize {
            candle_core::bail!("SynthID tensor vocabulary size must be in 1..=u32::MAX");
        }
        if context.len() < self.context_len || self.repeated_context(context, prompt_len) {
            return Ok(PreparedWatermark::identity(vocab_size, device));
        }
        let seed = self
            .tensor_seed
            .context(&context[context.len() - self.context_len..], device)?;
        let packed = crate::device_metadata::synthid(&seed, vocab_size, self.depth)?;
        PreparedWatermark::tournament(packed, vocab_size, self.depth)
    }

    /// Device-preserving tournament transform; the host samples the returned weights.
    pub fn apply_tensor(
        &self,
        probabilities: &candle_core::Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.prepare_tensor(
            probabilities.dims1()?,
            context,
            prompt_len,
            probabilities.device(),
        )?
        .apply(probabilities)
    }
}

#[cfg(feature = "candle")]
impl SynthIdText {
    /// Hash only the K actual candidate IDs; the result retains candidate order.
    pub fn prepare_indexed(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        context: &[u32],
        prompt_len: usize,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedWatermark> {
        use crate::tensor::{PreparedIndexedWatermark, PreparedWatermark};
        if prompt_len > context.len() {
            return Err(candle_core::Error::wrap(
                WatermarkError::PromptLengthExceedsContext,
            ));
        }
        if context.len() < self.context_len || self.repeated_context(context, prompt_len) {
            return Ok(PreparedIndexedWatermark::new(
                candidates,
                PreparedWatermark::identity(candidates.len(), candidates.device()),
            ));
        }
        let seed = self.tensor_seed.context(
            &context[context.len() - self.context_len..],
            candidates.device(),
        )?;
        let bits =
            crate::device_metadata::synthid_indexed(&seed, candidates.token_ids(), self.depth)?;
        Ok(PreparedIndexedWatermark::new(
            candidates,
            PreparedWatermark::tournament(bits, candidates.len(), self.depth)?,
        ))
    }

    /// Preparation from device history, including the exact 1,024-position
    /// generation repeat window. No token history or status is read back.
    pub fn prepare_indexed_device(
        &self,
        candidates: &crate::tensor::IndexedCandidates,
        history: &crate::tensor::DeviceHistory,
    ) -> candle_core::Result<crate::tensor::PreparedIndexedWatermark> {
        use crate::tensor::{PreparedIndexedWatermark, PreparedWatermark};
        let (seed, active) = history.seed(&self.tensor_seed, candidates, self.context_len, true)?;
        let bits =
            crate::device_metadata::synthid_indexed(&seed, candidates.token_ids(), self.depth)?;
        Ok(PreparedIndexedWatermark::new(
            candidates,
            PreparedWatermark::tournament(bits, candidates.len(), self.depth)?,
        )
        .with_active(active))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distr::Distribution, SeedableRng};
    use rand_isaac::Isaac64Rng;

    fn config() -> SynthIdConfig {
        SynthIdConfig::new(std::array::from_fn(|index| index as u8))
    }

    #[test]
    fn watermark_config_validation_and_redaction() {
        let mut config = config();
        assert!(format!("{config:?}").contains("[redacted]"));
        for depth in [0, MAX_DEPTH + 1] {
            config.depth = depth;
            assert_eq!(config.validate(), Err(WatermarkError::InvalidDepth));
        }
        config.depth = DEFAULT_DEPTH;
        for ngram_len in [0, 1, MAX_NGRAM_LEN + 1] {
            config.ngram_len = ngram_len;
            assert_eq!(config.validate(), Err(WatermarkError::InvalidNgramLen));
        }
    }

    #[test]
    fn watermark_hash_matches_independent_sha256_fixture() {
        let watermark = SynthIdText::new(&config()).unwrap();
        let values = SynthIdText::g_values(&watermark.context_hash(&[1, 2, 3, 4]), 7);
        let hex: String = values.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            hex,
            "ed2d3be6b1ef582545f37f2993070e8734f6bcc85029576ed929e34970e01856"
        );
    }

    #[test]
    fn watermark_matches_enumerated_pairwise_tournament() {
        let mut config = config();
        config.depth = 1;
        let watermark = SynthIdText::new(&config).unwrap();
        let context = [0, 2, 3, 4];
        let initial = [0.1f32, 0.2, 0.3, 0.4];
        let mut expected = [0.0; 4];
        let hash = watermark.context_hash(&context);
        let scores: Vec<_> = (0..4)
            .map(|token| SynthIdText::g_value(&SynthIdText::g_values(&hash, token), 0))
            .collect();
        assert!(scores.contains(&0.0) && scores.contains(&1.0));
        for a in 0..4 {
            for b in 0..4 {
                let mass = initial[a] * initial[b];
                if scores[a] == scores[b] {
                    expected[a] += mass / 2.0;
                    expected[b] += mass / 2.0;
                } else {
                    expected[if scores[a] > scores[b] { a } else { b }] += mass;
                }
            }
        }
        let mut actual = initial;
        watermark
            .apply(&mut actual, &context, context.len())
            .unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn watermark_preserves_support_and_point_masses() {
        let watermark = SynthIdText::new(&config()).unwrap();
        let mut probs = [0.0, 0.3, 0.0, 0.7];
        watermark.apply(&mut probs, &[1, 2, 3, 4], 4).unwrap();
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[2], 0.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|value| value.is_finite() && *value >= 0.0));
        let mut point_mass = [0.0, 1.0, 0.0];
        watermark.apply(&mut point_mass, &[1, 2, 3, 4], 4).unwrap();
        assert_eq!(point_mass, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn watermark_context_history_is_replayable_and_prompt_aware() {
        let watermark = SynthIdText::new(&config()).unwrap();
        let initial = vec![0.125; 8];
        for context in [&[1, 2, 3][..], &[1, 2, 3, 4, 1, 2, 3, 4][..]] {
            let mut probs = initial.clone();
            watermark.apply(&mut probs, context, 0).unwrap();
            assert_eq!(probs, initial);
        }
        let context = [1, 2, 3, 4, 1, 2, 3, 4];
        let mut first = initial.clone();
        watermark
            .apply(&mut first, &context, context.len())
            .unwrap();
        assert_ne!(first, initial);
        let mut retry = initial.clone();
        watermark
            .clone()
            .apply(&mut retry, &context, context.len())
            .unwrap();
        assert_eq!(first, retry);
        let mut unrelated = initial.clone();
        watermark.apply(&mut unrelated, &[9, 8, 7, 6], 4).unwrap();
        let mut after_rollback = initial;
        watermark
            .apply(&mut after_rollback, &context, context.len())
            .unwrap();
        assert_eq!(first, after_rollback);
    }

    #[test]
    fn watermark_detector_excludes_prompt_repeats_and_eos() {
        let watermark = SynthIdText::new(&config()).unwrap();
        assert!(watermark.detect(&[1], 2, &[]).is_err());
        for tokens in [&[][..], &[1, 2, 3, 4][..]] {
            let detection = watermark.detect(tokens, 0, &[]).unwrap();
            assert_eq!(detection.tokens_scored, 0);
            assert_eq!(detection.mean_g_value, None);
        }
        let detection = watermark.detect(&[1, 2, 3, 4, 7, 99, 8], 4, &[99]).unwrap();
        assert_eq!(detection.tokens_scored, 1);
        let values = SynthIdText::g_values(&watermark.context_hash(&[1, 2, 3, 4]), 7);
        let expected = (0..DEFAULT_DEPTH)
            .map(|layer| SynthIdText::g_value(&values, layer))
            .sum::<f64>()
            / DEFAULT_DEPTH as f64;
        assert_eq!(detection.mean_g_value, Some(expected));
        assert_eq!(
            watermark.detect(&[1; 100], 4, &[]).unwrap().tokens_scored,
            1
        );
    }

    #[test]
    fn watermark_signal_separates_marked_unmarked_and_wrong_key() {
        let watermark = SynthIdText::new(&config()).unwrap();
        let wrong_config = SynthIdConfig::new([0xff; KEY_BYTES]);
        let wrong_watermark = SynthIdText::new(&wrong_config).unwrap();
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let mut marked = vec![1, 2, 3, 4];
        let mut unmarked = marked.clone();
        for _ in 0..1000 {
            let mut probs = vec![1.0 / 128.0; 128];
            let baseline = rand::distr::weighted::WeightedIndex::new(&probs).unwrap();
            watermark.apply(&mut probs, &marked, 4).unwrap();
            let distribution = rand::distr::weighted::WeightedIndex::new(&probs).unwrap();
            marked.push(distribution.sample(&mut rng) as u32);
            unmarked.push(baseline.sample(&mut rng) as u32);
        }
        let marked_score = watermark
            .detect(&marked, 4, &[])
            .unwrap()
            .mean_g_value
            .unwrap();
        let unmarked_score = watermark
            .detect(&unmarked, 4, &[])
            .unwrap()
            .mean_g_value
            .unwrap();
        let wrong_score = wrong_watermark
            .detect(&marked, 4, &[])
            .unwrap()
            .mean_g_value
            .unwrap();
        assert!(marked_score > 0.60, "marked={marked_score}");
        assert!(
            (unmarked_score - 0.5).abs() < 0.02,
            "unmarked={unmarked_score}"
        );
        assert!((wrong_score - 0.5).abs() < 0.02, "wrong={wrong_score}");
    }

    #[test]
    fn watermark_preserves_distribution_over_independent_contexts() {
        let watermark = SynthIdText::new(&config()).unwrap();
        let expected = [0.1, 0.2, 0.3, 0.4];
        let mut totals = [0.0; 4];
        for nonce in 0..10000 {
            let mut probs = expected;
            watermark.apply(&mut probs, &[nonce, 2, 3, 4], 4).unwrap();
            for (total, prob) in totals.iter_mut().zip(probs) {
                *total += f64::from(prob);
            }
        }
        for (total, expected) in totals.into_iter().zip(expected) {
            assert!((total / 10000.0 - f64::from(expected)).abs() < 0.02);
        }
    }
}
