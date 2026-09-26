use sha2::{Digest, Sha256};

use crate::{CountDetection, WatermarkError};

pub(crate) fn validate_vocab(size: usize) -> Result<(), WatermarkError> {
    if size < 2 || size > u32::MAX as usize {
        return Err(WatermarkError::InvalidVocabSize);
    }
    Ok(())
}

pub(crate) fn validate_width(width: usize) -> Result<(), WatermarkError> {
    if !(1..=32).contains(&width) {
        return Err(WatermarkError::InvalidContextWidth);
    }
    Ok(())
}

pub(crate) fn green_count(size: usize, fraction: f64) -> Result<usize, WatermarkError> {
    if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
        return Err(WatermarkError::InvalidGreenFraction);
    }
    let count = (size as f64 * fraction).floor() as usize;
    if count == 0 || count >= size {
        return Err(WatermarkError::InvalidGreenFraction);
    }
    Ok(count)
}

pub(crate) fn validate_delta(delta: f64) -> Result<(), WatermarkError> {
    if !delta.is_finite() || delta < 0.0 {
        return Err(WatermarkError::InvalidDelta);
    }
    Ok(())
}

pub(crate) fn weights(probs: &[f32], vocab_size: usize) -> Result<f64, WatermarkError> {
    if probs.is_empty() {
        return Err(WatermarkError::EmptyDistribution);
    }
    if probs.len() != vocab_size {
        return Err(WatermarkError::VocabularySizeMismatch {
            expected: vocab_size,
            actual: probs.len(),
        });
    }
    let mut total = 0.0;
    for (index, &prob) in probs.iter().enumerate() {
        if !prob.is_finite() || prob < 0.0 {
            return Err(WatermarkError::InvalidProbability { index });
        }
        total += f64::from(prob);
    }
    if total == 0.0 {
        return Err(WatermarkError::ZeroProbabilityMass);
    }
    Ok(total)
}

pub(crate) fn token(token: u32, vocab_size: usize) -> Result<(), WatermarkError> {
    if token as usize >= vocab_size {
        return Err(WatermarkError::TokenOutOfRange { token, vocab_size });
    }
    Ok(())
}

pub(crate) fn context(
    tokens: &[u32],
    prompt_len: usize,
    vocab_size: usize,
) -> Result<(), WatermarkError> {
    if prompt_len > tokens.len() {
        return Err(WatermarkError::PromptLengthExceedsContext);
    }
    for &id in tokens {
        token(id, vocab_size)?;
    }
    Ok(())
}

pub(crate) fn completion_end(
    tokens: &[u32],
    prompt_len: usize,
    eos: &[u32],
    vocab_size: usize,
) -> Result<usize, WatermarkError> {
    if prompt_len > tokens.len() {
        return Err(WatermarkError::PromptLengthExceedsContext);
    }
    let end = (prompt_len..tokens.len())
        .find(|&i| eos.contains(&tokens[i]))
        .unwrap_or(tokens.len());
    context(&tokens[..end], prompt_len, vocab_size)?;
    Ok(end)
}

pub(crate) fn count_detection(
    trials: usize,
    successes: usize,
    expected_rate: f64,
) -> CountDetection {
    CountDetection {
        trials,
        successes,
        expected_rate,
        observed_rate: (trials > 0).then(|| successes as f64 / trials as f64),
        z_score: (trials > 0).then(|| {
            (successes as f64 - trials as f64 * expected_rate)
                / (trials as f64 * expected_rate * (1.0 - expected_rate)).sqrt()
        }),
    }
}

/// Equivalent to adding delta to favored logits, without exp(delta) overflow.
/// Subtract the largest occupied logit bias before exponentiation. If no favored
/// token has mass, normalize the original distribution instead of underflowing it.
pub(crate) fn boost(probs: &mut [f32], favored: impl Fn(usize) -> bool, delta: f64) {
    let has_favored = probs
        .iter()
        .enumerate()
        .any(|(i, &p)| p > 0.0 && favored(i));
    let factor = |i| {
        if has_favored && !favored(i) {
            (-delta).exp()
        } else {
            1.0
        }
    };
    let total: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| f64::from(p) * factor(i))
        .sum();
    for (i, p) in probs.iter_mut().enumerate() {
        *p = (f64::from(*p) * factor(i) / total) as f32;
    }
}

pub(crate) fn prefix(domain: &[u8], key: &[u8; 32], parameters: &[usize]) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(key);
    for &parameter in parameters {
        hash.update((parameter as u64).to_le_bytes());
    }
    hash
}

pub(crate) fn context_hash(prefix: &Sha256, context: &[u32]) -> Sha256 {
    let mut hash = prefix.clone();
    for token in context {
        hash.update(token.to_le_bytes());
    }
    hash
}

/// Counter-mode SHA-256 bytes. Bounded integers use rejection, not modulo bias.
pub(crate) struct HashRng {
    prefix: Sha256,
    counter: u64,
    block: [u8; 32],
    offset: usize,
}

impl HashRng {
    pub(crate) fn new(prefix: &Sha256, tag: u8) -> Self {
        let mut prefix = prefix.clone();
        prefix.update([tag]);
        Self {
            prefix,
            counter: 0,
            block: [0; 32],
            offset: 32,
        }
    }

    fn next_u64(&mut self) -> u64 {
        if self.offset == 32 {
            let mut hash = self.prefix.clone();
            hash.update(self.counter.to_le_bytes());
            self.block = hash.finalize().into();
            self.counter += 1;
            self.offset = 0;
        }
        let value =
            u64::from_le_bytes(self.block[self.offset..self.offset + 8].try_into().unwrap());
        self.offset += 8;
        value
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        // 52 bits leave both half-bin endpoints exactly representable in f64.
        ((self.next_u64() >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0
    }

    pub(crate) fn below(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

pub(crate) fn permutation(hash: &Sha256, size: usize) -> Vec<usize> {
    let mut order: Vec<_> = (0..size).collect();
    let mut rng = HashRng::new(hash, 0);
    for i in (1..size).rev() {
        order.swap(i, rng.below(i + 1));
    }
    order
}

pub(crate) fn green_mask(hash: &Sha256, size: usize, count: usize) -> Vec<bool> {
    let mut mask = vec![false; size];
    for id in permutation(hash, size).into_iter().take(count) {
        mask[id] = true;
    }
    mask
}
