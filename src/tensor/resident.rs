use super::*;

/// Immutable device-resident token prefix and its logical lengths. Tokens have
/// shape [capacity]; length and prompt_len each have shape [1], all U32. Padding
/// after length is ignored. The complete prefix is required for SynthID repeats.
#[derive(Clone)]
pub struct DeviceHistory {
    tokens: Tensor,
    length: Tensor,
    prompt_len: Tensor,
    vocab_size: usize,
}

impl DeviceHistory {
    /// Validate lengths and committed token IDs with one scalar status readback.
    pub fn new(
        tokens: &Tensor,
        length: &Tensor,
        prompt_len: &Tensor,
        vocab_size: usize,
    ) -> Result<Self> {
        let history = Self::new_trusted(tokens, length, prompt_len, vocab_size)?;
        history.validate()?;
        Ok(history)
    }

    /// No readback. Requires prompt_len <= length <= capacity and all committed
    /// IDs < vocab_size. Invalid values cannot cause out-of-bounds kernel reads.
    /// All inputs must remain immutable while this history/preparation is in use.
    pub fn new_trusted(
        tokens: &Tensor,
        length: &Tensor,
        prompt_len: &Tensor,
        vocab_size: usize,
    ) -> Result<Self> {
        let capacity = tokens.dims1()?;
        if capacity > (u32::MAX as usize) - 2 || vocab_size == 0 || vocab_size > u32::MAX as usize {
            candle_core::bail!("device history dimensions exceed kernel limits");
        }
        if length.dims() != [1] || prompt_len.dims() != [1] {
            candle_core::bail!("history length and prompt length must have shape [1]");
        }
        for tensor in [tokens, length, prompt_len] {
            if tensor.dtype() != DType::U32 || !tensor.device().same_device(tokens.device()) {
                candle_core::bail!("device history inputs must be U32 on the same device");
            }
        }
        Ok(Self {
            tokens: tokens.clone(),
            length: length.clone(),
            prompt_len: prompt_len.clone(),
            vocab_size,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let capacity = self.tokens.elem_count();
        let mut valid = self
            .length
            .le(capacity as u32)?
            .mul(&self.prompt_len.le(&self.length)?)?
            .to_dtype(DType::F32)?
            .min_all()?;
        if capacity > 0 {
            let padding = Tensor::arange(0u32, capacity as u32, self.tokens.device())?
                .broadcast_ge(&self.length)?;
            let in_range = self.tokens.lt(self.vocab_size as u32)?;
            let ones = Tensor::ones(capacity, DType::U8, self.tokens.device())?;
            let ids_valid = padding
                .where_cond(&ones, &in_range)?
                .to_dtype(DType::F32)?
                .min_all()?;
            valid = valid.mul(&ids_valid)?;
        }
        if valid.to_scalar::<f32>()? != 1.0 {
            candle_core::bail!("invalid device history lengths or committed token IDs");
        }
        Ok(())
    }

    pub(crate) fn seed(
        &self,
        prefix: &crate::device_metadata::Seed,
        candidates: &IndexedCandidates,
        width: usize,
        repeat: bool,
    ) -> Result<(Tensor, Tensor)> {
        candidates.check_vocab(self.vocab_size)?;
        if !candidates.device().same_device(self.tokens.device()) {
            candle_core::bail!("device history and candidate IDs must use the same device");
        }
        let input = Tensor::cat(&[&self.length, &self.prompt_len, &self.tokens], 0)?;
        let prefix = prefix.tensor(&[], self.tokens.device())?;
        crate::device_metadata::context_seed(&prefix, &input, width, repeat)
    }
}
