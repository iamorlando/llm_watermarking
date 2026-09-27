use super::*;

/// Actual vocabulary IDs in compact candidate order. IDs must be unique, in
/// range, and U32. Zero weights, rather than repeated IDs, represent exclusions.
#[derive(Clone)]
pub struct IndexedCandidates {
    ids: Tensor,
    vocab_size: usize,
}

impl IndexedCandidates {
    /// Validate candidate IDs with a scalar status readback.
    pub fn new(ids: &Tensor, vocab_size: usize) -> Result<Self> {
        let candidates = Self::new_trusted(ids, vocab_size)?;
        candidates.validate()?;
        Ok(candidates)
    }

    /// No readback. The caller guarantees unique IDs below `vocab_size`.
    /// Dimensions, dtype and vocabulary bounds are still checked. Invalid IDs
    /// cannot cause out-of-bounds GPU accesses, but give unspecified results.
    pub fn new_trusted(ids: &Tensor, vocab_size: usize) -> Result<Self> {
        let count = ids.dims1()?;
        if vocab_size == 0 || vocab_size > u32::MAX as usize || count == 0 || count > vocab_size {
            candle_core::bail!("indexed candidates require 1 <= K <= vocab_size <= u32::MAX");
        }
        if ids.dtype() != DType::U32 {
            candle_core::bail!("candidate token IDs must have dtype U32");
        }
        Ok(Self {
            ids: ids.contiguous()?,
            vocab_size,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let count = self.len();
        let mut valid = self
            .ids
            .lt(self.vocab_size as u32)?
            .to_dtype(DType::F32)?
            .min_all()?;
        if count > 1 {
            let (sorted, _) = crate::device_metadata::sort_u32(&self.ids)?;
            let unique = sorted
                .narrow(0, 1, count - 1)?
                .ne(&sorted.narrow(0, 0, count - 1)?)?
                .to_dtype(DType::F32)?
                .min_all()?;
            valid = valid.mul(&unique)?;
        }
        if valid.to_scalar::<f32>()? != 1.0 {
            candle_core::bail!("candidate token IDs must be unique and below vocab_size");
        }
        Ok(())
    }

    pub fn token_ids(&self) -> &Tensor {
        &self.ids
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn len(&self) -> usize {
        self.ids.elem_count()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn device(&self) -> &Device {
        self.ids.device()
    }

    pub(crate) fn check_vocab(&self, vocab_size: usize) -> Result<()> {
        if vocab_size != self.vocab_size {
            candle_core::bail!("candidate vocabulary does not match watermark vocabulary");
        }
        Ok(())
    }

    // Bounds protection even for a violated trusted precondition. Comparisons
    // remain integer operations, including for token IDs above 2^24.
    pub(crate) fn safe_ids(&self) -> Result<Tensor> {
        self.ids.clamp(0u32, (self.vocab_size - 1) as u32)
    }
}

/// Probability transformation bound to one immutable candidate ordering.
#[derive(Clone)]
pub struct PreparedIndexedWatermark {
    candidates: IndexedCandidates,
    prepared: PreparedWatermark,
    active: Option<Tensor>,
}

impl PreparedIndexedWatermark {
    pub(crate) fn new(candidates: &IndexedCandidates, prepared: PreparedWatermark) -> Self {
        Self {
            candidates: candidates.clone(),
            prepared,
            active: None,
        }
    }

    pub(crate) fn with_active(mut self, active: Tensor) -> Self {
        self.active = Some(active);
        self
    }

    pub fn candidates(&self) -> &IndexedCandidates {
        &self.candidates
    }

    /// Validate both IDs and weights. For a host-validated hot path use `apply_trusted`.
    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        self.candidates.validate()?;
        validate_probabilities(input)?;
        self.apply_trusted(input)
    }

    /// Return K weights in the original candidate order and input dtype/device,
    /// without readback. Requires the candidate and probability value contracts.
    pub fn apply_trusted(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.prepared.apply_trusted(input)?;
        match &self.active {
            Some(active) => active
                .broadcast_as(input.shape())?
                .where_cond(&output, input),
            None => Ok(output),
        }
    }
}

/// Keyed F32 scores in the original candidate order. Select argmax and map that
/// candidate index through `candidates().token_ids()` to get a vocabulary token.
#[derive(Clone)]
pub struct PreparedIndexedSampler {
    candidates: IndexedCandidates,
    prepared: PreparedSampler,
}

impl PreparedIndexedSampler {
    pub(crate) fn new(candidates: &IndexedCandidates, prepared: PreparedSampler) -> Self {
        Self {
            candidates: candidates.clone(),
            prepared,
        }
    }

    pub(crate) fn inverse(
        candidates: &IndexedCandidates,
        full_ranks: &Tensor,
        uniform: Tensor,
    ) -> Result<Self> {
        if candidates.vocab_size > MAX_INVERSE_TENSOR_VOCAB {
            candle_core::bail!("inverse-transform tensor vocabulary exceeds the F32 rank limit");
        }
        let candidate_ranks = full_ranks.index_select(&candidates.safe_ids()?, 0)?;
        let (score_ranks, order) = crate::device_metadata::sort_u32(&candidate_ranks)?;
        let (_, ranks) = crate::device_metadata::sort_u32(&order)?;
        Ok(Self::new(
            candidates,
            PreparedSampler {
                vocab_size: candidates.len(),
                device: candidates.device().clone(),
                operation: Selection::Inverse {
                    order,
                    ranks,
                    uniform,
                    score_ranks: Some(score_ranks),
                },
            },
        ))
    }

    pub fn candidates(&self) -> &IndexedCandidates {
        &self.candidates
    }

    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        self.candidates.validate()?;
        validate_probabilities(input)?;
        self.apply_trusted(input)
    }

    /// No readback; excluded candidates receive negative infinity.
    pub fn apply_trusted(&self, input: &Tensor) -> Result<Tensor> {
        self.prepared.apply_trusted(input)
    }
}

impl PreparedWatermark {
    /// Restrict existing dense metadata to a candidate set without expanding
    /// probabilities. Prefer scheme `prepare_indexed` for K-only hash generation.
    pub fn indexed(&self, candidates: &IndexedCandidates) -> Result<PreparedIndexedWatermark> {
        candidates.check_vocab(self.vocab_size)?;
        if !self.device.same_device(candidates.device()) {
            candle_core::bail!("candidate IDs and preparation must use the same device");
        }
        let ids = candidates.safe_ids()?;
        let prepared = match &self.operation {
            Reweighting::Identity => Self::identity(candidates.len(), &self.device),
            Reweighting::Bias { mask, delta } => Self::bias(mask.index_select(&ids, 0)?, *delta)?,
            Reweighting::Tournament { bits, depth } => {
                Self::tournament(bits.index_select(&ids, 1)?, candidates.len(), *depth)?
            }
        };
        Ok(PreparedIndexedWatermark::new(candidates, prepared))
    }
}

impl PreparedSampler {
    /// Restrict dense keyed metadata to a compact candidate set on the device.
    pub fn indexed(&self, candidates: &IndexedCandidates) -> Result<PreparedIndexedSampler> {
        candidates.check_vocab(self.vocab_size)?;
        if !self.device.same_device(candidates.device()) {
            candle_core::bail!("candidate IDs and preparation must use the same device");
        }
        match &self.operation {
            Selection::Exponential { gumbels } => Ok(PreparedIndexedSampler::new(
                candidates,
                Self::exponential(gumbels.index_select(&candidates.safe_ids()?, 0)?)?,
            )),
            Selection::Inverse { ranks, uniform, .. } => {
                PreparedIndexedSampler::inverse(candidates, ranks, uniform.clone())
            }
        }
    }
}

/// The host must dispatch the matching selection rule for each batch row.
#[derive(Clone)]
pub enum PreparedIndexedOperation {
    Probabilities(PreparedIndexedWatermark),
    SelectionScores(PreparedIndexedSampler),
}

impl PreparedIndexedOperation {
    pub fn candidates(&self) -> &IndexedCandidates {
        match self {
            Self::Probabilities(p) => p.candidates(),
            Self::SelectionScores(p) => p.candidates(),
        }
    }
    pub fn is_keyed_sampler(&self) -> bool {
        matches!(self, Self::SelectionScores(_))
    }
    pub fn apply_trusted(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Probabilities(p) => p.apply_trusted(input),
            Self::SelectionScores(p) => p.apply_trusted(input),
        }
    }
    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Probabilities(p) => p.apply(input),
            Self::SelectionScores(p) => p.apply(input),
        }
    }
}

/// Independent per-row metadata, including different keys, schemes and payloads.
/// A batch is an ordered collection of immutable preparations, not shared state.
/// Row operations are submitted on Candle's device; this is not a fused kernel.
#[derive(Clone)]
pub struct PreparedIndexedBatch {
    rows: Vec<PreparedIndexedOperation>,
}

impl PreparedIndexedBatch {
    pub fn new(rows: Vec<PreparedIndexedOperation>) -> Result<Self> {
        let Some(first) = rows.first() else {
            candle_core::bail!("watermark batch must not be empty");
        };
        for row in &rows {
            if row.candidates().len() != first.candidates().len()
                || !row
                    .candidates()
                    .device()
                    .same_device(first.candidates().device())
            {
                candle_core::bail!("watermark batch rows need the same candidate count and device");
            }
        }
        Ok(Self { rows })
    }
    pub fn rows(&self) -> &[PreparedIndexedOperation] {
        &self.rows
    }
    /// Return [B,K] F32 values. Each row retains its declared selection rule.
    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, false)
    }
    /// No readback. Every row must independently satisfy the trusted contracts.
    pub fn apply_trusted(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, true)
    }
    fn run(&self, input: &Tensor, trusted: bool) -> Result<Tensor> {
        if input.dims2()? != (self.rows.len(), self.rows[0].candidates().len()) {
            candle_core::bail!("watermark batch shape does not match preparations");
        }
        let output = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let weights = input.get(i)?;
                let result = if trusted {
                    row.apply_trusted(&weights)?
                } else {
                    row.apply(&weights)?
                };
                result.to_dtype(DType::F32)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&output, 0)
    }
}

impl PreparedIndexedWatermark {
    pub fn apply_traced(
        &self,
        input: &Tensor,
        options: &crate::trace::TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        self.candidates.validate()?;
        validate_probabilities(input)?;
        self.apply_traced_trusted(input, options)
    }
    pub fn apply_traced_trusted(
        &self,
        input: &Tensor,
        options: &crate::trace::TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        let trace = self
            .prepared
            .apply_traced_trusted(input, options)?
            .indexed(&self.candidates);
        match &self.active {
            Some(active) => trace.with_active(active),
            None => Ok(trace),
        }
    }
}

impl PreparedIndexedSampler {
    pub fn apply_traced(
        &self,
        input: &Tensor,
        options: &crate::trace::TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        self.candidates.validate()?;
        validate_probabilities(input)?;
        self.apply_traced_trusted(input, options)
    }
    pub fn apply_traced_trusted(
        &self,
        input: &Tensor,
        options: &crate::trace::TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        Ok(self
            .prepared
            .apply_traced_trusted(input, options)?
            .indexed(&self.candidates))
    }
}

impl PreparedIndexedOperation {
    pub fn apply_traced(
        &self,
        input: &Tensor,
        options: &crate::trace::TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        match self {
            Self::Probabilities(p) => p.apply_traced(input, options),
            Self::SelectionScores(p) => p.apply_traced(input, options),
        }
    }
    pub fn apply_traced_trusted(
        &self,
        input: &Tensor,
        options: &crate::trace::TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        match self {
            Self::Probabilities(p) => p.apply_traced_trusted(input, options),
            Self::SelectionScores(p) => p.apply_traced_trusted(input, options),
        }
    }
}

impl PreparedIndexedBatch {
    /// Per-row traces preserve the same keys, contexts, and selection rules as
    /// the ordinary batch. Each trace exposes its authoritative output row.
    pub fn apply_traced_trusted(
        &self,
        input: &Tensor,
        options: &[crate::trace::TraceOptions],
    ) -> Result<Vec<TensorSamplingTrace>> {
        if input.dims2()? != (self.rows.len(), self.rows[0].candidates().len())
            || options.len() != self.rows.len()
        {
            candle_core::bail!("trace batch shape/options do not match preparations");
        }
        self.rows
            .iter()
            .zip(options)
            .enumerate()
            .map(|(i, (row, options))| row.apply_traced_trusted(&input.get(i)?, options))
            .collect()
    }
}
