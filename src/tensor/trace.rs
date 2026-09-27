//! Device-resident, opt-in diagnostics. Trusted methods perform no host readback;
//! strict methods read scalar validation status.
use super::*;
use crate::trace::{SelectionScoreKind, TraceKind, TraceOptions, TraceView};

#[derive(Clone)]
struct LayerReduction {
    green_mass: Tensor,
    total_after: Tensor,
}

#[derive(Clone)]
enum Metadata {
    Identity,
    Bias {
        mask: Tensor,
        delta: f64,
    },
    Tournament {
        bits: Tensor,
        initial: Tensor,
        reductions: Vec<LayerReduction>,
        depth: usize,
    },
    Selection(SelectionScoreKind),
    Inverse {
        uniform: Tensor,
        threshold: Tensor,
        cdf: Tensor,
        ranks: Tensor,
        score_ranks: Tensor,
    },
}

/// Values for a single observed SynthID layer. This is a distribution update,
/// not a sampled contestant/winner bracket. Consult the snapshot's active flag.
#[derive(Clone)]
pub struct TensorTournamentLayer {
    pub index: usize,
    pub green_mass: Tensor,
    pub g_values: Tensor,
    pub input_probabilities: Tensor,
    pub probabilities: Tensor,
    pub output_normalizer: Tensor,
}

/// Tensor inverse CDF bounds/threshold use normalized ordered weights. The
/// threshold is uniform * final CDF, matching the actual finite-precision sampler.
#[derive(Clone)]
pub struct TensorInverseTrace {
    pub uniform: Tensor,
    pub threshold: Tensor,
    pub total_weight: Tensor,
    pub ranks: Tensor,
    pub cdf_lower: Tensor,
    pub cdf_upper: Tensor,
}

/// All tensors remain on the sampling device. Row tensors have shape [M];
/// `active` and each layer's `green_mass` have shape [1]. IDs are U32, masks U8,
/// diagnostic probabilities/scores F32, raw output weights retain input dtype.
#[derive(Clone)]
pub struct TensorTraceSnapshot {
    pub token_ids: Tensor,
    pub vocab_size: usize,
    pub kind: TraceKind,
    pub active: Tensor,
    pub input_weights: Tensor,
    pub output_weights: Option<Tensor>,
    pub input_probabilities: Option<Tensor>,
    pub output_probabilities: Option<Tensor>,
    pub output_log_probabilities: Option<Tensor>,
    pub selection_scores: Option<Tensor>,
    pub score_kind: Option<SelectionScoreKind>,
    pub inverse: Option<TensorInverseTrace>,
    pub favored_mask: Option<Tensor>,
    pub bias_delta: Option<f64>,
    pub effective_bias_delta: Option<f64>,
    pub layers: Vec<TensorTournamentLayer>,
    pub total_layers: usize,
    pub captured_layers: usize,
}

/// An opt-in capture containing the authoritative sampling output. Sampling
/// tensors and metadata are retained by reference; callers must not mutate their
/// storage. Only O(depth) reduction tensors are retained for SynthID. Candidate
/// layer details are reconstructed when a bounded snapshot is requested.
#[derive(Clone)]
pub struct TensorSamplingTrace {
    input: Tensor,
    output: Tensor,
    token_ids: Option<Tensor>,
    vocab_size: usize,
    kind: TraceKind,
    active: Tensor,
    metadata: Metadata,
}

impl TensorSamplingTrace {
    /// Exact output of the traced application, including dtype and skip behavior.
    pub fn output(&self) -> &Tensor {
        &self.output
    }
    pub fn row_count(&self) -> usize {
        self.input.elem_count()
    }
    pub fn is_keyed_sampler(&self) -> bool {
        matches!(
            self.metadata,
            Metadata::Selection(_) | Metadata::Inverse { .. }
        )
    }

    pub(super) fn indexed(mut self, candidates: &IndexedCandidates) -> Self {
        self.token_ids = Some(candidates.token_ids().clone());
        self.vocab_size = candidates.vocab_size();
        self
    }

    pub(super) fn with_active(mut self, active: &Tensor) -> Result<Self> {
        self.output = active
            .broadcast_as(self.input.shape())?
            .where_cond(&self.output, &self.input)?;
        self.active = active.clone();
        Ok(self)
    }

    /// Validate snapshot row indices with a scalar readback. These indices are
    /// positions in the dense/indexed input row, not vocabulary IDs for compact
    /// input. None selects all rows, subject to the view's explicit bounds.
    pub fn snapshot(&self, rows: Option<&Tensor>, view: &TraceView) -> Result<TensorTraceSnapshot> {
        self.check_rows(rows, view)?;
        if let Some(rows) = rows {
            if rows
                .lt(self.row_count() as u32)?
                .to_dtype(DType::F32)?
                .min_all()?
                .to_scalar::<f32>()?
                != 1.0
            {
                candle_core::bail!("trace row index is out of range");
            }
        }
        self.snapshot_trusted(rows, view)
    }

    fn check_rows(&self, rows: Option<&Tensor>, view: &TraceView) -> Result<()> {
        let count = if let Some(rows) = rows {
            if rows.dtype() != DType::U32 || !rows.device().same_device(self.input.device()) {
                candle_core::bail!("trace row indices must be U32 on the sampling device");
            }
            rows.dims1()?
        } else {
            self.row_count()
        };
        let captured = match &self.metadata {
            Metadata::Tournament { reductions, .. } => reductions.len(),
            _ => 0,
        };
        view.check(count, captured)
            .map_err(candle_core::Error::wrap)?;
        if self.row_count() > u32::MAX as usize {
            candle_core::bail!("trace row exceeds U32 indexing");
        }
        Ok(())
    }

    /// No readback. The host guarantees all row indices are in range. Bounds
    /// protection prevents invalid trusted indices from reading outside storage.
    /// Snapshot subsets are never renormalized; omitted mass stays omitted.
    pub fn snapshot_trusted(
        &self,
        rows: Option<&Tensor>,
        view: &TraceView,
    ) -> Result<TensorTraceSnapshot> {
        self.check_rows(rows, view)?;
        let rows = rows
            .map(|r| r.clamp(0u32, (self.row_count() - 1) as u32)?.contiguous())
            .transpose()?;
        let select = |tensor: &Tensor| -> Result<Tensor> {
            match &rows {
                Some(rows) => tensor.contiguous()?.index_select(rows, 0),
                None => Ok(tensor.clone()),
            }
        };
        let token_ids = match &self.token_ids {
            Some(ids) => select(ids)?,
            None => match &rows {
                Some(rows) => rows.clone(),
                None => Tensor::arange(0u32, self.row_count() as u32, self.input.device())?,
            },
        };
        let input_weights = select(&self.input)?;
        let output_weights = if self.is_keyed_sampler() {
            None
        } else {
            Some(select(&self.output)?)
        };
        let input_probabilities = if view.probabilities {
            Some(select(&normalize(&float_weights(&self.input)?)?)?)
        } else {
            None
        };
        let post = if !self.is_keyed_sampler() && (view.probabilities || view.log_probabilities) {
            Some(select(&normalize(&float_weights(&self.output)?)?)?)
        } else {
            None
        };
        let output_log_probabilities = if view.log_probabilities {
            post.as_ref().map(Tensor::log).transpose()?
        } else {
            None
        };
        let output_probabilities = if view.probabilities { post } else { None };
        let mut snapshot = TensorTraceSnapshot {
            token_ids,
            vocab_size: self.vocab_size,
            kind: self.kind,
            active: self.active.clone(),
            input_weights,
            output_weights,
            input_probabilities,
            output_probabilities,
            output_log_probabilities,
            selection_scores: None,
            score_kind: None,
            inverse: None,
            favored_mask: None,
            bias_delta: None,
            effective_bias_delta: None,
            layers: Vec::new(),
            total_layers: 0,
            captured_layers: 0,
        };
        match &self.metadata {
            Metadata::Identity => {}
            Metadata::Bias { mask, delta } => {
                snapshot.bias_delta = Some(*delta);
                snapshot.effective_bias_delta = Some(delta.min(512.0));
                if view.partition {
                    snapshot.favored_mask = Some(select(mask)?.broadcast_mul(&self.active)?);
                }
            }
            Metadata::Selection(kind) => {
                snapshot.selection_scores = Some(select(&self.output)?);
                snapshot.score_kind = Some(*kind);
            }
            Metadata::Inverse {
                uniform,
                threshold,
                cdf,
                ranks,
                score_ranks,
            } => {
                snapshot.selection_scores = Some(select(&self.output)?);
                snapshot.score_kind = Some(SelectionScoreKind::NegativeRank);
                let positions = select(ranks)?;
                let previous = positions
                    .clamp(1u32, (self.row_count() - 1).max(1) as u32)?
                    .broadcast_sub(&Tensor::new(1u32, self.input.device())?)?;
                let lower = cdf.index_select(&previous, 0)?;
                let zero = lower.zeros_like()?;
                snapshot.inverse = Some(TensorInverseTrace {
                    uniform: uniform.clone(),
                    threshold: threshold.clone(),
                    total_weight: cdf.narrow(0, self.row_count() - 1, 1)?,
                    ranks: score_ranks.index_select(&positions, 0)?,
                    cdf_lower: positions.gt(0u32)?.where_cond(&lower, &zero)?,
                    cdf_upper: cdf.index_select(&positions, 0)?,
                });
            }
            Metadata::Tournament {
                bits,
                initial,
                reductions,
                depth,
            } => {
                snapshot.total_layers = *depth;
                snapshot.captured_layers = reductions.len();
                if view.layers {
                    let bits = match &rows {
                        Some(rows) => bits.index_select(rows, 1)?,
                        None => bits.clone(),
                    };
                    let initial = select(initial)?;
                    let mut probabilities = initial.clone();
                    let active = self.active.broadcast_as(initial.shape())?;
                    for (index, reduction) in reductions.iter().enumerate() {
                        let g = layer_bits(&bits, index)?;
                        let factor = g.affine(1.0, 1.0)?.broadcast_sub(&reduction.green_mass)?;
                        let before = probabilities.clone();
                        probabilities = probabilities
                            .mul(&factor)?
                            .broadcast_div(&reduction.total_after)?;
                        // Inactive device-history preparations did not watermark this
                        // step. Mask placeholders; serializers should omit its layers.
                        snapshot.layers.push(TensorTournamentLayer {
                            index,
                            green_mass: reduction
                                .green_mass
                                .mul(&self.active.to_dtype(DType::F32)?)?,
                            g_values: g.to_dtype(DType::U8)?.mul(&active)?,
                            input_probabilities: active.where_cond(&before, &initial)?,
                            probabilities: active.where_cond(&probabilities, &initial)?,
                            output_normalizer: reduction.total_after.clone(),
                        });
                    }
                }
            }
        }
        Ok(snapshot)
    }
}

fn layer_bits(bits: &Tensor, layer: usize) -> Result<Tensor> {
    let packed = bits
        .narrow(0, layer / 8, 1)?
        .squeeze(0)?
        .to_dtype(DType::F32)?;
    let shifted = packed
        .affine(1.0 / (1u32 << (layer % 8)) as f64, 0.0)?
        .floor()?;
    let even = shifted.affine(0.5, 0.0)?.floor()?.affine(2.0, 0.0)?;
    shifted.sub(&even)
}

impl PreparedWatermark {
    pub fn apply_traced(
        &self,
        input: &Tensor,
        options: &TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        check_row(input, self.vocab_size, &self.device)?;
        validate_probabilities(input)?;
        self.apply_traced_trusted(input, options)
    }

    /// Separate opt-in entry point. Ordinary `apply_trusted` performs no trace
    /// branches, allocations or extra kernels. This method performs no readback.
    pub fn apply_traced_trusted(
        &self,
        input: &Tensor,
        options: &TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        options.validate().map_err(candle_core::Error::wrap)?;
        let (output, kind, metadata) = match &self.operation {
            Reweighting::Identity => (
                self.apply_trusted(input)?,
                TraceKind::Identity,
                Metadata::Identity,
            ),
            Reweighting::Bias { mask, delta } => (
                self.apply_trusted(input)?,
                TraceKind::Bias,
                Metadata::Bias {
                    mask: mask.clone(),
                    delta: *delta,
                },
            ),
            Reweighting::Tournament { bits, depth } => {
                let weights = check_row(input, self.vocab_size, &self.device)?;
                let mut probabilities = normalize(&weights)?;
                let initial = probabilities.clone();
                let mut reductions = Vec::with_capacity((*depth).min(options.max_layers));
                // Keep the ordinary operation order, including per-byte conversion,
                // to produce exactly the same authoritative output tensor.
                for byte in 0..depth.div_ceil(8) {
                    let packed = bits.narrow(0, byte, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
                    for bit in 0..8.min(depth - byte * 8) {
                        let shifted = packed.affine(1.0 / (1u32 << bit) as f64, 0.0)?.floor()?;
                        let even = shifted.affine(0.5, 0.0)?.floor()?.affine(2.0, 0.0)?;
                        let g = shifted.sub(&even)?;
                        let green_mass = probabilities.mul(&g)?.sum_keepdim(0)?.clamp(0.0, 1.0)?;
                        let factor = g.affine(1.0, 1.0)?.broadcast_sub(&green_mass)?;
                        probabilities = probabilities.mul(&factor)?;
                        let total_after = probabilities.sum_keepdim(0)?;
                        probabilities = probabilities.broadcast_div(&total_after)?;
                        if byte * 8 + bit < options.max_layers {
                            reductions.push(LayerReduction {
                                green_mass,
                                total_after,
                            });
                        }
                    }
                }
                (
                    probabilities.to_dtype(input.dtype())?,
                    TraceKind::SynthId,
                    Metadata::Tournament {
                        bits: bits.clone(),
                        initial,
                        reductions,
                        depth: *depth,
                    },
                )
            }
        };
        Ok(TensorSamplingTrace {
            input: input.clone(),
            output,
            token_ids: None,
            vocab_size: self.vocab_size,
            kind,
            active: Tensor::new(&[u8::from(kind != TraceKind::Identity)], input.device())?,
            metadata,
        })
    }
}

impl PreparedSampler {
    pub fn apply_traced(
        &self,
        input: &Tensor,
        options: &TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        check_row(input, self.vocab_size, &self.device)?;
        validate_probabilities(input)?;
        self.apply_traced_trusted(input, options)
    }

    pub fn apply_traced_trusted(
        &self,
        input: &Tensor,
        options: &TraceOptions,
    ) -> Result<TensorSamplingTrace> {
        options.validate().map_err(candle_core::Error::wrap)?;
        let (output, kind, metadata) = match &self.operation {
            Selection::Exponential { .. } => (
                self.apply_trusted(input)?,
                TraceKind::ExponentialRace,
                Metadata::Selection(SelectionScoreKind::GumbelMax),
            ),
            Selection::Inverse {
                order,
                ranks,
                uniform,
                score_ranks,
            } => {
                let weights = check_row(input, self.vocab_size, &self.device)?;
                let ordered = normalize(&weights)?.index_select(order, 0)?;
                let cdf = cumulative_sum(&ordered)?;
                let threshold = cdf
                    .narrow(0, self.vocab_size - 1, 1)?
                    .broadcast_mul(uniform)?;
                let eligible = cdf
                    .broadcast_gt(&threshold)?
                    .to_dtype(DType::F32)?
                    .mul(&ordered.gt(0.0)?.to_dtype(DType::F32)?)?
                    .gt(0.0)?;
                let score_ranks = match score_ranks {
                    Some(ranks) => ranks.clone(),
                    None => Tensor::arange(0u32, self.vocab_size as u32, &self.device)?,
                };
                let negative_rank = score_ranks.to_dtype(DType::F32)?.neg()?;
                let excluded = Tensor::full(f32::NEG_INFINITY, self.vocab_size, &self.device)?;
                let output = eligible
                    .where_cond(&negative_rank, &excluded)?
                    .index_select(ranks, 0)?;
                (
                    output,
                    TraceKind::InverseTransform,
                    Metadata::Inverse {
                        uniform: uniform.clone(),
                        threshold,
                        cdf,
                        ranks: ranks.clone(),
                        score_ranks,
                    },
                )
            }
        };
        Ok(TensorSamplingTrace {
            input: input.clone(),
            output,
            token_ids: None,
            vocab_size: self.vocab_size,
            kind,
            active: Tensor::new(&[1u8], input.device())?,
            metadata,
        })
    }
}
