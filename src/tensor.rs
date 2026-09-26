//! Device-preserving Candle operations on one dense vocabulary row at a time.
//! Prepared operations cache only keyed metadata, never model probabilities.
//! `apply` validates values using a scalar device-to-host status read;
//! `apply_trusted` checks shape/dtype/device but trusts the host's weight values.
#![doc = include_str!("../docs/candle.md")]

use candle_core::{DType, Device, Result, Tensor};

/// Largest inverse-transform vocabulary with distinct integer ranks in F32 scores.
pub const MAX_INVERSE_TENSOR_VOCAB: usize = 1 << 24;

/// A prepared probability transformation, bound to a vocabulary and device.
#[derive(Clone)]
pub struct PreparedWatermark {
    vocab_size: usize,
    device: Device,
    operation: Reweighting,
}

#[derive(Clone)]
enum Reweighting {
    Identity,
    Bias { mask: Tensor, delta: f64 },
    Tournament { bits: Tensor, depth: usize },
}

/// A prepared keyed sampling transformation. Outputs F32 selection scores;
/// the host must take argmax, not sample categorically or apply softmax to them.
#[derive(Clone)]
pub struct PreparedSampler {
    vocab_size: usize,
    device: Device,
    operation: Selection,
}

#[derive(Clone)]
enum Selection {
    Exponential {
        gumbels: Tensor,
    },
    Inverse {
        order: Tensor,
        ranks: Tensor,
        uniform: f64,
    },
}

pub(crate) fn float_weights(input: &Tensor) -> Result<Tensor> {
    match input.dtype() {
        DType::F16 | DType::BF16 | DType::F32 => input.to_dtype(DType::F32),
        dtype => candle_core::bail!("watermark tensors require F16, BF16, or F32, got {dtype:?}"),
    }
}

fn check_row(input: &Tensor, vocab_size: usize, device: &Device) -> Result<Tensor> {
    let actual = input.dims1()?;
    if actual != vocab_size || actual == 0 {
        candle_core::bail!(
            "expected a nonempty vocabulary row of length {vocab_size}, got {actual}"
        );
    }
    if !device.same_device(input.device()) {
        candle_core::bail!("prepared watermark and input must be on the same Candle device");
    }
    float_weights(input)
}

/// Validate a nonempty dense probability row without downloading its values.
/// One scalar status is read back, which synchronizes GPU execution.
pub fn validate_probabilities(input: &Tensor) -> Result<()> {
    if input.dims1()? == 0 {
        candle_core::bail!("probability tensor must not be empty");
    }
    let weights = float_weights(input)?;
    let nonnegative = weights.ge(0.0)?.to_dtype(DType::F32)?;
    let finite = weights.le(f32::MAX as f64)?.to_dtype(DType::F32)?;
    let valid = nonnegative.mul(&finite)?.min_all()?;
    let has_mass = weights.gt(0.0)?.to_dtype(DType::F32)?.max_all()?;
    if valid.mul(&has_mass)?.to_scalar::<f32>()? != 1.0 {
        candle_core::bail!("probabilities must be finite, nonnegative, and have positive mass");
    }
    Ok(())
}

fn normalize(weights: &Tensor) -> Result<Tensor> {
    // Direct division by F32::MAX can flush the GPU's reciprocal to zero.
    let scaled = weights
        .log()?
        .broadcast_sub(&weights.max_keepdim(0)?.log()?)?
        .exp()?;
    scaled.broadcast_div(&scaled.sum_keepdim(0)?)
}

impl PreparedWatermark {
    pub(crate) fn identity(vocab_size: usize, device: &Device) -> Self {
        Self {
            vocab_size,
            device: device.clone(),
            operation: Reweighting::Identity,
        }
    }

    pub(crate) fn bias(mask: Tensor, delta: f64) -> Result<Self> {
        Ok(Self {
            vocab_size: mask.dims1()?,
            device: mask.device().clone(),
            operation: Reweighting::Bias { mask, delta },
        })
    }

    pub(crate) fn tournament(bits: Tensor, vocab_size: usize, depth: usize) -> Result<Self> {
        Ok(Self {
            vocab_size,
            device: bits.device().clone(),
            operation: Reweighting::Tournament { bits, depth },
        })
    }

    /// Return probabilities on the same device and in the same dtype as input.
    /// Warmup/repeated SynthID contexts preserve input weights exactly.
    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        check_row(input, self.vocab_size, &self.device)?;
        validate_probabilities(input)?;
        self.apply_trusted(input)
    }

    /// No value readback. The host must provide finite nonnegative weights with
    /// positive mass; invalid values can produce NaNs. Shape/device/dtype are checked.
    pub fn apply_trusted(&self, input: &Tensor) -> Result<Tensor> {
        let weights = check_row(input, self.vocab_size, &self.device)?;
        let output = match &self.operation {
            Reweighting::Identity => return Ok(input.clone()),
            Reweighting::Bias { mask, delta } => {
                // Beyond 512, every red/green F32 mass ratio rounds to zero;
                // capping avoids infinity and loss of all within-green precision.
                let log_weights = weights.log()?;
                let boosted = log_weights.affine(1.0, delta.min(512.0))?;
                let logits = mask.where_cond(&boosted, &log_weights)?;
                let shifted = logits.broadcast_sub(&logits.max_keepdim(0)?)?;
                let exp = shifted.exp()?;
                exp.broadcast_div(&exp.sum_keepdim(0)?)?
            }
            Reweighting::Tournament { bits, depth } => {
                let mut probabilities = normalize(&weights)?;
                for byte in 0..depth.div_ceil(8) {
                    let packed = bits.narrow(0, byte, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
                    for bit in 0..8.min(depth - byte * 8) {
                        let shifted = packed.affine(1.0 / (1u32 << bit) as f64, 0.0)?.floor()?;
                        let even = shifted.affine(0.5, 0.0)?.floor()?.affine(2.0, 0.0)?;
                        let g = shifted.sub(&even)?;
                        let green_mass = probabilities.mul(&g)?.sum_keepdim(0)?.clamp(0.0, 1.0)?;
                        let factor = g.affine(1.0, 1.0)?.broadcast_sub(&green_mass)?;
                        probabilities = probabilities.mul(&factor)?;
                        probabilities =
                            probabilities.broadcast_div(&probabilities.sum_keepdim(0)?)?;
                    }
                }
                probabilities
            }
        };
        output.to_dtype(input.dtype())
    }
}

impl PreparedSampler {
    pub(crate) fn exponential(gumbels: Tensor) -> Result<Self> {
        Ok(Self {
            vocab_size: gumbels.dims1()?,
            device: gumbels.device().clone(),
            operation: Selection::Exponential { gumbels },
        })
    }

    pub(crate) fn inverse(order: Tensor, ranks: Tensor, uniform: f64) -> Result<Self> {
        let vocab_size = order.dims1()?;
        if vocab_size > MAX_INVERSE_TENSOR_VOCAB {
            candle_core::bail!("inverse-transform tensor vocabulary exceeds the F32 rank limit of {MAX_INVERSE_TENSOR_VOCAB}");
        }
        Ok(Self {
            vocab_size,
            device: order.device().clone(),
            operation: Selection::Inverse {
                order,
                ranks,
                uniform,
            },
        })
    }

    /// Return F32 scores on the input device, with excluded tokens at -infinity.
    /// The caller selects the token by argmax; no token ID is read back here.
    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        check_row(input, self.vocab_size, &self.device)?;
        validate_probabilities(input)?;
        self.apply_trusted(input)
    }

    /// Like `apply`, trusting the host's finite, nonnegative, positive-mass inputs.
    pub fn apply_trusted(&self, input: &Tensor) -> Result<Tensor> {
        let weights = check_row(input, self.vocab_size, &self.device)?;
        match &self.operation {
            Selection::Exponential { gumbels } => weights.log()?.add(gumbels),
            Selection::Inverse {
                order,
                ranks,
                uniform,
            } => {
                let ordered = normalize(&weights)?.index_select(order, 0)?;
                let cdf = cumulative_sum(&ordered)?;
                // Rounding a 52-bit uniform to F32 must not create the endpoint 1.
                let u = (*uniform as f32).min(f32::from_bits(1.0f32.to_bits() - 1));
                let target = cdf
                    .narrow(0, self.vocab_size - 1, 1)?
                    .affine(f64::from(u), 0.0)?;
                let eligible = cdf
                    .broadcast_gt(&target)?
                    .to_dtype(DType::F32)?
                    .mul(&ordered.gt(0.0)?.to_dtype(DType::F32)?)?
                    .gt(0.0)?;
                let negative_rank = Tensor::arange(0u32, self.vocab_size as u32, &self.device)?
                    .to_dtype(DType::F32)?
                    .neg()?;
                let excluded = Tensor::full(f32::NEG_INFINITY, self.vocab_size, &self.device)?;
                eligible
                    .where_cond(&negative_rank, &excluded)?
                    .index_select(ranks, 0)
            }
        }
    }
}

/// Candle's pinned cumsum builds a V-by-V matrix. A parallel prefix scan uses
/// O(V) temporary storage and O(log V) ordinary Candle operations instead.
fn cumulative_sum(input: &Tensor) -> Result<Tensor> {
    let size = input.dims1()?;
    let mut sum = input.clone();
    let mut offset = 1;
    while offset < size {
        let zeros = Tensor::zeros(offset, input.dtype(), input.device())?;
        let prefix = sum.narrow(0, 0, size - offset)?;
        let shifted = Tensor::cat(&[&zeros, &prefix], 0)?;
        sum = sum.add(&shifted)?;
        offset *= 2;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_scan_handles_non_power_of_two_lengths_and_exclusions() -> Result<()> {
        for values in [vec![3.0f32], vec![1.0, 0.0, 2.0, 0.0, 4.0], vec![1.0; 17]] {
            let input = Tensor::new(values.as_slice(), &Device::Cpu)?;
            let mut total = 0.0;
            let expected: Vec<_> = values
                .iter()
                .map(|v| {
                    total += v;
                    total
                })
                .collect();
            assert_eq!(cumulative_sum(&input)?.to_vec1::<f32>()?, expected);
        }
        Ok(())
    }
}
