use candle_core::{DType, Device, Result, Tensor};

use super::{SemStamp, SemStampDetection};
use crate::tensor::float_weights;

/// Reusable SemStamp hyperplanes on a Candle device. Embeddings stay on device;
/// only a validity status (strict calls) or signatures (detection) are read back.
#[derive(Clone)]
pub struct PreparedSemStamp {
    watermark: SemStamp,
    normals: Tensor,
    bit_weights: Tensor,
}

impl SemStamp {
    pub fn prepare_tensor(&self, device: &Device) -> Result<PreparedSemStamp> {
        let normals = Tensor::from_vec(
            self.normals
                .iter()
                .flatten()
                .map(|&x| x as f32)
                .collect::<Vec<_>>(),
            (self.normals.len(), self.embedding_dim),
            device,
        )?
        .t()?
        .contiguous()?;
        let bit_weights = Tensor::from_vec(
            (0..self.normals.len())
                .map(|i| (1u32 << i) as f32)
                .collect::<Vec<_>>(),
            (self.normals.len(), 1),
            device,
        )?;
        Ok(PreparedSemStamp {
            watermark: self.clone(),
            normals,
            bit_weights,
        })
    }

    /// Return a U8 acceptance mask for `[dimension]` or `[candidates, dimension]`.
    /// `previous_signature` is host-owned context from the last committed sentence.
    pub fn acceptance_tensor(
        &self,
        previous_signature: u32,
        candidates: &Tensor,
    ) -> Result<Tensor> {
        self.prepare_tensor(candidates.device())?
            .accepts(previous_signature, candidates)
    }

    /// Project `[sentences, dimension]` on device, then count transitions on CPU.
    /// Only one u32 signature per sentence is downloaded, never full embeddings.
    pub fn detect_tensor(
        &self,
        embeddings: &Tensor,
        prompt_len: usize,
    ) -> Result<SemStampDetection> {
        self.prepare_tensor(embeddings.device())?
            .detect(embeddings, prompt_len)
    }
}

impl PreparedSemStamp {
    fn embeddings(&self, input: &Tensor) -> Result<(Tensor, bool)> {
        if !input.device().same_device(self.normals.device()) {
            candle_core::bail!(
                "prepared SemStamp and embeddings must be on the same Candle device"
            );
        }
        let (matrix, single) = match *input.dims() {
            [dimension] if dimension == self.watermark.embedding_dim => (input.unsqueeze(0)?, true),
            [_, dimension] if dimension == self.watermark.embedding_dim => (input.clone(), false),
            _ => candle_core::bail!(
                "SemStamp expects [dimension] or [sentences, dimension] with dimension {}",
                self.watermark.embedding_dim
            ),
        };
        Ok((float_weights(&matrix)?, single))
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        let (matrix, _) = self.embeddings(input)?;
        if matrix.dim(0)? == 0 {
            return Ok(());
        }
        let absolute = matrix.abs()?;
        let finite = absolute
            .le(f32::MAX as f64)?
            .to_dtype(DType::F32)?
            .min_all()?;
        let nonzero = absolute.max(1)?.gt(0.0)?.to_dtype(DType::F32)?.min_all()?;
        if finite.mul(&nonzero)?.to_scalar::<f32>()? != 1.0 {
            candle_core::bail!("SemStamp embeddings must be finite and each have nonzero norm");
        }
        Ok(())
    }

    fn project_trusted(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let (matrix, single) = self.embeddings(input)?;
        if matrix.dim(0)? == 0 {
            return Ok((
                Tensor::zeros(0, DType::U32, input.device())?,
                Tensor::zeros(0, DType::F32, input.device())?,
            ));
        }
        // Log scaling avoids reciprocal underflow for large finite embeddings.
        let absolute = matrix.abs()?;
        let magnitude = absolute
            .log()?
            .broadcast_sub(&absolute.max_keepdim(1)?.log()?)?
            .exp()?;
        let scaled = matrix.ge(0.0)?.where_cond(&magnitude, &magnitude.neg()?)?;
        let normalized = scaled.broadcast_div(&scaled.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        let cosines = normalized.contiguous()?.matmul(&self.normals)?;
        let signatures = cosines
            .gt(0.0)?
            .to_dtype(DType::F32)?
            .matmul(&self.bit_weights)?
            .squeeze(1)?
            .to_dtype(DType::U32)?;
        let margins = cosines.abs()?.min(1)?;
        if single {
            Ok((signatures.squeeze(0)?, margins.squeeze(0)?))
        } else {
            Ok((signatures, margins))
        }
    }

    pub fn signatures(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.validate(embeddings)?;
        Ok(self.project_trusted(embeddings)?.0)
    }

    /// No readback; the host guarantees finite, nonzero embeddings.
    pub fn signatures_trusted(&self, embeddings: &Tensor) -> Result<Tensor> {
        Ok(self.project_trusted(embeddings)?.0)
    }

    pub fn accepts(&self, previous_signature: u32, candidates: &Tensor) -> Result<Tensor> {
        self.validate(candidates)?;
        self.accepts_trusted(previous_signature, candidates)
    }

    /// No value readback. Invalid embedding values can produce invalid masks.
    pub fn accepts_trusted(&self, previous_signature: u32, candidates: &Tensor) -> Result<Tensor> {
        if previous_signature as usize >= self.watermark.region_count {
            candle_core::bail!(
                "previous SemStamp signature is outside the configured region space"
            );
        }
        let (signatures, margins) = self.project_trusted(candidates)?;
        if signatures.elem_count() == 0 {
            return Tensor::zeros(signatures.shape(), DType::U8, candidates.device());
        }
        let mask = Tensor::from_vec(
            self.watermark
                .mask(previous_signature)
                .into_iter()
                .map(u8::from)
                .collect::<Vec<_>>(),
            self.watermark.region_count,
            candidates.device(),
        )?;
        let valid = mask
            .index_select(&signatures.flatten_all()?, 0)?
            .reshape(signatures.shape())?
            .to_dtype(DType::F32)?;
        valid
            .mul(&margins.ge(self.watermark.margin)?.to_dtype(DType::F32)?)?
            .gt(0.0)
    }

    pub fn detect(&self, embeddings: &Tensor, prompt_len: usize) -> Result<SemStampDetection> {
        let (sentences, _) = embeddings.dims2()?;
        if prompt_len > sentences {
            return Err(candle_core::Error::wrap(
                crate::WatermarkError::PromptLengthExceedsContext,
            ));
        }
        let signatures = self.signatures(embeddings)?.to_vec1::<u32>()?;
        self.watermark
            .detect_signatures(&signatures, prompt_len)
            .map_err(candle_core::Error::wrap)
    }
}
