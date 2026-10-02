//! Binary metadata operations: device IDs/history/positions plus a cached seed.
use super::*;
use candle_core::{CustomOp2, DType};

#[derive(Clone, Copy)]
pub(super) enum InputMode {
    SynthId = 0,
    Exponential = 1,
    Context = 2,
    Position = 3,
    Uniform = 4,
    Slot = 5,
    SortInit = 6,
    SortStep = 7,
}

pub(super) struct InputOp {
    mode: InputMode,
    seed_len: usize,
    input_len: usize,
    size: usize,
    parameter: usize,
    auxiliary: usize,
}

impl InputOp {
    pub(super) fn shape(&self) -> Shape {
        match self.mode {
            InputMode::SynthId => (self.parameter, self.size).into(),
            InputMode::SortInit | InputMode::SortStep => (2, self.size).into(),
            _ => self.size.into(),
        }
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn dtype(&self) -> DType {
        match self.mode {
            InputMode::SynthId | InputMode::Context | InputMode::Position => DType::U8,
            InputMode::Exponential | InputMode::Uniform => DType::F32,
            InputMode::Slot | InputMode::SortInit | InputMode::SortStep => DType::U32,
        }
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn parameters(&self) -> [u32; 6] {
        [
            self.seed_len as u32,
            self.input_len as u32,
            self.size as u32,
            self.mode as u32,
            self.parameter as u32,
            self.auxiliary as u32,
        ]
    }
}

impl CustomOp2 for InputOp {
    fn name(&self) -> &'static str {
        "watermark-device-input-v1"
    }
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (start, end) = offsets(l1)?;
        let seed = &s1.as_slice::<u8>()?[start..end];
        let (start, end) = offsets(l2)?;
        let input = &s2.as_slice::<u32>()?[start..end];
        debug_assert_eq!(seed.len(), self.seed_len);
        debug_assert_eq!(input.len(), self.input_len);
        let hash = Sha256::new().chain_update(seed);
        let output = match self.mode {
            InputMode::SynthId => {
                let mut output = vec![0; self.size * self.parameter];
                for (i, token) in input.iter().enumerate() {
                    let digest = hash.clone().chain_update(token.to_le_bytes()).finalize();
                    for b in 0..self.parameter {
                        output[b * self.size + i] = digest[b];
                    }
                }
                CpuStorage::U8(output)
            }
            InputMode::Exponential => CpuStorage::F32(
                input
                    .iter()
                    .map(|token| {
                        let h = hash.clone().chain_update(token.to_le_bytes());
                        let u = crate::common::HashRng::new(&h, 2).uniform();
                        -(-u.ln()).ln() as f32
                    })
                    .collect(),
            ),
            InputMode::Context => {
                let length = (input[0] as usize).min(input.len() - 2);
                let prompt = (input[1] as usize).min(length);
                let tokens = &input[2..2 + length];
                let width = self.parameter;
                let mut active = length >= width;
                if active && self.auxiliary != 0 {
                    let start = prompt.max(width).max(length.saturating_sub(1024));
                    active = !(start..length)
                        .any(|end| tokens[end - width..end] == tokens[length - width..]);
                }
                let mut bytes = seed.to_vec();
                for i in 0..width {
                    let token = if length >= width {
                        tokens[length - width + i]
                    } else {
                        0
                    };
                    bytes.extend_from_slice(&token.to_le_bytes());
                }
                bytes.push(u8::from(active));
                CpuStorage::U8(bytes)
            }
            InputMode::Position => {
                let mut bytes = seed.to_vec();
                bytes.extend_from_slice(
                    &((input[0] as usize % self.parameter) as u64).to_le_bytes(),
                );
                CpuStorage::U8(bytes)
            }
            InputMode::Uniform => {
                let uniform = crate::common::HashRng::new(&hash, 1).uniform() as f32;
                CpuStorage::F32(vec![uniform.min(f32::from_bits(1.0f32.to_bits() - 1))])
            }
            InputMode::Slot => CpuStorage::U32(vec![
                crate::common::HashRng::new(&hash, 1).below(self.parameter) as u32
            ]),
            InputMode::SortInit => {
                let mut output = vec![u32::MAX; 2 * self.size];
                output[..input.len()].copy_from_slice(input);
                for i in 0..self.size {
                    output[self.size + i] = i as u32;
                }
                CpuStorage::U32(output)
            }
            InputMode::SortStep => {
                let mut output = vec![0; 2 * self.size];
                for i in 0..self.size {
                    let j = i ^ self.parameter;
                    let a = (input[i], input[self.size + i]);
                    let b = (input[j], input[self.size + j]);
                    let take_min = ((i & self.auxiliary) == 0) == ((i & self.parameter) == 0);
                    let (value, index) = if take_min { a.min(b) } else { a.max(b) };
                    output[i] = value;
                    output[self.size + i] = index;
                }
                CpuStorage::U32(output)
            }
        };
        Ok((output, self.shape()))
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        super::metal::input_forward(self, s1, l1, s2, l2)
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        super::cuda::input_forward(self, s1, l1, s2, l2)
    }
}

pub(super) fn offsets(layout: &Layout) -> Result<(usize, usize)> {
    layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("device metadata input must be contiguous".into()))
}

fn run(
    seed: &Tensor,
    input: &Tensor,
    mode: InputMode,
    size: usize,
    parameter: usize,
    auxiliary: usize,
) -> Result<Tensor> {
    if seed.dtype() != DType::U8
        || input.dtype() != DType::U32
        || !seed.device().same_device(input.device())
    {
        candle_core::bail!("metadata seed and U32 input must use the same device");
    }
    for n in [
        seed.elem_count(),
        input.elem_count(),
        size,
        parameter,
        auxiliary,
    ] {
        if n > u32::MAX as usize {
            candle_core::bail!("device metadata input exceeds kernel limits");
        }
    }
    seed.contiguous()?.apply_op2_no_bwd(
        &input.contiguous()?,
        &InputOp {
            mode,
            seed_len: seed.elem_count(),
            input_len: input.elem_count(),
            size,
            parameter,
            auxiliary,
        },
    )
}

pub(crate) fn synthid_indexed(seed: &Tensor, ids: &Tensor, depth: usize) -> Result<Tensor> {
    run(
        seed,
        ids,
        InputMode::SynthId,
        ids.elem_count(),
        depth.div_ceil(8),
        0,
    )
}
pub(crate) fn exponential_indexed(seed: &Tensor, ids: &Tensor) -> Result<Tensor> {
    run(seed, ids, InputMode::Exponential, ids.elem_count(), 0, 0)
}
pub(crate) fn context_seed(
    seed: &Tensor,
    input: &Tensor,
    width: usize,
    repeat: bool,
) -> Result<(Tensor, Tensor)> {
    let len = seed.elem_count() + 4 * width;
    let both = run(
        seed,
        input,
        InputMode::Context,
        len + 1,
        width,
        usize::from(repeat),
    )?;
    Ok((both.narrow(0, 0, len)?, both.narrow(0, len, 1)?))
}
pub(crate) fn position_seed(seed: &Tensor, position: &Tensor, period: usize) -> Result<Tensor> {
    if position.dims() != [1] || position.dtype() != DType::U32 {
        candle_core::bail!("position must be a U32 tensor of shape [1]");
    }
    run(
        seed,
        position,
        InputMode::Position,
        seed.elem_count() + 8,
        period,
        0,
    )
}
pub(crate) fn uniform(seed: &Tensor) -> Result<Tensor> {
    let unused = Tensor::zeros(1, DType::U32, seed.device())?;
    run(seed, &unused, InputMode::Uniform, 1, 0, 0)
}
pub(crate) fn payload_slot(seed: &Tensor, bound: usize) -> Result<Tensor> {
    let unused = Tensor::zeros(1, DType::U32, seed.device())?;
    run(seed, &unused, InputMode::Slot, 1, bound, 0)
}

/// Candle's Metal argsort dispatches K threads in one threadgroup; large rows
/// exceed hardware limits. Use a global bitonic network above 1,024 entries,
/// with O(K) storage and no host indices or readback. Lexicographic comparison
/// on (value, original index) handles duplicates and padding deterministically.
pub(crate) fn sort_u32(input: &Tensor) -> Result<(Tensor, Tensor)> {
    let count = input.dims1()?;
    if count == 0 || input.dtype() != DType::U32 {
        candle_core::bail!("metadata sort requires a nonempty U32 row");
    }
    if count <= 1024 || input.device().is_cpu() {
        let input = input.contiguous()?;
        // Registry Candle 0.11 CPU argsort ignores the storage offset. A
        // contiguous view can still have an offset; materialize only that case.
        let input = if input.device().is_cpu() && input.layout().start_offset() != 0 {
            input.force_contiguous()?
        } else {
            input
        };
        return input.sort_last_dim(true);
    }
    let size = count
        .checked_next_power_of_two()
        .filter(|&n| n <= (u32::MAX as usize) / 2)
        .ok_or_else(|| candle_core::Error::Msg("candidate sort exceeds kernel limits".into()))?;
    let unused = Tensor::zeros(1, DType::U8, input.device())?;
    let mut sorted = run(&unused, input, InputMode::SortInit, size, 0, 0)?;
    let mut stage = 2;
    while stage <= size {
        let mut distance = stage / 2;
        while distance > 0 {
            sorted = run(
                &unused,
                &sorted.flatten_all()?,
                InputMode::SortStep,
                size,
                distance,
                stage,
            )?;
            distance /= 2;
        }
        stage *= 2;
    }
    Ok((
        sorted.get(0)?.narrow(0, 0, count)?,
        sorted.get(1)?.narrow(0, 0, count)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vectors(device: &Device) -> Result<()> {
        // IDs above the F32 exact-integer range must remain integer hash inputs.
        let ids = [u32::MAX - 1, 16_777_217, 0, 257, 7];
        let packed = Tensor::new(&[99u32, ids[0], ids[1], ids[2], ids[3], ids[4], 88], device)?;
        let candidates = packed.narrow(0, 1, ids.len())?;
        for len in [0, 1, 42, 43, 51, 52, 55, 56, 63, 64, 65, 119, 120, 257] {
            let mut data = vec![99u8; 3];
            data.extend((0..len).map(|i| (i * 37 + 11) as u8));
            let seed = Tensor::new(data.as_slice(), device)?.narrow(0, 3, len)?;
            let hash = Sha256::new().chain_update(&data[3..]);
            let bits = synthid_indexed(&seed, &candidates, 256)?.to_vec2::<u8>()?;
            let gumbels = exponential_indexed(&seed, &candidates)?.to_vec1::<f32>()?;
            for (i, &id) in ids.iter().enumerate() {
                let h = hash.clone().chain_update(id.to_le_bytes());
                let digest = h.clone().finalize();
                for b in 0..32 {
                    assert_eq!(bits[b][i], digest[b], "len={len}, id={id}");
                }
                let u = crate::common::HashRng::new(&h, 2).uniform();
                assert!((gumbels[i] as f64 - -(-u.ln()).ln()).abs() < 3e-6);
            }
            for bound in [1, 3, 17, 256, 100_003] {
                assert_eq!(
                    payload_slot(&seed, bound)?.to_vec1::<u32>()?[0] as usize,
                    crate::common::HashRng::new(&hash, 1).below(bound)
                );
            }
            let expected = (crate::common::HashRng::new(&hash, 1).uniform() as f32)
                .min(f32::from_bits(1.0f32.to_bits() - 1));
            assert!((uniform(&seed)?.to_vec1::<f32>()?[0] - expected).abs() <= f32::EPSILON);
            let position = Tensor::new(&[99u32, u32::MAX, 8], device)?.narrow(0, 1, 1)?;
            let mut expected_seed = data[3..].to_vec();
            expected_seed.extend_from_slice(&(u64::from(u32::MAX % 7)).to_le_bytes());
            assert_eq!(
                position_seed(&seed, &position, 7)?.to_vec1::<u8>()?,
                expected_seed
            );
        }
        Ok(())
    }

    fn sorting(device: &Device) -> Result<()> {
        for size in [1, 1023, 1024, 1025, 2049, 32769] {
            let mut values = vec![u32::MAX];
            values.extend((0..size).map(|i| {
                if i % 19 == 0 {
                    u32::MAX
                } else {
                    ((i * 7919) % 1009) as u32
                }
            }));
            let input = Tensor::new(values.as_slice(), device)?.narrow(0, 1, size)?;
            let (sorted, order) = sort_u32(&input)?;
            let mut expected = values[1..].to_vec();
            expected.sort();
            assert_eq!(sorted.to_vec1::<u32>()?, expected);
            assert_eq!(input.index_select(&order, 0)?.to_vec1::<u32>()?, expected);
            let mut indices = order.to_vec1::<u32>()?;
            indices.sort();
            assert_eq!(indices, (0..size as u32).collect::<Vec<_>>());
        }
        Ok(())
    }
    #[test]
    fn indexed_cpu_sorting() -> Result<()> {
        sorting(&Device::Cpu)
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn indexed_metal_sorting() -> Result<()> {
        sorting(&Device::new_metal(0)?)
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn indexed_cuda_sorting() -> Result<()> {
        sorting(&Device::new_cuda(0)?)
    }

    #[test]
    fn indexed_cpu_metadata_vectors() -> Result<()> {
        vectors(&Device::Cpu)
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn indexed_metal_metadata_vectors() -> Result<()> {
        vectors(&Device::new_metal(0)?)
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn indexed_cuda_metadata_vectors() -> Result<()> {
        vectors(&Device::new_cuda(0)?)
    }
}
