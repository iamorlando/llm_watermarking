//! Keyed metadata only. Allocation, command submission and stream ordering belong
//! to Candle. The only uploads on changing contexts are small seed suffixes.
use std::sync::{Arc, Mutex};

use candle_core::{CpuStorage, CustomOp1, Device, Layout, Result, Shape, Tensor};
use sha2::{Digest, Sha256};

#[cfg(feature = "cuda")]
#[allow(unsafe_code)] // Audited Candle/cudarc allocation and kernel launches only.
#[path = "device_metadata/cuda.rs"]
mod cuda;
#[cfg(feature = "metal")]
#[path = "device_metadata/metal.rs"]
mod metal;

/// One entry per Candle device, never one entry per context or generation step.
#[derive(Clone)]
pub(crate) struct DeviceCache<T>(Arc<Mutex<Vec<(Device, T)>>>);

impl<T> Default for DeviceCache<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}

impl<T: Clone> DeviceCache<T> {
    pub(crate) fn get(&self, device: &Device, make: impl FnOnce() -> Result<T>) -> Result<T> {
        let mut cache = self
            .0
            .lock()
            .map_err(|_| candle_core::Error::Msg("watermark device cache poisoned".into()))?;
        if let Some((_, value)) = cache.iter().find(|(d, _)| d.same_device(device)) {
            return Ok(value.clone());
        }
        let value = make()?;
        cache.push((device.clone(), value.clone()));
        Ok(value)
    }
}

/// Retains the exact versioned hash input, not a replacement hash construction.
#[derive(Clone)]
pub(crate) struct Seed {
    bytes: Vec<u8>,
    devices: DeviceCache<Tensor>,
}

impl Seed {
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            devices: DeviceCache::default(),
        }
    }

    pub(crate) fn prefix(domain: &[u8], key: &[u8; 32], parameters: &[usize]) -> Self {
        let mut bytes = domain.to_vec();
        bytes.extend_from_slice(key);
        for &p in parameters {
            bytes.extend_from_slice(&(p as u64).to_le_bytes());
        }
        Self::new(bytes)
    }

    pub(crate) fn tensor(&self, suffix: &[u8], device: &Device) -> Result<Tensor> {
        let prefix = self
            .devices
            .get(device, || Tensor::new(self.bytes.as_slice(), device))?;
        if suffix.is_empty() {
            return Ok(prefix);
        }
        Tensor::cat(&[prefix, Tensor::new(suffix, device)?], 0)
    }

    pub(crate) fn context(&self, context: &[u32], device: &Device) -> Result<Tensor> {
        let suffix: Vec<_> = context.iter().flat_map(|id| id.to_le_bytes()).collect();
        self.tensor(&suffix, device)
    }
}

#[derive(Clone, Copy)]
enum Mode {
    SynthId(usize),
    Exponential,
    Permutation,
}

struct MetadataOp {
    mode: Mode,
    size: usize,
}

impl MetadataOp {
    fn shape(&self) -> Shape {
        match self.mode {
            Mode::SynthId(bytes) => (bytes, self.size).into(),
            Mode::Exponential => self.size.into(),
            Mode::Permutation => (2, self.size).into(),
        }
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn parameters(&self, seed_len: usize) -> [u32; 4] {
        [
            seed_len as u32,
            self.size as u32,
            match self.mode {
                Mode::SynthId(n) => n as u32,
                _ => 0,
            },
            0,
        ]
    }
}

impl CustomOp1 for MetadataOp {
    fn name(&self) -> &'static str {
        "watermark-keyed-metadata-v1"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("metadata seed must be contiguous".into()))?;
        let mut hash = Sha256::new();
        hash.update(&storage.as_slice::<u8>()?[start..end]);
        let output = match self.mode {
            Mode::SynthId(bytes) => {
                let mut output = vec![0; bytes * self.size];
                for token in 0..self.size {
                    let mut h = hash.clone();
                    h.update((token as u32).to_le_bytes());
                    let digest = h.finalize();
                    for b in 0..bytes {
                        output[b * self.size + token] = digest[b];
                    }
                }
                CpuStorage::U8(output)
            }
            Mode::Exponential => CpuStorage::F32(
                (0..self.size)
                    .map(|token| {
                        let mut h = hash.clone();
                        h.update((token as u32).to_le_bytes());
                        let u = crate::common::HashRng::new(&h, 2).uniform();
                        (-(-u.ln()).ln()) as f32
                    })
                    .collect(),
            ),
            Mode::Permutation => {
                let order = crate::common::permutation(&hash, self.size);
                let mut output = vec![0; 2 * self.size];
                for (rank, &token) in order.iter().enumerate() {
                    output[rank] = token as u32;
                    output[self.size + token] = rank as u32;
                }
                CpuStorage::U32(output)
            }
        };
        Ok((output, self.shape()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::forward(self, storage, layout)
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(self, storage, layout)
    }
}

fn generate(seed: &Tensor, mode: Mode, size: usize) -> Result<Tensor> {
    if size == 0 || size > u32::MAX as usize || seed.elem_count() > u32::MAX as usize {
        candle_core::bail!("watermark metadata dimensions exceed kernel limits");
    }
    seed.apply_op1_no_bwd(&MetadataOp { mode, size })
}

pub(crate) fn synthid(seed: &Tensor, size: usize, depth: usize) -> Result<Tensor> {
    generate(seed, Mode::SynthId(depth.div_ceil(8)), size)
}

pub(crate) fn exponential(seed: &Tensor, size: usize) -> Result<Tensor> {
    generate(seed, Mode::Exponential, size)
}

pub(crate) fn permutation(seed: &Tensor, size: usize) -> Result<(Tensor, Tensor)> {
    let both = generate(seed, Mode::Permutation, size)?;
    Ok((both.get(0)?, both.get(1)?))
}

pub(crate) fn green_mask(seed: &Tensor, size: usize, count: usize) -> Result<Tensor> {
    permutation(seed, size)?.1.lt(count as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vectors(device: &Device) -> Result<()> {
        // Exercise SHA padding across blocks, arbitrary domains and offsets.
        for len in [0, 1, 42, 43, 51, 52, 55, 56, 63, 64, 65, 119, 120, 257] {
            let mut data = vec![99u8; 3];
            data.extend((0..len).map(|i| (i * 37 + 11) as u8));
            data.push(88);
            let seed = Tensor::new(data.as_slice(), device)?.narrow(0, 3, len)?;
            let cpu = Tensor::new(data.as_slice(), &Device::Cpu)?.narrow(0, 3, len)?;
            for size in [1, 2, 3, 4, 5, 257] {
                assert_eq!(
                    synthid(&seed, size, 256)?.to_vec2::<u8>()?,
                    synthid(&cpu, size, 256)?.to_vec2::<u8>()?,
                    "SHA len={len}, size={size}"
                );
                let (order, ranks) = permutation(&seed, size)?;
                let (cpu_order, cpu_ranks) = permutation(&cpu, size)?;
                assert_eq!(
                    order.to_vec1::<u32>()?,
                    cpu_order.to_vec1::<u32>()?,
                    "shuffle len={len}, size={size}"
                );
                assert_eq!(ranks.to_vec1::<u32>()?, cpu_ranks.to_vec1::<u32>()?);
                let expected = exponential(&cpu, size)?.to_vec1::<f32>()?;
                for (actual, expected) in exponential(&seed, size)?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .zip(expected)
                {
                    assert!(
                        actual.is_finite() && (actual - expected).abs() < 3e-6,
                        "Gumbel len={len}: {actual} != {expected}"
                    );
                }
            }
        }
        // A realistic, non-power-of-two vocabulary spans many hash counters and
        // checks exact permutation/rank identity, independently of probabilities.
        let seed = Seed::prefix(b"large-metadata-test\0", &[213; 32], &[32769]);
        let input = seed.tensor(&[255, 0, 128, 17], device)?;
        let (order, ranks) = permutation(&input, 32769)?;
        let (cpu_order, cpu_ranks) =
            permutation(&seed.tensor(&[255, 0, 128, 17], &Device::Cpu)?, 32769)?;
        assert_eq!(order.to_vec1::<u32>()?, cpu_order.to_vec1::<u32>()?);
        assert_eq!(ranks.to_vec1::<u32>()?, cpu_ranks.to_vec1::<u32>()?);
        Ok(())
    }

    #[test]
    fn cpu_metadata_vectors() -> Result<()> {
        vectors(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn metal_metadata_vectors() -> Result<()> {
        vectors(&Device::new_metal(0)?)?;
        // Distinct Candle devices for the same physical GPU must not share
        // cached tensors or pipelines tied to a different execution context.
        let a = Device::new_metal(0)?;
        let b = Device::new_metal(0)?;
        let seed = Seed::new(vec![1, 2, 3]);
        let left = seed.tensor(&[], &a)?;
        let right = seed.clone().tensor(&[], &b)?;
        assert!(left.device().same_device(&a));
        assert!(right.device().same_device(&b));
        assert!(!left.device().same_device(right.device()));
        assert_eq!(
            synthid(&left, 17, 256)?.to_vec2::<u8>()?,
            synthid(&right, 17, 256)?.to_vec2::<u8>()?
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_metadata_vectors() -> Result<()> {
        vectors(&Device::new_cuda(0)?)
    }
}
