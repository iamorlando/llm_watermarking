use std::sync::{Mutex, OnceLock};

use candle_core::{
    backend::BackendStorage,
    metal_backend::{DeviceId, MetalDevice},
    DType, Layout, MetalStorage, Result, Shape,
};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLMathMode, MTLSize};

use super::{MetadataOp, Mode};

const SOURCE: &str = concat!(
    include_str!("kernels/prefix.metal"),
    "\n",
    include_str!("kernels/metadata.h"),
    "\n",
    include_str!("kernels/entry.metal")
);
type Pipelines = Vec<(DeviceId, [ComputePipeline; 5])>;
static PIPELINES: OnceLock<Mutex<Pipelines>> = OnceLock::new();

fn pipelines(device: &MetalDevice) -> Result<[ComputePipeline; 5]> {
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("watermark Metal pipeline cache poisoned".into()))?;
    if let Some((_, pipelines)) = cache.iter().find(|(id, _)| *id == device.id()) {
        return Ok(pipelines.clone());
    }
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    let library = device
        .metal_device()
        .new_library_with_source(SOURCE, Some(&options))
        .map_err(|e| candle_core::Error::Msg(format!("watermark Metal compilation: {e}")))?;
    let make = |name| -> Result<ComputePipeline> {
        let function = library
            .get_function(name, None)
            .map_err(|e| candle_core::Error::Msg(format!("watermark Metal function: {e}")))?;
        device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| candle_core::Error::Msg(format!("watermark Metal pipeline: {e}")))
    };
    let pipelines = [
        make("watermark_init")?,
        make("watermark_repair")?,
        make("watermark_links")?,
        make("watermark_parents")?,
        make("watermark_resolve")?,
    ];
    cache.push((device.id(), pipelines.clone()));
    Ok(pipelines)
}

pub(super) fn forward(
    op: &MetadataOp,
    storage: &MetalStorage,
    layout: &Layout,
) -> Result<(MetalStorage, Shape)> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("metadata seed must be contiguous".into()))?;
    let device = storage.device();
    let pipelines = pipelines(device)?;
    let shape = op.shape();
    let dtype = match op.mode {
        Mode::SynthId(_) => DType::U8,
        Mode::Exponential => DType::F32,
        Mode::Permutation => DType::U32,
    };
    let output = device
        .new_buffer_builder()
        .with_size_for(shape.elem_count(), dtype)
        .with_label("watermark metadata")
        .build()?;
    let draws = if matches!(op.mode, Mode::Permutation) {
        op.size.div_ceil(4) * 8 + op.size * 5 + 1
    } else {
        1
    };
    let random = device
        .new_buffer_builder()
        .with_zeros(draws * DType::U32.size_in_bytes())
        .with_label("watermark hash stream")
        .build()?;
    let mut parameters = op.parameters(end - start);
    parameters[3] = match op.mode {
        Mode::SynthId(_) => 0,
        Mode::Exponential => 1,
        Mode::Permutation => 2,
    };
    let passes = if matches!(op.mode, Mode::Permutation) {
        5
    } else {
        1
    };
    for (pass, pipeline) in pipelines.iter().enumerate().take(passes) {
        // A separate Candle encoder guard for each pass preserves its resource
        // hazard tracking across hash, repair, linked-list and resolution passes.
        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_input_buffer(0, Some(storage.buffer()), start);
        encoder.set_output_buffer(1, Some(&output), 0);
        encoder.set_output_buffer(2, Some(&random), 0);
        encoder.set_bytes(3, &parameters);
        let threads = if pass == 1 { 1 } else { op.size };
        encoder.dispatch_threads(
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads.min(64),
                height: 1,
                depth: 1,
            },
        );
    }
    Ok((
        MetalStorage::new(output, device.clone(), shape.elem_count(), dtype),
        shape,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn metal_rejection_repairs_exhausted_prefetch() -> Result<()> {
        let candle_core::Device::Metal(device) = candle_core::Device::new_metal(0)? else {
            unreachable!()
        };
        let size = 17usize; // Its first bound rejects zero, unlike powers of two.
        let base = size.div_ceil(4) * 8;
        let available = base / 2;
        let seed = [71u8, 9, 128, 3];
        // Force every prefetched draw to be rejected at the first bound. Repair
        // must resume hashing beyond the buffer, then keep shifted stream offsets.
        let mut scratch = vec![0u32; base + size * 5 + 1];
        scratch[base + size..base + size * 2].fill(u32::MAX);
        scratch[base + size * 5] = 1;
        let random = device.new_buffer_builder().with_data(&scratch).build()?;
        let input = device.new_buffer_builder().with_data(&seed).build()?;
        let output = device
            .new_buffer_builder()
            .with_size_for(size * 2, DType::U32)
            .build()?;
        let p = [seed.len() as u32, size as u32, 0, 2];
        for (pass, pipeline) in pipelines(&device)?.iter().enumerate().skip(1) {
            let guard = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = guard.as_ref();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_input_buffer(0, Some(&input), 0);
            encoder.set_output_buffer(1, Some(&output), 0);
            encoder.set_output_buffer(2, Some(&random), 0);
            encoder.set_bytes(3, &p);
            let threads = if pass == 1 { 1 } else { size };
            encoder.dispatch_threads(
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let actual = MetalStorage::new(output, device, size * 2, DType::U32).to_cpu_storage()?;
        let actual = actual.as_slice::<u32>()?;
        let mut expected: Vec<_> = (0..size as u32).collect();
        let mut cursor = 0usize;
        for i in (1..size).rev() {
            let bound = (i + 1) as u64;
            let threshold = bound.wrapping_neg() % bound;
            let value = loop {
                let value = if cursor < available {
                    0
                } else {
                    let mut h = Sha256::new();
                    h.update(seed);
                    h.update([0]);
                    h.update(((cursor / 4) as u64).to_le_bytes());
                    let digest = h.finalize();
                    let start = (cursor % 4) * 8;
                    u64::from_le_bytes(digest[start..start + 8].try_into().unwrap())
                };
                cursor += 1;
                if value >= threshold {
                    break value;
                }
            };
            expected.swap(i, (value % bound) as usize);
        }
        assert!(cursor > available);
        assert_eq!(&actual[..size], expected);
        for (rank, &token) in expected.iter().enumerate() {
            assert_eq!(actual[size + token as usize], rank as u32);
        }
        Ok(())
    }
}
