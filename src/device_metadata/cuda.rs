use std::sync::OnceLock;

use candle_core::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::{
            driver::{LaunchConfig, PushKernelArg},
            nvrtc::{compile_ptx_with_opts, CompileOptions},
        },
        WrapErr,
    },
    CudaStorage, Layout, Result, Shape,
};

use super::{MetadataOp, Mode};

const SOURCE: &str = concat!(
    include_str!("kernels/prefix.cu"),
    "\n",
    include_str!("kernels/metadata.h"),
    "\n",
    include_str!("kernels/input.h"),
    "\n",
    include_str!("kernels/entry.cu")
);
static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

pub(super) fn forward(
    op: &MetadataOp,
    storage: &CudaStorage,
    layout: &Layout,
) -> Result<(CudaStorage, Shape)> {
    let ptx = ptx()?;
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("metadata seed must be contiguous".into()))?;
    let device = storage.device();
    let seed = storage.as_cuda_slice::<u8>()?.slice(start..end);
    let shape = op.shape();
    let draws = if matches!(op.mode, Mode::Permutation) {
        op.size.div_ceil(4) * 8 + op.size * 5 + 1
    } else {
        1
    };
    // The rejection flag starts at zero; all other scratch values read by a
    // pass are written by an earlier pass on this same Candle stream.
    let mut random = device.alloc_zeros::<u32>(draws)?;
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
    macro_rules! dispatch {
        ($dtype:ty) => {{
            // SAFETY: the kernels initialize all returned elements; the private
            // op constructs bounds/shape and cannot receive caller dimensions.
            let mut output = unsafe { device.alloc::<$dtype>(shape.elem_count()) }?;
            for (pass, name) in [
                "watermark_init",
                "watermark_repair",
                "watermark_links",
                "watermark_parents",
                "watermark_resolve",
            ]
            .into_iter()
            .enumerate()
            .take(passes)
            {
                let function =
                    device.get_or_load_custom_func(name, "llm_watermarking_metadata_v1", ptx)?;
                let mut builder = function.builder();
                builder.arg(&seed);
                builder.arg(&mut output);
                builder.arg(&mut random);
                for parameter in &parameters {
                    builder.arg(parameter);
                }
                let threads = if pass == 1 { 1 } else { op.size as u32 };
                // SAFETY: pointers are Candle-owned slices kept alive through
                // submission; mutable arguments retain cudarc event tracking.
                unsafe { builder.launch(LaunchConfig::for_num_elems(threads)) }.w()?;
            }
            CudaStorage::wrap_cuda_slice(output, device.clone())
        }};
    }
    let output = match op.mode {
        Mode::SynthId(_) => dispatch!(u8),
        Mode::Exponential => dispatch!(f32),
        Mode::Permutation => dispatch!(u32),
    };
    Ok((output, shape))
}

fn ptx() -> Result<&'static String> {
    PTX.get_or_init(|| {
        compile_ptx_with_opts(
            SOURCE,
            CompileOptions {
                options: vec!["--std=c++17".into()],
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| candle_core::Error::Msg(format!("watermark CUDA compilation: {e}")))
}

pub(super) fn input_forward(
    op: &super::input::InputOp,
    seed: &CudaStorage,
    seed_layout: &Layout,
    input: &CudaStorage,
    input_layout: &Layout,
) -> Result<(CudaStorage, Shape)> {
    let (start, end) = super::input::offsets(seed_layout)?;
    let seed_slice = seed.as_cuda_slice::<u8>()?.slice(start..end);
    let (start, end) = super::input::offsets(input_layout)?;
    let input_slice = input.as_cuda_slice::<u32>()?.slice(start..end);
    let device = seed.device();
    let shape = op.shape();
    let parameters = op.parameters();
    let function = device.get_or_load_custom_func(
        "watermark_input",
        "llm_watermarking_metadata_v1",
        ptx()?,
    )?;
    macro_rules! dispatch {
        ($dtype:ty) => {{
            // SAFETY: the private op checks dimensions, and the kernel writes
            // every output element and bounds all device-provided history lengths.
            let mut output = unsafe { device.alloc::<$dtype>(shape.elem_count()) }?;
            let mut builder = function.builder();
            builder.arg(&seed_slice).arg(&input_slice).arg(&mut output);
            for p in &parameters {
                builder.arg(p);
            }
            // SAFETY: all pointers are live Candle allocations on its stream;
            // mutable output maintains cudarc's resource dependency tracking.
            unsafe { builder.launch(LaunchConfig::for_num_elems(parameters[2])) }.w()?;
            CudaStorage::wrap_cuda_slice(output, device.clone())
        }};
    }
    let output = match op.dtype() {
        candle_core::DType::U8 => dispatch!(u8),
        candle_core::DType::U32 => dispatch!(u32),
        candle_core::DType::F32 => dispatch!(f32),
        _ => unreachable!(),
    };
    Ok((output, shape))
}
