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
    include_str!("kernels/entry.cu")
);
static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

pub(super) fn forward(
    op: &MetadataOp,
    storage: &CudaStorage,
    layout: &Layout,
) -> Result<(CudaStorage, Shape)> {
    let ptx = PTX
        .get_or_init(|| {
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
        .map_err(|e| candle_core::Error::Msg(format!("watermark CUDA compilation: {e}")))?;
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
