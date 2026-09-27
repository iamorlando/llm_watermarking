#![cfg(feature = "candle")]
use llm_watermarking::{
    candle_core::{DType, Device, Result, Tensor},
    exponential::ExponentialRace,
    inverse_transform::InverseTransform,
    kgw::{Kgw, KgwConfig},
    mpac::{Mpac, MpacConfig},
    sampling::SamplingConfig,
    synthid::{SynthIdConfig, SynthIdText},
    tensor::{
        DeviceHistory, IndexedCandidates, PreparedIndexedBatch, PreparedIndexedOperation,
        TensorSamplingTrace,
    },
    trace::{TraceKind, TraceOptions, TraceView},
    unigram::{Unigram, UnigramConfig},
};

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.to_vec1::<f32>()
}
fn exact(a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(a.dims(), b.dims());
    assert!(a.device().same_device(b.device()));
    assert_eq!(
        values(a)?.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        values(b)?.into_iter().map(f32::to_bits).collect::<Vec<_>>()
    );
    Ok(())
}

fn snapshot(trace: &TensorSamplingTrace, ids: &[u32], device: &Device) -> Result<()> {
    let indices = Tensor::new(&[99u32, 4, 1, 2, 88], device)?.narrow(0, 1, 3)?;
    let view = TraceView::default();
    let s = trace.snapshot_trusted(Some(&indices), &view)?;
    let strict = trace.snapshot(Some(&indices), &view)?;
    assert_eq!(s.token_ids.to_vec1::<u32>()?, [ids[4], ids[1], ids[2]]);
    assert_eq!(s.token_ids.dtype(), DType::U32);
    for tensor in [&s.token_ids, &s.active, &s.input_weights] {
        assert!(tensor.device().same_device(device));
    }
    exact(&s.input_weights, &strict.input_weights)?;
    if let Some(scores) = &s.selection_scores {
        assert!(s.output_weights.is_none());
        assert!(s.output_probabilities.is_none());
        assert!(s.output_log_probabilities.is_none());
        exact(
            scores,
            &trace
                .output()
                .contiguous()?
                .index_select(&indices.contiguous()?, 0)?,
        )?;
    } else {
        exact(
            s.output_weights.as_ref().unwrap(),
            &trace
                .output()
                .contiguous()?
                .index_select(&indices.contiguous()?, 0)?,
        )?;
        let output = values(trace.output())?;
        let total: f64 = output.iter().map(|&p| f64::from(p)).sum();
        let probabilities = values(s.output_probabilities.as_ref().unwrap())?;
        let logs = values(s.output_log_probabilities.as_ref().unwrap())?;
        for (i, row) in [4, 1, 2].into_iter().enumerate() {
            let expected = f64::from(output[row]) / total;
            assert!((f64::from(probabilities[i]) - expected).abs() < 2e-6);
            if expected == 0.0 {
                assert_eq!(logs[i], f32::NEG_INFINITY);
            } else {
                assert!((logs[i] - probabilities[i].ln()).abs() < 2e-6);
            }
        }
    }
    for layer in &s.layers {
        let before = values(&layer.input_probabilities)?;
        let after = values(&layer.probabilities)?;
        let g = layer.g_values.to_vec1::<u8>()?;
        let mass = values(&layer.green_mass)?[0];
        let norm = values(&layer.output_normalizer)?[0];
        if s.active.to_vec1::<u8>()?[0] != 0 {
            for i in 0..3 {
                assert!(
                    (after[i] - before[i] * (1.0 + f32::from(g[i]) - mass) / norm).abs() < 2e-6
                );
            }
        } else {
            assert_eq!(before, after);
            assert_eq!(g, [0, 0, 0]);
        }
    }
    if let Some(inv) = s.inverse {
        let full = trace.snapshot_trusted(None, &view)?;
        let all = full.inverse.unwrap();
        let scores = values(trace.output())?;
        let winner = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let lower = values(&all.cdf_lower)?;
        let upper = values(&all.cdf_upper)?;
        let target = values(&inv.threshold)?[0];
        assert!(lower[winner] <= target && target < upper[winner]);
        assert_eq!(s.kind, TraceKind::InverseTransform);
        assert_eq!(inv.ranks.dtype(), DType::U32);
    }
    assert!(trace
        .snapshot(Some(&Tensor::new(&[99u32], device)?), &view)
        .is_err());
    assert!(trace
        .snapshot_trusted(
            None,
            &TraceView {
                max_rows: 1,
                ..view
            }
        )
        .is_err());
    assert!(trace
        .snapshot_trusted(
            None,
            &TraceView {
                max_elements: 1,
                ..view
            }
        )
        .is_err());
    let lean = trace.snapshot_trusted(
        Some(&indices),
        &TraceView {
            layers: false,
            probabilities: false,
            log_probabilities: false,
            partition: false,
            ..view
        },
    )?;
    assert!(lean.layers.is_empty());
    assert!(lean.input_probabilities.is_none());
    assert!(lean.favored_mask.is_none());
    Ok(())
}

fn parity(device: &Device) -> Result<()> {
    let ids = [36u32, 7, 0, 19, 29];
    let token_ids = Tensor::new(&[99u32, 36, 7, 0, 19, 29, 88], device)?.narrow(0, 1, 5)?;
    let candidates = IndexedCandidates::new_trusted(&token_ids, 37)?;
    let mut sc = SynthIdConfig::new([6; 32]);
    sc.depth = 30;
    let synth = SynthIdText::new(&sc).unwrap();
    let kgw = Kgw::new(&KgwConfig::new([3; 32], 37)).unwrap();
    let uni = Unigram::new(&UnigramConfig::new([4; 32], 37)).unwrap();
    let mut mc = MpacConfig::new([5; 32], 37, 3);
    mc.radix = 4;
    let mpac = Mpac::new(&mc).unwrap();
    let mut config = SamplingConfig::new([31; 32], 37);
    config.sequence_len = 7;
    let exp = ExponentialRace::new(&config).unwrap();
    let inv = InverseTransform::new(&config).unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let input = Tensor::new(
            &[
                [99f32, 1.0],
                [99.0, 2.0],
                [99.0, 0.0],
                [99.0, 3.0],
                [99.0, 7.0],
            ],
            device,
        )?
        .to_dtype(dtype)?
        .narrow(1, 1, 1)?
        .squeeze(1)?;
        for context in [vec![1u32, 2, 3, 4], vec![], vec![1, 2, 3, 4, 1, 2, 3, 4]] {
            let prompt = context.len().min(4);
            let mut padded = context.clone();
            padded.push(u32::MAX);
            let history = DeviceHistory::new_trusted(
                &Tensor::new(padded.as_slice(), device)?,
                &Tensor::new(&[context.len() as u32], device)?,
                &Tensor::new(&[prompt as u32], device)?,
                37,
            )?;
            let position = Tensor::new(&[23u32], device)?;
            let operations = vec![
                PreparedIndexedOperation::Probabilities(synth.prepare_indexed(
                    &candidates,
                    &context,
                    prompt,
                )?),
                PreparedIndexedOperation::Probabilities(
                    synth.prepare_indexed_device(&candidates, &history)?,
                ),
                PreparedIndexedOperation::Probabilities(
                    kgw.prepare_indexed_device(&candidates, &history)?,
                ),
                PreparedIndexedOperation::Probabilities(uni.prepare_indexed(&candidates)?),
                PreparedIndexedOperation::Probabilities(mpac.prepare_indexed_device(
                    &candidates,
                    &history,
                    &[3, 0, 2],
                )?),
                PreparedIndexedOperation::SelectionScores(
                    exp.prepare_indexed_device(&candidates, &position)?,
                ),
                PreparedIndexedOperation::SelectionScores(
                    inv.prepare_indexed_device(&candidates, &position)?,
                ),
            ];
            for limit in [0, 2, 30] {
                let options = TraceOptions { max_layers: limit };
                for op in &operations {
                    let normal = op.apply_trusted(&input)?;
                    let trace = op.apply_traced_trusted(&input, &options)?;
                    exact(trace.output(), &normal)?;
                    snapshot(&trace, &ids, device)?;
                    let again = op.apply_traced(&input, &options)?;
                    exact(again.output(), trace.output())?;
                    let s = trace.snapshot_trusted(None, &TraceView::default())?;
                    if s.kind == TraceKind::SynthId {
                        assert_eq!(s.captured_layers, limit);
                        assert_eq!(s.total_layers, 30);
                    }
                }
            }
            let batch = PreparedIndexedBatch::new(operations)?;
            let inputs = Tensor::stack(&vec![input.clone(); batch.rows().len()], 0)?;
            let untraced = batch.apply_trusted(&inputs)?;
            let traced = batch.apply_traced_trusted(
                &inputs,
                &vec![TraceOptions { max_layers: 2 }; batch.rows().len()],
            )?;
            for (i, trace) in traced.iter().enumerate() {
                exact(&trace.output().to_dtype(DType::F32)?, &untraced.get(i)?)?;
            }
        }
    }
    // K=1 still carries original vocabulary ID/rank and a valid CDF lower bound.
    let singleton = IndexedCandidates::new_trusted(&Tensor::new(&[36u32], device)?, 37)?;
    let single_input = Tensor::new(&[f32::MAX], device)?;
    let one = inv.prepare_indexed(&singleton, 23)?;
    let trace = one.apply_traced_trusted(&single_input, &TraceOptions::default())?;
    exact(trace.output(), &one.apply_trusted(&single_input)?)?;
    let snap = trace.snapshot_trusted(None, &TraceView::default())?;
    assert_eq!(snap.token_ids.to_vec1::<u32>()?, [36]);
    let cdf = snap.inverse.unwrap();
    assert_eq!(values(&cdf.cdf_lower)?, [0.0]);
    assert_eq!(values(&cdf.cdf_upper)?, [1.0]);
    assert!(values(&cdf.threshold)?[0] < 1.0);
    for keyed in [
        exp.prepare_tensor(23, device)?,
        inv.prepare_tensor(23, device)?,
    ] {
        let dense_input = Tensor::new(
            (0..37)
                .map(|i| if i == 0 { 0.0f32 } else { i as f32 })
                .collect::<Vec<_>>(),
            device,
        )?;
        let capture = keyed.apply_traced_trusted(&dense_input, &TraceOptions::default())?;
        exact(capture.output(), &keyed.apply_trusted(&dense_input)?)?;
        let rows = Tensor::new(&[36u32, 7, 0], device)?;
        let snap = capture.snapshot_trusted(Some(&rows), &TraceView::default())?;
        assert_eq!(snap.token_ids.to_vec1::<u32>()?, [36, 7, 0]);
        assert_eq!(
            values(snap.selection_scores.as_ref().unwrap())?[2],
            f32::NEG_INFINITY
        );
    }
    // Dense prepared operations use the same explicit trace surface.
    let input = Tensor::new(&[1f32, 0.0, 2.0, 3.0, 7.0], device)?;
    let dense = synth.prepare_tensor(5, &[1, 2, 3, 4], 4, device)?;
    let trace = dense.apply_traced_trusted(&input, &TraceOptions { max_layers: 2 })?;
    exact(trace.output(), &dense.apply_trusted(&input)?)?;
    let rows = Tensor::new(&[4u32, 0, 2], device)?;
    assert_eq!(
        trace
            .snapshot_trusted(Some(&rows), &TraceView::default())?
            .token_ids
            .to_vec1::<u32>()?,
        [4, 0, 2]
    );
    if !device.is_cpu() {
        assert!(trace
            .snapshot_trusted(Some(&rows.to_device(&Device::Cpu)?), &TraceView::default())
            .is_err());
    }
    Ok(())
}

#[test]
fn tensor_trace_cpu_parity() -> Result<()> {
    parity(&Device::Cpu)
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a Metal GPU"]
fn tensor_trace_metal_parity() -> Result<()> {
    parity(&Device::new_metal(0)?)
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn tensor_trace_cuda_parity() -> Result<()> {
    parity(&Device::new_cuda(0)?)
}
