#![cfg(feature = "candle")]

use llm_watermarking::{
    candle_core::{DType, Device, Result, Tensor},
    exponential::ExponentialRace,
    inverse_transform::InverseTransform,
    kgw::{Kgw, KgwConfig},
    mpac::{Mpac, MpacConfig},
    sampling::SamplingConfig,
    synthid::{SynthIdConfig, SynthIdText},
    tensor::{DeviceHistory, IndexedCandidates, PreparedIndexedBatch, PreparedIndexedOperation},
    unigram::{Unigram, UnigramConfig},
};

fn compare(output: &Tensor, expected: &[f32], tolerance: f32) -> Result<()> {
    let actual = output.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a == e || (a - e).abs() < tolerance,
            "candidate {i}: {a} != {e}"
        );
        if e == 0.0 {
            assert_eq!(a, 0.0);
        }
    }
    Ok(())
}

fn history(tokens: &[u32], prompt: usize, vocab: usize, device: &Device) -> Result<DeviceHistory> {
    // An offset, padded view tests that the logical length, not capacity, is used.
    let mut padded = vec![u32::MAX];
    padded.extend_from_slice(tokens);
    padded.extend_from_slice(&[u32::MAX; 3]);
    let values = Tensor::new(padded.as_slice(), device)?.narrow(0, 1, tokens.len() + 2)?;
    DeviceHistory::new(
        &values,
        &Tensor::new(&[tokens.len() as u32], device)?,
        &Tensor::new(&[prompt as u32], device)?,
        vocab,
    )
}

fn parity(device: &Device) -> Result<()> {
    let vocab = 257;
    let kgw = Kgw::new(&KgwConfig::new([17; 32], vocab)).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new([19; 32], vocab)).unwrap();
    let mut mc = MpacConfig::new([23; 32], vocab, 3);
    mc.radix = 4;
    mc.context_width = 2;
    let mpac = Mpac::new(&mc).unwrap();
    let payload = [3, 0, 2];
    let mut sc = SamplingConfig::new([71; 32], vocab);
    sc.sequence_len = 7;
    let exponential = ExponentialRace::new(&sc).unwrap();
    let inverse = InverseTransform::new(&sc).unwrap();
    for depth in [1, 9, 30, 256] {
        let mut config = SynthIdConfig::new([7; 32]);
        config.depth = depth;
        let synthid = SynthIdText::with_domain(&config, &[91; 87]).unwrap();
        for ids in [vec![256, 7, 128, 0, 19], vec![128]] {
            let interleaved: Vec<_> = ids.iter().flat_map(|&id| [u32::MAX, id]).collect();
            let token_ids = Tensor::from_vec(interleaved, (ids.len(), 2), device)?
                .narrow(1, 1, 1)?
                .squeeze(1)?;
            let candidates = IndexedCandidates::new(&token_ids, vocab)?;
            let base: Vec<f32> = (0..ids.len())
                .map(|i| if i == 2 { 0.0 } else { (i + 1) as f32 * 3.0 })
                .collect();
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let values: Vec<_> = base.iter().flat_map(|&p| [999.0, p]).collect();
                let weights = Tensor::from_vec(values, (ids.len(), 2), device)?
                    .to_dtype(dtype)?
                    .narrow(1, 1, 1)?
                    .squeeze(1)?;
                let quantized = weights.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let mut dense = vec![0.0; vocab];
                for (&id, &p) in ids.iter().zip(&quantized) {
                    dense[id as usize] = p;
                }
                let contexts = [
                    vec![],
                    vec![1],
                    vec![1, 2, 3, 4],
                    vec![1, 2, 3, 4, 1, 2, 3, 4],
                    vec![5, 1, 2, 3, 4],
                ];
                for context in contexts {
                    let prompt = context.len().min(4);
                    let resident = history(&context, prompt, vocab, device)?;
                    let mut synth_ref = dense.clone();
                    synthid.apply(&mut synth_ref, &context, prompt).unwrap();
                    let mut kgw_ref = dense.clone();
                    kgw.apply(&mut kgw_ref, &context, prompt).unwrap();
                    let mut uni_ref = dense.clone();
                    unigram.apply(&mut uni_ref).unwrap();
                    let mut mpac_ref = dense.clone();
                    mpac.apply(&mut mpac_ref, &context, prompt, &payload)
                        .unwrap();
                    let preparations = [
                        (
                            synthid.prepare_indexed(&candidates, &context, prompt)?,
                            synth_ref.clone(),
                        ),
                        (
                            synthid.prepare_indexed_device(&candidates, &resident)?,
                            synth_ref,
                        ),
                        (
                            kgw.prepare_indexed(&candidates, &context, prompt)?,
                            kgw_ref.clone(),
                        ),
                        (kgw.prepare_indexed_device(&candidates, &resident)?, kgw_ref),
                        (unigram.prepare_indexed(&candidates)?, uni_ref),
                        (
                            mpac.prepare_indexed(&candidates, &context, prompt, &payload)?,
                            mpac_ref.clone(),
                        ),
                        (
                            mpac.prepare_indexed_device(&candidates, &resident, &payload)?,
                            mpac_ref,
                        ),
                    ];
                    for (prepared, reference) in preparations {
                        let expected: Vec<_> =
                            ids.iter().map(|&id| reference[id as usize]).collect();
                        let output = prepared.apply(&weights)?;
                        assert_eq!(output.dtype(), dtype);
                        assert!(output.device().same_device(device));
                        compare(
                            &output,
                            &expected,
                            if dtype == DType::F32 { 4e-5 } else { 0.008 },
                        )?;
                        compare(
                            &prepared.clone().apply_trusted(&weights)?,
                            &output.to_dtype(DType::F32)?.to_vec1::<f32>()?,
                            1e-7,
                        )?;
                    }
                }
                for position in [0, 1, 6, 7, 14, 33, u32::MAX as usize] {
                    let positions =
                        Tensor::new(&[19u32, position as u32, 7], device)?.narrow(0, 1, 1)?;
                    let expected_exp = exponential.sample(&dense, position).unwrap();
                    let expected_inv = inverse.sample(&dense, position).unwrap();
                    for (prepared, expected) in [
                        (
                            exponential.prepare_indexed(&candidates, position)?,
                            expected_exp,
                        ),
                        (
                            exponential.prepare_indexed_device(&candidates, &positions)?,
                            expected_exp,
                        ),
                        (
                            inverse.prepare_indexed(&candidates, position)?,
                            expected_inv,
                        ),
                        (
                            inverse.prepare_indexed_device(&candidates, &positions)?,
                            expected_inv,
                        ),
                    ] {
                        let output = prepared.apply_trusted(&weights)?;
                        assert_eq!(output.dtype(), DType::F32);
                        assert!(output.device().same_device(device));
                        let chosen = output.argmax(0)?.to_scalar::<u32>()? as usize;
                        assert_eq!(ids[chosen], expected);
                        let scores = output.to_vec1::<f32>()?;
                        for (i, &p) in quantized.iter().enumerate() {
                            if p == 0.0 {
                                assert_eq!(scores[i], f32::NEG_INFINITY);
                            }
                        }
                        compare(&prepared.clone().apply(&weights)?, &scores, 1e-7)?;
                    }
                }
                compare(&weights, &quantized, 0.0)?;
            }
        }
    }
    Ok(())
}

fn validation(device: &Device) -> Result<()> {
    for (ids, vocab) in [
        (vec![], 10),
        (vec![1, 1], 10),
        (vec![1, 10], 10),
        (vec![u32::MAX], 10),
        (vec![0], 0),
    ] {
        let input = if ids.is_empty() {
            Tensor::new(&[0u32], device)?.narrow(0, 0, 0)?
        } else {
            Tensor::new(ids.as_slice(), device)?
        };
        assert!(IndexedCandidates::new(&input, vocab).is_err());
    }
    assert!(IndexedCandidates::new_trusted(&Tensor::new(&[1f32], device)?, 10).is_err());
    let ids = Tensor::new(&[9u32, 1, 5], device)?;
    let candidates = IndexedCandidates::new(&ids, 10)?;
    let unigram = Unigram::new(&UnigramConfig::new([2; 32], 10)).unwrap();
    let prepared = unigram.prepare_indexed(&candidates)?;
    for data in [
        [0.0f32; 3],
        [f32::NAN, 1.0, 2.0],
        [-1.0, 1.0, 2.0],
        [f32::INFINITY, 1.0, 2.0],
    ] {
        assert!(prepared.apply(&Tensor::new(&data, device)?).is_err());
    }
    assert!(prepared
        .apply_trusted(&Tensor::ones(2, DType::F32, device)?)
        .is_err());
    assert!(prepared
        .apply_trusted(&Tensor::ones((1, 3), DType::F32, device)?)
        .is_err());
    assert!(prepared
        .apply_trusted(&Tensor::ones(3, DType::U32, device)?)
        .is_err());
    let wrong_vocab = IndexedCandidates::new(&ids, 11)?;
    assert!(unigram.prepare_indexed(&wrong_vocab).is_err());
    for (tokens, len, prompt) in [
        (vec![1u32, 2], 3, 0),
        (vec![1, 2], 1, 2),
        (vec![1, 10], 2, 0),
    ] {
        assert!(DeviceHistory::new(
            &Tensor::new(tokens.as_slice(), device)?,
            &Tensor::new(&[len], device)?,
            &Tensor::new(&[prompt], device)?,
            10
        )
        .is_err());
    }
    let empty = history(&[], 0, 10, device)?;
    let synthid = SynthIdText::new(&SynthIdConfig::new([9; 32])).unwrap();
    let weights = Tensor::new(&[3f32, 0.0, 7.0], device)?;
    compare(
        &synthid
            .prepare_indexed_device(&candidates, &empty)?
            .apply(&weights)?,
        &[3.0, 0.0, 7.0],
        0.0,
    )?;
    // Strict apply still validates IDs even if constructed with the trusted API.
    let invalid = IndexedCandidates::new_trusted(&Tensor::new(&[10u32, 1, 5], device)?, 10)?;
    assert!(unigram.prepare_indexed(&invalid)?.apply(&weights).is_err());
    if !device.is_cpu() {
        assert!(prepared
            .apply_trusted(&weights.to_device(&Device::Cpu)?)
            .is_err());
        assert!(unigram
            .prepare_tensor(&Device::Cpu)?
            .indexed(&candidates)
            .is_err());
        let exp = ExponentialRace::new(&SamplingConfig::new([3; 32], 10)).unwrap();
        assert!(exp
            .prepare_indexed_device(&candidates, &Tensor::new(&[0u32], &Device::Cpu)?)
            .is_err());
    }
    Ok(())
}

fn batch_and_repeat(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[[15u32, 1, 9], [2, 8, 0], [7, 4, 12], [3, 1, 11]], device)?;
    let weights = Tensor::new(
        &[
            [1f32, 0.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
            [7.0, 0.0, 2.0],
        ],
        device,
    )?;
    let mut rows = Vec::new();
    for row in 0..4 {
        let c = IndexedCandidates::new(&ids.get(row)?, 16)?;
        let h = history(&[1, 2, 3, 4, row as u32], 4, 16, device)?;
        rows.push(match row {
            0 => PreparedIndexedOperation::Probabilities(
                SynthIdText::new(&SynthIdConfig::new([1; 32]))
                    .unwrap()
                    .prepare_indexed_device(&c, &h)?,
            ),
            1 => PreparedIndexedOperation::Probabilities(
                Mpac::new(&MpacConfig::new([2; 32], 16, 2))
                    .unwrap()
                    .prepare_indexed_device(&c, &h, &[1, 0])?,
            ),
            2 => PreparedIndexedOperation::SelectionScores(
                ExponentialRace::new(&SamplingConfig::new([3; 32], 16))
                    .unwrap()
                    .prepare_indexed_device(&c, &Tensor::new(&[99u32], device)?)?,
            ),
            _ => PreparedIndexedOperation::SelectionScores(
                InverseTransform::new(&SamplingConfig::new([4; 32], 16))
                    .unwrap()
                    .prepare_indexed_device(&c, &Tensor::new(&[13u32], device)?)?,
            ),
        });
    }
    let batch = PreparedIndexedBatch::new(rows.clone())?;
    assert_eq!(
        batch
            .rows()
            .iter()
            .map(|r| r.is_keyed_sampler())
            .collect::<Vec<_>>(),
        [false, false, true, true]
    );
    let output = batch.apply_trusted(&weights)?;
    let strict = batch.clone().apply(&weights)?;
    for (i, row) in rows.iter().enumerate() {
        let expected = row.apply_trusted(&weights.get(i)?)?.to_vec1::<f32>()?;
        compare(&output.get(i)?, &expected, 0.0)?;
        compare(&strict.get(i)?, &expected, 0.0)?;
    }
    // Scheduler reordering changes only row order; retry does not advance state.
    let order = Tensor::new(&[3u32, 1, 0, 2], device)?;
    let reordered =
        PreparedIndexedBatch::new([3, 1, 0, 2].iter().map(|&i| rows[i].clone()).collect())?;
    let shuffled = reordered.apply_trusted(&weights.index_select(&order, 0)?)?;
    for (i, j) in [3, 1, 0, 2].into_iter().enumerate() {
        compare(&shuffled.get(i)?, &output.get(j)?.to_vec1::<f32>()?, 0.0)?;
    }
    assert!(PreparedIndexedBatch::new(vec![]).is_err());
    assert!(batch.apply_trusted(&weights.get(0)?).is_err());

    let synth = SynthIdText::new(&SynthIdConfig::new([11; 32])).unwrap();
    let c = IndexedCandidates::new(&ids.get(0)?, 16)?;
    // Prompt-only repeats are not generation repeats. Exercise both sides of
    // the 1,024-position generation window with a deliberately unique suffix.
    for length in [1027, 1028, 1029, 1100] {
        let mut context = vec![8; length];
        context[0..4].copy_from_slice(&[1, 2, 3, 4]);
        context[length - 4..].copy_from_slice(&[1, 2, 3, 4]);
        for prompt in [0, 4, 5, length] {
            let h = history(&context, prompt, 16, device)?;
            let expected = synth
                .prepare_indexed(&c, &context, prompt)?
                .apply_trusted(&weights.get(0)?)?;
            compare(
                &synth
                    .prepare_indexed_device(&c, &h)?
                    .apply_trusted(&weights.get(0)?)?,
                &expected.to_vec1::<f32>()?,
                1e-7,
            )?;
        }
    }
    Ok(())
}

fn large_candidates(device: &Device) -> Result<()> {
    let vocab = 4099;
    let ids: Vec<_> = (0..1537).map(|i| ((i * 37) % vocab) as u32).collect();
    let values: Vec<_> = (0..ids.len())
        .map(|i| if i % 5 == 0 { 0.0 } else { (i % 13 + 1) as f32 })
        .collect();
    let mut dense = vec![0.0; vocab];
    for (&id, &weight) in ids.iter().zip(&values) {
        dense[id as usize] = weight;
    }
    let candidates = IndexedCandidates::new(&Tensor::new(ids.as_slice(), device)?, vocab)?;
    let weights = Tensor::new(values.as_slice(), device)?;
    let inverse = InverseTransform::new(&SamplingConfig::new([83; 32], vocab)).unwrap();
    for position in [0, 17, 1024, 2345] {
        let prepared = inverse
            .prepare_indexed_device(&candidates, &Tensor::new(&[position as u32], device)?)?;
        let scores = prepared.apply(&weights)?;
        let index = scores.argmax(0)?.to_scalar::<u32>()? as usize;
        assert_eq!(ids[index], inverse.sample(&dense, position).unwrap());
    }
    // The public compact API also accepts IDs that cannot be represented in F32.
    let big = IndexedCandidates::new(
        &Tensor::new(&[16_777_217u32, u32::MAX - 1, 3], device)?,
        u32::MAX as usize,
    )?;
    assert_eq!(
        big.token_ids().to_vec1::<u32>()?,
        [16_777_217, u32::MAX - 1, 3]
    );
    let synth = SynthIdText::new(&SynthIdConfig::new([7; 32])).unwrap();
    let marked = synth
        .prepare_indexed(&big, &[1, 2, 3, 4], 4)?
        .apply(&Tensor::new(&[1f32, 2.0, 0.0], device)?)?;
    let expected = synth
        .prepare_indexed(
            &IndexedCandidates::new(&big.token_ids().to_device(&Device::Cpu)?, u32::MAX as usize)?,
            &[1, 2, 3, 4],
            4,
        )?
        .apply(&Tensor::new(&[1f32, 2.0, 0.0], &Device::Cpu)?)?
        .to_vec1::<f32>()?;
    compare(&marked, &expected, 4e-5)?;
    Ok(())
}

#[test]
fn indexed_cpu_large_candidates() -> Result<()> {
    large_candidates(&Device::Cpu)
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a Metal GPU"]
fn indexed_metal_large_candidates() -> Result<()> {
    large_candidates(&Device::new_metal(0)?)
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn indexed_cuda_large_candidates() -> Result<()> {
    large_candidates(&Device::new_cuda(0)?)
}

#[test]
fn indexed_cpu_parity() -> Result<()> {
    parity(&Device::Cpu)
}
#[test]
fn indexed_cpu_validation() -> Result<()> {
    validation(&Device::Cpu)
}
#[test]
fn indexed_cpu_batch_and_repeat() -> Result<()> {
    batch_and_repeat(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a Metal GPU"]
fn indexed_metal_parity() -> Result<()> {
    let device = Device::new_metal(0)?;
    parity(&device)?;
    validation(&device)?;
    batch_and_repeat(&device)
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn indexed_cuda_parity() -> Result<()> {
    let device = Device::new_cuda(0)?;
    parity(&device)?;
    validation(&device)?;
    batch_and_repeat(&device)
}
