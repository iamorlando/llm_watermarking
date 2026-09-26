#![cfg(feature = "candle")]

use llm_watermarking::{
    candle_core::{DType, Device, Tensor},
    exponential::ExponentialRace,
    inverse_transform::InverseTransform,
    kgw::{Kgw, KgwConfig},
    mpac::{Mpac, MpacConfig},
    sampling::SamplingConfig,
    semstamp::{SemStamp, SemStampConfig},
    synthid::{SynthIdConfig, SynthIdText},
    tensor::validate_probabilities,
    unigram::{Unigram, UnigramConfig},
};

fn close(output: &Tensor, expected: &[f32], input: &Tensor, tolerance: f32) {
    assert!(output.device().same_device(input.device()));
    assert_eq!(output.dtype(), input.dtype());
    assert_eq!(output.dims(), input.dims());
    let actual = output
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && a >= 0.0, "invalid output at {index}: {a}");
        assert!(
            (a - e).abs() < tolerance,
            "index {index}: tensor={a}, reference={e}"
        );
        if e == 0.0 {
            assert_eq!(a, 0.0, "excluded token {index}");
        }
    }
}

fn reweighting_parity(device: &Device) {
    let weights: Vec<_> = (0..37)
        .map(|i| {
            if i % 5 == 0 {
                0.0
            } else {
                (i + 1) as f32 / 7.0
            }
        })
        .collect();
    let input = Tensor::from_vec(weights.clone(), weights.len(), device).unwrap();
    let context = [1, 2, 3, 4];
    for depth in [1, 7, 8, 9, 30, 256] {
        let mut config = SynthIdConfig::new([42; 32]);
        config.depth = depth;
        let watermark = SynthIdText::with_domain(&config, b"candle-parity-v1\0").unwrap();
        let mut reference = weights.clone();
        watermark.apply(&mut reference, &context, 4).unwrap();
        let prepared = watermark
            .prepare_tensor(weights.len(), &context, 4, device)
            .unwrap();
        close(&prepared.apply(&input).unwrap(), &reference, &input, 3e-5);
        close(
            &prepared.clone().apply_trusted(&input).unwrap(),
            &reference,
            &input,
            3e-5,
        );
    }
    let mut kgw_config = KgwConfig::new([42; 32], weights.len());
    kgw_config.context_width = 2;
    let kgw = Kgw::new(&kgw_config).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new([42; 32], weights.len())).unwrap();
    let mut mpac_config = MpacConfig::new([42; 32], weights.len(), 3);
    mpac_config.radix = 4;
    let mpac = Mpac::new(&mpac_config).unwrap();
    for (output, reference) in [
        (kgw.apply_tensor(&input, &context, 4).unwrap(), {
            let mut p = weights.clone();
            kgw.apply(&mut p, &context, 4).unwrap();
            p
        }),
        (unigram.apply_tensor(&input).unwrap(), {
            let mut p = weights.clone();
            unigram.apply(&mut p).unwrap();
            p
        }),
        (
            mpac.apply_tensor(&input, &context, 4, &[0, 2, 3]).unwrap(),
            {
                let mut p = weights.clone();
                mpac.apply(&mut p, &context, 4, &[0, 2, 3]).unwrap();
                p
            },
        ),
    ] {
        close(&output, &reference, &input, 2e-6);
    }
    // Offset, strided rows from an existing host tensor must work without copying to CPU.
    let interleaved: Vec<_> = weights.iter().flat_map(|&v| [99.0, v]).collect();
    let strided = Tensor::from_vec(interleaved, (weights.len(), 2), device)
        .unwrap()
        .narrow(1, 1, 1)
        .unwrap()
        .squeeze(1)
        .unwrap();
    assert!(!strided.is_contiguous());
    let mut reference = weights.clone();
    kgw.apply(&mut reference, &context, 4).unwrap();
    close(
        &kgw.apply_tensor(&strided, &context, 4).unwrap(),
        &reference,
        &strided,
        2e-6,
    );
    // Half-precision input uses F32 accumulation and returns the original dtype.
    for (dtype, tolerance) in [(DType::F16, 0.001), (DType::BF16, 0.005)] {
        let half = input.to_dtype(dtype).unwrap();
        let mut reference = half.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        unigram.apply(&mut reference).unwrap();
        close(
            &unigram.apply_tensor(&half).unwrap(),
            &reference,
            &half,
            tolerance,
        );
    }
}

fn skip_and_invalid_inputs(device: &Device) {
    let synthid = SynthIdText::new(&SynthIdConfig::new([42; 32])).unwrap();
    let kgw = Kgw::new(&KgwConfig::new([42; 32], 4)).unwrap();
    let mpac = Mpac::new(&MpacConfig::new([42; 32], 4, 1)).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new([42; 32], 4)).unwrap();
    let prepared = unigram.prepare_tensor(device).unwrap();
    let input = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device).unwrap();
    for context in [&[1, 2][..], &[1, 2, 3, 4, 1, 2, 3, 4][..]] {
        close(
            &synthid.apply_tensor(&input, context, 0).unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
            &input,
            f32::EPSILON,
        );
    }
    close(
        &kgw.apply_tensor(&input, &[], 0).unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        &input,
        f32::EPSILON,
    );
    close(
        &mpac.apply_tensor(&input, &[], 0, &[0]).unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        &input,
        f32::EPSILON,
    );
    for values in [
        [0.0; 4],
        [-1.0, 1.0, 1.0, 1.0],
        [f32::NAN, 1.0, 1.0, 1.0],
        [f32::INFINITY, 1.0, 1.0, 1.0],
    ] {
        let invalid = Tensor::new(&values, device).unwrap();
        assert!(validate_probabilities(&invalid).is_err());
        assert!(prepared.apply(&invalid).is_err());
        assert!(synthid.apply_tensor(&invalid, &[1], 1).is_err());
    }
    assert!(prepared
        .apply(&Tensor::zeros(3, DType::F32, device).unwrap())
        .is_err());
    assert!(prepared.apply(&input.reshape((2, 2)).unwrap()).is_err());
    assert!(prepared
        .apply(&input.to_dtype(DType::U32).unwrap())
        .is_err());
    assert!(synthid.apply_tensor(&input, &[1], 2).is_err());
    assert!(kgw.apply_tensor(&input, &[9], 1).is_err());
    assert!(mpac.apply_tensor(&input, &[1], 1, &[3]).is_err());
    assert_eq!(input.to_vec1::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    if !device.is_cpu() {
        let cpu = unigram.prepare_tensor(&Device::Cpu).unwrap();
        assert!(cpu.apply(&input).is_err());
        assert!(cpu.apply_trusted(&input).is_err());
    }
}

fn extreme_probabilities(device: &Device) {
    for delta in [0.0, 20.0, 1000.0, f64::MAX] {
        let mut config = UnigramConfig::new([42; 32], 8);
        config.delta = delta;
        let watermark = Unigram::new(&config).unwrap();
        for token in [0, 1] {
            // Known green and red tokens for this key.
            let mut weights = [0.0; 8];
            weights[token] = f32::MAX;
            let input = Tensor::new(&weights, device).unwrap();
            watermark.apply(&mut weights).unwrap();
            close(
                &watermark.apply_tensor(&input).unwrap(),
                &weights,
                &input,
                1e-6,
            );
        }
        let mut weights = [1.0, f32::MAX, 3.0, 0.0, 5.0, 7.0, 0.0, 9.0];
        let input = Tensor::new(&weights, device).unwrap();
        watermark.apply(&mut weights).unwrap();
        close(
            &watermark.apply_tensor(&input).unwrap(),
            &weights,
            &input,
            1e-5,
        );
    }
}

fn sampling_parity(device: &Device) {
    let config = SamplingConfig::new([42; 32], 8);
    let exponential = ExponentialRace::new(&config).unwrap();
    let inverse = InverseTransform::new(&config).unwrap();
    let weights = [0.0f32, 1.0, 4.0, 0.0, 5.0, 8.0, 1.0, 2.0];
    let input = Tensor::new(&weights, device).unwrap();
    for position in 0..64 {
        for (scores, reference) in [
            (
                exponential
                    .selection_scores_tensor(&input, position)
                    .unwrap(),
                exponential.sample(&weights, position).unwrap(),
            ),
            (
                inverse.selection_scores_tensor(&input, position).unwrap(),
                inverse.sample(&weights, position).unwrap(),
            ),
        ] {
            assert!(scores.device().same_device(device));
            assert_eq!(scores.dtype(), DType::F32);
            let token = scores.argmax(0).unwrap().to_scalar::<u32>().unwrap();
            assert_eq!(token, reference, "position {position}");
            let values = scores.to_vec1::<f32>().unwrap();
            assert_eq!(values[0], f32::NEG_INFINITY);
            assert_eq!(values[3], f32::NEG_INFINITY);
            assert!(values.iter().all(|v| !v.is_nan()));
        }
    }
    for token in 0..8 {
        let mut weights = [0.0f32; 8];
        weights[token] = f32::MAX;
        let input = Tensor::new(&weights, device).unwrap();
        for (label, prepared) in [
            (
                "exponential",
                exponential.prepare_tensor(5, device).unwrap(),
            ),
            ("inverse", inverse.prepare_tensor(5, device).unwrap()),
        ] {
            let scores = prepared.apply_trusted(&input).unwrap();
            assert_eq!(
                scores.argmax(0).unwrap().to_scalar::<u32>().unwrap(),
                token as u32,
                "{label} point mass: {:?}",
                scores.to_vec1::<f32>().unwrap()
            );
        }
    }
    let invalid = Tensor::zeros(8, DType::F32, device).unwrap();
    assert!(inverse.selection_scores_tensor(&invalid, 0).is_err());
    assert!(exponential.selection_scores_tensor(&invalid, 0).is_err());
    let half = input.to_dtype(DType::F16).unwrap();
    assert_eq!(
        inverse.selection_scores_tensor(&half, 0).unwrap().dtype(),
        DType::F32
    );
    // A practical vocabulary would make Candle's matrix-based cumsum prohibitively large.
    let large = InverseTransform::new(&SamplingConfig::new([42; 32], 32000)).unwrap();
    let input = Tensor::ones(32000, DType::F32, device).unwrap();
    let chosen = large
        .selection_scores_tensor(&input, 0)
        .unwrap()
        .argmax(0)
        .unwrap()
        .to_scalar::<u32>()
        .unwrap();
    assert_eq!(chosen, large.sample(&vec![1.0; 32000], 0).unwrap());
}

fn semstamp_parity(device: &Device) {
    let watermark = SemStamp::new(&SemStampConfig::new([42; 32], 16)).unwrap();
    let prepared = watermark.prepare_tensor(device).unwrap();
    let embeddings: Vec<Vec<f32>> = (0..40)
        .map(|row| {
            (0..16)
                .map(|col| ((row * 17 + col + 1) as f32 * 1.718).sin())
                .collect()
        })
        .collect();
    let input = Tensor::from_vec(
        embeddings.iter().flatten().copied().collect::<Vec<_>>(),
        (40, 16),
        device,
    )
    .unwrap();
    let signatures = prepared.signatures(&input).unwrap();
    assert!(signatures.device().same_device(device));
    assert_eq!(signatures.dtype(), DType::U32);
    let expected: Vec<_> = embeddings
        .iter()
        .map(|e| watermark.signature(e).unwrap())
        .collect();
    assert_eq!(signatures.to_vec1::<u32>().unwrap(), expected);
    assert_eq!(
        prepared
            .signatures_trusted(&input)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap(),
        expected
    );
    let accepted = prepared.accepts(expected[0], &input).unwrap();
    assert!(accepted.device().same_device(device));
    let expected_accepted: Vec<_> = embeddings
        .iter()
        .map(|e| u8::from(watermark.accepts(&embeddings[0], e).unwrap()))
        .collect();
    assert_eq!(accepted.to_vec1::<u8>().unwrap(), expected_accepted);
    assert_eq!(
        watermark
            .acceptance_tensor(expected[0], &input)
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        expected_accepted
    );
    assert_eq!(
        watermark.detect_tensor(&input, 1).unwrap(),
        watermark.detect(&embeddings, 1).unwrap()
    );
    let row = input.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
    assert_eq!(
        prepared
            .signatures(&row)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap(),
        expected[0]
    );
    assert!(prepared.accepts(u32::MAX, &row).is_err());
    assert!(prepared
        .signatures(&Tensor::zeros(16, DType::F32, device).unwrap())
        .is_err());
    assert!(prepared
        .signatures(&Tensor::full(f32::NAN, 16, device).unwrap())
        .is_err());
    assert!(prepared
        .signatures(&Tensor::ones(15, DType::F32, device).unwrap())
        .is_err());
    assert!(prepared.detect(&input, 41).is_err());
    let empty = Tensor::zeros((0, 16), DType::F32, device).unwrap();
    assert_eq!(prepared.signatures(&empty).unwrap().dims(), &[0]);
    assert_eq!(prepared.accepts(expected[0], &empty).unwrap().dims(), &[0]);
    assert_eq!(prepared.detect(&empty, 0).unwrap().valid_fraction, None);
    let large = vec![f32::MAX; 16];
    let large_tensor = Tensor::new(large.as_slice(), device).unwrap();
    assert_eq!(
        prepared
            .signatures(&large_tensor)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap(),
        watermark.signature(&large).unwrap()
    );
}

#[test]
fn candle_cpu_reweighting_matches_scalar_schemes() {
    reweighting_parity(&Device::Cpu);
}

#[test]
fn candle_cpu_validation_and_skips() {
    skip_and_invalid_inputs(&Device::Cpu);
}

#[test]
fn candle_cpu_extreme_probabilities() {
    extreme_probabilities(&Device::Cpu);
}

#[test]
fn candle_cpu_sampling_matches_scalar_schemes() {
    sampling_parity(&Device::Cpu);
}

#[test]
fn candle_cpu_semstamp_matches_scalar_schemes() {
    semstamp_parity(&Device::Cpu);
}

fn changing_contexts(device: &Device) {
    let vocab = 257;
    let mut kgw_config = KgwConfig::new([91; 32], vocab);
    kgw_config.context_width = 3;
    let kgw = Kgw::new(&kgw_config).unwrap();
    let mut mpac_config = MpacConfig::new([91; 32], vocab, 5);
    mpac_config.context_width = 3;
    mpac_config.radix = 7; // Uneven partition leaves tokens outside every group.
    let mpac = Mpac::new(&mpac_config).unwrap();
    let synthid = SynthIdText::new(&SynthIdConfig::new([91; 32])).unwrap();
    let weights: Vec<f32> = (0..vocab).map(|i| (i % 17) as f32).collect();
    let input = Tensor::new(weights.as_slice(), device).unwrap();
    for step in 0..17 {
        let context = [1, 5, 200, step];
        let payload = [0, 1, 3, 6, (step % 7) as u8];
        let mut reference = weights.clone();
        kgw.apply(&mut reference, &context, 4).unwrap();
        close(
            &kgw.clone()
                .prepare_tensor(&context, 4, device)
                .unwrap()
                .apply_trusted(&input)
                .unwrap(),
            &reference,
            &input,
            1e-6,
        );
        reference.clone_from(&weights);
        mpac.apply(&mut reference, &context, 4, &payload).unwrap();
        close(
            &mpac
                .prepare_tensor(&context, 4, &payload, device)
                .unwrap()
                .apply_trusted(&input)
                .unwrap(),
            &reference,
            &input,
            1e-6,
        );
        reference.clone_from(&weights);
        synthid.apply(&mut reference, &context, 4).unwrap();
        close(
            &synthid
                .prepare_tensor(vocab, &context, 4, device)
                .unwrap()
                .apply_trusted(&input)
                .unwrap(),
            &reference,
            &input,
            2e-5,
        );
    }
    let mut config = SamplingConfig::new([91; 32], vocab);
    config.sequence_len = 3;
    let exponential = ExponentialRace::new(&config).unwrap();
    let inverse = InverseTransform::new(&config).unwrap();
    for position in [0, 1, 2, 3, 4, 8, usize::MAX] {
        for (scores, expected) in [
            (
                exponential
                    .prepare_tensor(position, device)
                    .unwrap()
                    .apply_trusted(&input)
                    .unwrap(),
                exponential.sample(&weights, position).unwrap(),
            ),
            (
                inverse
                    .clone()
                    .prepare_tensor(position, device)
                    .unwrap()
                    .apply_trusted(&input)
                    .unwrap(),
                inverse.sample(&weights, position).unwrap(),
            ),
        ] {
            assert_eq!(
                scores.argmax(0).unwrap().to_scalar::<u32>().unwrap(),
                expected
            );
        }
    }
}

#[test]
fn candle_cpu_changing_metadata() {
    changing_contexts(&Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a Metal GPU; run with --features metal -- --ignored"]
fn candle_metal_parity() {
    let device = Device::new_metal(0).expect("Metal GPU required; never fall back to CPU");
    reweighting_parity(&device);
    skip_and_invalid_inputs(&device);
    extreme_probabilities(&device);
    sampling_parity(&device);
    semstamp_parity(&device);
    changing_contexts(&device);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU; run with --features cuda -- --ignored"]
fn candle_cuda_parity() {
    let device = Device::new_cuda(0).expect("CUDA GPU required; never fall back to CPU");
    reweighting_parity(&device);
    skip_and_invalid_inputs(&device);
    extreme_probabilities(&device);
    sampling_parity(&device);
    semstamp_parity(&device);
    changing_contexts(&device);
}
