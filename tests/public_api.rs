use llm_watermarking::{
    synthid::{SynthIdConfig, SynthIdText, HASH_DOMAIN},
    WatermarkError,
};

#[test]
fn invalid_inputs_do_not_modify_weights() {
    let watermark = SynthIdText::new(&SynthIdConfig::new([7; 32])).unwrap();
    for (mut probs, expected) in [
        (vec![], WatermarkError::EmptyDistribution),
        (vec![0.0, 0.0], WatermarkError::ZeroProbabilityMass),
        (
            vec![1.0, -0.1],
            WatermarkError::InvalidProbability { index: 1 },
        ),
        (
            vec![1.0, f32::NAN],
            WatermarkError::InvalidProbability { index: 1 },
        ),
        (
            vec![1.0, f32::INFINITY],
            WatermarkError::InvalidProbability { index: 1 },
        ),
    ] {
        let before: Vec<_> = probs.iter().map(|prob| prob.to_bits()).collect();
        assert_eq!(watermark.apply(&mut probs, &[1, 2, 3, 4], 4), Err(expected));
        assert_eq!(
            probs.iter().map(|prob| prob.to_bits()).collect::<Vec<_>>(),
            before
        );
    }
    let mut probs = [0.25; 4];
    assert_eq!(
        watermark.apply(&mut probs, &[1, 2], 3),
        Err(WatermarkError::PromptLengthExceedsContext)
    );
    assert_eq!(probs, [0.25; 4]);
}

#[test]
fn domains_are_explicit_and_reproducible() {
    let config = SynthIdConfig::new([7; 32]);
    let default = SynthIdText::new(&config).unwrap();
    let explicit = SynthIdText::with_domain(&config, HASH_DOMAIN).unwrap();
    let custom = SynthIdText::with_domain(&config, b"example-deployment-v1\0").unwrap();
    let context = [1, 2, 3, 4];
    let mut a = [0.125; 8];
    let mut b = a;
    let mut c = a;
    default.apply(&mut a, &context, 4).unwrap();
    explicit.apply(&mut b, &context, 4).unwrap();
    custom.apply(&mut c, &context, 4).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    let tokens = [1, 2, 3, 4, 7, 6, 5];
    assert_eq!(
        default.detect(&tokens, 4, &[]),
        explicit.detect(&tokens, 4, &[])
    );
}

#[test]
fn unnormalized_weights_and_single_token_support_are_supported() {
    let watermark = SynthIdText::new(&SynthIdConfig::new([7; 32])).unwrap();
    let mut weights = [0.0, 7.0, 0.0];
    watermark.apply(&mut weights, &[1, 2, 3, 4], 4).unwrap();
    assert_eq!(weights, [0.0, 1.0, 0.0]);
    let mut warmup = [1.0, 3.0];
    watermark.apply(&mut warmup, &[1], 1).unwrap();
    assert_eq!(warmup, [1.0, 3.0]);
}
