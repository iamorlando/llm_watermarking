use llm_watermarking::{
    exponential::{ExponentialRace, ExponentialRaceConfig},
    inverse_transform::{InverseTransform, InverseTransformConfig},
    kgw::{Kgw, KgwConfig},
    mpac::{Mpac, MpacConfig},
    sampling::{AlignmentConfig, SamplingConfig},
    semstamp::{SemStamp, SemStampConfig, SentenceSamplingError},
    unigram::{Unigram, UnigramConfig},
    WatermarkError,
};
use rand::{distr::weighted::WeightedIndex, distr::Distribution, Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

fn generate(
    vocab_size: usize,
    length: usize,
    seed: u64,
    apply: impl Fn(&mut [f32], &[u32]),
) -> Vec<u32> {
    let mut rng = Isaac64Rng::seed_from_u64(seed);
    let mut tokens = vec![1, 2, 3];
    for _ in 0..length {
        let mut weights = vec![1.0; vocab_size];
        apply(&mut weights, &tokens);
        tokens.push(WeightedIndex::new(&weights).unwrap().sample(&mut rng) as u32);
    }
    tokens
}

#[test]
fn partitions_match_independent_python_sha256_fisher_yates_fixtures() {
    // Independently computed with Python hashlib, little-endian struct.pack,
    // counter-mode digest bytes, and rejection-sampled Fisher-Yates swaps.
    let kgw = Kgw::new(&KgwConfig::new([42; 32], 8)).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new([42; 32], 8)).unwrap();
    assert_eq!(kgw.green_list(&[3]).unwrap(), [1, 2, 5, 6]);
    assert_eq!(unigram.green_list(), [0, 2, 4, 7]);
    assert_eq!(kgw.clone().green_list(&[7, 3]), kgw.green_list(&[3]));
    assert_ne!(kgw.green_list(&[3]), kgw.green_list(&[4]));
}

#[test]
fn soft_bias_matches_logit_addition_and_uses_rounded_null_rate() {
    let mut config = KgwConfig::new([42; 32], 8);
    config.delta = 3.0_f64.ln();
    let kgw = Kgw::new(&config).unwrap();
    let original = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut actual = original;
    kgw.apply(&mut actual, &[3], 1).unwrap();
    let green = [1, 2, 5, 6];
    let total: f64 = original
        .iter()
        .enumerate()
        .map(|(i, &p)| f64::from(p) * if green.contains(&i) { 3.0 } else { 1.0 })
        .sum();
    for (i, (&before, &after)) in original.iter().zip(&actual).enumerate() {
        let expected = f64::from(before) * if green.contains(&i) { 3.0 } else { 1.0 } / total;
        assert!((f64::from(after) - expected).abs() < 1e-7);
    }
    let rounded = Kgw::new(&KgwConfig::new([42; 32], 7)).unwrap();
    assert_eq!(rounded.green_list(&[1]).unwrap().len(), 3);
    let score = rounded.detect(&[1, 2], 1, &[]).unwrap();
    assert_eq!(score.expected_rate, 3.0 / 7.0);
    let expected_z = (score.successes as f64 - 3.0 / 7.0) / (12.0_f64 / 49.0).sqrt();
    assert!((score.z_score.unwrap() - expected_z).abs() < 1e-12);
}

#[test]
fn unigram_partition_is_independent_of_position_and_neighbor_edits() {
    let unigram = Unigram::new(&UnigramConfig::new([42; 32], 8)).unwrap();
    let a = unigram.detect(&[0, 1, 2, 3], 0, &[]).unwrap();
    let b = unigram.detect(&[3, 2, 1, 0], 0, &[]).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.successes, 2);
    let edited = unigram.detect(&[0, 4, 2, 3], 0, &[]).unwrap();
    assert_eq!(edited.successes, a.successes + 1);
    let mut weights = [1.0; 8];
    unigram.apply(&mut weights).unwrap();
    assert!((weights[0] / weights[1] - 2.0_f32.exp()).abs() < 1e-5);
}

#[test]
fn reweighting_rejects_invalid_inputs_atomically() {
    let kgw = Kgw::new(&KgwConfig::new([7; 32], 4)).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new([7; 32], 4)).unwrap();
    let mpac = Mpac::new(&MpacConfig::new([7; 32], 4, 1)).unwrap();
    for kind in 0..3 {
        for (mut weights, expected) in [
            (vec![], WatermarkError::EmptyDistribution),
            (
                vec![1.0],
                WatermarkError::VocabularySizeMismatch {
                    expected: 4,
                    actual: 1,
                },
            ),
            (vec![0.0; 4], WatermarkError::ZeroProbabilityMass),
            (
                vec![1.0, -1.0, 0.0, 0.0],
                WatermarkError::InvalidProbability { index: 1 },
            ),
            (
                vec![1.0, f32::NAN, 0.0, 0.0],
                WatermarkError::InvalidProbability { index: 1 },
            ),
            (
                vec![1.0, f32::INFINITY, 0.0, 0.0],
                WatermarkError::InvalidProbability { index: 1 },
            ),
        ] {
            let before: Vec<_> = weights.iter().map(|v| v.to_bits()).collect();
            let result = match kind {
                0 => kgw.apply(&mut weights, &[1], 1),
                1 => unigram.apply(&mut weights),
                _ => mpac.apply(&mut weights, &[1], 1, &[0]),
            };
            assert_eq!(result, Err(expected));
            assert_eq!(
                weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                before
            );
        }
    }
    let mut weights = [1.0; 4];
    assert_eq!(
        kgw.apply(&mut weights, &[1], 2),
        Err(WatermarkError::PromptLengthExceedsContext)
    );
    assert_eq!(
        kgw.apply(&mut weights, &[4], 1),
        Err(WatermarkError::TokenOutOfRange {
            token: 4,
            vocab_size: 4
        })
    );
    assert_eq!(
        mpac.apply(&mut weights, &[1], 1, &[]),
        Err(WatermarkError::InvalidPayloadLength)
    );
    assert_eq!(
        mpac.apply(&mut weights, &[1], 1, &[2]),
        Err(WatermarkError::InvalidPayloadSymbol { index: 0 })
    );
    assert_eq!(weights, [1.0; 4]);
}

#[test]
fn extreme_bias_zero_bias_and_point_masses_stay_finite() {
    for delta in [0.0, 2.0, f64::MAX] {
        let mut config = UnigramConfig::new([42; 32], 8);
        config.delta = delta;
        let unigram = Unigram::new(&config).unwrap();
        for token in [0, 1] {
            // A green and a red point mass, respectively.
            let mut weights = [0.0; 8];
            weights[token] = f32::MAX;
            unigram.apply(&mut weights).unwrap();
            assert_eq!(weights[token], 1.0);
            assert_eq!(weights.iter().sum::<f32>(), 1.0);
        }
        let mut weights = [f32::MAX, 0.0, f32::MIN_POSITIVE, 1.0, 1.0, 0.0, 1.0, 1.0];
        unigram.apply(&mut weights).unwrap();
        assert!(weights.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[5], 0.0);
    }
    let mut config = KgwConfig::new([7; 32], 4);
    config.delta = 0.0;
    let mut weights = [1.0, 2.0, 3.0, 4.0];
    Kgw::new(&config)
        .unwrap()
        .apply(&mut weights, &[1], 1)
        .unwrap();
    assert_eq!(weights, [0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn token_detectors_respect_prompt_eos_warmup_and_repeat_policies() {
    let kgw = Kgw::new(&KgwConfig::new([7; 32], 4)).unwrap();
    let mpac = Mpac::new(&MpacConfig::new([7; 32], 4, 2)).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new([7; 32], 4)).unwrap();
    let tokens = [1, 2, 1, 2, 99, u32::MAX];
    assert_eq!(kgw.detect(&tokens, 1, &[99]).unwrap().trials, 2);
    assert_eq!(mpac.detect(&tokens, 1, &[99]).unwrap().tokens_scored, 2);
    assert_eq!(unigram.detect(&tokens, 1, &[99]).unwrap().trials, 3);
    let mut unique_config = UnigramConfig::new([7; 32], 4);
    unique_config.ignore_repeated_tokens = true;
    assert_eq!(
        Unigram::new(&unique_config)
            .unwrap()
            .detect(&tokens, 1, &[99])
            .unwrap()
            .trials,
        2
    );
    assert_eq!(kgw.detect(&[1, 2], 1, &[2]).unwrap().z_score, None);
    assert_eq!(kgw.detect(&[], 0, &[]).unwrap().observed_rate, None);
    assert!(kgw.detect(&[1], 2, &[]).is_err());
    assert!(mpac.detect(&[1], 2, &[]).is_err());
    assert!(unigram.detect(&[1], 2, &[]).is_err());
    assert!(kgw.detect(&[1, 4], 1, &[]).is_err());
    assert!(mpac.detect(&[1, 4], 1, &[]).is_err());
    assert!(unigram.detect(&[4], 0, &[]).is_err());
    let mut config = KgwConfig::new([7; 32], 4);
    config.context_width = 3;
    let kgw = Kgw::new(&config).unwrap();
    let mut weights = [1.0; 4];
    kgw.apply(&mut weights, &[1, 2], 0).unwrap();
    assert_eq!(weights, [1.0; 4]);
    assert_eq!(
        kgw.green_list(&[1]),
        Err(WatermarkError::InsufficientContext)
    );
    assert_eq!(kgw.detect(&[1, 2, 3], 0, &[]).unwrap().trials, 0);
    let mut config = MpacConfig::new([7; 32], 4, 2);
    config.context_width = 3;
    Mpac::new(&config)
        .unwrap()
        .apply(&mut weights, &[1], 0, &[0, 1])
        .unwrap();
    assert_eq!(weights, [1.0; 4]);
}

#[test]
fn green_list_signals_separate_marked_unmarked_and_wrong_keys() {
    let mut config = KgwConfig::new([42; 32], 128);
    config.context_width = 2;
    let kgw = Kgw::new(&config).unwrap();
    config.key = [43; 32];
    let wrong_kgw = Kgw::new(&config).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new([42; 32], 128)).unwrap();
    let wrong_unigram = Unigram::new(&UnigramConfig::new([43; 32], 128)).unwrap();
    let kgw_tokens = generate(128, 1000, 42, |p, t| kgw.apply(p, t, 3).unwrap());
    let unigram_tokens = generate(128, 1000, 42, |p, _| unigram.apply(p).unwrap());
    let unmarked = generate(128, 1000, 43, |_, _| {});
    for (marked, unmarked, wrong) in [
        (
            kgw.detect(&kgw_tokens, 3, &[]).unwrap(),
            kgw.detect(&unmarked, 3, &[]).unwrap(),
            wrong_kgw.detect(&kgw_tokens, 3, &[]).unwrap(),
        ),
        (
            unigram.detect(&unigram_tokens, 3, &[]).unwrap(),
            unigram.detect(&unmarked, 3, &[]).unwrap(),
            wrong_unigram.detect(&unigram_tokens, 3, &[]).unwrap(),
        ),
    ] {
        assert!(marked.observed_rate.unwrap() > 0.8, "{marked:?}");
        assert!(marked.z_score.unwrap() > 18.0);
        assert!(
            (unmarked.observed_rate.unwrap() - 0.5).abs() < 0.1,
            "{unmarked:?}"
        );
        assert!(
            (wrong.observed_rate.unwrap() - 0.5).abs() < 0.2,
            "{wrong:?}"
        );
    }
}

#[test]
fn mpac_recovers_a_radix_four_payload_and_scores_a_predeclared_payload() {
    let payload = [0, 1, 2, 3, 3, 2, 1, 0];
    let mut config = MpacConfig::new([42; 32], 128, payload.len());
    config.radix = 4;
    config.context_width = 3;
    let mpac = Mpac::new(&config).unwrap();
    let tokens = generate(128, 2000, 42, |p, t| mpac.apply(p, t, 3, &payload).unwrap());
    let detection = mpac.detect(&tokens, 3, &[]).unwrap();
    assert_eq!(detection.payload, payload.map(Some));
    assert!(detection.winning_fraction.unwrap() > 0.65);
    assert_eq!(
        detection.votes.iter().flatten().sum::<usize>(),
        detection.tokens_scored
    );
    let known = mpac.detect_payload(&tokens, 3, &[], &payload).unwrap();
    assert!(known.z_score.unwrap() > 30.0);
    let wrong_payload = payload.map(|s| (s + 1) % 4);
    assert!(
        mpac.detect_payload(&tokens, 3, &[], &wrong_payload)
            .unwrap()
            .z_score
            .unwrap()
            < 0.0
    );
    config.key = [43; 32];
    assert!(
        Mpac::new(&config)
            .unwrap()
            .detect(&tokens, 3, &[])
            .unwrap()
            .winning_fraction
            .unwrap()
            < 0.4
    );
    let unmarked = generate(128, 2000, 43, |_, _| {});
    assert!(
        mpac.detect(&unmarked, 3, &[])
            .unwrap()
            .winning_fraction
            .unwrap()
            < 0.4
    );
    let empty = mpac.detect(&[], 0, &[]).unwrap();
    assert_eq!(empty.payload, vec![None; payload.len()]);
    assert_eq!(empty.winning_fraction, None);
}

#[test]
fn sampling_matches_independent_python_fixtures_and_replays_without_state() {
    let mut config = SamplingConfig::new([42; 32], 8);
    config.sequence_len = 16;
    let exponential = ExponentialRace::new(&config).unwrap();
    let inverse = InverseTransform::new(&config).unwrap();
    let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let exp: Vec<_> = (0..8)
        .map(|i| exponential.sample(&weights, i).unwrap())
        .collect();
    let its: Vec<_> = (0..8)
        .map(|i| inverse.sample(&weights, i).unwrap())
        .collect();
    assert_eq!(exp, [5, 6, 6, 5, 6, 6, 7, 7]);
    assert_eq!(its, [3, 6, 7, 6, 7, 7, 5, 6]);
    assert!(
        (exponential
            .detect(&exp, 0, &[], 0)
            .unwrap()
            .mean_cost
            .unwrap()
            - 0.14579897255308472)
            .abs()
            < 1e-14
    );
    assert!(
        (inverse.detect(&its, 0, &[], 0).unwrap().mean_cost.unwrap() - 0.03879154311497349).abs()
            < 1e-14
    );
    for i in 0..8 {
        assert_eq!(
            exponential.clone().sample(&weights, i + 16).unwrap(),
            exp[i]
        );
        assert_eq!(inverse.clone().sample(&weights, i + 16).unwrap(), its[i]);
        let scaled = weights.map(|w| w * 4.0);
        assert_eq!(exponential.sample(&scaled, i).unwrap(), exp[i]);
        assert_eq!(inverse.sample(&scaled, i).unwrap(), its[i]);
    }
}

#[test]
fn mpac_keeps_equal_colorlists_and_leaves_remainders_uncolored() {
    let config = MpacConfig::new([42; 32], 5, 1);
    let mpac = Mpac::new(&config).unwrap();
    let mut group_sizes = [0; 2];
    let mut remainder = Vec::new();
    for token in 0..5 {
        let detection = mpac.detect(&[1, token], 1, &[]).unwrap();
        assert_eq!(detection.tokens_scored, 1);
        match detection.payload[0] {
            Some(symbol) => group_sizes[symbol as usize] += 1,
            None => remainder.push(token),
        }
    }
    assert_eq!(group_sizes, [2, 2]);
    assert_eq!(remainder.len(), 1);
    let only_remainder = mpac
        .detect_payload(&[1, remainder[0]], 1, &[], &[0])
        .unwrap();
    assert_eq!(only_remainder.successes, 0);
    assert_eq!(only_remainder.expected_rate, 2.0 / 5.0);
    // Two colorlisted observations in different groups must produce an erasure.
    let tied = (0..5)
        .flat_map(|a| (0..5).map(move |b| [1, a, b]))
        .map(|tokens| mpac.detect(&tokens, 1, &[]).unwrap())
        .find(|d| d.votes[0] == [1, 1])
        .expect("fixture contains a two-vote tie");
    assert_eq!(tied.payload, [None]);
    assert_eq!(tied.winning_fraction, Some(0.5));
}

#[test]
fn fresh_sampling_rows_preserve_the_categorical_distribution() {
    let mut config = SamplingConfig::new([42; 32], 4);
    config.sequence_len = 16384;
    let exponential = ExponentialRace::new(&config).unwrap();
    let inverse = InverseTransform::new(&config).unwrap();
    let probabilities = [0.1, 0.2, 0.3, 0.4];
    let mut counts = [[0; 4]; 2];
    for position in 0..10000 {
        counts[0][exponential.sample(&probabilities, position).unwrap() as usize] += 1;
        counts[1][inverse.sample(&probabilities, position).unwrap() as usize] += 1;
    }
    for row in counts {
        for (count, p) in row.into_iter().zip(probabilities) {
            assert!(
                (count as f64 / 10000.0 - f64::from(p)).abs() < 0.02,
                "{row:?}"
            );
        }
    }
}

#[test]
fn sampling_rejects_invalid_weights_and_preserves_point_masses() {
    let exponential = ExponentialRace::new(&ExponentialRaceConfig::new([7; 32], 4)).unwrap();
    let inverse = InverseTransform::new(&InverseTransformConfig::new([7; 32], 4)).unwrap();
    for weights in [
        vec![],
        vec![1.0],
        vec![0.0; 4],
        vec![-1.0; 4],
        vec![f32::NAN; 4],
        vec![f32::INFINITY; 4],
    ] {
        assert!(exponential.sample(&weights, 0).is_err());
        assert!(inverse.sample(&weights, 0).is_err());
    }
    for position in [0, 1, 1023, usize::MAX] {
        for mass in [f32::from_bits(1), 1.0, f32::MAX] {
            assert_eq!(
                exponential
                    .sample(&[0.0, 0.0, mass, 0.0], position)
                    .unwrap(),
                2
            );
            assert_eq!(inverse.sample(&[0.0, 0.0, mass, 0.0], position).unwrap(), 2);
        }
    }
    for detector in [
        exponential.detect(&[], 0, &[], 0),
        inverse.detect(&[], 0, &[], 0),
    ] {
        assert_eq!(detector.unwrap().mean_cost, None);
    }
    assert!(exponential.detect(&[0], 2, &[], 0).is_err());
    assert!(inverse.detect(&[4], 0, &[], 0).is_err());
    assert_eq!(
        exponential
            .detect(&[0, 1, 99, 4], 1, &[99], usize::MAX)
            .unwrap()
            .tokens_scored,
        1
    );
    assert_eq!(
        inverse
            .detect(&[0, 1, 99, 4], 1, &[99], usize::MAX)
            .unwrap()
            .tokens_scored,
        1
    );
}

#[test]
fn sampling_scores_separate_keys_and_find_cropped_and_edited_alignments() {
    let mut config = SamplingConfig::new([42; 32], 64);
    config.sequence_len = 64;
    let exponential = ExponentialRace::new(&config).unwrap();
    let inverse = InverseTransform::new(&config).unwrap();
    config.key = [43; 32];
    let wrong_exp = ExponentialRace::new(&config).unwrap();
    let wrong_its = InverseTransform::new(&config).unwrap();
    let weights = [1.0; 64];
    let exp: Vec<_> = (17..57)
        .map(|i| exponential.sample(&weights, i).unwrap())
        .collect();
    let its: Vec<_> = (17..57)
        .map(|i| inverse.sample(&weights, i).unwrap())
        .collect();
    assert!(
        exponential
            .detect(&exp, 0, &[], 17)
            .unwrap()
            .mean_cost
            .unwrap()
            < 0.06
    );
    assert!(inverse.detect(&its, 0, &[], 17).unwrap().mean_cost.unwrap() < 0.02);
    assert!(
        wrong_exp
            .detect(&exp, 0, &[], 17)
            .unwrap()
            .mean_cost
            .unwrap()
            > 0.5
    );
    assert!(
        wrong_its
            .detect(&its, 0, &[], 17)
            .unwrap()
            .mean_cost
            .unwrap()
            > 0.2
    );
    let mut alignment = AlignmentConfig::new(30);
    for detection in [
        exponential.detect_aligned(&exp[5..35], 0, &[], &alignment),
        inverse.detect_aligned(&its[5..35], 0, &[], &alignment),
    ] {
        let detection = detection.unwrap();
        assert_eq!(detection.key_offset, Some(22));
        assert_eq!(detection.text_offset, Some(0));
    }
    let mutate = |mut text: Vec<u32>| {
        text.remove(12);
        text.insert(25, 1);
        text
    };
    let edited_exp = mutate(exp);
    let edited_its = mutate(its);
    alignment.block_size = 40;
    let straight_exp = exponential
        .detect_aligned(&edited_exp, 0, &[], &alignment)
        .unwrap()
        .mean_cost
        .unwrap();
    let straight_its = inverse
        .detect_aligned(&edited_its, 0, &[], &alignment)
        .unwrap()
        .mean_cost
        .unwrap();
    alignment.edit_penalty = Some(0.2);
    let aligned_exp = exponential
        .detect_aligned(&edited_exp, 0, &[], &alignment)
        .unwrap()
        .mean_cost
        .unwrap();
    let aligned_its = inverse
        .detect_aligned(&edited_its, 0, &[], &alignment)
        .unwrap()
        .mean_cost
        .unwrap();
    assert!(aligned_exp < straight_exp && aligned_exp < 0.08);
    assert!(aligned_its < straight_its && aligned_its < 0.04);
    assert!(
        wrong_exp
            .detect_aligned(&edited_exp, 0, &[], &alignment)
            .unwrap()
            .mean_cost
            .unwrap()
            > aligned_exp + 0.1
    );
    assert!(
        wrong_its
            .detect_aligned(&edited_its, 0, &[], &alignment)
            .unwrap()
            .mean_cost
            .unwrap()
            > aligned_its + 0.05
    );
}

#[test]
fn alignment_checks_input_limits_empty_samples_and_offsets() {
    let config = SamplingConfig::new([42; 32], 8);
    let exp = ExponentialRace::new(&config).unwrap();
    let its = InverseTransform::new(&config).unwrap();
    let mut alignment = AlignmentConfig::new(2);
    assert_eq!(
        exp.detect_aligned(&[1], 0, &[], &alignment)
            .unwrap()
            .mean_cost,
        None
    );
    assert_eq!(
        its.detect_aligned(&[], 0, &[], &alignment)
            .unwrap()
            .text_offset,
        None
    );
    alignment.max_cells = 1;
    assert_eq!(
        exp.detect_aligned(&[1, 2], 0, &[], &alignment),
        Err(WatermarkError::AlignmentTooLarge)
    );
    assert_eq!(
        its.detect_aligned(&[1, 2], 0, &[], &alignment),
        Err(WatermarkError::AlignmentTooLarge)
    );
    alignment.block_size = 0;
    assert_eq!(alignment.validate(), Err(WatermarkError::InvalidAlignment));
    alignment.block_size = 2;
    for penalty in [-1.0, f64::INFINITY, f64::NAN] {
        alignment.edit_penalty = Some(penalty);
        assert_eq!(alignment.validate(), Err(WatermarkError::InvalidAlignment));
    }
    let text = [
        7,
        its.sample(&[1.0; 8], 9).unwrap(),
        its.sample(&[1.0; 8], 10).unwrap(),
        99,
        u32::MAX,
    ];
    let result = its
        .detect_aligned(&text, 1, &[99], &AlignmentConfig::new(2))
        .unwrap();
    assert_eq!(result.tokens_scored, 2);
    assert_eq!(result.text_offset, Some(1));
}

fn embedding(rng: &mut Isaac64Rng) -> Vec<f32> {
    (0..32).map(|_| rng.random_range(-1.0..1.0)).collect()
}

#[test]
fn semstamp_rejection_sampling_and_detection_separate_marked_embeddings() {
    let mut config = SemStampConfig::new([42; 32], 32);
    config.max_attempts = 1000;
    let watermark = SemStamp::new(&config).unwrap();
    config.key = [43; 32];
    let wrong = SemStamp::new(&config).unwrap();
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let mut marked = vec![embedding(&mut rng)];
    let mut unmarked = marked.clone();
    for i in 0..160 {
        let sample = watermark
            .sample_sentence(marked.last().unwrap(), || {
                Ok::<_, std::convert::Infallible>((i, embedding(&mut rng)))
            })
            .unwrap();
        assert!(sample.accepted);
        assert!(watermark
            .accepts(marked.last().unwrap(), &sample.embedding)
            .unwrap());
        assert_eq!(sample.sentence, i);
        marked.push(sample.embedding);
        unmarked.push(embedding(&mut rng));
    }
    let score = watermark.detect(&marked, 1).unwrap();
    assert_eq!(score.valid_fraction, Some(1.0));
    assert!(score.sentences_scored > 100);
    assert!(score.z_score.unwrap() > 15.0);
    assert!(wrong.detect(&marked, 1).unwrap().valid_fraction.unwrap() < 0.5);
    assert!(
        watermark
            .detect(&unmarked, 1)
            .unwrap()
            .valid_fraction
            .unwrap()
            < 0.5
    );
    let scaled: Vec<_> = marked
        .iter()
        .map(|e| e.iter().map(|x| x * 4.0).collect::<Vec<_>>())
        .collect();
    assert_eq!(watermark.detect(&scaled, 1), watermark.detect(&marked, 1));
    assert_eq!(
        watermark.clone().signature(&marked[0]),
        watermark.signature(&marked[0])
    );
}

#[test]
fn semstamp_retry_exhaustion_and_generator_errors_are_explicit() {
    let mut config = SemStampConfig::new([42; 32], 32);
    config.margin = 0.99;
    config.max_attempts = 3;
    let watermark = SemStamp::new(&config).unwrap();
    let previous = [1.0; 32];
    assert!(!watermark.accepts(&previous, &previous).unwrap());
    let mut calls = 0;
    let sample = watermark
        .sample_sentence(&previous, || {
            calls += 1;
            Ok::<_, std::convert::Infallible>(("fallback", previous.to_vec()))
        })
        .unwrap();
    assert!(!sample.accepted);
    assert_eq!(calls, 3);
    assert_eq!(sample.attempts, 3);
    assert_eq!(sample.sentence, "fallback");
    let failure = watermark.sample_sentence(&previous, || Err::<((), Vec<f32>), _>("offline"));
    assert!(matches!(
        failure,
        Err(SentenceSamplingError::Generator("offline"))
    ));
    let invalid = watermark.sample_sentence(&previous, || {
        Ok::<_, std::convert::Infallible>(((), vec![0.0; 32]))
    });
    assert!(matches!(
        invalid,
        Err(SentenceSamplingError::Watermark(
            WatermarkError::ZeroEmbedding
        ))
    ));
    let mut called = false;
    let invalid_prompt = watermark.sample_sentence(&[], || {
        called = true;
        Ok::<_, std::convert::Infallible>(((), previous.to_vec()))
    });
    assert!(invalid_prompt.is_err());
    assert!(!called);
}

#[test]
fn semstamp_validates_embeddings_and_sentence_boundaries() {
    let watermark = SemStamp::new(&SemStampConfig::new([42; 32], 2)).unwrap();
    assert!(matches!(
        watermark.signature(&[1.0]),
        Err(WatermarkError::EmbeddingDimensionMismatch { .. })
    ));
    assert_eq!(
        watermark.signature(&[0.0, 0.0]),
        Err(WatermarkError::ZeroEmbedding)
    );
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            watermark.signature(&[1.0, value]),
            Err(WatermarkError::InvalidEmbedding { index: 1 })
        );
    }
    assert!(watermark.signature(&[f32::MAX; 2]).is_ok());
    assert!(watermark.signature(&[f32::from_bits(1); 2]).is_ok());
    assert_eq!(watermark.detect::<Vec<f32>>(&[], 0).unwrap().z_score, None);
    assert_eq!(
        watermark.detect(&[[1.0, 1.0]], 0).unwrap().sentences_scored,
        0
    );
    assert!(watermark.detect(&[[1.0, 1.0]], 2).is_err());
    assert_eq!(
        watermark
            .detect(&[[1.0, 1.0]; 10], 1)
            .unwrap()
            .sentences_scored,
        1
    );
    let mut config = SemStampConfig::new([42; 32], 2);
    config.ignore_repeated_transitions = false;
    assert_eq!(
        SemStamp::new(&config)
            .unwrap()
            .detect(&[[1.0, 1.0]; 10], 1)
            .unwrap()
            .sentences_scored,
        9
    );
}

#[test]
fn configuration_boundaries_and_key_debug_redaction() {
    macro_rules! invalid {
        ($config:expr, $field:ident, $values:expr, $error:expr) => {
            for value in $values {
                let mut config = $config;
                config.$field = value;
                assert_eq!(config.validate(), Err($error));
            }
        };
    }
    invalid!(
        KgwConfig::new([7; 32], 8),
        vocab_size,
        [0, 1, usize::MAX],
        WatermarkError::InvalidVocabSize
    );
    invalid!(
        KgwConfig::new([7; 32], 8),
        context_width,
        [0, 33],
        WatermarkError::InvalidContextWidth
    );
    invalid!(
        KgwConfig::new([7; 32], 8),
        green_fraction,
        [0.0, 0.01, 1.0, f64::NAN, f64::INFINITY],
        WatermarkError::InvalidGreenFraction
    );
    invalid!(
        UnigramConfig::new([7; 32], 8),
        delta,
        [-1.0, f64::NAN, f64::INFINITY],
        WatermarkError::InvalidDelta
    );
    invalid!(
        SamplingConfig::new([7; 32], 8),
        sequence_len,
        [0, 65537],
        WatermarkError::InvalidSequenceLength
    );
    invalid!(
        MpacConfig::new([7; 32], 8, 1),
        radix,
        [0, 1, 9, 257],
        WatermarkError::InvalidRadix
    );
    invalid!(
        MpacConfig::new([7; 32], 8, 1),
        payload_len,
        [0, 65537],
        WatermarkError::InvalidPayloadLength
    );
    invalid!(
        SemStampConfig::new([7; 32], 32),
        embedding_dim,
        [0, 65537],
        WatermarkError::InvalidEmbeddingDimension
    );
    invalid!(
        SemStampConfig::new([7; 32], 32),
        num_hyperplanes,
        [0, 17],
        WatermarkError::InvalidHyperplaneCount
    );
    invalid!(
        SemStampConfig::new([7; 32], 32),
        margin,
        [-1.0, 1.0, f64::NAN],
        WatermarkError::InvalidMargin
    );
    invalid!(
        SemStampConfig::new([7; 32], 32),
        max_attempts,
        [0],
        WatermarkError::InvalidMaxAttempts
    );
    for debug in [
        format!("{:?}", KgwConfig::new([123; 32], 8)),
        format!("{:?}", UnigramConfig::new([123; 32], 8)),
        format!("{:?}", SamplingConfig::new([123; 32], 8)),
        format!("{:?}", MpacConfig::new([123; 32], 8, 2)),
        format!("{:?}", SemStampConfig::new([123; 32], 32)),
    ] {
        assert!(debug.contains("[redacted]"));
        assert!(!debug.contains("123"));
    }
}
