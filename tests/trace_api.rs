use llm_watermarking::{
    exponential::ExponentialRace,
    inverse_transform::InverseTransform,
    kgw::{Kgw, KgwConfig},
    mpac::{Mpac, MpacConfig},
    sampling::SamplingConfig,
    synthid::{SynthIdConfig, SynthIdText},
    trace::{SelectionScoreKind, TraceOptions, TraceStatus, TraceView},
    unigram::{Unigram, UnigramConfig},
};

#[test]
fn scalar_trace_preserves_all_outputs_and_projects_actual_ids() {
    let vocab = 37;
    let original: Vec<f32> = (0..vocab)
        .map(|i| if i % 5 == 0 { 0.0 } else { (i + 1) as f32 })
        .collect();
    let context = [1, 2, 3, 4];
    let rows = [36u32, 7, 0, 19];
    let view = TraceView::default();
    let kgw = Kgw::new(&KgwConfig::new([3; 32], vocab)).unwrap();
    let uni = Unigram::new(&UnigramConfig::new([4; 32], vocab)).unwrap();
    let mut mc = MpacConfig::new([5; 32], vocab, 3);
    mc.radix = 4;
    let mpac = Mpac::new(&mc).unwrap();
    for limit in [0, 1, 8, 32, 256] {
        let options = TraceOptions { max_layers: limit };
        let mut sc = SynthIdConfig::new([6; 32]);
        sc.depth = 30;
        let synth = SynthIdText::new(&sc).unwrap();
        for scheme in 0..4 {
            let mut normal = original.clone();
            let mut observed = original.clone();
            let trace = match scheme {
                0 => {
                    kgw.apply(&mut normal, &context, 4).unwrap();
                    kgw.apply_traced(&mut observed, &context, 4, &options)
                        .unwrap()
                }
                1 => {
                    uni.apply(&mut normal).unwrap();
                    uni.apply_traced(&mut observed, &options).unwrap()
                }
                2 => {
                    mpac.apply(&mut normal, &context, 4, &[3, 0, 2]).unwrap();
                    mpac.apply_traced(&mut observed, &context, 4, &[3, 0, 2], &options)
                        .unwrap()
                }
                _ => {
                    synth.apply(&mut normal, &context, 4).unwrap();
                    synth
                        .apply_traced(&mut observed, &context, 4, &options)
                        .unwrap()
                }
            };
            assert_eq!(normal, observed);
            assert_eq!(trace.output_weights().unwrap(), normal);
            let snapshot = trace.snapshot(Some(&rows), &view).unwrap();
            assert_eq!(snapshot.token_ids, rows);
            assert_eq!(snapshot.status, TraceStatus::Applied);
            let total: f64 = normal.iter().map(|&p| f64::from(p)).sum();
            for (i, &id) in rows.iter().enumerate() {
                assert_eq!(
                    snapshot.output_weights.as_ref().unwrap()[i],
                    normal[id as usize]
                );
                let expected = f64::from(normal[id as usize]) / total;
                assert_eq!(snapshot.output_probabilities.as_ref().unwrap()[i], expected);
                assert_eq!(
                    snapshot.output_log_probabilities.as_ref().unwrap()[i],
                    expected.ln()
                );
            }
            assert!(
                snapshot
                    .output_probabilities
                    .as_ref()
                    .unwrap()
                    .iter()
                    .sum::<f64>()
                    < 0.999
            );
            match scheme {
                0 | 1 => {
                    let green = if scheme == 0 {
                        kgw.green_list(&context).unwrap()
                    } else {
                        uni.green_list()
                    };
                    assert_eq!(
                        snapshot.favored_mask.unwrap(),
                        rows.iter().map(|id| green.contains(id)).collect::<Vec<_>>()
                    );
                }
                2 => {
                    let position = snapshot.payload_position.unwrap();
                    assert_eq!(snapshot.payload_symbol, Some([3, 0, 2][position]));
                    let full = trace.snapshot(None, &view).unwrap();
                    assert_eq!(
                        full.favored_mask.unwrap().iter().filter(|&&x| x).count(),
                        vocab / 4
                    );
                }
                _ => {
                    assert_eq!(snapshot.total_layers, 30);
                    assert_eq!(snapshot.layers.len(), limit.min(30));
                    for layer in snapshot.layers {
                        for i in 0..rows.len() {
                            assert_eq!(
                                layer.probabilities[i],
                                layer.input_probabilities[i]
                                    * (1.0 + f64::from(layer.g_values[i]) - layer.green_mass)
                            );
                        }
                        let mut prefix = SynthIdConfig::new([6; 32]);
                        prefix.depth = layer.index + 1;
                        let mut expected = original.clone();
                        SynthIdText::new(&prefix)
                            .unwrap()
                            .apply(&mut expected, &context, 4)
                            .unwrap();
                        for (i, &id) in rows.iter().enumerate() {
                            assert_eq!(layer.probabilities[i] as f32, expected[id as usize]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn scalar_keyed_traces_preserve_selection_and_cdf_semantics() {
    let mut config = SamplingConfig::new([31; 32], 37);
    config.sequence_len = 7;
    let exp = ExponentialRace::new(&config).unwrap();
    let inv = InverseTransform::new(&config).unwrap();
    let options = TraceOptions::default();
    for weights in [
        vec![1.0; 37],
        (0..37)
            .map(|i| if i % 3 == 0 { 0.0 } else { (i + 1) as f32 })
            .collect(),
        (0..37)
            .map(|i| if i == 35 { f32::MAX } else { 0.0 })
            .collect(),
    ] {
        for position in [0, 1, 6, 7, 23, usize::MAX] {
            for inverse in [false, true] {
                let expected = if inverse {
                    inv.sample(&weights, position)
                } else {
                    exp.sample(&weights, position)
                }
                .unwrap();
                let (token, trace) = if inverse {
                    inv.sample_traced(&weights, position, &options)
                } else {
                    exp.sample_traced(&weights, position, &options)
                }
                .unwrap();
                assert_eq!(token, expected);
                let rows = [token, 0, 36];
                let snapshot = trace.snapshot(Some(&rows), &TraceView::default()).unwrap();
                assert_eq!(snapshot.selected_token, Some(token));
                assert_eq!(snapshot.key_position, Some(position % 7));
                assert!(snapshot.output_probabilities.is_none());
                assert!(snapshot.output_log_probabilities.is_none());
                let full = trace.snapshot(None, &TraceView::default()).unwrap();
                let scores = full.selection_scores.unwrap();
                let best = (0..scores.len())
                    .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then_with(|| b.cmp(&a)))
                    .unwrap();
                assert_eq!(best, token as usize);
                for (score, &weight) in scores.iter().zip(&weights) {
                    if weight == 0.0 {
                        assert_eq!(*score, f64::NEG_INFINITY);
                    }
                }
                if inverse {
                    assert_eq!(snapshot.score_kind, Some(SelectionScoreKind::NegativeRank));
                    let cdf = snapshot.inverse.unwrap();
                    assert!(cdf.cdf_lower[0] <= cdf.threshold && cdf.threshold < cdf.cdf_upper[0]);
                    assert_eq!(cdf.threshold, cdf.uniform * cdf.total_weight);
                    assert_eq!(
                        snapshot.selection_scores.unwrap()[0],
                        -f64::from(cdf.ranks[0])
                    );
                }
            }
        }
    }
}

#[test]
fn trace_skips_errors_and_bounds_do_not_change_sampling() {
    let synth = SynthIdText::new(&SynthIdConfig::new([8; 32])).unwrap();
    let original = [1f32, 0.0, 3.0, 2.0];
    for (context, prompt, status) in [
        (vec![1], 1, TraceStatus::Warmup),
        (
            vec![1, 2, 3, 4, 1, 2, 3, 4],
            4,
            TraceStatus::RepeatedContext,
        ),
    ] {
        let mut observed = original;
        let trace = synth
            .apply_traced(
                &mut observed,
                &context,
                prompt,
                &TraceOptions { max_layers: 256 },
            )
            .unwrap();
        assert_eq!(observed, original);
        let s = trace
            .snapshot(Some(&[3, 1]), &TraceView::default())
            .unwrap();
        assert_eq!(s.status, status);
        assert!(s.layers.is_empty());
    }
    let mut observed = original;
    assert!(synth
        .apply_traced(
            &mut observed,
            &[1, 2, 3, 4],
            4,
            &TraceOptions { max_layers: 257 }
        )
        .is_err());
    assert_eq!(observed, original);
    for values in [[0f32; 4], [-1.0, 2.0, 3.0, 4.0], [f32::NAN, 1.0, 2.0, 3.0]] {
        let mut observed = values;
        assert!(synth
            .apply_traced(&mut observed, &[1, 2, 3, 4], 4, &TraceOptions::default())
            .is_err());
        assert_eq!(observed.map(f32::to_bits), values.map(f32::to_bits));
    }
    let trace = synth
        .apply_traced(
            &mut observed,
            &[1, 2, 3, 4],
            4,
            &TraceOptions { max_layers: 4 },
        )
        .unwrap();
    assert!(trace.snapshot(Some(&[4]), &TraceView::default()).is_err());
    assert!(trace.snapshot(Some(&[]), &TraceView::default()).is_err());
    assert!(trace
        .snapshot(
            None,
            &TraceView {
                max_rows: 3,
                ..TraceView::default()
            }
        )
        .is_err());
    assert!(trace
        .snapshot(
            None,
            &TraceView {
                max_elements: 1,
                ..TraceView::default()
            }
        )
        .is_err());
    let reduced = trace
        .snapshot(
            Some(&[2]),
            &TraceView {
                layers: false,
                partition: false,
                probabilities: false,
                log_probabilities: false,
                ..TraceView::default()
            },
        )
        .unwrap();
    assert!(reduced.layers.is_empty());
    assert!(reduced.input_probabilities.is_none());
    assert!(reduced.output_log_probabilities.is_none());
}
