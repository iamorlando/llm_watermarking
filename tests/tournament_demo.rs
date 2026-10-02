use llm_watermarking::{
    synthid::{
        tournament::{
            TournamentDemo, TournamentOptions, TournamentOrigin, TournamentSide, TournamentSource,
            TournamentWinReason,
        },
        SynthIdConfig, SynthIdText,
    },
    trace::{TraceOptions, TraceStatus, TraceView},
    WatermarkError,
};

fn watermark(depth: usize) -> SynthIdText {
    let mut config = SynthIdConfig::new([6; 32]);
    config.depth = depth;
    SynthIdText::new(&config).unwrap()
}

fn check_tree(demo: &TournamentDemo) {
    assert_eq!(demo.origin, TournamentOrigin::Demonstration);
    assert_eq!(demo.draws.len(), 1 << demo.rounds);
    assert_eq!(demo.matches.len(), demo.draws.len() - 1);
    let mut references = vec![0; demo.matches.len()];
    let mut draw_references = vec![0; demo.draws.len()];
    for (index, game) in demo.matches.iter().enumerate() {
        assert_eq!(game.match_id, index);
        assert!(game.round < demo.rounds);
        for entrant in [&game.left, &game.right] {
            assert!(entrant.g_value <= 1);
            match entrant.source {
                TournamentSource::Draw(id) => {
                    assert_eq!(game.round, 0);
                    assert_eq!(entrant.draw_id, id);
                    draw_references[id] += 1;
                }
                TournamentSource::Match(id) => {
                    assert!(id < game.match_id);
                    assert_eq!(demo.matches[id].round + 1, game.round);
                    assert_eq!(demo.matches[id].winner_draw_id, entrant.draw_id);
                    references[id] += 1;
                }
            }
        }
        let (winner, loser) = match game.winner {
            TournamentSide::Left => (&game.left, &game.right),
            TournamentSide::Right => (&game.right, &game.left),
        };
        assert_eq!(winner.draw_id, game.winner_draw_id);
        match game.reason {
            TournamentWinReason::HigherScore => assert!(winner.g_value > loser.g_value),
            TournamentWinReason::RandomTieBreak => assert_eq!(winner.g_value, loser.g_value),
        }
    }
    assert!(draw_references.iter().all(|&uses| uses == 1));
    assert!(references[..references.len() - 1]
        .iter()
        .all(|&uses| uses == 1));
    assert_eq!(references.last(), Some(&0));
    let winner = demo.winner.as_ref().unwrap();
    assert_eq!(winner.match_id, demo.matches.len() - 1);
    assert_eq!(winner.draw_id, demo.matches.last().unwrap().winner_draw_id);
    assert_eq!(winner.token_id, demo.draws[winner.draw_id].token_id);
}

#[test]
fn bracket_matches_independent_python_sha256_draw_and_tie_fixture() {
    // Independent hashlib + struct implementation of the documented v1 format.
    let demo = watermark(30)
        .tournament_demo(
            &[1.0, 2.0, 0.0, 3.0, 7.0],
            &[1, 2, 3, 4],
            4,
            &TournamentOptions {
                rounds: 4,
                seed: 42,
            },
        )
        .unwrap();
    check_tree(&demo);
    assert_eq!(
        demo.draws.iter().map(|d| d.token_id).collect::<Vec<_>>(),
        [0, 4, 3, 3, 4, 4, 1, 1, 1, 4, 0, 3, 0, 3, 0, 4]
    );
    let expected = [
        (0, 0, 1, 1, 0, 0, false),
        (0, 2, 3, 1, 1, 3, true),
        (0, 4, 5, 0, 0, 4, true),
        (0, 6, 7, 1, 1, 7, true),
        (0, 8, 9, 1, 0, 8, false),
        (0, 10, 11, 1, 1, 11, true),
        (0, 12, 13, 1, 1, 12, true),
        (0, 14, 15, 1, 0, 14, false),
        (1, 0, 3, 1, 0, 0, false),
        (1, 4, 7, 0, 0, 4, true),
        (1, 8, 11, 0, 0, 8, true),
        (1, 12, 14, 1, 1, 14, true),
        (2, 0, 4, 0, 1, 4, false),
        (2, 8, 14, 0, 0, 8, true),
        (3, 4, 8, 0, 0, 8, true),
    ];
    for (game, expected) in demo.matches.iter().zip(expected) {
        assert_eq!(
            (
                game.round,
                game.left.draw_id,
                game.right.draw_id,
                game.left.g_value,
                game.right.g_value,
                game.winner_draw_id,
                game.reason == TournamentWinReason::RandomTieBreak
            ),
            expected
        );
    }
    // Lower original model probability can still win because its g score is higher.
    assert!(demo.draws[0].probability < demo.draws[1].probability);
    assert_eq!(demo.matches[0].winner_draw_id, 0);
}

#[test]
fn scores_match_actual_trace_including_custom_domains_and_probabilities_use_full_input() {
    let config = SynthIdConfig::new([6; 32]);
    for domain in [
        b"custom-debug-fixture\0".as_slice(),
        llm_watermarking::synthid::HASH_DOMAIN,
    ] {
        let watermark = SynthIdText::with_domain(&config, domain).unwrap();
        let original = [1.0f32, 2.0, 0.0, 3.0, 7.0];
        let demo = watermark
            .tournament_demo(
                &original,
                &[99, 1, 2, 3, 4],
                5,
                &TournamentOptions::default(),
            )
            .unwrap();
        check_tree(&demo);
        let mut weights = original;
        let trace = watermark
            .apply_traced(
                &mut weights,
                &[99, 1, 2, 3, 4],
                5,
                &TraceOptions { max_layers: 4 },
            )
            .unwrap();
        let snapshot = trace.snapshot(None, &TraceView::default()).unwrap();
        for game in demo.matches {
            for entrant in [game.left, game.right] {
                let draw = &demo.draws[entrant.draw_id];
                assert_eq!(
                    draw.probability,
                    f64::from(original[draw.token_id as usize]) / 13.0
                );
                assert_eq!(
                    entrant.g_value,
                    snapshot.layers[game.round].g_values[draw.token_id as usize]
                );
            }
        }
    }
}

#[test]
fn duplicates_are_distinct_and_point_masses_and_extreme_weights_are_supported() {
    let watermark = watermark(30);
    for mass in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
        let mut weights = vec![0.0; 257];
        weights[256] = mass;
        let demo = watermark
            .tournament_demo(
                &weights,
                &[1, 2, 3, 4],
                4,
                &TournamentOptions {
                    rounds: 4,
                    seed: u64::MAX,
                },
            )
            .unwrap();
        check_tree(&demo);
        for (id, draw) in demo.draws.iter().enumerate() {
            assert_eq!(draw.draw_id, id);
            assert_eq!(draw.token_id, 256);
            assert_eq!(draw.probability, 1.0);
        }
        assert!(demo
            .matches
            .iter()
            .all(|game| game.reason == TournamentWinReason::RandomTieBreak));
        assert_eq!(demo.winner.unwrap().token_id, 256);
    }
    // All-positive large values normalize without overflowing F32.
    let demo = watermark
        .tournament_demo(
            &[f32::MAX, f32::MAX],
            &[1, 2, 3, 4],
            4,
            &TournamentOptions::default(),
        )
        .unwrap();
    assert!(demo.draws.iter().all(|draw| draw.probability == 0.5));
    let singleton = watermark
        .tournament_demo(&[1.0], &[1, 2, 3, 4], 4, &TournamentOptions::default())
        .unwrap();
    check_tree(&singleton);
    assert_eq!(singleton.winner.unwrap().token_id, 0);
}

#[test]
fn rounds_are_bounded_replayable_and_never_exceed_configured_depth() {
    for depth in [1, 2, 4, 30, 256] {
        let watermark = watermark(depth);
        for rounds in 1..=4 {
            let options = TournamentOptions { rounds, seed: 42 };
            if rounds > depth {
                assert_eq!(
                    watermark.tournament_demo(&[1.0, 2.0, 3.0, 7.0], &[1, 2, 3, 4], 4, &options),
                    Err(WatermarkError::InvalidTournamentRounds)
                );
                continue;
            }
            let first = watermark
                .tournament_demo(&[1.0, 2.0, 3.0, 7.0], &[1, 2, 3, 4], 4, &options)
                .unwrap();
            let second = watermark
                .tournament_demo(&[1.0, 2.0, 3.0, 7.0], &[1, 2, 3, 4], 4, &options)
                .unwrap();
            assert_eq!(first, second);
            assert_eq!(first.rounds, rounds);
            assert_eq!(first.requested_rounds, rounds);
            assert_eq!(first.configured_depth, depth);
            check_tree(&first);
        }
    }
    let watermark = watermark(30);
    let first = watermark
        .tournament_demo(
            &[1.0; 5],
            &[1, 2, 3, 4],
            4,
            &TournamentOptions { rounds: 4, seed: 1 },
        )
        .unwrap();
    let second = watermark
        .tournament_demo(
            &[1.0; 5],
            &[1, 2, 3, 4],
            4,
            &TournamentOptions { rounds: 4, seed: 2 },
        )
        .unwrap();
    assert_ne!(first.draws, second.draws);
}

#[test]
fn validation_and_skips_match_generation_without_fabricated_games() {
    let watermark = watermark(30);
    for rounds in [0, 5, usize::MAX] {
        assert_eq!(
            watermark.tournament_demo(
                &[1.0],
                &[1, 2, 3, 4],
                4,
                &TournamentOptions { rounds, seed: 0 }
            ),
            Err(WatermarkError::InvalidTournamentRounds)
        );
    }
    let options = TournamentOptions::default();
    for weights in [
        vec![],
        vec![0.0],
        vec![-1.0, 1.0],
        vec![f32::NAN],
        vec![f32::INFINITY],
    ] {
        let mut generation = weights.clone();
        let expected = watermark
            .apply(&mut generation, &[1, 2, 3, 4], 4)
            .unwrap_err();
        assert_eq!(
            watermark
                .tournament_demo(&weights, &[1, 2, 3, 4], 4, &options)
                .unwrap_err(),
            expected
        );
    }
    assert_eq!(
        watermark.tournament_demo(&[1.0], &[1], 2, &options),
        Err(WatermarkError::PromptLengthExceedsContext)
    );
    for (context, prompt, status) in [
        (vec![1, 2], 2, TraceStatus::Warmup),
        (
            vec![1, 2, 3, 4, 1, 2, 3, 4],
            4,
            TraceStatus::RepeatedContext,
        ),
    ] {
        let demo = watermark
            .tournament_demo(&[1.0, 2.0], &context, prompt, &options)
            .unwrap();
        assert_eq!(demo.status, status);
        assert_eq!(demo.rounds, 0);
        assert!(demo.draws.is_empty());
        assert!(demo.matches.is_empty());
        assert!(demo.winner.is_none());
    }
    // Prompt-only occurrences are not repeats in generation.
    assert_eq!(
        watermark
            .tournament_demo(&[1.0, 2.0], &[1, 2, 3, 4, 1, 2, 3, 4], 8, &options)
            .unwrap()
            .status,
        TraceStatus::Applied
    );
}

#[test]
fn diagnostic_calls_do_not_change_production_or_existing_trace_results() {
    let watermark = watermark(30);
    let original = [1.0f32, 2.0, 0.0, 3.0, 7.0];
    let mut expected = original;
    watermark.apply(&mut expected, &[1, 2, 3, 4], 4).unwrap();
    for seed in 0..10 {
        let demo = watermark
            .tournament_demo(
                &original,
                &[1, 2, 3, 4],
                4,
                &TournamentOptions { rounds: 4, seed },
            )
            .unwrap();
        assert_eq!(demo.configured_depth, 30);
        let mut actual = original;
        let trace = watermark
            .apply_traced(
                &mut actual,
                &[1, 2, 3, 4],
                4,
                &TraceOptions { max_layers: 4 },
            )
            .unwrap();
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
        assert_eq!(trace.output_weights().unwrap(), expected);
    }
}

#[test]
fn empirical_winner_distribution_matches_actual_prefix_depth_transform() {
    // Fixed seed sweep is reproducible; generous bounds catch reversed score/tie
    // rules or wrong layer ordering without a statistical/flaky CI assertion.
    let original = [1.0f32, 2.0, 0.0, 3.0, 7.0];
    let samples = 4096;
    for rounds in 1..=4 {
        let mut expected = original;
        watermark(rounds)
            .apply(&mut expected, &[1, 2, 3, 4], 4)
            .unwrap();
        let demo_watermark = watermark(30);
        let mut counts = [0; 5];
        for seed in 0..samples {
            let demo = demo_watermark
                .tournament_demo(
                    &original,
                    &[1, 2, 3, 4],
                    4,
                    &TournamentOptions { rounds, seed },
                )
                .unwrap();
            counts[demo.winner.unwrap().token_id as usize] += 1;
        }
        for token in 0..5 {
            let actual = f64::from(counts[token]) / samples as f64;
            assert!(
                (actual - f64::from(expected[token])).abs() < 0.035,
                "rounds={rounds}, token={token}, observed={actual}, expected={}",
                expected[token]
            );
        }
        assert_eq!(counts[2], 0);
    }
}
