use std::collections::{BTreeMap, BTreeSet};

use llm_watermarking::{
    synthid::{
        generation_tournament::*,
        tournament::{TournamentSide, TournamentWinReason},
        SynthIdConfig, SynthIdText,
    },
    WatermarkError,
};
use rand::{RngCore, SeedableRng};
use rand_isaac::Isaac64Rng;

fn watermark(depth: usize) -> SynthIdText {
    let mut config = SynthIdConfig::new([6; 32]);
    config.depth = depth;
    SynthIdText::with_domain(&config, b"production-test\0").unwrap()
}

fn info() -> GenerationRngInfo<'static> {
    GenerationRngInfo {
        rng_version: "rand_isaac/0.4.0/Isaac64Rng/seed_from_u64",
        effective_seed: "42",
    }
}

fn check_graph(trace: &GenerationTournament, selected: u32) {
    assert_eq!(trace.origin, "production");
    assert!(trace.used_for_generation);
    assert_eq!(trace.status, GenerationTournamentStatus::Applied);
    assert_eq!(Some(trace.rounds), trace.configured_depth);
    assert_eq!(trace.total_draws, 1 << trace.rounds);
    assert_eq!(trace.total_matches, trace.total_draws - 1);
    let draws: BTreeMap<_, _> = trace.draws.iter().map(|d| (d.draw_id, d)).collect();
    let games: BTreeMap<_, _> = trace.matches.iter().map(|m| (m.match_id, m)).collect();
    let collapsed: BTreeMap<_, _> = trace
        .collapsed_subtrees
        .iter()
        .map(|s| (s.root_match_id, s))
        .collect();
    assert_eq!(draws.len(), trace.draws.len());
    assert_eq!(games.len(), trace.matches.len());
    assert_eq!(collapsed.len(), trace.collapsed_subtrees.len());
    assert!(games.keys().all(|id| !collapsed.contains_key(id)));
    let mut referenced = BTreeSet::new();
    let mut retained_draws = BTreeSet::new();
    for game in &trace.matches {
        for entrant in [&game.left, &game.right] {
            let draw = draws[&entrant.draw_id];
            assert_eq!(draw.token_id, entrant.token_id);
            assert!(draw.probability > 0.0 && draw.probability <= 1.0);
            match entrant.source {
                GenerationTournamentSource::Draw(id) => {
                    assert_eq!(game.round, 0);
                    assert_eq!(id, entrant.draw_id);
                    assert!(retained_draws.insert(id));
                }
                GenerationTournamentSource::Match(id) => {
                    assert!(referenced.insert(id));
                    let source = games[&id];
                    assert_eq!(source.round + 1, game.round);
                    assert_eq!(source.winner_draw_id, entrant.draw_id);
                }
                GenerationTournamentSource::CollapsedSubtree(id) => {
                    assert!(referenced.insert(id));
                    let source = collapsed[&id];
                    assert_eq!(source.round + 1, game.round);
                    assert_eq!(source.winner_draw_id, entrant.draw_id);
                    assert_eq!(source.token_id, entrant.token_id);
                }
            }
        }
        let (win, lose) = match game.winner {
            TournamentSide::Left => (&game.left, &game.right),
            TournamentSide::Right => (&game.right, &game.left),
        };
        assert_eq!(game.winner_draw_id, win.draw_id);
        match game.reason {
            TournamentWinReason::HigherScore => assert!(win.g_value > lose.g_value),
            TournamentWinReason::RandomTieBreak => assert_eq!(win.g_value, lose.g_value),
        }
    }
    for subtree in &trace.collapsed_subtrees {
        assert!(retained_draws.insert(subtree.winner_draw_id));
        assert_eq!(draws[&subtree.winner_draw_id].token_id, subtree.token_id);
        assert!(
            (subtree.first_draw_id..subtree.first_draw_id + subtree.draw_count)
                .contains(&subtree.winner_draw_id)
        );
        assert_eq!(subtree.draw_count, 1 << (subtree.round + 1));
        assert_eq!(subtree.match_count, subtree.draw_count - 1);
    }
    assert_eq!(retained_draws.len(), trace.draws.len());
    assert_eq!(
        trace.matches.len()
            + trace
                .collapsed_subtrees
                .iter()
                .map(|s| s.match_count)
                .sum::<usize>(),
        trace.total_matches
    );
    let winner = trace.winner.as_ref().unwrap();
    assert_eq!(winner.token_id, selected);
    assert_eq!(draws[&winner.draw_id].token_id, selected);
    assert_eq!(winner.match_id, trace.total_matches - 1);
    assert!(!referenced.contains(&winner.match_id));
    assert_eq!(referenced.len() + 1, games.len() + collapsed.len());
    if let Some(game) = games.get(&winner.match_id) {
        assert_eq!(game.winner_draw_id, winner.draw_id);
    } else {
        assert_eq!(collapsed[&winner.match_id].winner_draw_id, winner.draw_id);
    }
    assert!(trace.draws.len() <= trace.matches.len() + 1);
}

#[test]
fn actual_matches_match_independent_sha256_and_live_rng_fixture() {
    let wm = watermark(3);
    let sampler = wm.tournament_sampler().unwrap();
    let mut calls = 0u64;
    let mut next = || {
        calls += 1;
        calls.wrapping_mul(0x9e3779b97f4a7c15)
    };
    let (token, trace) = sampler
        .sample_traced(
            &[1.0, 2.0, 0.0, 3.0, 7.0],
            &[1, 2, 3, 4],
            4,
            &mut next,
            &GenerationTournamentOptions::default(),
            GenerationRngInfo {
                rng_version: "fixture/weyl64-v1",
                effective_seed: "0",
            },
        )
        .unwrap();
    assert_eq!(calls, 13);
    check_graph(&trace, token);
    assert_eq!(token, 1);
    assert_eq!(trace.winner.as_ref().unwrap().draw_id, 6);
    assert_eq!(
        trace.draws.iter().map(|d| d.token_id).collect::<Vec<_>>(),
        [4, 3, 4, 1, 3, 4, 1, 4]
    );
    let expected = [
        (0, 0, 0, 1, 0, 0, true, 1, true),
        (1, 0, 2, 3, 0, 0, false, 2, true),
        (2, 0, 4, 5, 0, 0, true, 5, true),
        (3, 0, 6, 7, 0, 0, false, 6, true),
        (4, 1, 1, 2, 1, 0, false, 1, false),
        (5, 1, 5, 6, 0, 0, true, 6, true),
        (6, 2, 1, 6, 0, 1, true, 6, false),
    ];
    for (game, expected) in trace.matches.iter().zip(expected) {
        assert_eq!(
            (
                game.match_id,
                game.round,
                game.left.draw_id,
                game.right.draw_id,
                game.left.g_value,
                game.right.g_value,
                game.winner == TournamentSide::Right,
                game.winner_draw_id,
                game.reason == TournamentWinReason::RandomTieBreak
            ),
            expected
        );
    }
    for draw in &trace.draws {
        assert_eq!(
            draw.probability,
            [1.0, 2.0, 0.0, 3.0, 7.0][draw.token_id as usize] / 13.0
        );
    }
    assert!(!trace.truncated);
}

#[test]
fn capture_limits_do_not_change_tokens_depth_or_rng_state() {
    for depth in [1, 4, 10, 13] {
        let wm = watermark(depth);
        let sampler = wm.tournament_sampler().unwrap();
        for seed in [0, 42, u64::MAX] {
            let mut baseline = Isaac64Rng::seed_from_u64(seed);
            let mut baseline_calls = 0;
            let expected = sampler
                .sample(&[1.0, 2.0, 0.0, 3.0, 7.0], &[1, 2, 3, 4], 4, &mut || {
                    baseline_calls += 1;
                    baseline.next_u64()
                })
                .unwrap();
            let continuation: Vec<_> = (0..16).map(|_| baseline.next_u64()).collect();
            for (max_matches, max_layers) in [(0, 0), (1, 32), (2, 32), (9, 2), (4095, 32)] {
                let mut rng = Isaac64Rng::seed_from_u64(seed);
                let mut calls = 0;
                let options = GenerationTournamentOptions {
                    max_matches,
                    max_layers,
                };
                let seed_text = seed.to_string();
                let (token, trace) = sampler
                    .sample_traced(
                        &[1.0, 2.0, 0.0, 3.0, 7.0],
                        &[1, 2, 3, 4],
                        4,
                        &mut || {
                            calls += 1;
                            rng.next_u64()
                        },
                        &options,
                        GenerationRngInfo {
                            effective_seed: &seed_text,
                            ..info()
                        },
                    )
                    .unwrap();
                assert_eq!(token, expected);
                assert_eq!(calls, baseline_calls);
                assert_eq!(
                    (0..16).map(|_| rng.next_u64()).collect::<Vec<_>>(),
                    continuation
                );
                assert_eq!(trace.effective_seed, seed_text);
                assert_eq!(trace.sampling_version, Some(SAMPLING_VERSION));
                assert!(trace.matches.len() <= max_matches);
                check_graph(&trace, token);
                if max_matches < trace.total_matches || max_layers < depth {
                    assert!(trace.truncated);
                    assert!(trace.truncation_reason.is_some());
                }
            }
        }
    }
}

#[test]
fn collapsed_subtrees_are_actual_advancers_from_the_same_run() {
    let wm = watermark(8);
    let sampler = wm.tournament_sampler().unwrap();
    let sample = |options| {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        sampler
            .sample_traced(
                &[1.0, 2.0, 3.0, 7.0],
                &[1, 2, 3, 4],
                4,
                &mut || rng.next_u64(),
                &options,
                info(),
            )
            .unwrap()
    };
    let (token, full) = sample(GenerationTournamentOptions::default());
    for (max_matches, max_layers) in [(0, 32), (1, 32), (6, 32), (15, 2), (64, 3)] {
        let (actual, bounded) = sample(GenerationTournamentOptions {
            max_matches,
            max_layers,
        });
        assert_eq!(actual, token);
        check_graph(&bounded, actual);
        for game in &bounded.matches {
            let mut expected = full.matches[game.match_id].clone();
            expected.left.source = game.left.source;
            expected.right.source = game.right.source;
            assert_eq!(game, &expected);
        }
        // Source kinds change when their subtree is collapsed; compare actual
        // match decisions independently of how their child records are captured.
        for subtree in &bounded.collapsed_subtrees {
            let original = &full.matches[subtree.root_match_id];
            assert_eq!(subtree.winner_draw_id, original.winner_draw_id);
            assert_eq!(
                subtree.token_id,
                full.draws[subtree.winner_draw_id].token_id
            );
        }
        for draw in &bounded.draws {
            assert_eq!(draw, &full.draws[draw.draw_id]);
        }
    }
}

#[test]
fn repeated_tokens_have_separate_draws_and_real_tie_coins() {
    let wm = watermark(5);
    let mut calls = 0;
    let (token, trace) = wm
        .tournament_sampler()
        .unwrap()
        .sample_traced(
            &[0.0, f32::MAX, 0.0],
            &[1, 2, 3, 4],
            4,
            &mut || {
                calls += 1;
                1
            },
            &GenerationTournamentOptions::default(),
            info(),
        )
        .unwrap();
    check_graph(&trace, token);
    assert_eq!(calls, 63); // All 32 draws and all 31 actual tie coins.
    assert_eq!(trace.draws.len(), 32);
    assert!(trace
        .draws
        .iter()
        .all(|d| d.token_id == 1 && d.probability == 1.0));
    assert!(trace
        .matches
        .iter()
        .all(|m| m.reason == TournamentWinReason::RandomTieBreak));
    assert_eq!(trace.winner.unwrap().draw_id, 31);
}

#[test]
fn warmup_and_repeat_skips_preserve_the_generation_draw() {
    let wm = watermark(3);
    let sampler = wm.tournament_sampler().unwrap();
    for (context, prompt_len, status) in [
        (vec![1, 2, 3], 3, GenerationTournamentStatus::Warmup),
        (
            vec![1, 2, 3, 4, 1, 2, 3, 4],
            4,
            GenerationTournamentStatus::RepeatedContext,
        ),
    ] {
        let mut calls = 0;
        let (token, trace) = sampler
            .sample_traced(
                &[0.0, 1.0],
                &context,
                prompt_len,
                &mut || {
                    calls += 1;
                    42
                },
                &GenerationTournamentOptions::default(),
                info(),
            )
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(token, 1);
        assert_eq!(trace.status, status);
        assert_eq!(trace.rounds, 0);
        assert!(!trace.used_for_generation);
        assert!(trace.winner.is_none() && trace.draws.is_empty() && trace.matches.is_empty());
        let mut untraced_calls = 0;
        assert_eq!(
            sampler
                .sample(&[0.0, 1.0], &context, prompt_len, &mut || {
                    untraced_calls += 1;
                    42
                })
                .unwrap(),
            token
        );
        assert_eq!(untraced_calls, calls);
    }
}

#[test]
fn invalid_inputs_and_unsupported_depth_fail_before_using_rng() {
    assert!(matches!(
        watermark(30).tournament_sampler(),
        Err(WatermarkError::UnsupportedTournamentDepth { configured: 30, .. })
    ));
    assert!(watermark(MAX_GENERATION_TOURNAMENT_DEPTH)
        .tournament_sampler()
        .is_ok());
    let wm = watermark(3);
    let sampler = wm.tournament_sampler().unwrap();
    let mut never = || panic!("validation must not consume generation randomness");
    for weights in [
        &[][..],
        &[0.0, 0.0],
        &[f32::NAN, 1.0],
        &[-1.0, 1.0],
        &[f32::INFINITY, 1.0],
    ] {
        assert!(sampler
            .sample(weights, &[1, 2, 3, 4], 4, &mut never)
            .is_err());
    }
    assert!(sampler.sample(&[1.0], &[1], 2, &mut never).is_err());
    for options in [
        GenerationTournamentOptions {
            max_matches: MAX_CAPTURED_MATCHES + 1,
            max_layers: 1,
        },
        GenerationTournamentOptions {
            max_matches: 1,
            max_layers: 257,
        },
    ] {
        assert_eq!(
            sampler
                .sample_traced(&[1.0], &[1, 2, 3, 4], 4, &mut never, &options, info())
                .unwrap_err(),
            WatermarkError::InvalidGenerationTournamentOptions
        );
    }
    for seed in ["", "-1", "1.2", " 42", "0x42"] {
        assert_eq!(
            sampler
                .sample_traced(
                    &[1.0],
                    &[1, 2, 3, 4],
                    4,
                    &mut never,
                    &GenerationTournamentOptions::default(),
                    GenerationRngInfo {
                        effective_seed: seed,
                        ..info()
                    }
                )
                .unwrap_err(),
            WatermarkError::InvalidGenerationRngInfo
        );
    }
}

#[test]
fn probability_update_backend_reports_no_bracket_without_a_fake_winner() {
    for (reason, status) in [
        (
            NoTournamentReason::ProbabilityUpdates,
            GenerationTournamentStatus::NoProductionBracket,
        ),
        (
            NoTournamentReason::WatermarkDisabled,
            GenerationTournamentStatus::WatermarkDisabled,
        ),
        (
            NoTournamentReason::Greedy,
            GenerationTournamentStatus::Greedy,
        ),
        (
            NoTournamentReason::UnsupportedSampling,
            GenerationTournamentStatus::UnsupportedSampling,
        ),
        (
            NoTournamentReason::UnsupportedDepth,
            GenerationTournamentStatus::UnsupportedDepth,
        ),
    ] {
        let trace = GenerationTournament::not_run(Some(30), reason, info()).unwrap();
        assert_eq!(trace.status, status);
        assert_eq!(trace.configured_depth, Some(30));
        assert!(!trace.used_for_generation);
        assert_eq!(trace.rounds, 0);
        assert!(
            trace.winner.is_none()
                && trace.matches.is_empty()
                && trace.collapsed_subtrees.is_empty()
        );
        assert!(trace.sampling_version.is_none());
        assert!(!trace.truncated);
    }
}

#[test]
fn explicit_samples_follow_the_probability_update_distribution() {
    let wm = watermark(2);
    let sampler = wm.tournament_sampler().unwrap();
    let weights = [1.0, 2.0, 0.0, 3.0, 7.0];
    let mut expected = weights;
    wm.apply(&mut expected, &[1, 2, 3, 4], 4).unwrap();
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let mut counts = [0usize; 5];
    for _ in 0..20_000 {
        counts[sampler
            .sample(&weights, &[1, 2, 3, 4], 4, &mut || rng.next_u64())
            .unwrap() as usize] += 1;
    }
    for (count, p) in counts.into_iter().zip(expected) {
        assert!((count as f64 / 20_000.0 - f64::from(p)).abs() < 0.015);
    }
    assert_eq!(counts[2], 0);
}
