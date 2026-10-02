//! Bounded, opt-in SynthID tournament demonstrations for debugging and research.
//!
//! Probability-transform generation does not run this demonstration bracket.
//! No generation or trace method calls this module implicitly. Actual generation
//! brackets belong to `super::generation_tournament`.
use sha2::{Digest, Sha256};

use super::SynthIdText;
use crate::{common, trace::TraceStatus, WatermarkError};

pub const MAX_TOURNAMENT_ROUNDS: usize = 4;
const RANDOM_DOMAIN: &[u8] = b"llm-watermarking-synthid-tournament-demo-v1\0";

/// Diagnostic randomness is isolated from generation and from the watermark key.
/// Equal seeds replay the same draws/tie coins for the same inputs. For independent
/// observations, supply a different seed per step/attempt without drawing from the
/// generation RNG. Rounds must not exceed the configured SynthID depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TournamentOptions {
    pub rounds: usize,
    pub seed: u64,
}

impl Default for TournamentOptions {
    fn default() -> Self {
        Self { rounds: 4, seed: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentOrigin {
    Demonstration,
}

/// IDs are local to one demonstration. Draws remain distinct even when their
/// token IDs are equal. A match source means the winner of that previous match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentSource {
    Draw(usize),
    Match(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentWinReason {
    HigherScore,
    RandomTieBreak,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TournamentDraw {
    pub draw_id: usize,
    pub token_id: u32,
    /// Normalized over the full filtered input row, before watermarking.
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TournamentEntrant {
    pub source: TournamentSource,
    /// Original draw ID, including when entering as a previous match's winner.
    pub draw_id: usize,
    pub g_value: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TournamentMatch {
    /// Zero-based, increasing in round order. Also its index in `matches`.
    pub match_id: usize,
    /// Zero-based SynthID layer index; the first matches use g bit zero.
    pub round: usize,
    pub left: TournamentEntrant,
    pub right: TournamentEntrant,
    pub winner: TournamentSide,
    pub winner_draw_id: usize,
    pub reason: TournamentWinReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TournamentWinner {
    pub match_id: usize,
    pub draw_id: usize,
    pub token_id: u32,
}

/// A demonstration, never an explanation of how the actual generated token was
/// selected. Token text and actual generated token ID belong to the host.
#[derive(Clone, Debug, PartialEq)]
pub struct TournamentDemo {
    pub origin: TournamentOrigin,
    pub status: TraceStatus,
    pub seed: u64,
    pub requested_rounds: usize,
    pub configured_depth: usize,
    /// The requested number of rounds, or zero when skipped.
    pub rounds: usize,
    pub draws: Vec<TournamentDraw>,
    pub matches: Vec<TournamentMatch>,
    /// Absent for warmup/repeated-context skips; no contestants are drawn then.
    pub winner: Option<TournamentWinner>,
}

impl SynthIdText {
    /// Sample a bounded teaching bracket from the complete pre-watermark row.
    /// Uses this instance's actual key/domain/context and first `rounds` g bits.
    /// Higher g wins; equal g uses an independent fair diagnostic coin.
    ///
    /// At most 16 draws and 15 matches are retained. Sampling scans the full row
    /// with O(2^rounds) extra storage, with no V-sized CDF/mask or V-by-depth trace.
    /// All inputs are immutable. No production RNG, tensor readback, shared state,
    /// or existing trace/generation path is involved. Skips match `apply`.
    pub fn tournament_demo(
        &self,
        weights: &[f32],
        context: &[u32],
        prompt_len: usize,
        options: &TournamentOptions,
    ) -> Result<TournamentDemo, WatermarkError> {
        if !(1..=MAX_TOURNAMENT_ROUNDS).contains(&options.rounds) || options.rounds > self.depth {
            return Err(WatermarkError::InvalidTournamentRounds);
        }
        if prompt_len > context.len() {
            return Err(WatermarkError::PromptLengthExceedsContext);
        }
        if weights.len() > u32::MAX as usize {
            return Err(WatermarkError::InvalidVocabSize);
        }
        let total = common::weights(weights, weights.len())?;
        let status = if context.len() < self.context_len {
            TraceStatus::Warmup
        } else if self.repeated_context(context, prompt_len) {
            TraceStatus::RepeatedContext
        } else {
            TraceStatus::Applied
        };
        let mut demo = TournamentDemo {
            origin: TournamentOrigin::Demonstration,
            status,
            seed: options.seed,
            requested_rounds: options.rounds,
            configured_depth: self.depth,
            rounds: 0,
            draws: Vec::new(),
            matches: Vec::new(),
            winner: None,
        };
        if status != TraceStatus::Applied {
            return Ok(demo);
        }
        demo.rounds = options.rounds;
        let count = 1usize << demo.rounds;
        // Separate draw/tie streams make initial contestants independent of how
        // many equal-score matches occurred. Neither stream uses the secret key.
        let random_prefix = Sha256::new()
            .chain_update(RANDOM_DOMAIN)
            .chain_update(options.seed.to_le_bytes());
        let mut draw_rng = common::HashRng::new(&random_prefix, 0);
        let mut tie_rng = common::HashRng::new(&random_prefix, 1);
        let mut targets: Vec<_> = (0..count)
            .map(|id| (id, draw_rng.uniform() * total))
            .collect();
        targets.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let mut tokens = vec![0u32; count];
        let mut next = 0;
        let mut cumulative = 0.0;
        let mut last_positive = 0;
        for (token, &weight) in weights.iter().enumerate() {
            if weight > 0.0 {
                last_positive = token as u32;
                cumulative += f64::from(weight);
                while next < count && targets[next].1 < cumulative {
                    tokens[targets[next].0] = token as u32;
                    next += 1;
                }
            }
        }
        // Endpoint rounding must never select a zero-weight token.
        for &(id, _) in &targets[next..] {
            tokens[id] = last_positive;
        }
        demo.draws = tokens
            .iter()
            .enumerate()
            .map(|(draw_id, &token_id)| TournamentDraw {
                draw_id,
                token_id,
                probability: f64::from(weights[token_id as usize]) / total,
            })
            .collect();
        let hash = self.context_hash(&context[context.len() - self.context_len..]);
        let scores: Vec<_> = tokens
            .iter()
            .map(|&token| Self::g_values(&hash, token))
            .collect();
        let mut entrants: Vec<_> = (0..count)
            .map(|id| (TournamentSource::Draw(id), id))
            .collect();
        demo.matches.reserve(count - 1);
        for round in 0..demo.rounds {
            let mut advancing = Vec::with_capacity(entrants.len() / 2);
            for pair in entrants.chunks_exact(2) {
                let entrant = |(source, draw_id): (TournamentSource, usize)| TournamentEntrant {
                    source,
                    draw_id,
                    g_value: Self::g_value(&scores[draw_id], round) as u8,
                };
                let left = entrant(pair[0]);
                let right = entrant(pair[1]);
                let (winner, reason) = match left.g_value.cmp(&right.g_value) {
                    std::cmp::Ordering::Greater => {
                        (TournamentSide::Left, TournamentWinReason::HigherScore)
                    }
                    std::cmp::Ordering::Less => {
                        (TournamentSide::Right, TournamentWinReason::HigherScore)
                    }
                    std::cmp::Ordering::Equal => (
                        if tie_rng.below(2) == 0 {
                            TournamentSide::Left
                        } else {
                            TournamentSide::Right
                        },
                        TournamentWinReason::RandomTieBreak,
                    ),
                };
                let winner_draw_id = match winner {
                    TournamentSide::Left => left.draw_id,
                    TournamentSide::Right => right.draw_id,
                };
                let match_id = demo.matches.len();
                advancing.push((TournamentSource::Match(match_id), winner_draw_id));
                demo.matches.push(TournamentMatch {
                    match_id,
                    round,
                    left,
                    right,
                    winner,
                    winner_draw_id,
                    reason,
                });
            }
            entrants = advancing;
        }
        let draw_id = entrants[0].1;
        demo.winner = Some(TournamentWinner {
            match_id: demo.matches.len() - 1,
            draw_id,
            token_id: tokens[draw_id],
        });
        Ok(demo)
    }
}
