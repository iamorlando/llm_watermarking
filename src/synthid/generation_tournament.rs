//! Explicit SynthID token selection, with optional capture of the actual matches.
//! The host supplies its live generation RNG and emits the returned token.
//! Existing probability-transform APIs never enter this module.

use sha2::Sha256;

use super::{
    tournament::{TournamentDraw, TournamentSide, TournamentWinReason, TournamentWinner},
    SynthIdText, MAX_DEPTH,
};
use crate::{common, WatermarkError};

/// A work limit, independent of capture limits: depth 20 executes 1,048,576 draws.
/// The default probability-update depth of 30 is intentionally unsupported here.
pub const MAX_GENERATION_TOURNAMENT_DEPTH: usize = 20;
pub const MAX_CAPTURED_MATCHES: usize = 65_535;
/// Defines traversal, categorical mapping, tie coins, and RNG consumption.
pub const SAMPLING_VERSION: &str = "synthid_explicit_tournament_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationTournamentOptions {
    pub max_matches: usize,
    /// Capture rounds nearest the final winner. Never limits generation depth.
    pub max_layers: usize,
}

impl Default for GenerationTournamentOptions {
    fn default() -> Self {
        Self {
            max_matches: 4095,
            max_layers: 32,
        }
    }
}

impl GenerationTournamentOptions {
    pub fn validate(&self) -> Result<(), WatermarkError> {
        if self.max_matches > MAX_CAPTURED_MATCHES || self.max_layers > MAX_DEPTH {
            return Err(WatermarkError::InvalidGenerationTournamentOptions);
        }
        Ok(())
    }
}

/// Host-supplied provenance for the SAME RNG passed to sampling. This is a label,
/// never a seed used to instantiate or reset a second RNG. Include RNG/seeding
/// algorithm versions; a seed alone does not describe an advanced RNG's state.
#[derive(Clone, Copy, Debug)]
pub struct GenerationRngInfo<'a> {
    pub rng_version: &'a str,
    pub effective_seed: &'a str,
}

impl GenerationRngInfo<'_> {
    fn validate(&self) -> Result<(), WatermarkError> {
        if self.rng_version.trim().is_empty()
            || self.effective_seed.is_empty()
            || !self.effective_seed.bytes().all(|c| c.is_ascii_digit())
        {
            return Err(WatermarkError::InvalidGenerationRngInfo);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationTournamentStatus {
    Applied,
    Warmup,
    RepeatedContext,
    /// Probability updates followed by a categorical draw execute no bracket.
    NoProductionBracket,
    WatermarkDisabled,
    Greedy,
    UnsupportedSampling,
    UnsupportedDepth,
}

impl GenerationTournamentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::Warmup => "warmup",
            Self::RepeatedContext => "repeated_context",
            Self::NoProductionBracket => "no_production_bracket",
            Self::WatermarkDisabled => "watermark_disabled",
            Self::Greedy => "greedy",
            Self::UnsupportedSampling => "unsupported_sampling",
            Self::UnsupportedDepth => "unsupported_depth",
        }
    }
}

/// Reasons a host can report without claiming matches were executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoTournamentReason {
    ProbabilityUpdates,
    WatermarkDisabled,
    Greedy,
    UnsupportedSampling,
    UnsupportedDepth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationTournamentSource {
    Draw(usize),
    Match(usize),
    /// Refers to the root_match_id of an explicitly collapsed subtree.
    CollapsedSubtree(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationTournamentEntrant {
    pub source: GenerationTournamentSource,
    pub draw_id: usize,
    pub token_id: u32,
    pub g_value: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationTournamentMatch {
    /// Logical round-order ID; capture can be sparse. Never index the Vec by ID.
    pub match_id: usize,
    pub round: usize,
    pub left: GenerationTournamentEntrant,
    pub right: GenerationTournamentEntrant,
    pub winner: TournamentSide,
    pub winner_draw_id: usize,
    pub reason: TournamentWinReason,
}

/// All matches in this subtree really executed; only their capture was omitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollapsedTournamentSubtree {
    pub root_match_id: usize,
    pub round: usize,
    pub first_draw_id: usize,
    pub draw_count: usize,
    pub match_count: usize,
    pub winner_draw_id: usize,
    pub token_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentTruncationReason {
    MaxMatches,
    MaxLayers,
    MaxMatchesAndLayers,
}

impl TournamentTruncationReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxMatches => "max_matches",
            Self::MaxLayers => "max_layers",
            Self::MaxMatchesAndLayers => "max_matches_and_layers",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationTournament {
    /// Always "production"; unlike the separate demonstration API.
    pub origin: &'static str,
    pub used_for_generation: bool,
    pub status: GenerationTournamentStatus,
    /// None only if the host reports that no SynthID instance was configured.
    pub configured_depth: Option<usize>,
    pub rounds: usize,
    pub total_draws: usize,
    pub total_matches: usize,
    /// Captured real draws, including advancing draws from collapsed subtrees.
    /// The host decodes text using its actual tokenizer, only when exporting.
    pub draws: Vec<TournamentDraw>,
    pub matches: Vec<GenerationTournamentMatch>,
    pub collapsed_subtrees: Vec<CollapsedTournamentSubtree>,
    pub winner: Option<TournamentWinner>,
    pub rng_version: String,
    pub effective_seed: String,
    pub sampling_version: Option<&'static str>,
    pub truncated: bool,
    pub truncation_reason: Option<TournamentTruncationReason>,
}

impl GenerationTournament {
    /// Metadata only: does not sample, call an RNG, or fabricate a winner.
    pub fn not_run(
        configured_depth: Option<usize>,
        reason: NoTournamentReason,
        rng: GenerationRngInfo<'_>,
    ) -> Result<Self, WatermarkError> {
        rng.validate()?;
        let status = match reason {
            NoTournamentReason::ProbabilityUpdates => {
                GenerationTournamentStatus::NoProductionBracket
            }
            NoTournamentReason::WatermarkDisabled => GenerationTournamentStatus::WatermarkDisabled,
            NoTournamentReason::Greedy => GenerationTournamentStatus::Greedy,
            NoTournamentReason::UnsupportedSampling => {
                GenerationTournamentStatus::UnsupportedSampling
            }
            NoTournamentReason::UnsupportedDepth => GenerationTournamentStatus::UnsupportedDepth,
        };
        Ok(Self::empty(configured_depth, status, rng))
    }

    fn empty(
        depth: Option<usize>,
        status: GenerationTournamentStatus,
        rng: GenerationRngInfo<'_>,
    ) -> Self {
        Self {
            origin: "production",
            used_for_generation: false,
            status,
            configured_depth: depth,
            rounds: 0,
            total_draws: 0,
            total_matches: 0,
            draws: Vec::new(),
            matches: Vec::new(),
            collapsed_subtrees: Vec::new(),
            winner: None,
            rng_version: rng.rng_version.into(),
            effective_seed: rng.effective_seed.into(),
            sampling_version: None,
            truncated: false,
            truncation_reason: None,
        }
    }
}

/// Select this sampling algorithm independently of whether tracing is requested.
/// Same instance, input, and live RNG stream yield the same token with or without
/// capture. This is not seed-for-seed equivalent to sampling the marginal update.
pub struct ProductionTournamentSampler<'a> {
    watermark: &'a SynthIdText,
}

impl SynthIdText {
    /// Validates the exponential generation work before a request starts.
    pub fn tournament_sampler(&self) -> Result<ProductionTournamentSampler<'_>, WatermarkError> {
        if self.depth > MAX_GENERATION_TOURNAMENT_DEPTH {
            return Err(WatermarkError::UnsupportedTournamentDepth {
                configured: self.depth,
                maximum: MAX_GENERATION_TOURNAMENT_DEPTH,
            });
        }
        Ok(ProductionTournamentSampler { watermark: self })
    }
}

impl ProductionTournamentSampler<'_> {
    pub fn configured_depth(&self) -> usize {
        self.watermark.depth
    }

    /// Performs no trace allocation: a zero-sized recorder is specialized
    /// away. The callback borrows the host's live generation RNG (`next_u64`).
    pub fn sample(
        &self,
        weights: &[f32],
        context: &[u32],
        prompt_len: usize,
        next_u64: &mut impl FnMut() -> u64,
    ) -> Result<u32, WatermarkError> {
        let input = Input::new(self.watermark, weights, context, prompt_len)?;
        Ok(input.sample(next_u64, &mut NoCapture, ()).token_id)
    }

    /// Returns the token to emit and the bracket from that very sampling call.
    /// Neither options nor provenance affect generation randomness or depth.
    pub fn sample_traced(
        &self,
        weights: &[f32],
        context: &[u32],
        prompt_len: usize,
        next_u64: &mut impl FnMut() -> u64,
        options: &GenerationTournamentOptions,
        rng: GenerationRngInfo<'_>,
    ) -> Result<(u32, GenerationTournament), WatermarkError> {
        options.validate()?;
        rng.validate()?;
        let input = Input::new(self.watermark, weights, context, prompt_len)?;
        let mut capture = Capture {
            trace: GenerationTournament::empty(Some(self.watermark.depth), input.status, rng),
            weights,
            total: input.total,
            leaves: 1 << self.watermark.depth,
        };
        capture.trace.sampling_version = Some(SAMPLING_VERSION);
        let plan = Plan {
            visible: true,
            matches: options.max_matches,
            layers: options.max_layers,
        };
        let selected = input.sample(next_u64, &mut capture, plan);
        if input.status == GenerationTournamentStatus::Applied {
            capture.trace.used_for_generation = true;
            capture.trace.rounds = self.watermark.depth;
            capture.trace.total_draws = capture.leaves;
            capture.trace.total_matches = capture.leaves - 1;
            capture.trace.winner = Some(TournamentWinner {
                match_id: capture.leaves - 2,
                draw_id: selected.tag,
                token_id: selected.token_id,
            });
            capture.trace.draws.sort_unstable_by_key(|d| d.draw_id);
            capture.trace.matches.sort_unstable_by_key(|m| m.match_id);
            capture
                .trace
                .collapsed_subtrees
                .sort_unstable_by_key(|s| s.root_match_id);
        }
        Ok((selected.token_id, capture.trace))
    }
}

struct Input {
    cdf: Vec<(u32, f64)>,
    total: f64,
    hash: Option<Sha256>,
    depth: usize,
    status: GenerationTournamentStatus,
}

impl Input {
    fn new(
        w: &SynthIdText,
        weights: &[f32],
        context: &[u32],
        prompt_len: usize,
    ) -> Result<Self, WatermarkError> {
        if prompt_len > context.len() {
            return Err(WatermarkError::PromptLengthExceedsContext);
        }
        if weights.len() > u32::MAX as usize {
            return Err(WatermarkError::InvalidVocabSize);
        }
        let total = common::weights(weights, weights.len())?;
        let status = if context.len() < w.context_len {
            GenerationTournamentStatus::Warmup
        } else if w.repeated_context(context, prompt_len) {
            GenerationTournamentStatus::RepeatedContext
        } else {
            GenerationTournamentStatus::Applied
        };
        let hash = (status == GenerationTournamentStatus::Applied)
            .then(|| w.context_hash(&context[context.len() - w.context_len..]));
        let mut cumulative = 0.0;
        let cdf = weights
            .iter()
            .enumerate()
            .filter_map(|(id, &weight)| {
                if weight == 0.0 {
                    return None;
                }
                cumulative += f64::from(weight);
                Some((id as u32, cumulative))
            })
            .collect();
        Ok(Self {
            cdf,
            total,
            hash,
            depth: w.depth,
            status,
        })
    }

    fn draw<T: DrawTag>(&self, id: usize, rng: &mut impl FnMut() -> u64) -> Player<T> {
        let uniform = ((rng() >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0;
        let target = uniform * self.total;
        let index = self
            .cdf
            .partition_point(|&(_, cumulative)| cumulative <= target)
            .min(self.cdf.len() - 1);
        let token_id = self.cdf[index].0;
        let scores = self
            .hash
            .as_ref()
            .map(|h| SynthIdText::g_values(h, token_id))
            .unwrap_or([0; 32]);
        Player {
            token_id,
            scores,
            tag: T::new(id),
        }
    }

    fn sample<C: Recorder>(
        &self,
        rng: &mut impl FnMut() -> u64,
        recorder: &mut C,
        plan: C::Plan,
    ) -> Player<C::Tag> {
        if self.status != GenerationTournamentStatus::Applied {
            return self.draw(0, rng);
        }
        self.subtree(self.depth, 0, rng, recorder, plan)
    }

    fn subtree<C: Recorder>(
        &self,
        height: usize,
        first: usize,
        rng: &mut impl FnMut() -> u64,
        recorder: &mut C,
        plan: C::Plan,
    ) -> Player<C::Tag> {
        if height == 0 {
            let draw = self.draw(first, rng);
            recorder.draw(&draw, plan);
            return draw;
        }
        let (left_plan, right_plan) = C::split(plan);
        let left = self.subtree(height - 1, first, rng, recorder, left_plan);
        let right = self.subtree(
            height - 1,
            first + (1 << (height - 1)),
            rng,
            recorder,
            right_plan,
        );
        let round = height - 1;
        let left_g = SynthIdText::g_value(&left.scores, round) as u8;
        let right_g = SynthIdText::g_value(&right.scores, round) as u8;
        let (side, reason) = match left_g.cmp(&right_g) {
            std::cmp::Ordering::Greater => (TournamentSide::Left, TournamentWinReason::HigherScore),
            std::cmp::Ordering::Less => (TournamentSide::Right, TournamentWinReason::HigherScore),
            std::cmp::Ordering::Equal => (
                if rng() & 1 == 0 {
                    TournamentSide::Left
                } else {
                    TournamentSide::Right
                },
                TournamentWinReason::RandomTieBreak,
            ),
        };
        let winner = if side == TournamentSide::Left {
            left
        } else {
            right
        };
        recorder.game(
            Game {
                height,
                first,
                left,
                right,
                left_g,
                right_g,
                side,
                reason,
                winner,
            },
            plan,
        );
        winner
    }
}

trait DrawTag: Copy {
    fn new(id: usize) -> Self;
}
impl DrawTag for () {
    #[inline(always)]
    fn new(_: usize) {}
}
impl DrawTag for usize {
    fn new(id: usize) -> Self {
        id
    }
}

#[derive(Clone, Copy)]
struct Player<T: Copy> {
    token_id: u32,
    scores: [u8; 32],
    tag: T,
}
struct Game<T: Copy> {
    height: usize,
    first: usize,
    left: Player<T>,
    right: Player<T>,
    left_g: u8,
    right_g: u8,
    side: TournamentSide,
    reason: TournamentWinReason,
    winner: Player<T>,
}

trait Recorder {
    type Tag: DrawTag;
    type Plan: Copy;
    fn split(plan: Self::Plan) -> (Self::Plan, Self::Plan);
    fn draw(&mut self, player: &Player<Self::Tag>, plan: Self::Plan);
    fn game(&mut self, game: Game<Self::Tag>, plan: Self::Plan);
}
struct NoCapture;
impl Recorder for NoCapture {
    type Tag = ();
    type Plan = ();
    #[inline(always)]
    fn split(_: ()) -> ((), ()) {
        ((), ())
    }
    #[inline(always)]
    fn draw(&mut self, _: &Player<()>, _: ()) {}
    #[inline(always)]
    fn game(&mut self, _: Game<()>, _: ()) {}
}

#[derive(Clone, Copy)]
struct Plan {
    visible: bool,
    matches: usize,
    layers: usize,
}
impl Plan {
    fn retained(self) -> bool {
        self.visible && self.matches > 0 && self.layers > 0
    }
}
struct Capture<'a> {
    trace: GenerationTournament,
    weights: &'a [f32],
    total: f64,
    leaves: usize,
}

impl Capture<'_> {
    fn id(&self, height: usize, first: usize) -> usize {
        self.leaves - (self.leaves >> (height - 1)) + (first >> height)
    }
    fn retain_draw(&mut self, player: &Player<usize>) {
        self.trace.draws.push(TournamentDraw {
            draw_id: player.tag,
            token_id: player.token_id,
            probability: f64::from(self.weights[player.token_id as usize]) / self.total,
        });
    }
    fn source(
        &self,
        height: usize,
        first: usize,
        player: &Player<usize>,
        plan: Plan,
    ) -> GenerationTournamentSource {
        if height == 0 {
            GenerationTournamentSource::Draw(player.tag)
        } else if plan.retained() {
            GenerationTournamentSource::Match(self.id(height, first))
        } else {
            GenerationTournamentSource::CollapsedSubtree(self.id(height, first))
        }
    }
}

impl Recorder for Capture<'_> {
    type Tag = usize;
    type Plan = Plan;
    fn split(plan: Plan) -> (Plan, Plan) {
        let visible = plan.retained();
        let remaining = plan.matches.saturating_sub(1);
        let child = |matches| Plan {
            visible,
            matches,
            layers: plan.layers.saturating_sub(1),
        };
        (child(remaining.div_ceil(2)), child(remaining / 2))
    }
    fn draw(&mut self, player: &Player<usize>, plan: Plan) {
        if plan.visible {
            self.retain_draw(player);
        }
    }
    fn game(&mut self, game: Game<usize>, plan: Plan) {
        if !plan.visible {
            return;
        }
        let match_id = self.id(game.height, game.first);
        let round = game.height - 1;
        if plan.retained() {
            let (left_plan, right_plan) = Self::split(plan);
            let entrant = |p: Player<usize>, first, g_value, plan| GenerationTournamentEntrant {
                source: self.source(round, first, &p, plan),
                draw_id: p.tag,
                token_id: p.token_id,
                g_value,
            };
            self.trace.matches.push(GenerationTournamentMatch {
                match_id,
                round,
                left: entrant(game.left, game.first, game.left_g, left_plan),
                right: entrant(
                    game.right,
                    game.first + (1 << round),
                    game.right_g,
                    right_plan,
                ),
                winner: game.side,
                winner_draw_id: game.winner.tag,
                reason: game.reason,
            });
        } else {
            self.retain_draw(&game.winner);
            self.trace
                .collapsed_subtrees
                .push(CollapsedTournamentSubtree {
                    root_match_id: match_id,
                    round,
                    first_draw_id: game.first,
                    draw_count: 1 << game.height,
                    match_count: (1 << game.height) - 1,
                    winner_draw_id: game.winner.tag,
                    token_id: game.winner.token_id,
                });
            self.trace.truncated = true;
            let reason = match (plan.matches == 0, plan.layers == 0) {
                (true, true) => TournamentTruncationReason::MaxMatchesAndLayers,
                (true, false) => TournamentTruncationReason::MaxMatches,
                _ => TournamentTruncationReason::MaxLayers,
            };
            self.trace.truncation_reason = Some(match self.trace.truncation_reason {
                None => reason,
                Some(previous) if previous == reason => reason,
                Some(_) => TournamentTruncationReason::MaxMatchesAndLayers,
            });
        }
    }
}
