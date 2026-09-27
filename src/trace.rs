//! Opt-in sampling diagnostics. Ordinary generation methods do not call this
//! module. Model logits, sampling filters, token selection, and HTTP belong to
//! the host. Log probabilities below are canonical ln(p), not model logits.
use sha2::{Digest, Sha256};

use crate::WatermarkError;

/// Capture the first `max_layers` SynthID reductions. All configured layers
/// still execute. Zero captures no tournament details; the maximum is 256.
#[derive(Clone, Copy, Debug, Default)]
pub struct TraceOptions {
    pub max_layers: usize,
}

impl TraceOptions {
    pub(crate) fn validate(&self) -> Result<(), WatermarkError> {
        if self.max_layers > crate::synthid::MAX_DEPTH {
            return Err(WatermarkError::InvalidTraceOptions);
        }
        Ok(())
    }
}

/// Bounds and fields for a snapshot of explicitly selected rows. Subsets retain
/// their probability in the complete sampling distribution, not a renormalized
/// distribution over the snapshot. Raise bounds explicitly for full-vocabulary
/// inspection. These limits apply to returned diagnostic elements, not sampling.
#[derive(Clone, Copy, Debug)]
pub struct TraceView {
    pub max_rows: usize,
    pub max_elements: usize,
    pub probabilities: bool,
    pub log_probabilities: bool,
    pub partition: bool,
    pub layers: bool,
}

impl Default for TraceView {
    fn default() -> Self {
        Self {
            max_rows: 256,
            max_elements: 1_000_000,
            probabilities: true,
            log_probabilities: true,
            partition: true,
            layers: true,
        }
    }
}

impl TraceView {
    pub(crate) fn check(&self, rows: usize, layers: usize) -> Result<(), WatermarkError> {
        // Conservative upper bound across both sampling and transform schemas.
        let columns = 16usize
            + 2 * usize::from(self.probabilities)
            + usize::from(self.log_probabilities)
            + usize::from(self.partition)
            + if self.layers { 3 * layers } else { 0 };
        let elements = rows.saturating_mul(columns).saturating_add(4 * layers);
        if rows == 0 || rows > self.max_rows || elements > self.max_elements {
            return Err(WatermarkError::TraceTooLarge);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceKind {
    Identity,
    Bias,
    SynthId,
    ExponentialRace,
    InverseTransform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceStatus {
    Applied,
    Warmup,
    RepeatedContext,
}

/// Scalar exponential scores use -(-ln(U)/weight), whereas Candle returns
/// ln(weight)-ln(-ln(U)). Both select argmax; neither is a categorical weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionScoreKind {
    NegativeExponentialCost,
    GumbelMax,
    NegativeRank,
}

#[derive(Clone, Debug)]
pub struct ScalarTournamentLayer {
    pub index: usize,
    pub green_mass: f64,
    pub g_values: Vec<u8>,
    pub input_probabilities: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub input_normalizer: f64,
}

/// Scalar inverse CDF bounds and threshold are in original weight units.
#[derive(Clone, Debug)]
pub struct ScalarInverseTrace {
    pub uniform: f64,
    pub threshold: f64,
    pub total_weight: f64,
    pub ranks: Vec<u32>,
    pub cdf_lower: Vec<f64>,
    pub cdf_upper: Vec<f64>,
}

#[derive(Clone)]
pub(crate) struct ScalarInverseMetadata {
    pub uniform: f64,
    pub threshold: f64,
    pub total_weight: f64,
    pub ranks: Vec<usize>,
    pub cdf_lower: Vec<f64>,
    pub cdf_upper: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct ScalarTraceSnapshot {
    pub token_ids: Vec<u32>,
    pub kind: TraceKind,
    pub status: TraceStatus,
    pub input_weights: Vec<f32>,
    pub output_weights: Option<Vec<f32>>,
    pub input_probabilities: Option<Vec<f64>>,
    pub output_probabilities: Option<Vec<f64>>,
    pub output_log_probabilities: Option<Vec<f64>>,
    pub selection_scores: Option<Vec<f64>>,
    pub score_kind: Option<SelectionScoreKind>,
    pub selected_token: Option<u32>,
    pub favored_mask: Option<Vec<bool>>,
    pub bias_delta: Option<f64>,
    pub payload_position: Option<usize>,
    pub payload_symbol: Option<u8>,
    pub key_position: Option<usize>,
    pub inverse: Option<ScalarInverseTrace>,
    pub layers: Vec<ScalarTournamentLayer>,
    pub total_layers: usize,
    pub captured_layers: usize,
}

#[derive(Clone)]
pub(crate) struct ScalarReduction {
    pub total_before: f64,
    pub green_mass: f64,
}

#[derive(Clone)]
pub(crate) enum ScalarMetadata {
    Identity,
    Bias {
        mask: Vec<bool>,
        delta: f64,
        payload: Option<(usize, u8)>,
    },
    Tournament {
        hash: Sha256,
        reductions: Vec<ScalarReduction>,
        depth: usize,
    },
    Selection {
        scores: Vec<f64>,
        kind: SelectionScoreKind,
        selected: u32,
        position: usize,
        inverse: Option<ScalarInverseMetadata>,
    },
}

/// An immutable observation of a scalar call. No secret keys are exposed or
/// printed. Snapshot rows can be chosen after the host has selected a token.
/// SynthID retains O(depth) reductions, never a V-by-depth probability history.
#[derive(Clone)]
pub struct ScalarSamplingTrace {
    pub(crate) input: Vec<f32>,
    pub(crate) output: Option<Vec<f32>>,
    pub(crate) kind: TraceKind,
    pub(crate) status: TraceStatus,
    pub(crate) metadata: ScalarMetadata,
}

impl ScalarSamplingTrace {
    pub fn row_count(&self) -> usize {
        self.input.len()
    }
    pub fn output_weights(&self) -> Option<&[f32]> {
        self.output.as_deref()
    }

    pub(crate) fn identity(input: &[f32], status: TraceStatus) -> Self {
        Self {
            input: input.to_vec(),
            output: Some(input.to_vec()),
            kind: TraceKind::Identity,
            status,
            metadata: ScalarMetadata::Identity,
        }
    }

    pub(crate) fn bias(probs: &mut [f32], mask: Vec<bool>, delta: f64) -> Self {
        let input = probs.to_vec();
        crate::common::boost(probs, |i| mask[i], delta);
        Self {
            input,
            output: Some(probs.to_vec()),
            kind: TraceKind::Bias,
            status: TraceStatus::Applied,
            metadata: ScalarMetadata::Bias {
                mask,
                delta,
                payload: None,
            },
        }
    }

    pub(crate) fn with_payload(mut self, position: usize, symbol: u8) -> Self {
        if let ScalarMetadata::Bias { payload, .. } = &mut self.metadata {
            *payload = Some((position, symbol));
        }
        self
    }

    /// `token_ids=None` requests every vocabulary row, subject to view bounds.
    /// Selected-token/top-N snapshots can be requested after sampling without
    /// replaying the watermark or rerunning its vocabulary-wide reductions.
    pub fn snapshot(
        &self,
        token_ids: Option<&[u32]>,
        view: &TraceView,
    ) -> Result<ScalarTraceSnapshot, WatermarkError> {
        let (depth, captured) = match &self.metadata {
            ScalarMetadata::Tournament {
                reductions, depth, ..
            } => (*depth, reductions.len()),
            _ => (0, 0),
        };
        let count = token_ids.map_or(self.input.len(), <[u32]>::len);
        view.check(count, captured)?;
        if self.input.len() > u32::MAX as usize {
            return Err(WatermarkError::InvalidVocabSize);
        }
        let ids: Vec<u32> = match token_ids {
            Some(ids) => ids.to_vec(),
            None => (0..self.input.len() as u32).collect(),
        };
        for &id in &ids {
            crate::common::token(id, self.input.len())?;
        }
        let input_weights: Vec<_> = ids.iter().map(|&id| self.input[id as usize]).collect();
        let output_weights: Option<Vec<_>> = self
            .output
            .as_ref()
            .map(|p| ids.iter().map(|&id| p[id as usize]).collect());
        let input_total: f64 = self.input.iter().map(|&p| f64::from(p)).sum();
        let input_probabilities = view.probabilities.then(|| {
            input_weights
                .iter()
                .map(|&p| f64::from(p) / input_total)
                .collect()
        });
        let output_total: Option<f64> = self
            .output
            .as_ref()
            .map(|p| p.iter().map(|&v| f64::from(v)).sum());
        let output_probabilities = if view.probabilities {
            output_weights.as_ref().map(|p| {
                p.iter()
                    .map(|&p| f64::from(p) / output_total.unwrap())
                    .collect()
            })
        } else {
            None
        };
        let output_log_probabilities = if view.log_probabilities {
            output_weights.as_ref().map(|p| {
                p.iter()
                    .map(|&p| (f64::from(p) / output_total.unwrap()).ln())
                    .collect()
            })
        } else {
            None
        };
        let mut snapshot = ScalarTraceSnapshot {
            token_ids: ids,
            kind: self.kind,
            status: self.status,
            input_weights,
            output_weights,
            input_probabilities,
            output_probabilities,
            output_log_probabilities,
            selection_scores: None,
            score_kind: None,
            selected_token: None,
            favored_mask: None,
            bias_delta: None,
            layers: Vec::new(),
            total_layers: depth,
            captured_layers: captured,
            payload_position: None,
            payload_symbol: None,
            key_position: None,
            inverse: None,
        };
        match &self.metadata {
            ScalarMetadata::Identity => {}
            ScalarMetadata::Bias {
                mask,
                delta,
                payload,
            } => {
                snapshot.payload_position = payload.map(|p| p.0);
                snapshot.payload_symbol = payload.map(|p| p.1);
                snapshot.bias_delta = Some(*delta);
                if view.partition {
                    snapshot.favored_mask = Some(
                        snapshot
                            .token_ids
                            .iter()
                            .map(|&id| mask[id as usize])
                            .collect(),
                    );
                }
            }
            ScalarMetadata::Selection {
                scores,
                kind,
                selected,
                position,
                inverse,
            } => {
                snapshot.key_position = Some(*position);
                if let Some(inverse) = inverse {
                    snapshot.inverse = Some(ScalarInverseTrace {
                        uniform: inverse.uniform,
                        threshold: inverse.threshold,
                        total_weight: inverse.total_weight,
                        ranks: snapshot
                            .token_ids
                            .iter()
                            .map(|&id| inverse.ranks[id as usize] as u32)
                            .collect(),
                        cdf_lower: snapshot
                            .token_ids
                            .iter()
                            .map(|&id| inverse.cdf_lower[id as usize])
                            .collect(),
                        cdf_upper: snapshot
                            .token_ids
                            .iter()
                            .map(|&id| inverse.cdf_upper[id as usize])
                            .collect(),
                    });
                }
                snapshot.selection_scores = Some(
                    snapshot
                        .token_ids
                        .iter()
                        .map(|&id| scores[id as usize])
                        .collect(),
                );
                snapshot.score_kind = Some(*kind);
                snapshot.selected_token = Some(*selected);
            }
            ScalarMetadata::Tournament {
                hash, reductions, ..
            } if view.layers => {
                let digests: Vec<[u8; 32]> = snapshot
                    .token_ids
                    .iter()
                    .map(|id| {
                        hash.clone()
                            .chain_update(id.to_le_bytes())
                            .finalize()
                            .into()
                    })
                    .collect();
                let mut probs: Vec<f64> = snapshot
                    .input_weights
                    .iter()
                    .map(|&p| f64::from(p))
                    .collect();
                for (index, reduction) in reductions.iter().enumerate() {
                    let g_values: Vec<u8> = digests
                        .iter()
                        .map(|d| (d[index / 8] >> (index % 8)) & 1)
                        .collect();
                    let input_probabilities =
                        probs.iter().map(|p| *p / reduction.total_before).collect();
                    for (p, &g) in probs.iter_mut().zip(&g_values) {
                        *p = (*p / reduction.total_before)
                            * (1.0 + f64::from(g) - reduction.green_mass);
                    }
                    snapshot.layers.push(ScalarTournamentLayer {
                        index,
                        green_mass: reduction.green_mass,
                        g_values,
                        input_probabilities,
                        probabilities: probs.clone(),
                        input_normalizer: reduction.total_before,
                    });
                }
            }
            ScalarMetadata::Tournament { .. } => {}
        }
        Ok(snapshot)
    }
}
