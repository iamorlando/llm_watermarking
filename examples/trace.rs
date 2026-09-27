//! Run with `cargo run --example trace`.
use llm_watermarking::{
    synthid::{SynthIdConfig, SynthIdText},
    trace::{TraceOptions, TraceView},
};
use rand::{distr::weighted::WeightedIndex, prelude::Distribution, SeedableRng};
use rand_isaac::Isaac64Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let watermark = SynthIdText::new(&SynthIdConfig::new([42; 32]))?;
    let mut weights = [0.1f32, 0.2, 0.3, 0.4];
    let trace = watermark.apply_traced(
        &mut weights,
        &[10, 11, 12, 13],
        4,
        &TraceOptions { max_layers: 8 },
    )?;
    // The host's real draw runs exactly once, after watermarking.
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let selected = WeightedIndex::new(weights)?.sample(&mut rng) as u32;
    let mut ids = vec![selected];
    for id in [3, 1] {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let snapshot = trace.snapshot(
        Some(&ids),
        &TraceView {
            max_rows: 3,
            ..TraceView::default()
        },
    )?;
    println!("Selected token: {selected}; trace: {snapshot:?}");
    Ok(())
}
