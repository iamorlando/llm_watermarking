use llm_watermarking::mpac::{Mpac, MpacConfig};
use rand::{distr::weighted::WeightedIndex, distr::Distribution, SeedableRng};
use rand_isaac::Isaac64Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Four radix-4 symbols carry eight bits; leading zero symbols are preserved.
    let payload = [0, 1, 2, 3];
    let mut config = MpacConfig::new([42; 32], 128, payload.len());
    config.radix = 4;
    config.context_width = 2;
    let watermark = Mpac::new(&config)?;
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let mut tokens = vec![1, 2];
    for _ in 0..1000 {
        let mut weights = [1.0; 128];
        watermark.apply(&mut weights, &tokens, 2, &payload)?;
        tokens.push(WeightedIndex::new(weights)?.sample(&mut rng) as u32);
    }
    let recovered = watermark.detect(&tokens, 2, &[])?;
    println!("sent: {payload:?}; recovered: {:?}", recovered.payload);
    println!(
        "blind recovery winning fraction: {:?}",
        recovered.winning_fraction
    );
    // This payload was fixed before seeing the text, so it can be scored directly.
    println!(
        "known payload: {:?}",
        watermark.detect_payload(&tokens, 2, &[], &payload)?
    );
    Ok(())
}
