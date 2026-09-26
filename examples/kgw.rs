use llm_watermarking::kgw::{Kgw, KgwConfig};
use rand::{distr::weighted::WeightedIndex, distr::Distribution, SeedableRng};
use rand_isaac::Isaac64Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Demonstration key only. Use securely generated key bytes in production.
    let mut config = KgwConfig::new([42; 32], 128);
    config.context_width = 2;
    let watermark = Kgw::new(&config)?;
    config.key = [43; 32];
    let wrong_key = Kgw::new(&config)?;
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let mut marked = vec![1, 2];
    let mut unmarked = marked.clone();
    let baseline = WeightedIndex::new([1.0; 128])?;
    for _ in 0..1000 {
        let mut weights = [1.0; 128];
        watermark.apply(&mut weights, &marked, 2)?;
        marked.push(WeightedIndex::new(weights)?.sample(&mut rng) as u32);
        unmarked.push(baseline.sample(&mut rng) as u32);
    }
    println!("marked: {:?}", watermark.detect(&marked, 2, &[])?);
    println!("unmarked: {:?}", watermark.detect(&unmarked, 2, &[])?);
    println!("wrong key: {:?}", wrong_key.detect(&marked, 2, &[])?);
    Ok(())
}
