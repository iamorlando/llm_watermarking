use llm_watermarking::unigram::{Unigram, UnigramConfig};
use rand::{distr::weighted::WeightedIndex, distr::Distribution, SeedableRng};
use rand_isaac::Isaac64Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Demonstration keys and a synthetic vocabulary; no language model required.
    let watermark = Unigram::new(&UnigramConfig::new([42; 32], 128))?;
    let wrong_key = Unigram::new(&UnigramConfig::new([43; 32], 128))?;
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let mut weights = [1.0; 128];
    watermark.apply(&mut weights)?;
    let distribution = WeightedIndex::new(weights)?;
    let baseline = WeightedIndex::new([1.0; 128])?;
    let marked: Vec<_> = (0..1000)
        .map(|_| distribution.sample(&mut rng) as u32)
        .collect();
    let unmarked: Vec<_> = (0..1000)
        .map(|_| baseline.sample(&mut rng) as u32)
        .collect();
    println!("marked: {:?}", watermark.detect(&marked, 0, &[])?);
    println!("unmarked: {:?}", watermark.detect(&unmarked, 0, &[])?);
    println!("wrong key: {:?}", wrong_key.detect(&marked, 0, &[])?);
    Ok(())
}
