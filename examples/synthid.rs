use llm_watermarking::synthid::{SynthIdConfig, SynthIdText};
use rand::{distr::weighted::WeightedIndex, distr::Distribution, SeedableRng};
use rand_isaac::Isaac64Rng;

const VOCAB_SIZE: usize = 128;
const GENERATED_TOKENS: usize = 1000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reproducible demonstration keys are unsuitable for production.
    let watermark = SynthIdText::new(&SynthIdConfig::new([0x42; 32]))?;
    let wrong_key = SynthIdText::new(&SynthIdConfig::new([0x43; 32]))?;
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let mut marked = vec![1, 2, 3, 4];
    let mut unmarked = marked.clone();
    let prompt_len = marked.len();
    let uniform = vec![1.0 / VOCAB_SIZE as f32; VOCAB_SIZE];
    let baseline = WeightedIndex::new(&uniform)?;
    for _ in 0..GENERATED_TOKENS {
        let mut probs = uniform.clone();
        watermark.apply(&mut probs, &marked, prompt_len)?;
        marked.push(WeightedIndex::new(&probs)?.sample(&mut rng) as u32);
        unmarked.push(baseline.sample(&mut rng) as u32);
    }
    for (label, detector, tokens) in [
        ("marked", &watermark, &marked),
        ("unmarked", &watermark, &unmarked),
        ("wrong key", &wrong_key, &marked),
    ] {
        let evidence = detector.detect(tokens, prompt_len, &[])?;
        println!("{label}: {evidence:?}");
    }
    Ok(())
}
