use llm_watermarking::semstamp::{SemStamp, SemStampConfig};
use rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
use std::convert::Infallible;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let watermark = SemStamp::new(&SemStampConfig::new([42; 32], 32))?;
    let wrong_key = SemStamp::new(&SemStampConfig::new([43; 32], 32))?;
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let mut embeddings = vec![(0..32)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<f32>>()];
    let mut attempts = 0;
    let mut fallbacks = 0;
    for _ in 0..100 {
        let sample = watermark.sample_sentence(embeddings.last().unwrap(), || {
            attempts += 1;
            // Replace this callback with sentence generation from the committed
            // history, followed by embedding with a fixed sentence encoder.
            let sentence = format!("Synthetic candidate {attempts}");
            let embedding = (0..32).map(|_| rng.random_range(-1.0..1.0)).collect();
            Ok::<_, Infallible>((sentence, embedding))
        })?;
        if !sample.accepted {
            fallbacks += 1;
        }
        // Commit sample.sentence to the host's history only here.
        embeddings.push(sample.embedding);
    }
    println!("Synthetic embeddings only; this is not a paraphrase robustness benchmark.");
    println!("{attempts} candidates; {fallbacks} maxout fallbacks");
    println!("marked: {:?}", watermark.detect(&embeddings, 1)?);
    println!("wrong key: {:?}", wrong_key.detect(&embeddings, 1)?);
    Ok(())
}
