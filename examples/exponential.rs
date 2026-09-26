use llm_watermarking::{
    exponential::{ExponentialRace, ExponentialRaceConfig},
    sampling::AlignmentConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = ExponentialRaceConfig::new([42; 32], 128);
    config.sequence_len = 128;
    let watermark = ExponentialRace::new(&config)?;
    config.key = [43; 32];
    let wrong_key = ExponentialRace::new(&config)?;
    // Choose a host-random starting offset in real use. The key rows wrap at
    // sequence_len; this 64-token example never reuses a row.
    let tokens: Vec<_> = (17..81)
        .map(|position| watermark.sample(&[1.0; 128], position))
        .collect::<Result<_, _>>()?;
    println!(
        "marked (lower is stronger): {:?}",
        watermark.detect(&tokens, 0, &[], 17)?
    );
    println!("wrong key: {:?}", wrong_key.detect(&tokens, 0, &[], 17)?);
    let mut edited = tokens;
    edited.remove(12);
    edited.insert(25, 7);
    let mut alignment = AlignmentConfig::new(edited.len());
    alignment.edit_penalty = Some(0.2);
    println!(
        "edited, unknown offset: {:?}",
        watermark.detect_aligned(&edited, 0, &[], &alignment)?
    );
    Ok(())
}
