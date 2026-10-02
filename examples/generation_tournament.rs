//! The token printed here comes from the same call that records the bracket.
use llm_watermarking::synthid::{
    generation_tournament::{GenerationRngInfo, GenerationTournamentOptions},
    SynthIdConfig, SynthIdText,
};
use rand::{RngCore, SeedableRng};
use rand_isaac::Isaac64Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let capture = !std::env::args().any(|arg| arg == "--no-trace");
    let mut config = SynthIdConfig::new([6; 32]);
    // Generation depth is the same with and without tracing.
    config.depth = 5;
    let watermark = SynthIdText::new(&config)?;
    let sampler = watermark.tournament_sampler()?;
    let seed = 42u64;
    let seed_text = seed.to_string();
    let mut rng = Isaac64Rng::seed_from_u64(seed);
    let words = ["fox", "owl", "cat", "tree", "dog"];
    let weights = [1.0f32, 2.0, 0.0, 3.0, 7.0];
    let context = [1, 2, 3, 4];
    let token = if capture {
        let (token, trace) = sampler.sample_traced(
            &weights,
            &context,
            context.len(),
            &mut || rng.next_u64(),
            &GenerationTournamentOptions {
                max_matches: 3,
                max_layers: 32,
            },
            GenerationRngInfo {
                rng_version: "rand_isaac/0.4.0/Isaac64Rng/seed_from_u64",
                effective_seed: &seed_text,
            },
        )?;
        assert_eq!(trace.winner.as_ref().unwrap().token_id, token);
        println!(
            "{}: {} rounds, {} actual draws, {} captured matches, {} collapsed subtrees",
            trace.origin,
            trace.rounds,
            trace.total_draws,
            trace.matches.len(),
            trace.collapsed_subtrees.len()
        );
        for draw in &trace.draws {
            println!(
                "draw {}: token {}, text {:?}, input probability {}",
                draw.draw_id, draw.token_id, words[draw.token_id as usize], draw.probability
            );
        }
        token
    } else {
        sampler.sample(&weights, &context, context.len(), &mut || rng.next_u64())?
    };
    // The host emits this exact token; there is no categorical draw afterward.
    println!("Emitted token {token}: {}", words[token as usize]);
    println!("Next generation RNG word: {}", rng.next_u64());
    Ok(())
}
