//! Run with `cargo run --example tournament`. The host owns token decoding.
use llm_watermarking::synthid::{
    tournament::{TournamentOptions, TournamentWinReason},
    SynthIdConfig, SynthIdText,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let watermark = SynthIdText::new(&SynthIdConfig::new([6; 32]))?;
    // Example tokenizer labels; a real integration decodes the returned IDs.
    let words = ["fox", "owl", "cat", "tree", "dog"];
    let input = [1.0f32, 2.0, 0.0, 3.0, 7.0];
    let demo = watermark.tournament_demo(
        &input,
        &[1, 2, 3, 4],
        4,
        &TournamentOptions {
            rounds: 4,
            seed: 42,
        },
    )?;
    println!(
        "Teaching demonstration: {} rounds, {} draws; production depth {}.",
        demo.rounds,
        demo.draws.len(),
        demo.configured_depth
    );
    println!("This bracket is not used for generation.");
    for game in &demo.matches {
        let left = &demo.draws[game.left.draw_id];
        let right = &demo.draws[game.right.draw_id];
        let winner = &demo.draws[game.winner_draw_id];
        let reason = match game.reason {
            TournamentWinReason::HigherScore => "higher watermark score",
            TournamentWinReason::RandomTieBreak => "random tie-break",
        };
        println!("Round {}, match {}: {} #{} (p={:.3}, g={}) vs {} #{} (p={:.3}, g={}); {} #{} advances: {}.",
            game.round + 1, game.match_id,
            words[left.token_id as usize], left.draw_id, left.probability, game.left.g_value,
            words[right.token_id as usize], right.draw_id, right.probability, game.right.g_value,
            words[winner.token_id as usize], winner.draw_id, reason);
    }
    if let Some(winner) = demo.winner {
        println!(
            "Demonstration winner: {} #{}.",
            words[winner.token_id as usize], winner.draw_id
        );
    }
    Ok(())
}
