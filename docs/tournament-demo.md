# SynthID teaching tournament API

This opt-in diagnostic API returns a real sampled demonstration bracket for
research, debugging and teaching. Production SynthID generation still computes
its distribution directly with `apply`/`apply_traced`; it does not run this
bracket. The existing sampling trace describes those actual probability updates.
The two results have different meanings and must remain visibly separate.

An [explicit production sampler](generation-tournament.md) is also available.
It uses the host's generation RNG and returns the actual selected token with an
optional bracket; this demonstration method remains separate and unchanged.

Only SynthID exposes this method. No existing generation, prepared tensor,
trace, or other watermark method calls it. Their method bodies and state layouts
are unchanged, with no new runtime flags, allocations, kernels or transfers.
This is a guarantee about added work on those paths, not a timing benchmark.
The new API uses the existing `sha2` dependency and adds no runtime dependencies.

## Calling the library

```rust
use llm_watermarking::synthid::{
    tournament::TournamentOptions,
    SynthIdConfig, SynthIdText,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let synthid = SynthIdText::new(&SynthIdConfig::new([6; 32]))?;
let filtered_weights = [1.0f32, 2.0, 0.0, 3.0, 7.0];
let demo = synthid.tournament_demo(
    &filtered_weights,
    &[1, 2, 3, 4],
    4,
    &TournamentOptions { rounds: 4, seed: 42 },
)?;
assert_eq!(demo.draws.len(), 16);
assert_eq!(demo.matches.len(), 15);
// Decode demo.draws[*].token_id with the host's actual tokenizer.
// Keep the production-selected token ID separate from demo.winner.
# Ok(())
# }
```

The signature is `SynthIdText::tournament_demo(&[f32], &[u32], usize,
&TournamentOptions) -> Result<TournamentDemo, WatermarkError>`. Use the same
configured instance as generation, including its custom hash domain, prefix and
prompt length. Pass the **full filtered input before watermarking**, not the
post-watermark output or bounded display candidate rows. Input indices are
actual vocabulary IDs, as with scalar `apply`; this is a dense host API.

`rounds` must be 1–4 and must not exceed configured SynthID depth. Invalid rounds
return `InvalidTournamentRounds`; no silent truncation occurs. The library
options default to four rounds and seed zero, so set rounds explicitly for a
configuration with depth below four. Finite nonnegative weights with positive
mass are required; they need not sum to one. Zero-weight tokens cannot be drawn.
Singleton distributions are supported. All inputs and the watermark are immutable.

Warmup and repeated-context rules are identical to generation, including the
prompt boundary and repeat window. Such results contain the accurate
`TraceStatus`, `rounds: 0`, empty draws/matches and no winner. The requested
rounds and configured depth remain available. No diagnostic random numbers are
drawn for skipped results. The host handles its own greedy bypass separately.

See [the runnable example](../examples/tournament.rs):
`cargo run --example tournament`. Its first match demonstrates a lower-probability
“fox” beating “dog” because its actual keyed score is higher.

## Records and interpretation

All types are in `llm_watermarking::synthid::tournament`. IDs are zero-based and
local to a single demonstration; the host adds step/attempt identity and text.

| Type | Data |
| --- | --- |
| `TournamentDemo` | `origin: TournamentOrigin::Demonstration`, `status`, diagnostic `seed`, `requested_rounds`, `configured_depth`, executed `rounds`, `draws`, `matches`, optional `winner` |
| `TournamentDraw` | Unique `draw_id`, actual `token_id`, full-input normalized `probability` |
| `TournamentEntrant` | Typed `source` (`TournamentSource::Draw(id)` or `Match(id)`), original `draw_id`, this round's binary `g_value` |
| `TournamentMatch` | `match_id`, `round`, `left`/`right` entrants, `winner: TournamentSide::{Left,Right}`, `winner_draw_id`, `reason` |
| `TournamentWinReason` | `HigherScore` or `RandomTieBreak` |
| `TournamentWinner` | Final `match_id`, original `draw_id`, actual `token_id` |

Draw `i` is `draws[i]`; match `i` is `matches[i]`. A source of `Match(id)` means
that match's winning draw. The first round pairs draws 0/1, 2/3, etc.; subsequent
rounds pair the previous round's adjacent winners. Match IDs increase in round
order. Round zero uses the same g bit as production layer zero, round one uses
layer one, and so on. G values are computed with the actual SynthID key, domain
and context, exactly as in the existing probability-update trace.

The higher g value wins. Equal scores, including two occurrences of the same
token, use a fresh fair diagnostic coin. Draws are independent and made with
replacement: **never deduplicate contestants by token ID**. A repeated token
still has a different draw ID and a distinct place in the bracket.

The original input probability is attached to each draw. It is not the
probability that draw wins its match or the tournament. No per-bracket logits,
production selection probabilities, or contestant brackets for actual generation
are fabricated. The demonstration winner can differ from the generated token,
especially when production uses more layers. Across seeds, the demo winner
follows the tournament distribution for its first R layers, subject to normal
finite-precision sampling effects; one bracket is not a distribution estimate.

The method validates/scans the input row and samples all targets in one further
CDF scan. It retains at most 16 draws, 15 matches, and 16 token hashes: O(2^R)
extra storage, no vocabulary-sized CDF or mask and no V × depth history. Output
bounds are independent of the vocabulary size and production depth.

## Reproducible diagnostic randomness

The generator is isolated: it accepts a diagnostic seed and has no access to a
production RNG or shared mutable state. Its version-1 identity can be serialized
as `sha256_tournament_demo_v1`. It uses the existing library SHA-256 counter stream:

1. The prefix bytes are `llm-watermarking-synthid-tournament-demo-v1\0` followed
   by `LE64(seed)` and one tag byte. Draws use tag 0; tie coins use tag 1.
2. Block `c` is `SHA256(prefix || LE64(c))`, starting at zero. Consume each block
   as four little-endian u64 values in order.
3. For each initial draw, map value `x` to `u = ((x >> 12) + 0.5) / 2^52`, then
   select the first token whose cumulative original F64 weight exceeds
   `u * total_weight`. Last-positive-token fallback handles endpoint rounding.
   All draws are generated in draw-ID order; sorting thresholds is only an
   optimization of the common CDF scan.
4. For each equal-score match in match-ID order, consume a tie-stream value.
   An even value chooses left, odd chooses right. Non-tied matches do not
   consume tie values. Both outcomes have equal probability.

The random seed does not contain the secret watermark key or context. Those
only determine scores; changing the key does not redraw the initial contestants.
The same seed/inputs/domain/configuration reproduce the entire bracket. Never
borrow or clone the generation RNG to construct the diagnostic seed.

## Mistral integration handoff

Use the already integrated [sampling trace path](sampling-trace.md), on chat and
completions endpoints, with a separate optional nested object:

```json
{
  "logprobs": true,
  "sampling_trace": {
    "max_steps": 32,
    "max_candidates": 32,
    "max_layers": 8,
    "teaching_tournament": {"rounds": 4, "seed": 42}
  }
}
```

The request also needs the normal SynthID watermark configuration. Reject the
object for absent/non-SynthID watermark or invalid rounds/depth with HTTP 400.
Retain existing logprobs, n=1 and ordinary-generation restrictions. The consumer
may default an empty object to rounds=2, seed=0, as proposed; absence must disable
it completely. `max_layers` independently limits the actual generation trace
and may be zero while a teaching demonstration is requested.

`Sampler::sample_traced` already has the full filtered `before` row and the
configured Mistral-domain SynthID instance. Invoke `tournament_demo` only when
the nested option is present and the existing trace budget permits capture.
Use that `before` row and instance without converting to a top-candidate subset,
constructing a default-domain instance or changing generation. This requires no
additional GPU readback. Do not add a device teaching export or touch the compact
GPU path for this integration.

Suggested per-position effective seed:
`LE64(first 8 bytes of SHA256("mistralrs-synthid-teaching-seed-v1\0" ||
LE64(request_seed) || LE64(generated_index)))`. Keep the same seed on grammar
retry and use the accepted attempt's actual masked weights. Document the
chosen derivation and return its effective seed for replay. Serialize a u64
seed as a decimal string in response metadata to preserve exact values in
JavaScript clients. Diagnostic RNG state must never span requests or retries.

Map the result to a separate optional `step.teaching_tournament`:

- `origin: "teaching_simulation"`, `used_for_generation: false`, the RNG version,
  requested/effective rounds, configured depth, layer indices, effective seed
  and status (`demonstrated`, `warmup`, `repeated_context`, or host `greedy`).
- Contestants retaining draw IDs, token IDs, original probabilities and decoded
  text. They need not belong to the ordinary bounded `step.candidates` rows.
- Matches with typed source references, both g values, winning side/draw ID,
  and reason (`higher_g` or `random_tie`). Preserve all referenced records.
- A separate demo winner. The existing `step.selected_token_id` remains the
  actual generation result. Do not overwrite it or the production `layers`.

For greedy, report a skipped demo with empty draws/matches and no winner without
calling the library method. For warmup/repeat, map the library's accurate status.
No invented matches for skipped observations. Keep generation RNG draws and
all existing trace fields unchanged whether the demo is requested or absent.

Charge draw/match records to the existing request observation limit:

`max_steps * (max_candidates * max(1,max_layers) + (2^(rounds+1)-1)) <= 65536`

The added term is zero when absent. Validate rounds before checked arithmetic.
Reuse max_steps and the 8 MiB serialized byte budget. Include the whole demo
when sizing the step and drop/truncate observation at a whole-step boundary;
never emit part of a bracket with dangling references. Generation continues
normally when tracing stops. Existing SSE generated-index alignment, accepted
attempt ownership, delivery once per step and final truncation summary apply.

Update request/response types, SynthID adapter dispatch, trace sampler, HTTP
validation, OpenAPI, user guide, Python/SDK bindings as needed, and the server
example. Add tests for disabled-field omission, full-support contestants,
duplicates/lineage, same-seed replay, exact generation RNG/token/trace parity,
retries, HTTP/SSE mapping and request/byte bounds. No changes are needed to the
other watermark implementations.

## Library verification

`tests/tournament_demo.rs` verifies independent SHA-256 draw/tie/score fixtures,
complete graph lineage, custom-domain score agreement with existing traces,
full-row probabilities, zero-weight exclusion, duplicate contestants, singleton
and extreme distributions, invalid rounds including depth overflow, warmup/
repeat behavior, deterministic replay and unchanged production results. A fixed
seed sweep checks the winner distribution against each actual prefix-depth
transform for rounds 1–4.

Validation passed on 2026-09-27: the default and Candle CPU test suites (including
seven new tournament tests and existing trace/regression tests), Metal-feature
all-target Clippy with warnings denied, formatting and diff checks, and
`cargo run --example tournament`. This change adds a host-only diagnostic method;
no device implementation or readback path was changed.
