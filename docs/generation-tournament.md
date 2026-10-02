# Production tournament sampling and capture

`SynthIdText::tournament_sampler()` provides explicit, host-side tournament
sampling. `sample_traced` returns **both the selected token and the matches that
selected it, from one call**. The host emits that token directly. This library
does not add an HTTP server, tokenizer, model, or Mistral dependency.

The existing `apply`, tensor/indexed/resident operations, and probability-update
traces remain unchanged. They compute a tournament-equivalent distribution;
sampling that distribution does not execute a bracket. For those paths, report
`no_production_bracket` instead of attaching a sampled demonstration.

## API and generation randomness

```rust
use llm_watermarking::synthid::{
    generation_tournament::{GenerationRngInfo, GenerationTournamentOptions},
    SynthIdConfig, SynthIdText,
};
use rand::{RngCore, SeedableRng};
use rand_isaac::Isaac64Rng;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut config = SynthIdConfig::new([6; 32]);
config.depth = 5; // Ordinary generation setting, independent of tracing.
let watermark = SynthIdText::new(&config)?;
let sampler = watermark.tournament_sampler()?; // Validate before generation.
let generation_seed = 42u64;
let seed_text = generation_seed.to_string();
let mut generation_rng = Isaac64Rng::seed_from_u64(generation_seed);
let (selected_token_id, trace) = sampler.sample_traced(
    &[1.0f32, 2.0, 0.0, 3.0, 7.0], // Complete filtered weights BEFORE watermarking.
    &[1, 2, 3, 4],
    4,
    &mut || generation_rng.next_u64(),
    &GenerationTournamentOptions { max_matches: 3, max_layers: 32 },
    GenerationRngInfo {
        rng_version: "rand_isaac/0.4.0/Isaac64Rng/seed_from_u64",
        effective_seed: &seed_text,
    },
)?;
assert_eq!(trace.winner.as_ref().unwrap().token_id, selected_token_id);
assert_eq!(trace.rounds, 5); // Capture limits did not reduce generation depth.
// Emit selected_token_id. Decode retained draw IDs only when exporting the trace.
# Ok(())
# }
```

All production types are in `synthid::generation_tournament`. No additional
runtime dependency is needed: the RNG interface is `&mut impl FnMut() -> u64`.
The host supplies successive words from its existing generation RNG. The library
never seeds, clones, resets, or draws from another RNG. `GenerationRngInfo` is
provenance supplied by the host, not an instruction to instantiate randomness.
It accepts a nonempty RNG version and a decimal seed string, including integers
larger than u64. The library can validate this format, but only the host can
attest that the labels describe the RNG it supplied.

The method without capture is:

```text
sampler.sample(weights, context, prompt_len, &mut next_u64) -> Result<u32>
```

For the same inputs and RNG state, `sample` and `sample_traced` select the same
token and leave the RNG in the same state. This remains true with zero capture
limits, skipped contexts, partial brackets, and after the host stops capturing.
Switching from the old probability-update/categorical sampler to this sampler
does change how generation randomness is consumed. Therefore **select the
sampling algorithm independently of whether tracing is requested**.

`SAMPLING_VERSION = "synthid_explicit_tournament_v1"` fixes:

- A depth-first, left-before-right traversal of a full depth-D binary tree.
- One RNG word per leaf. Map it to `U = ((word >> 12) + 0.5) / 2^52`, then use
  the first positive-weight CDF interval strictly above `U * total_weight`.
- A match at height H uses the instance's actual g bit H-1. Higher g wins.
  Equal g consumes one additional RNG word: an even word chooses left, odd right.
- No tie word for unequal g, including no extra randomness for capture.
- Warmup/repeated contexts perform one ordinary input-distribution draw, no
  matches, no tie coins, and report the corresponding skipped status.

The CDF is in vocabulary-token order, uses f64 accumulation, and retains original
vocabulary IDs. Zero-weight tokens are excluded; unnormalized finite nonnegative
weights and singleton distributions are supported. Probability is always the
draw's weight divided by the full filtered input total. The original input and
watermark are immutable. Inputs/options/provenance are validated before RNG use.
Context/domain/repeat rules agree with the ordinary SynthID methods.

## Work and capture limits

A real depth-D bracket performs **2^D draws and 2^D - 1 matches**, even if only
one match is captured. There is no sampling shortcut conditioned on the answer.
This implementation supports generation depths 1 through 20, with
`MAX_GENERATION_TOURNAMENT_DEPTH` defining the work limit. The factory rejects
higher depths with `UnsupportedTournamentDepth`, before consuming randomness.
The ordinary probability-update default, depth 30, would need 1,073,741,824
draws for an explicit bracket and is unsupported here. No depth is silently
reduced and no automatic fallback selects a different algorithm.

The untraced explicit sampler uses O(V) CDF storage and O(D) stack storage, not
an exponential bracket allocation. Each leaf uses a binary CDF lookup and a
candidate hash; each match compares the actual advancing candidates. Capture
adds O(M) retained records, where M is the requested match budget.

`GenerationTournamentOptions` contains only capture limits:

| Field | Meaning |
| --- | --- |
| `max_matches` | Maximum retained match records; 0 through 65,535; default 4,095 |
| `max_layers` | Maximum visible rounds nearest the final match; 0 through 256; default 32 |

Capture retains a connected tree near the final match, distributing its remaining
match budget between the two child subtrees. Hidden subtrees still execute all
their matches. Each records its actual advancing draw as a `CollapsedSubtree`
source. If either limit is zero, the entire tree is one collapsed subtree;
the actual final match ID and winner remain available.

`rounds` is always the actual number of generation rounds, not the visible count.
`total_draws`/`total_matches` include omitted records. `truncated` and
`truncation_reason` explicitly report `max_matches`, `max_layers`, or both.
At most M+1 draw records and M+1 collapsed-subtree records are retained. The
general sampling trace's candidate-display limit is separate: repeated draws of
one token are distinct contestants, not a list of unique vocabulary candidates.
Hosts must budget bracket records/serialized bytes separately.

## Record identity and lineage

`GenerationTournament` uses `origin: "production"`. When applied,
`used_for_generation` is true and `winner.token_id` equals the returned token.
The host owns whether that token is committed/emitted, including grammar retries.

Draw IDs are leaf positions. Repeated tokens keep distinct IDs. Match IDs are
logical round-order IDs in the complete bracket:

```text
match_id = total_draws - (total_draws >> round) + (first_draw_id >> (round + 1))
```

Returned vectors are sorted by ID, but IDs can be sparse: **do not index a
captured vector by ID**. Match execution is depth-first; ID order does not claim
RNG execution order. Use ID maps and source references to build the bracket.

- Draws carry `draw_id`, `token_id`, and full-input `probability`.
- Matches carry their ID/round, both entrants' token/draw IDs, source and g value,
  winning side, winning draw ID, and reason. Serialize `HigherScore` as
  `higher_g` and `RandomTieBreak` as `random_tie`.
- Sources are `Draw`, `Match`, or `CollapsedSubtree`. A match source advances the
  winner of that match; a collapsed source advances that subtree's actual winner.
- Collapsed subtrees carry their root match ID/round, contiguous leaf range,
  omitted draw/match counts, actual advancing draw ID and token ID. The draw
  record for the advancing contestant is retained, including its probability.
- The final winner carries `match_id`, `draw_id`, and `token_id`. Its match can
  be present in `matches` or represented by a collapsed root.

No captured match is independently resampled or reconstructed after token
selection. Collapse changes retained observations, not the actual matches.

## Honest absence and unsupported paths

`GenerationTournament::not_run(depth, reason, rng_info)` makes an explicit
no-bracket report without accessing a sampler/RNG. Its `used_for_generation` is
false, `rounds` and total bracket counts are zero, and `winner` is absent.
`sampling_version` is absent because no explicit sampler ran. Available reasons
map to `no_production_bracket`, `watermark_disabled`, `greedy`,
`unsupported_sampling`, and `unsupported_depth`. `depth` is the actual configured
SynthID depth if known, otherwise None. Warmup and repeated-context statuses are
returned directly by the explicit sampler's traced call.

The library currently implements explicit sampling on **host probability slices**.
It does not add a CUDA/Metal explicit tournament sampler or implicitly download
tensors. A device-only/fused host path must report `unsupported_sampling` or
`no_production_bracket`, as appropriate. Backend selection or a deliberate CPU
sampling policy must also be independent of the tracing flag.

## No trace work when absent

No existing scalar, tensor, indexed, resident, or prepared method calls this
module or adds a trace flag, allocation, kernel, readback, or RNG call. The new
explicit sampler has a separate untraced entry point using a zero-sized,
statically dispatched no-op recorder. Its optimized specialization retains no
draw IDs, match records, provenance strings, decoding, or capture-limit checks.
The CDF and actual matches are generation work intrinsic to selecting explicit
tournament sampling, irrespective of trace settings.

Hosts must only call `sample_traced`, construct provenance, decode draws, and
serialize records when requested and while their capture budget permits. Once
capture ends, continue with `sample` on the same live RNG. Never fall back to the
probability-update sampler merely because a capture budget expired.

## Example and validation

```bash
cargo run --example generation_tournament
cargo run --example generation_tournament -- --no-trace
cargo test --test generation_tournament
```

The example prints the same emitted token and next RNG word with either flag.
Tests cover independent SHA-256/live-word match fixtures, exact RNG continuation,
traced/untraced selection, collapsed lineage versus a fully captured run,
depth exceeding capture limits, repeated-token draw identity, point masses,
warmup/repeat skips, invalid inputs/provenance and unsupported depths, and
agreement with the probability-update distribution over repeated samples.

See [the host/HTTP migration guide](generation-tournament-migration.md) for
attaching this trace to the token actually emitted by a completion endpoint.
