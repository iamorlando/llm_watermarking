# Host migration: authoritative generation brackets

This is an integration contract for inference hosts, including Mistral. The
library implementation is generic; it changes no host repository, HTTP route,
tokenizer, inference engine, generation RNG, or speculative decoder.
See [the Rust API](generation-tournament.md) and
[runnable example](../examples/generation_tournament.rs).

## Preserve the endpoint and make capture opt-in

Keep `POST /v1/completions`. Add the optional nested setting to the existing
sampling trace request; leave prompt, watermark settings, temperature, and
generation seed in their ordinary request fields:

```json
{
  "sampling_trace": {
    "max_steps": 1,
    "max_candidates": 128,
    "max_layers": 32,
    "generation_tournament": {
      "max_matches": 4095
    }
  }
}
```

This fragment configures capture only. It does not select an algorithm, change
watermark depth, or supply another seed. When the nested object is absent, do
not construct/export a production tournament trace. Preserve the existing
outer trace behavior if it was requested independently.

Map nested `max_matches` and outer `max_layers` to
`GenerationTournamentOptions`. Accept zero for either to capture only a
collapsed root, if the host's broader trace contract permits zero. Validate
bounds before generation. `max_candidates` continues to limit the ordinary
candidate table, not the number of distinct draw identities in the bracket.
Bound bracket records with `max_matches` and the request's byte budget. Repeated
draws must not be deduplicated by token ID.

## Choose generation policy independently of tracing

The current probability-update sampler has no production bracket. For that
backend, requesting this field should attach an explicit no-bracket report:

```rust
use llm_watermarking::synthid::generation_tournament::{
    GenerationRngInfo, GenerationTournament, NoTournamentReason,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let report = GenerationTournament::not_run(
    Some(30), // Actual configured SynthID depth, not the display-layer budget.
    NoTournamentReason::ProbabilityUpdates,
    GenerationRngInfo {
        rng_version: "host-generation-rng/version-and-seeding-policy",
        effective_seed: "42", // The actual generation seed resolved by the host.
    },
)?;
assert_eq!(report.status.as_str(), "no_production_bracket");
assert!(!report.used_for_generation);
assert!(report.winner.is_none());
# Ok(())
# }
```

Do not relabel `teaching_tournament`, `layers`, a rerun, or a bracket conditioned
on the selected token as a production bracket. If the host keeps the existing
sampler, this no-bracket report is the truthful implementation of the request.

To return an applied production bracket, add an explicit tournament **generation
policy** independently of the trace option. The host may expose that choice
inside its ordinary watermark settings. Its name/serialization belong to the
host; this crate has no HTTP configuration type. Construct
`watermark.tournament_sampler()` once during configuration validation. Its
generation depth limit is 20, independent of capture limits. Reject unsupported
explicit configurations before streaming starts; never clamp depth 30 to 12
because `max_matches` is 4095. The probability-update backend remains available
for larger depths, with no claim of an executed bracket.

For a request using the explicit policy, use this flow for **every** step:

```text
model logits -> penalties/processors -> temperature -> probability filters
             -> complete pre-watermark weights
             -> sample OR sample_traced using the same live generation RNG
             -> host commits and emits the returned token directly
```

Do not first call `apply`/`apply_traced` on those weights: that would apply the
watermark twice. Do not make another categorical draw after the tournament.
Turning capture on or off must select only `sample_traced` versus `sample`, not
a different sampling policy. Once `max_steps` or a capture budget expires,
continue with `sample` on the same explicit sampler and RNG.

The library's explicit implementation consumes a host slice. Do not enable a
CPU fallback or full GPU tensor download solely because tracing is requested.
For device/fused paths without an explicit bracket implementation, report
`no_production_bracket` or `unsupported_sampling` as appropriate. Any host policy
that deliberately chooses scalar explicit sampling must choose it regardless
of tracing. No CUDA/Metal explicit bracket implementation is added here.

## Use the actual generation RNG

Borrow the live generation RNG and pass its `next_u64` operation. With a locked
RNG such as Mistral's ISAAC64 generator, hold that same RNG guard while sampling.
Do not create a demo RNG, derive another seed, clone the RNG to replay the step,
or consume a random word to name the trace. The Rust example uses `rand` only to
adapt its host RNG; the library has no `rand` runtime dependency.

Set `rng_version` from the actual generator and seeding policy, and serialize
the already resolved generation seed as a decimal string. Include the library's
`sampling_version` when present: generator identity alone does not describe
draw traversal and tie consumption. A seed is a replay identifier, not a snapshot
of a partially consumed RNG; preserve generated index, attempt, and host policies
that affect advancement. If the host cannot establish actual RNG provenance,
report that limitation explicitly rather than inventing an effective seed.

## Attach to the emitted step

Put the result at:

```text
choices[i].sampling_trace.steps[j].generation_tournament
```

For an applied trace, enforce before committing the response:

```text
generation_tournament.winner.token_id
    == step.selected_token_id
    == token committed to the sequence and emitted by the host
```

Keep a trace attached to its own sampling attempt. If a grammar rejects the
token and the host samples again with masked weights, discard that rejected
attempt from the emitted-token trace or expose it separately as a rejected
attempt. Never attach its bracket to the replacement token. Preserve the host's
normal retry/RNG policy in traced and untraced execution.

Speculative acceptance/rejection or later token replacement needs its own
integration that associates an executed bracket with each committed token.
Until then report `unsupported_sampling`; do not pretend a proposal's winner
selected a different committed token. For streaming, deliver each committed
step once, aligned to its token event, and flush pending trace data before the
terminal event. For multiple choices/sequences, keep RNG/context/capture budgets
and attempt ownership separate and respect any existing unsupported combinations.

## Wire fields

Library records are transport-neutral. The host maps enums to strings and
decodes token text only for captured draws, using its actual tokenizer.

| Wire field | Source and required meaning |
| --- | --- |
| `origin` | Library `origin`, always `"production"` |
| `used_for_generation` | True only when actual matches selected the returned token; false for explicit no-tournament statuses |
| `status` | `status.as_str()`: `applied`, `warmup`, `repeated_context`, `no_production_bracket`, `watermark_disabled`, `greedy`, `unsupported_sampling`, or `unsupported_depth` |
| `configured_depth` | Actual depth, not a capture limit; null only if there is no configured SynthID depth |
| `rounds` | Actual full rounds executed; zero if no tournament ran |
| `total_draws`, `total_matches` | Counts including omitted subtree records |
| `draws` | Retained actual `draw_id`, `token_id`, decoded `text`, and full-input `probability` |
| `matches` | Actual retained match IDs/rounds, both entrants with source references/token/draw IDs/g values, winning side, `winner_draw_id`, and reason |
| `winner` | Final actual match ID, draw ID and token ID; null if no tournament ran |
| `collapsed_subtrees` | Actual omitted subtree root match ID/round, leaf range/count, match count, advancing draw ID and token ID |
| `rng_version`, `effective_seed` | Actual generation provenance; seed is a decimal string |
| `sampling_version` | Versioned explicit sampling convention, when that sampler was used |
| `truncated`, `truncation_reason` | Capture omitted records, with the stated limits; never means fewer generation rounds |

Map winning sides to `left`/`right`. Map `TournamentWinReason::HigherScore` to
`higher_g` and `RandomTieBreak` to `random_tie`. An entrant's source can use
the existing tagged bracket structure:

```json
{"kind": "draw", "id": 7}
```

```json
{"kind": "match", "id": 12}
```

```json
{"kind": "collapsed_subtree", "id": 28}
```

The latter ID is the omitted subtree's `root_match_id`. Its advancing token and
draw are real generation results, not newly sampled examples. Preserve those
records when the candidate table or text display is filtered. Draw IDs and
match IDs are different namespaces. IDs may be sparse; arrays are not ID-indexed.
Even an entirely collapsed bracket preserves its final winner and root match
identity. An omitted loser is not evidence it was never drawn.

If the host cannot decode a token, report an explicit decode failure according
to its trace contract; do not substitute a different token's text. Never expose
watermark keys. Existing reporting logprobs retain their original host meaning.
An explicit bracket does not itself compute the full final marginal probability
vector; do not present the winning draw's input probability as its post-watermark
selection probability. Any optional marginal visualization must be labelled and
must never drive another draw or masquerade as actual matches.

Capture limits must preserve the current step's winner. Reserve space for a
minimal collapsed-root report before starting capture. If exporting a retained
tree exceeds a byte/text budget, collapse from those already captured actual
records and state the additional reason, without resampling. Stop requesting
traces on later steps when the request budget is exhausted. Do not change
generation depth, discard the current winner, or reset the RNG to fit a response.

## Mistral implementation checklist

In the inspected Mistral layout, the host work belongs in:

1. `sampling_trace.rs` and its tournament wire types: optional request/step
   fields, bound validation, enum mapping, decoded draws, collapsed sources,
   provenance and omission when not requested.
2. The watermark configuration/adapter: select and validate explicit sampling
   independently of capture; preserve the existing domain/key/context settings.
3. `sampler.rs` and `sampler/trace.rs`: use the same live generation RNG and
   pre-watermark filtered input; directly return the library's selected token.
   The old probability-update branch only reports absence of a bracket.
4. `pipeline/sampling.rs` and sequence trace state: retain attempt ownership,
   grammar retry alignment, committed token identity, and capture budgets.
5. Server completion request/response schemas, OpenAPI and examples: keep
   `/v1/completions`, document policy selection separately from trace capture,
   and add the optional nested `generation_tournament` contract.

Required host acceptance tests compare traced and untraced requests under the
**same sampling policy**, seed, model, prompt, filters and depth. Verify identical
tokens, subsequent RNG state, logprobs, and grammar retry behavior. Verify
absent fields/no decoding/no new readback when absent, exact winner-to-emitted
token equality, duplicate draw IDs remaining distinct, zero/partial/full capture,
explicit unsupported/skip/no-bracket statuses, step and byte budgets, streaming
alignment and independent choices. A library test cannot prove HTTP token
emission or host RNG provenance; those checks belong to the consumer.
