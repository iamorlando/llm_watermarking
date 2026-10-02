# Sampling trace API and Mistral HTTP handoff

Implemented in `llm-watermarking` on 2026-09-26, following the Mistral consumer
agent's inspection and agreed requirements. This extends the
[GPU integration response](mistral-integration.md). HTTP implementation belongs
to Mistral; this document is its implementation handoff.

For sampled teaching brackets, use the separate [SynthID tournament demonstration
API](tournament-demo.md). Existing `layers` remain the actual generation
probability updates; the optional demonstration is not used for generation.

For matches that actually selected the emitted token, use the separate
[production tournament sampler](generation-tournament.md) and
[host migration contract](generation-tournament-migration.md). Its returned token
is authoritative; selecting that generation policy must be independent of trace
capture. Probability-update backends must report `no_production_bracket`.

## Opt-in boundary

All existing scalar `apply`/`sample`, prepared `apply`/`apply_trusted`, preparation
methods and prepared struct fields are unchanged. They perform no trace checks,
allocations, additional kernels, synchronization or serialization. Call the new
methods only when requested and while the host's capture budget remains. This is
a structural guarantee about added work, not a measured wall-clock benchmark.

The host owns model/processed logits, penalties, temperature, filters,
normalization, RNG draws, final token selection, decoding and transport. The
library records its actual weight transforms and keyed selection quantities.
`output_log_probabilities` is canonical `ln(p)`, **not raw model logits**.
Expose raw/processed logits only from the corresponding host sampling stages.

## Scalar API: initial Mistral integration

Import `llm_watermarking::trace::{TraceOptions, TraceView, ScalarSamplingTrace}`.
Every traced call validates the existing input contract; errors leave mutable
input weights unchanged.

| Scheme | New method | Result |
| --- | --- | --- |
| KGW | `apply_traced(&mut weights, context, prompt_len, &options)` | `ScalarSamplingTrace` |
| Unigram | `apply_traced(&mut weights, &options)` | `ScalarSamplingTrace` |
| MPAC | `apply_traced(&mut weights, context, prompt_len, payload, &options)` | `ScalarSamplingTrace` |
| SynthID | `apply_traced(&mut weights, context, prompt_len, &options)` | `ScalarSamplingTrace` |
| Exponential race | `sample_traced(&weights, position, &options)` | `(u32, ScalarSamplingTrace)` |
| Inverse transform | `sample_traced(&weights, position, &options)` | `(u32, ScalarSamplingTrace)` |

These methods return `Result<_, WatermarkError>`. Reweighting mutates the same
slice exactly as `apply`. Use the normal host categorical draw **once** afterward.
Keyed calls return the actual selected token; use it directly. No trace method
consumes host RNG or advances mutable history. Do not replay sampling to inspect
an extra candidate.

`TraceOptions { max_layers: 8 }` captures the first eight SynthID layer reductions;
all configured layers still execute. The library accepts 0–256, default 0.
Capture retains input/output weights and mask or selection metadata. SynthID
retains O(depth) reduction scalars and its opaque context hash, then reconstructs
only requested candidate rows; it does not retain V × depth probabilities.

After the real selection, choose bounded actual vocabulary IDs and call:

```rust
let snapshot = trace.snapshot(Some(&selected_and_top_ids), &TraceView {
    max_rows: max_candidates,
    max_elements: 1_000_000,
    ..TraceView::default()
})?;
```

`None` requests all vocabulary IDs, subject to explicit bounds. Default view
limits are 256 rows and 1,000,000 diagnostic elements; `probabilities`,
`log_probabilities`, `partition`, and `layers` default true and can be disabled.
Bounds reject the snapshot, never truncate the sampling distribution. IDs may
be arbitrary and retain the supplied order. Deduplicate them in the consumer.
Probabilities use the **entire input/output row** as denominator, never the
reported subset. See [the runnable scalar example](../examples/trace.rs).

`row_count()` returns full row size. `output_weights()` exposes transformed
weights for choosing top output candidates without a full snapshot. Keyed
traces have no output weights; rank keyed candidates using a bounded input
candidate pool and their snapshot scores (plus the selected token), or explicitly
raise view bounds if full score ranking is needed.

## Meaning of the exported fields

`ScalarTraceSnapshot` includes `token_ids`, `kind`, `status`, input/output
weights, optional normalized probabilities and canonical log probabilities,
optional favored membership, and scheme-specific fields. `TraceKind::Bias`
groups KGW, Unigram and MPAC; the host supplies the exact configured scheme.
`TraceStatus` distinguishes `Applied`, `Warmup`, and `RepeatedContext`.
A skipped call has identity output and no fabricated membership/layers.

| Scheme | Fields and interpretation |
| --- | --- |
| KGW / Unigram | `favored_mask` is green; false is red. `bias_delta` is the configured additive log-weight bias, applied relatively and normalized stably. Zero-weight tokens remain excluded. |
| MPAC | `favored_mask`, `payload_position`, `payload_symbol`, `bias_delta`. Call false **unfavored**, never red. It can include another color or the unassigned vocabulary remainder. Full color IDs are not exported in this version. |
| SynthID | `layers`, `total_layers`, `captured_layers`. Each layer has `index`, full-distribution `green_mass`, row-aligned binary `g_values`, `input_probabilities`, and output `probabilities`. These are tournament-equivalent reweighting steps, not sampled contestant/winner brackets. |
| Exponential scalar | `selected_token`, wrapped `key_position`, `selection_scores`, `score_kind=NegativeExponentialCost`. Score is `-(-ln(U)/weight)`; maximize it. Exact ties retain the lowest vocabulary ID, matching the existing scalar scan. |
| Exponential tensor | F32 scores with `score_kind=GumbelMax`, `ln(weight)-ln(-ln(U))`. Use the host's existing argmax and tie policy. These are selection scores, not categorical weights. |
| Inverse transform | `score_kind=NegativeRank`, eligible score `-rank`, excluded score negative infinity. `inverse` contains `uniform`, `threshold`, `total_weight`, candidate `ranks`, `cdf_lower`, `cdf_upper`. Scalar also returns `selected_token` and wrapped `key_position`. |

For scalar SynthID, `input_normalizer` is the full-support total immediately
before that layer. The update is `p_out = p_in * (1 + g - green_mass)`, where
`p_in` is already divided by that normalizer. Internal arithmetic is F64; the
final authoritative weight slice is F32. Layers describe their actual internal
stage, which can differ slightly from rounded final weights.

Scalar inverse CDF bounds and threshold are in original weight units. Tensor
inverse CDF bounds use normalized ordered weights; `total_weight` is the last
finite-precision CDF entry and `threshold = uniform * total_weight`. Do not mix
these units. Compact tensor ranks retain original vocabulary permutation rank.
The tensor implementation emits all eligible scores; argmax chooses the first
eligible rank. Scalar retains the existing last-positive-token endpoint fallback.

Keyed snapshots deliberately have **no** output categorical weights,
probabilities or log probabilities. Never softmax these scores or label an
internal acceptance point mass as a model/selection probability.

## Candle: retain the device path

`PreparedWatermark`, `PreparedSampler`, `PreparedIndexedWatermark`,
`PreparedIndexedSampler` and `PreparedIndexedOperation` expose
`apply_traced_trusted(&weights, &options) -> Result<TensorSamplingTrace>` and a
strict `apply_traced` variant. Existing host/device-history preparation works
unchanged. `PreparedIndexedBatch::apply_traced_trusted(&weights, &options_per_row)`
returns independent per-row traces. Batch ordinary output is F32: cast trace
outputs to F32 before stacking if reproducing that aggregate format.

`trace.output()` is the authoritative output tensor; use it in the existing
sampler. It preserves ordinary dtype/skip behavior. `trace.snapshot_trusted(
Some(&row_indices), &view)` projects diagnostics without any value readback.
Indices are U32 **row positions**, not compact vocabulary IDs. Exported
`token_ids` maps them back to actual vocabulary IDs. `None` exports all rows
within the budget. Noncontiguous inputs are supported.

Every returned tensor remains on the sampling device. IDs/ranks are U32,
masks/active flags U8, diagnostic probabilities/scores F32, raw weights retain
their original dtype. Candidate fields are `[M]`; `active`, CDF thresholds and
layer reductions are `[1]`. The caller must not mutate retained tensor storage.
Trusted inputs require valid finite nonnegative weights with positive mass,
valid unique candidate IDs and valid row positions. Shapes/device/dtypes remain
checked. Use trusted methods inside a validated GPU loop: strict application
reads validation status, and strict `snapshot` validates indices with a scalar
readback. No trusted trace method calls `to_vec`, `to_scalar`, CPU transfer or
synchronization. Snapshot arithmetic and gathers incur work only when requested.

Tensor traces retain O(depth) green-mass/normalizer tensors and reconstruct
requested rows using device operations. Each layer additionally exports
`output_normalizer`; its formula is
`p_out = p_in * (1 + g - green_mass) / output_normalizer`.
`output_weights` remains authoritative after dtype rounding. Bias snapshots
include configured `bias_delta` and `effective_bias_delta=min(delta,512)`,
matching the existing tensor implementation.

Device-history preparation keeps the `active` flag on device. When false,
output is unchanged; membership placeholders are zeroed and layers must be
omitted when serializing. This generic prepared API does not distinguish
warmup versus repeated context; use status `skipped`, reason `unavailable`.
Do not infer the reason. It also does not compute MPAC payload slot/symbol or
wrapped key position: the host supplies already-known metadata, or marks it
unavailable. This preserves existing preparation fields and kernel behavior.

Mistral's current compact GPU path selects **on the host after one packed
readback**. A future integration must pack all bounded compact trace rows and
small metadata into that existing export before selection. Gathering selected
IDs on GPU after that host draw would require another readback and is unsuitable.
A future device-side selector can gather there before the one export. Do not
scatter to V, download full vocabulary tensors, or add an export for trace data.
Full GPU HTTP tracing is a follow-up; this API provides its device-resident data.

## Initial HTTP implementation in Mistral

Proceed with the existing scalar logprobs path. Ordinary `return_logprobs`
already uses `Sampler::speculative_probs` and scalar `Watermark::apply`, despite
that method's name. It already brings logits to the host. Require this path for
initial tracing, so trace adds no device readback and requests without tracing
retain their original execution path.

1. Add optional `sampling_trace` to both `/v1/chat/completions` and
   `/v1/completions`, request propagation, response choices and SSE choices.
   Absence must omit response fields (`skip_serializing_if`) and do no capture.
2. Require ordinary generation, one choice, and existing logprobs enabled
   (`true` for chat, the existing numeric option for completions). Reject
   speculative tracing explicitly with HTTP 400. Allow watermark omitted for
   baseline observation. Greedy must identify its selection rule and skipped
   watermark, honoring existing behavior.
3. Branch at the existing watermark call only for capture-enabled steps.
   Retain reporting distribution, original/processed logits if exposed, filtered
   input and final host-normalized categorical weights from their real stages.
   Use traced scalar apply/sample in the scheme adapter, carrying the trace
   beside the sample result. Stop requesting traces when the budget expires.
4. Select rows after the actual draw: selected ID first, then deterministic
   bounded union of leading input/output probabilities; tie by actual ID.
   Keyed rules use leading input candidates and typed scores, with no fabricated
   post probability. A documented narrower bounded input policy is acceptable.
   Call `snapshot` once with those IDs; never run sampling again.
5. Grammar can reject a first draw and retry with masked logits. Keep each trace
   with its attempt result; commit only the final accepted attempt. Avoid
   sampler-global mutable observation state across sequences or retries.
6. Emit only newly committed steps in ordinary SSE chunk framing. Use explicit
   `generated_index`; decoded text can lag behind tokens due to UTF-8, stops,
   reasoning or tool parsing. Trace-only empty-delta chunks are valid. Flush
   pending trace before terminal chunk and `[DONE]`, emit each step once, and
   include the final truncation summary. Retain bounded history only.

Suggested chat extension:

```json
{
  "logprobs": true,
  "top_logprobs": 10,
  "sampling_trace": {"max_steps": 32, "max_candidates": 32, "max_layers": 8}
}
```

Use defaults 32/32/8; maxima 256 steps, 128 candidates, 32 captured layers.
Require positive steps/candidates, allow zero layers, and enforce
`steps * candidates * max(1,layers) <= 65536` with checked arithmetic. Reject
invalid bounds with 400. Enforce an 8 MiB serialized trace budget per request;
stop observation, not generation, with an explicit truncation reason. Include
framing/final-summary allowance so the bound is real. Library view limits do
not replace request-wide element and byte limits.

Put nonstreaming trace at `choices[i].sampling_trace`, containing `steps`,
`truncated`, and `truncation_reason`. Each step should contain:

- `generated_index`, `context_length`, `attempt`, `selected_token_id`, and
  `selection_rule` (`categorical`, `greedy`, or the exact keyed rule).
- Full effective `candidate_count`, `candidates_truncated`, configured watermark
  `scheme` and accurate `status`; baseline uses `watermark: null`.
- `candidates`: actual `token_id`, decoded token piece `text`,
  `reporting_probability`, `pre_watermark_probability`, optional
  `post_watermark_probability`/canonical log probability, and membership.
  For keyed rules, use typed score/CDF fields instead of a post probability.
- `layers`: index, green mass and candidate-ID-aligned g/input/output fields;
  `layers_truncated` distinguishes capture limits from algorithm depth.

Reporting logprobs remain after penalties/processors/temperature and before
filtering/watermarking. Pre-watermark probability is the normalized filtered
input. Post-watermark probability is the actual normalized distribution passed
to the host categorical draw. Reuse that host stage if its normalization differs
in floating-point rounding from library diagnostic normalization (scalar F64,
tensor F32); never renormalize just the displayed candidates. Baseline pre/post
values match. Omitted candidates are not necessarily zero mass.

If requested, expose `raw_logit`/`processed_logit` only from retained host logits
and name their stage. The library's `ln(p)` is a different field. JSON must not
contain NaN or infinity: encode zero-mass log probabilities or excluded score
sentinels as `null` with a documented status. Do not serialize keys or prompt/
context text. MPAC initial membership is favored/unfavored with slot and symbol;
full color and unassigned-remainder labeling remains unavailable.

## Consumer files and acceptance checks

Update core `sampler.rs`, `watermark/schemes.rs`, `request.rs`, `sequence.rs`,
`pipeline/sampling.rs`, and `response.rs`; gate speculative requests at the
appropriate verifier/driver boundary. Server changes belong in `openai.rs`,
`chat_completion.rs`, `completions.rs` and `openapi.rs`. Update
`docs/openapi.json`, `examples/server/watermarking.py`, and the watermarking
user guide. The GPU adapter `sampler/watermark.rs` should retain its ordinary
path in this first integration.

Verify absent/present requests, baseline and all six schemes, seed replay,
unchanged existing logprobs, selected-token inclusion, full-distribution
normalization, grammar retry ownership, SSE token alignment/one-time delivery,
bounds/truncation and rejected unsupported combinations. Confirm no trace field
or extra readback for ordinary requests.

Library validation covers exact traced/untraced outputs and keyed selections,
capture/projection limits, full-support probabilities, SynthID layer updates,
warmup/repeat skips, wrapped key positions, MPAC favored mask/remainder count,
zero mass, K=1, arbitrary compact IDs, noncontiguous storage, independent batches,
F32/F16/BF16 and dense/indexed layouts. Validation passed on 2026-09-26:

- `cargo test --no-default-features`
- `cargo test --features candle` (including documentation examples)
- `cargo test --features metal -- --ignored` on the real Metal GPU, including
  existing dense/indexed/metadata tests and the new trace parity test
- `cargo clippy --features metal --all-targets -- -D warnings`
- `cargo fmt --all -- --check` and `cargo run --example trace`

CUDA hardware/toolchain validation is pending on this Mac. No CUDA-specific
kernels were added or changed by the trace implementation.
