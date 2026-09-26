# Candle tensor boundary

The host supplies its existing Candle tensor and receives a new tensor on the
same device. This crate provides watermark transformations and keyed metadata;
Candle owns allocation and CPU/CUDA/Metal execution. The host retains inference,
filters, batching, scheduling, token selection, and public service interfaces.

## Dependency identity and features

`candle-core` is optional and uses the same source, version, and revision as the
inspected mistral-rs workspace:

```toml
candle-core = { git = "https://github.com/huggingface/candle.git", rev = "66a8cf184a5a519671454066b1b9efd446ec9f5c", version = "0.11.0", default-features = false }
```

Features are `candle`, `cuda = ["candle", "candle-core/cuda"]`, and
`metal = ["candle", "candle-core/metal"]`. There is no separate GPU runtime,
allocator, inference framework, or mistral-rs dependency in this crate. Its
`candle_core` re-export exposes the exact dependency's types. The pinned dependency
tree requires Rust 1.88, including `zip` and the CUDA loader's requirements.

A registry `candle-core = "0.11"`, another Git URL/revision selector, or a path copy
can resolve to another package and produce incompatible `Tensor` types. Coordinate
pin updates in both projects. Use `cargo tree -d` and `cargo metadata` in the final
consumer to verify that only the intended Candle package is used. A downstream
workspace's patch must apply consistently to all Candle consumers.

## Probability transformations

```rust
use llm_watermarking::{
    candle_core::{Device, Tensor},
    synthid::{SynthIdConfig, SynthIdText},
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let watermark = SynthIdText::new(&SynthIdConfig::new([42; 32]))?;
let device = Device::Cpu; // In a host, use its existing tensor.device().
let probabilities = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &device)?;
let marked = watermark.apply_tensor(&probabilities, &[1, 2, 3, 4], 4)?;
assert!(marked.device().same_device(probabilities.device()));
// Return `marked` to the host's categorical sampler, without downloading it.
# Ok(())
# }
```

These operations accept a dense one-dimensional `[vocab_size]` tensor. Index `i`
must be token ID `i`; filtering masks tokens with zero weight. Compact top-k
vectors need the host to restore vocabulary indexing before calling this API.
Weights may be unnormalized but must be finite, nonnegative, and have positive
mass. F16, BF16, and F32 are supported; computation uses F32, and probability
transforms return the input dtype and shape. F64 and integer probabilities are
rejected consistently, including on CPU.

| Scheme | One-call API | Reusable preparation |
| --- | --- | --- |
| SynthID | `apply_tensor(weights, context, prompt_len)` | `prepare_tensor(vocab_size, context, prompt_len, device)` |
| KGW | `apply_tensor(weights, context, prompt_len)` | `prepare_tensor(context, prompt_len, device)` |
| Unigram | `apply_tensor(weights)` | `prepare_tensor(device)` |
| MPAC | `apply_tensor(weights, context, prompt_len, payload)` | `prepare_tensor(context, prompt_len, payload, device)` |

Preparation returns `PreparedWatermark`. Its `apply` validates values, then runs
the transformation. `apply_trusted` performs shape, dtype, and device checks but
trusts the host's value contract. It never reads probability values back to CPU:

```rust
use llm_watermarking::{
    candle_core::{Device, Tensor},
    unigram::{Unigram, UnigramConfig},
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let device = Device::Cpu;
let watermark = Unigram::new(&UnigramConfig::new([42; 32], 4))?;
let prepared = watermark.prepare_tensor(&device)?; // Reuse for every Unigram step.
let filtered = Tensor::new(&[0.0f32, 0.3, 0.2, 0.5], &device)?;
let marked = prepared.apply_trusted(&filtered)?;
assert!(marked.device().same_device(&device));
# Ok(())
# }
```

Strict validation reduces validity to one scalar and reads it back, synchronizing
the GPU. `validate_probabilities` exposes the same check for hosts that validate
once separately. Trusted calls on invalid values can return NaNs; this is a
numerical precondition, not a memory-safety escape hatch.

Prepared operations are immutable and cloneable. Reusing them is valid only for
the same key/context/payload/position represented at preparation. Use new
preparation for a changed context or sequence position. A prepared operation
rejects an input on another device; it never silently migrates it. Warmup and
repeated SynthID contexts return the original tensor unchanged after validation.
Input storage is never mutated, including on errors or speculative retries.

The host owns batching. It can take device-resident rows from a batch, apply the
corresponding per-sequence operation, and stack the results with Candle. Supplying
a whole matrix to a token transformation is an error rather than implicitly
applying one sequence's context to the entire batch.

## Keyed sampling and host-owned selection

Exponential-race and inverse-transform are sampling watermarks, not probability
biases. Their `prepare_tensor(position, device)` returns `PreparedSampler`, whose
`apply` and `apply_trusted` return F32 **selection scores** on the input device.
`selection_scores_tensor(weights, position)` combines preparation and strict
application. The host completes the algorithm with `argmax`:

```rust
use llm_watermarking::{
    candle_core::{Device, Tensor},
    exponential::{ExponentialRace, ExponentialRaceConfig},
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let watermark = ExponentialRace::new(&ExponentialRaceConfig::new([42; 32], 4))?;
let weights = Tensor::new(&[0.0f32, 0.3, 0.2, 0.5], &Device::Cpu)?;
let scores = watermark.selection_scores_tensor(&weights, 17)?;
let token_on_device = scores.argmax(0)?; // Owned by the host, not the library.
assert!(token_on_device.device().same_device(weights.device()));
# Ok(())
# }
```

Do not apply softmax, another categorical draw, or further filters to those scores.
Exponential scores are `ln(weight) - ln(-ln(U))`. Inverse-transform scores favor
the earliest occupied interval above the keyed CDF target in permuted order.
Both keep zero-weight tokens at negative infinity. The original filtered
probabilities remain available to the host for reporting or other inference work.
Inverse-transform tensor vocabularies are limited to `2^24` entries so every
integer rank has a distinct F32 score (`tensor::MAX_INVERSE_TENSOR_VOCAB`).
Sequence position, period wrapping, and randomness assumptions match the scalar
API. The library does not advance position or mutate sequence state.

The pinned Candle `cumsum` constructs a square matrix. Inverse transform instead
uses a doubling prefix scan made from ordinary Candle operations, taking
`O(V log V)` arithmetic and `O(V)` temporary storage rather than `O(V²)` storage.

## SemStamp embeddings

`SemStamp::prepare_tensor(device)` uploads reusable hyperplanes and returns
`PreparedSemStamp`. It accepts `[embedding_dim]` or `[candidates, embedding_dim]`
F16/BF16/F32 tensors. Projection and cosine margins run in F32 on the input device.

- `signatures(embeddings)` returns U32 signatures, scalar or one per row.
- `accepts(previous_signature, candidates)` returns a U8 acceptance mask with
  the same scalar/row structure, combining region membership and cosine margins.
- `signatures_trusted` and `accepts_trusted` avoid validity readback when the host
  guarantees finite, nonzero embeddings. Shape and device checks still apply.
- `detect(embeddings, prompt_len)` accepts `[sentences, embedding_dim]`, downloads
  only one U32 signature per sentence, and counts transitions with the existing
  detector. `SemStamp::detect_tensor` is the one-call equivalent.

`SemStamp::acceptance_tensor(previous_signature, candidates)` offers the one-call
acceptance path. The previous signature is host-owned context from the last
committed sentence. A tensor-only host can obtain it by reading the single
signature of its accepted sentence. The host supplies its own sentence encoder
and candidate generation, and commits only the selected candidate after handling
acceptance or an explicit retry-limit fallback.

Token detectors continue to accept host token IDs; they need no probability
tensor or model inference. No watermark detector downloads full model weights.

## What remains on CPU

Key/context hashing, exact Fisher-Yates permutations, payload allocation, and
SemStamp hyperplane generation use the existing CPU implementation. Preparation
uploads only deterministic watermark metadata; transformations never download
probability vectors. SynthID uploads packed g-value bytes and unpacks their bits
with tensor arithmetic. This preserves the versioned hash formats without adding
custom CUDA/Metal kernels or a second GPU stack.

Preparation has CPU and upload costs proportional to the relevant metadata.
Unigram masks and SemStamp hyperplanes can be reused across steps; other prepared
operations can be reused for identical contexts or positions and speculative
retries. This implementation does not claim GPU-resident keyed hashing or a
fully asynchronous one-call path. A future hash optimization can use Candle's
`CustomOp1` backend hooks without changing the host-facing boundary.

F32 accumulation, half-precision casts, GPU math, and parallel reductions can
differ from the scalar f64 reference near ties, CDF cutoffs, or semantic region
boundaries. Keys and partitions match exactly; floating-point outputs need not.
Extreme biases and dynamic ranges can underflow additional probabilities.

## Mistral-rs insertion point

Package compatibility alone does not install a hook in mistral-rs. The host must
expose its **filtered probability tensor before drawing a token**. The current
`CustomLogitsProcessor` is earlier, before temperature, softmax, and probability
filters, so passing logits to these probability APIs is incorrect.

For the probability-transform schemes, host execution should be:

```text
model logits -> penalties/processors -> temperature/softmax -> probability filters
             -> apply_tensor / prepared.apply_trusted -> host categorical selection
```

For exponential-race or inverse-transform, replace the final transformation and
selection with `selection_scores_tensor -> host argmax`. For fused samplers that
combine filtering and drawing, split at this boundary or provide an equivalent
host-owned path when a watermark is enabled. Preserve per-sequence context,
prompt lengths, original vocabulary indexing, reporting probabilities, and any
host policy for greedy decoding. Ordinary and speculative generation must use
consistent watermarked distributions and sequence positions. This crate does
not modify mistral-rs sampling plans or own that integration code.

## Verification

```bash
cargo test --features candle
cargo check --features metal --all-targets
cargo test --features metal --test candle_api candle_metal_parity -- --ignored
cargo check --features cuda --all-targets
cargo test --features cuda --test candle_api candle_cuda_parity -- --ignored
cargo run --features metal --example candle -- metal
```

GPU tests are explicitly ignored by default because they require real hardware.
When selected they require the requested device and never fall back to CPU. They
compare all schemes with scalar references, verify device/dtype preservation,
exercise strided inputs and errors, and test a realistic inverse-transform
vocabulary size. Metal requires a supported macOS GPU; CUDA requires the Candle
toolchain's CUDA dependencies and a compatible GPU.
