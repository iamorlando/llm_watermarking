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

Features are `candle`, `cuda` (forwarding `candle-core/cuda`), and `metal`
(forwarding `candle-core/metal` and enabling Candle's Metal kernel wrappers).
The `candle-metal-kernels` package uses the same Git revision as `candle-core`.
There is no separate GPU runtime,
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
vectors use the [indexed candidate APIs](mistral-integration.md), which retain
actual vocabulary IDs without expanding the probability row.
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

For dense APIs the host can take device-resident rows from a batch, apply the
corresponding per-sequence operation, and stack the results with Candle. Indexed
APIs also provide `PreparedIndexedBatch` for independent row preparations. Supplying
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

## Device metadata generation

The tensor preparation paths use Candle `CustomOp1`/`CustomOp2` backend hooks
for the missing keyed primitives. CPU uses the existing Rust SHA-256 and Fisher–Yates implementation;
CUDA and Metal use a shared watermark-specific SHA-256 kernel and exact shuffle.
Candle owns the allocations, CUDA stream, Metal encoders and resource dependencies.
No GPU probability or embedding vector is read back for metadata preparation.

| Scheme | Preparation on the target device | Reuse |
| --- | --- | --- |
| SynthID | Parallel candidate SHA-256, producing packed g-value bytes | Same context |
| KGW | Parallel SHA-256 random stream, exact permutation, green mask | Same context |
| Unigram | Exact permutation and fixed mask | Automatically cached per device |
| MPAC | Exact permutation and payload group mask | Same context and payload |
| Exponential race | Parallel candidate SHA-256 and F32 Gumbel values | Same wrapped sequence position |
| Inverse transform | Exact permutation and inverse ranks | Automatically cached per device, across positions |
| SemStamp | Exact previous-region permutation and membership mask | Hyperplanes reused through `PreparedSemStamp` |

The fixed key/domain prefix is cached on each Candle device. Changing contexts or
positions upload only their small seed suffix, not a vocabulary-sized mask or hash
table. Clones share these caches. There is no growing cache of per-token contexts.
Unigram and inverse-transform preparations reuse their device tensors automatically.
Caller-retained prepared operations also support retries of identical steps.

The kernels preserve the versioned hash inputs, byte order, unbiased integer
rejection sampling and permutation order used by the scalar detectors. Random
stream hashes and bounded choices run in parallel. The GPU reconstructs the exact
Fisher–Yates result using per-target lists and increasing dependency chains,
without executing all swaps serially. Atomic insertion order cannot change the
result: each dependency selects the smallest eligible higher swap index. A rare
rejected integer draw triggers an on-device serial repair of stream offsets;
there is no CPU fallback or readback. Scratch storage is O(V); work depends on
keyed list lengths and chain depths. No new partition format or detector is needed.
Device residency alone does not guarantee a speedup; measure the target workload.

Metal libraries/pipelines are compiled lazily and cached for each Candle device.
CUDA compiles the small kernel source through Candle's re-exported NVRTC, caches
the PTX, and loads it into the existing Candle CUDA device. CUDA therefore requires
NVRTC at runtime in addition to Candle's build dependencies. Warm preparation
before latency-sensitive generation to exclude first-use compilation costs.

In the host-metadata APIs, token-history validation, SynthID repeat checks, the
single MPAC payload-slot choice and inverse-transform uniform remain CPU
bookkeeping. `prepare_indexed_device` moves this changing metadata to the GPU;
`DeviceHistory` supplies resident prefixes, lengths and prompt lengths. See the
[indexed integration contract](mistral-integration.md) for trusted validation
preconditions, per-row positions, retry semantics, and the remaining O(V)
partition work. SemStamp's reusable hyperplanes are generated once on CPU and uploaded during preparation. Scalar
Unigram/inverse constructors also retain CPU partitions for their scalar APIs and
detectors. None of these require per-step vocabulary-sized CPU uploads. Strict
value validation still reads one status scalar; trusted applications avoid it.

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
selection with `selection_scores_tensor -> host argmax`. Fused samplers that
combine filtering, drawing or speculative acceptance must remain excluded until
the host exposes this boundary and uses consistent watermarked distributions.
The indexed/device APIs alone do not make removing those exclusions correct.
Preserve per-sequence context,
prompt lengths, original vocabulary indexing, reporting probabilities, and any
host policy for greedy decoding. Ordinary and speculative generation must use
consistent watermarked distributions and sequence positions. This crate does
not modify mistral-rs sampling plans or own that integration code.

## Verification

```bash
cargo test --features candle
cargo check --features metal --all-targets
cargo test --features metal -- --ignored
cargo check --features cuda --all-targets
cargo test --features cuda -- --ignored
cargo run --features metal --example candle -- metal
```

GPU tests are explicitly ignored by default because they require real hardware.
When selected they require the requested device and never fall back to CPU. They
compare all schemes with scalar references, verify device/dtype preservation,
exercise strided inputs and errors, and test a realistic inverse-transform
vocabulary size. Metal requires a supported macOS GPU; CUDA requires the Candle
toolchain's CUDA dependencies and a compatible GPU.

The ignored GPU tests also compare raw SHA-256 bytes and exact permutations across
padding boundaries, offset seed tensors, changing contexts/payloads, period
wrapping, distinct device instances (Metal), and a 32,769-token vocabulary.
A forced-rejection Metal test exhausts the prefetched stream and verifies exact
continuation. Floating-point sampling values are compared within tolerance.

Measure warm preparation separately from inference with:

```bash
cargo run --release --features metal --example metadata_bench -- metal 32000 20
cargo run --release --features cuda --example metadata_bench -- cuda 128000 20
```

This example compares CPU and the requested GPU, synchronizes after each step,
and excludes first-use compilation/fixed preparation. It reports metadata latency,
not end-to-end generation throughput. Benchmark the final host sampling path too.
