# Mistral integration follow-up

This is the library-side response to Mistral's `watermarking-gpu.md` feedback
(reviewed 2026-09-26). The Candle package identity and existing dense APIs are
unchanged. Mistral can replace its scatter/transform/gather workaround with the
indexed APIs below. Updating the Mistral adapter remains consumer work.

## Compact candidates

`tensor::IndexedCandidates` binds a device U32 `[K]` token-ID tensor to the original
vocabulary size. IDs are unique and in range; their order is arbitrary. Filtered
weights have shape `[K]` in that same order. A zero weight excludes a candidate.
There is no vocabulary-sized probability allocation or probability readback.

```rust
use llm_watermarking::{
    candle_core::{Device, Tensor},
    kgw::{Kgw, KgwConfig},
    tensor::IndexedCandidates,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let device = Device::Cpu; // Use the host's existing CUDA/Metal device.
let watermark = Kgw::new(&KgwConfig::new([42; 32], 32_000))?;
let ids = Tensor::new(&[31_999u32, 17, 400], &device)?;
let filtered_weights = Tensor::new(&[0.3f32, 0.0, 0.7], &device)?;
let candidates = IndexedCandidates::new_trusted(&ids, 32_000)?;
let prepared = watermark.prepare_indexed(&candidates, &[10, 11], 2)?;
let marked = prepared.apply_trusted(&filtered_weights)?;
assert_eq!(marked.dims(), &[3]);
// Continue the existing host categorical draw, then map its index through ids.
# Ok(())
# }
```

For Mistral's current top-k adapter, replace:

```text
zeros(V).scatter_add(ids, sampling)
    -> dense watermark application
    -> index_select(ids)
```

with:

```text
IndexedCandidates::new_trusted(ids, V)
    -> scheme.prepare_indexed(candidates, ...)
    -> prepared.apply_trusted(sampling)
```

| Scheme | Host metadata preparation | Device metadata preparation |
| --- | --- | --- |
| SynthID | `prepare_indexed(candidates, context, prompt_len)` | `prepare_indexed_device(candidates, history)` |
| KGW | `prepare_indexed(candidates, context, prompt_len)` | `prepare_indexed_device(candidates, history)` |
| Unigram | `prepare_indexed(candidates)` | Same; no changing context |
| MPAC | `prepare_indexed(candidates, context, prompt_len, payload)` | `prepare_indexed_device(candidates, history, payload)` |
| Exponential race | `prepare_indexed(candidates, position)` | `prepare_indexed_device(candidates, position_tensor)` |
| Inverse transform | `prepare_indexed(candidates, position)` | `prepare_indexed_device(candidates, position_tensor)` |

The first four return `PreparedIndexedWatermark`: categorical weights in the
input dtype. The last two return `PreparedIndexedSampler`: F32 scores for
**argmax**. Argmax returns a candidate index, which the host maps to its actual
vocabulary token ID. Neither API performs selection or changes the reporting
probabilities. Keep Mistral's original reporting probabilities and existing
compact readback alongside the returned values and token IDs.

Existing `PreparedWatermark` and `PreparedSampler` also provide `.indexed(&candidates)`
for adapting already-prepared dense metadata. Prefer the scheme-level methods:
SynthID and exponential race then hash only K actual token IDs. The full
vocabulary definition remains part of each scheme's original seed/partition
contract; candidate ranks never replace vocabulary IDs in hashes.

KGW and MPAC still reconstruct the exact full-vocabulary Fisher–Yates permutation
on each changed context. Unigram and inverse transform cache their full-vocabulary
metadata per device. This O(V) partition work is required by the current format;
the indexed APIs remove dense **probability** work, not all vocabulary-sized
metadata. Inverse transform sorts candidate ranks and runs its CDF scan on K
entries. For K above 1,024, sorting uses device-wide bitonic passes, avoiding the
pinned Candle Metal argsort's single-threadgroup limit. That path uses O(K) storage
and O(K log² K) work. There is no blanket throughput claim.

## Device history and per-sequence state

`DeviceHistory::new_trusted(tokens, length, prompt_len, vocab_size)` accepts U32
`tokens[capacity]`, `length[1]`, and `prompt_len[1]` on one device. Length is the
number of committed tokens in the complete prompt-plus-generation prefix;
padding after length is ignored. It requires `prompt_len <= length <= capacity`
and all committed token IDs to be in range. For a padded batch, use a row view of
the history and a `[1]` narrow of each length tensor.

`prepare_indexed_device` constructs the exact seed bytes on device. SynthID's
warmup and last-1,024-generation-position repeat checks run there too, with the
same prompt-only exclusion as the scalar implementation. KGW and MPAC preserve
warmup identity; MPAC chooses its payload slot on device. MPAC payloads remain
small host-supplied configuration, independently supplied for each row.

Exponential and inverse-transform positions are U32 `[1]` tensors on the candidate
device. The library wraps each position to its configured period on device.
Positions are independent of prompt length. For absolute counters above U32,
the host must supply the modulo-period position, not truncate the absolute count.
Inverse-transform uniform generation stays on device. No committed tokens,
positions, repeat flags, payload-slot choices, or uniforms are read back.

The library uses Candle's allocation, stream ordering, and resource tracking.
Nonzero offsets and strided input views are supported; any needed copies stay
on the device. Keys and fixed prefixes are uploaded/cached per device. Device
history preparation currently assembles a contiguous history input on device;
it is not a fused inference or KV-cache operation.

Preparations capture one candidate ordering and one key/context/position/payload.
Reuse them for retries of that exact step. Prepare again when any of these change.
The host must keep input storage immutable while it is in use, including when
using Candle's in-place operations. The library never advances sequence state.

## Independent batches

`PreparedIndexedBatch::new(Vec<PreparedIndexedOperation>)` collects independent
row preparations. Each entry is `Probabilities(prepared)` or
`SelectionScores(prepared)`. Rows can have different keys, schemes, contexts,
positions, payloads, and vocabularies; they must share device and K. Input and
output are `[B,K]`; batch output is F32. Use each row's selection rule, available
through `rows()[i].is_keyed_sampler()`.

The batch helper submits the existing row operations on the device. It does not
fuse their kernels or assign one row's metadata to another. Reorder preparations
with scheduler rows. Retain or rebuild a branch's preparation from its own prefix
for speculative retries. Ragged K values can use separate row calls. Do not pad
with duplicate token IDs; zero-weight entries still require distinct in-range IDs.

## Validation and remaining host boundaries

`IndexedCandidates::new` checks range and uniqueness with a scalar readback;
`DeviceHistory::new` checks committed IDs and lengths with a scalar readback.
Their `new_trusted` versions check shape, dtype, and device only. Indexed `apply`
validates candidate IDs and probabilities and synchronizes for those statuses.
Use the trusted constructors and `apply_trusted` only when the host guarantees
the documented value contracts. The entire trusted preparation/application path
performs no validation readback. Invalid trusted values have unspecified numerical
results, with device indices/lengths bounded against out-of-range memory access.

The fused CUDA batch/resident and sparse speculative samplers still need an
explicit host insertion point between filtering and selection/acceptance. These
library APIs do not make removing their exclusions correct. Acceptance must use
the appropriate watermarked distributions and consistent branch positions. Keep
those exclusions until Mistral exposes composable stages; a second inference
sampler is unnecessary.

SemStamp remains an embedding/sentence-retry workflow. Its existing GPU operations
are available, but it cannot be treated as a token probability transform. Ordinary
token-generation requests should continue to reject it unless the host adds a
sentence encoder and candidate-sentence retry hook.

## Verification and consumer handoff

Local verification for this change passed the scalar/default and Candle CPU
suites, all Metal hardware suites (including large-candidate sampling), Clippy
with Metal enabled, formatting, and the CPU example. CUDA build verification
could not run on this Mac because `nvcc` is unavailable; validate on CUDA hardware
before enabling that consumer path.

`tests/indexed_api.rs` compares all token schemes against scalar references with
unsorted actual IDs, filtered zeros, single-candidate rows, F32/F16/BF16, offset
and strided views, warmup, repeat-window boundaries, positions, independent batch
rows, row reordering, and replay. Metadata fixtures check exact hash bytes, large
integer token IDs, payload slots, seed bytes and large candidate sorting. Run:

```bash
cargo test --features candle
cargo test --features metal -- --ignored
cargo test --features cuda -- --ignored
cargo run --features candle --example indexed
cargo run --features metal --example indexed -- metal
```

The CUDA commands require a CUDA toolkit/NVRTC and GPU. Floating-point comparisons
use tolerances; partitions and hash bytes must match exactly. After migrating the
adapter, rerun Mistral's watermark tests, including the real top-k entry point, on
its target backend. Benchmark the intended model, V, K, batch sizes, and scheme;
these synthetic tests do not establish generation quality or performance.
