# llm-watermarking

A model-independent Rust library with seven watermarking schemes and their
detectors. Token schemes operate on probability slices and token IDs. SemStamp
operates on caller-supplied sentence embeddings. The host owns tokenization,
model inference, sampling filters, sentence segmentation, embedding models,
serialization, and language or HTTP bindings.

The only runtime dependency is `sha2`. Algorithm errors implement
`std::error::Error`; no tensor framework or inference engine is required.

| Scheme | Generation | Detection | Example |
| --- | --- | --- | --- |
| [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) | Two-candidate tournaments | Mean binary g-value | [synthid](examples/synthid.rs) |
| [KGW / soft green-list](https://arxiv.org/abs/2301.10226) | Context-keyed partition and additive logit bias | Favored-token count and nominal z-score | [kgw](examples/kgw.rs) |
| [Unigram](https://arxiv.org/abs/2306.17439) | Fixed keyed partition and additive logit bias | Favored-token count and nominal z-score | [unigram](examples/unigram.rs) |
| [Exponential race / Gumbel](https://arxiv.org/abs/2307.15593) | Direct keyed exponential-minimum sampling | Keyed exponential cost, optional edit alignment | [exponential](examples/exponential.rs) |
| [Inverse transform](https://arxiv.org/abs/2307.15593) | Direct CDF sampling in a keyed token order | Token-rank distance, optional edit alignment | [inverse_transform](examples/inverse_transform.rs) |
| [MPAC](https://aclanthology.org/2024.naacl-long.224/) | Context-keyed message position and vocabulary colorlists | Payload recovery or known-payload scoring | [mpac](examples/mpac.rs) |
| [SemStamp](https://aclanthology.org/2024.naacl-long.226/) | Sentence rejection using embedding regions and margins | Valid-region sentence count and nominal z-score | [semstamp](examples/semstamp.rs) |

These are implementations of the published constructions using this crate's
versioned SHA-256 formats, not byte-compatible ports of the authors' PRNGs or
provider configurations. See [format and implementation choices](docs/formats.md).

## Use

The package is named `llm-watermarking` and its Rust import is `llm_watermarking`.
Until it is published or hosted remotely, add it as a local dependency:

```toml
[dependencies]
llm-watermarking = { path = "../llm_watermarking", version = "0.1.0" }
```

```rust
use llm_watermarking::synthid::{SynthIdConfig, SynthIdText};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Example only: supply securely generated key bytes in production.
let config = SynthIdConfig::new([0x42; 32]);
let watermark = SynthIdText::new(&config)?;

let context = [10, 11, 12, 13];
let prompt_len = context.len();
let mut probabilities = [0.1, 0.2, 0.3, 0.4];
watermark.apply(&mut probabilities, &context, prompt_len)?;
// Draw a token with the host's categorical sampler using these weights.

let tokens = [10, 11, 12, 13, 2];
let evidence = watermark.detect(&tokens, prompt_len, &[])?;
assert_eq!(evidence.tokens_scored, 1);
# Ok(())
# }
```

`SynthIdConfig` accepts a caller-owned `[u8; 32]` key and defaults to `ngram_len=5`
and `depth=30`. Change its public `ngram_len` or `depth` fields before constructing
`SynthIdText`. Supported ranges are 2 through 32 and 1 through 256 respectively.
There is no default key, automatic key storage, or model-specific configuration.
Debug formatting redacts the key.

## SynthID probability and history contract

`apply` takes a dense vocabulary slice: index `i` is token ID `i`. Supply finite,
nonnegative weights with positive total mass, after temperature and all desired
token filters. Weights need not sum to one. An active tournament normalizes the
result; skipped contexts leave the input weights unchanged. Zero-weight tokens
remain excluded. The host can normalize again before drawing its next token.

`context` contains the complete prompt and generated token prefix; `prompt_len`
marks where generation began. Incomplete contexts and contexts seen in the last
1,024 generation positions are skipped. Prompt-only occurrences are not counted.
The algorithm derives history from the supplied prefix and stores no mutable
sequence state. Clones and speculative branch retries are independent.

`apply` rejects empty, all-zero, negative, NaN, or infinite weights and a prompt
length beyond the context. Errors leave the probability slice unchanged.

## SynthID detection

`detect(tokens, prompt_len, eos_token_ids)` returns `WatermarkDetection`, with a
scored-token count and optional mean g-value. It excludes prompt tokens, repeated
contexts, and the first generated EOS and everything after it. Empty or short
samples have no score. With completion-only IDs, pass zero for `prompt_len`;
the first `ngram_len - 1` IDs supply context and are not scored.

Use the same tokenizer, key, parameters, and hash domain as generation. Prefer
original generated IDs, since detokenization and re-encoding can change IDs.
Detection excludes every repeated context, even beyond the generation window.

This is an uncalibrated evidence score, not a probability of AI authorship.
Independent unmarked text has an expected score near 0.5. Establish thresholds
and false-positive rates using representative held-out data at relevant lengths.
Greedy output, low-entropy distributions, short text, editing, and retokenization
can weaken the signal. Possessing the key permits producing the signal; the
watermark is not an authorship signature.

## SynthID algorithm and format

The algorithm follows the published [SynthID-Text tournament method](https://www.nature.com/articles/s41586-024-08025-4).
For each layer, let `g` be its keyed binary token scores and `G = sum(p * g)`.
The exact two-candidate tournament distribution is `p_next = p * (1 + g - G)`.
Repeating this transformation avoids drawing `2^depth` candidate tokens.

The default g-function is:

```text
digest = SHA256(
    b"llm-watermarking-synthid-text-v1\0" || key_bytes || LE32(ngram_len) ||
    LE32(context_token_1) || ... || LE32(context_token_(ngram_len - 1)) ||
    LE32(candidate_token)
)
g[layer] = (digest[layer / 8] >> (layer % 8)) & 1
```

`SynthIdText::with_domain(&config, domain_bytes)` allows applications to select an
explicit domain separator, including an already-deployed format. The bytes are
hashed exactly as provided, including any terminating zero byte. Generation and
detection must use the same domain. The default is exported as `synthid::HASH_DOMAIN`.

This keyed-hash implementation does not reproduce private provider configurations
or the public Transformers sampling-table format. Computation runs on CPU, with
tournament work proportional to the number of surviving tokens times depth.

## KGW and Unigram

```rust
use llm_watermarking::{kgw::{Kgw, KgwConfig}, unigram::{Unigram, UnigramConfig}};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut config = KgwConfig::new([0x42; 32], 128);
config.context_width = 2;
config.green_fraction = 0.25;
config.delta = 2.0;
let kgw = Kgw::new(&config)?;
let mut weights = [1.0; 128];
kgw.apply(&mut weights, &[10, 11], 2)?;
// Draw the next token with the host's categorical sampler.
let evidence = kgw.detect(&[10, 11, 12], 2, &[])?;
assert_eq!(evidence.trials, 1);

let unigram = Unigram::new(&UnigramConfig::new([0x42; 32], 128))?;
let mut weights = [1.0; 128];
unigram.apply(&mut weights)?; // No context: the partition is fixed.
let evidence = unigram.detect(&[10, 11, 12], 0, &[])?;
assert_eq!(evidence.trials, 3);
# Ok(())
# }
```

Both configurations default to `green_fraction=0.5` and `delta=2.0`; KGW defaults
to one preceding token. The partition contains exactly
`floor(green_fraction * vocab_size)` tokens. Detection uses that realized fraction,
which can differ from the requested fraction for small vocabularies.

Reweighting multiplies favored probabilities by `exp(delta)` and normalizes,
implemented with a common scaling factor to avoid overflow. Increasing delta
strengthens the bias and can affect text quality. Zero delta just normalizes.
KGW leaves weights unchanged during its context warmup. Its detector skips
duplicate `(context, token)` n-grams by default; set
`ignore_repeated_ngrams=false` to count all occurrences. Unigram counts all
occurrences by default, with optional `ignore_repeated_tokens=true` distinct-token
scoring. Repeated identities are correlated for Unigram, so its nominal z-score
must not be treated as a calibrated false-positive probability.

## Exponential race and inverse transform

```rust
use llm_watermarking::{
    exponential::{ExponentialRace, ExponentialRaceConfig},
    inverse_transform::InverseTransform,
    sampling::AlignmentConfig,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut config = ExponentialRaceConfig::new([0x42; 32], 128);
config.sequence_len = 128;
let exponential = ExponentialRace::new(&config)?;
let inverse = InverseTransform::new(&config)?; // Both use SamplingConfig.
let weights = [1.0; 128];
let tokens: Vec<_> = (0..64).map(|i| exponential.sample(&weights, i))
    .collect::<Result<_, _>>()?;
let evidence = exponential.detect(&tokens, 0, &[], 0)?;
assert_eq!(evidence.tokens_scored, 64);
let _inverse_token = inverse.sample(&weights, 0)?;

let mut alignment = AlignmentConfig::new(32);
alignment.edit_penalty = Some(0.2);
let aligned = exponential.detect_aligned(&tokens, 0, &[], &alignment)?;
assert_eq!(aligned.tokens_scored, 32);
# Ok(())
# }
```

These methods **select the token themselves**. Do not apply another categorical
sampler afterward. Supply the model's final filtered weights at each step.
`sample(weights, position)` is stateless and deterministic. Position is an index
in the watermark key sequence, independent of the prompt, wrapping at
`sequence_len` (default 1,024; supported range 1 through 65,536). A host can choose
a random initial offset; `detect` takes that same offset for the first generated
token. `detect_aligned` searches all offsets when it is unknown.

Exponential sampling minimizes `-ln(U[token]) / weight[token]`. Inverse transform
uses a fixed keyed vocabulary permutation and one keyed uniform per position.
Their distribution-preservation claim assumes fresh independent uniform
randomness; here SHA-256 supplies pseudorandom values. Reusing a key row,
including after wrapping the period or across requests, does not provide fresh
independent randomness. A fixed key and position do not yield random retries.

Both detectors work without model probabilities. Exponential evidence is mean
`-ln(U[observed_token])`; inverse-transform evidence is mean absolute distance
between the uniform value and normalized token rank. **Lower is stronger**.
Alignment searches every text window of `block_size` and every cyclic key offset.
With `edit_penalty=Some(...)`, it uses Levenshtein dynamic programming with the
chosen insertion/deletion cost. Shorter inputs return no score. This search has
no calibrated p-value: thresholds must account for the entire search, for example
by repeating it with independently sampled null keys.

For `m` completion tokens, period `n`, and block size `k`, direct search costs
`O((m-k+1) * n * k)` and edit search costs `O((m-k+1) * n * k²)`. Cached costs use
`O(m*n)` memory, plus `O(k)` alignment rows. `max_cells` defaults to 4,194,304;
larger matrices return `AlignmentTooLarge`. It is a memory limit, not a time limit.

## MPAC payloads

```rust
use llm_watermarking::mpac::{Mpac, MpacConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let payload = [0, 1, 2, 3];
let mut config = MpacConfig::new([0x42; 32], 128, payload.len());
config.radix = 4;
let watermark = Mpac::new(&config)?;
let mut weights = [1.0; 128];
watermark.apply(&mut weights, &[10], 1, &payload)?;
// Host samples from weights and accumulates enough tokens for each position.
let recovered = watermark.detect(&[10, 11, 12], 1, &[])?;
assert_eq!(recovered.payload.len(), payload.len());
let known = watermark.detect_payload(&[10, 11, 12], 1, &[], &payload)?;
assert_eq!(known.trials, 2);
# Ok(())
# }
```

MPAC allocates each context to a message position, then boosts the colorlist
selected by that position's payload symbol. `payload_len` counts symbols in
radix 2 through 256; the default radix is 2. For radix 4, four symbols carry
eight bits. Each group has `floor(vocab_size / radix)` tokens; remainder tokens
receive neither bias nor a vote, but still count as detection trials. Defaults
are one context token, delta 2, and duplicate-n-gram exclusion.

Blind `detect` returns per-position votes, recovered symbols, and a winning
fraction. Tied and unobserved positions are `None`, never a fabricated zero.
The winning fraction is inflated by selecting the best symbol at each position;
it is not a z-score or confidence probability. `detect_payload` scores a payload
chosen before inspecting the text using `CountDetection`. Do not pass a payload
recovered from the same text and interpret that nominal z-score as significance.
Payload framing, conversion from bytes, error correction, and authentication
belong to the host. Recovery requires enough diverse contexts and token entropy.

## SemStamp sentence integration

```rust
use llm_watermarking::semstamp::{SemStamp, SemStampConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let watermark = SemStamp::new(&SemStampConfig::new([0x42; 32], 4))?;
// Illustrative vectors only: use your sentence encoder for real text.
let previous = [0.2, 0.7, -0.1, 0.4];
let candidate = [0.1, 0.6, -0.2, 0.5];
let accepted = watermark.accepts(&previous, &candidate)?;
// Commit an accepted sentence; otherwise generate a fresh candidate.
let evidence = watermark.detect(&[previous, candidate], 1)?;
assert_eq!(evidence.sentences_scored, 1);
# let _ = accepted;
# Ok(())
# }
```

SemStamp hashes embeddings using keyed Gaussian hyperplanes, then chooses valid
regions from the previous sentence's signature. `accepts` requires both region
membership and a minimum absolute cosine margin to every hyperplane normal.
Defaults are 8 hyperplanes, valid fraction 0.25, margin 0.02, and 100 attempts.
Embeddings may be unnormalized but must have the configured dimension, finite
components, and nonzero norm. Hyperplane count is limited to 16.

`sample_sentence(previous_embedding, callback)` handles bounded retries. The
callback returns `Result<(sentence, Vec<f32>), E>` and must generate each candidate
from the same committed history. It returns the accepted sentence, its embedding,
and the attempt count. On exhaustion, it returns the last candidate with
`accepted=false`; the host can commit that explicit fallback or reject it.
Callback errors and invalid embeddings propagate immediately.

Detection counts valid-region transitions, without requiring the generation
margin. `prompt_len` counts sentences, and completion-only input uses its first
sentence as context. Repeated signature pairs are excluded by default; disable
`ignore_repeated_transitions` for raw counts. Use the same segmentation and
embedding model at generation and detection. This crate does not supply or train
an encoder, and its synthetic example does not establish paraphrase robustness.

## Shared contracts for the added schemes

All keys are caller-owned 32-byte arrays and are redacted in configuration Debug
output. Constructors validate public configuration fields and capture their
values; editing a config afterward does not change an existing watermark.
Vocabulary size is fixed at construction (at least 2). Probability slices must
have that exact length, with index `i` representing token ID `i`, even after
filtering. All weights must be finite and nonnegative with positive mass.
They need not be normalized. Zero-weight tokens remain excluded; extreme biases
may underflow additional weights to zero at finite precision. Validation errors
leave mutable weights unchanged.

Token detectors exclude prompt tokens and the first generated EOS and everything
after it. EOS IDs can be sentinels outside the vocabulary; earlier IDs, including
prompt context, must be in range. Context-based detectors also exclude warmup
tokens. Use the same tokenizer, key, vocabulary, and parameters for generation
and detection. All APIs are stateless and cloneable.

`CountDetection` reports `trials`, `successes`, `expected_rate`, optional
`observed_rate`, and optional `z_score`. Empty eligible samples have no score.
Nominal z-scores, recovered payloads, and alignment costs are evidence, not
probabilities of authorship. Thresholds and false-positive rates require
representative held-out data, including repetitions and the actual decoding
policy. The watermark is not a signature; anyone holding its key can reproduce it.

## Development

```bash
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run --example synthid
cargo run --example kgw
cargo run --example unigram
cargo run --example exponential
cargo run --example inverse_transform
cargo run --example mpac
cargo run --example semstamp
```

The examples use synthetic distributions or embeddings without loading models.
Tests cover independent hash/sampling fixtures, reweighting math, categorical
sampling frequencies, payload recovery, wrong-key and unmarked controls, cropping
and edit alignment, sentence rejection and retry exhaustion, invalid inputs,
prompt/EOS handling, and existing SynthID behavior. `cargo test` also compiles and
runs the README snippets. If a shared Cargo target directory is unwritable, set
`CARGO_TARGET_DIR=target` when running build commands.
