# llm-watermarking

A model-independent Rust library for text watermarking algorithms. The first
implementation is SynthID-Text two-candidate tournament sampling and mean-g-value
detection. The library operates on probability slices and token IDs. The host
owns tokenization, model inference, sampling filters, random token selection,
serialization, and language or HTTP bindings.

The only runtime dependency is `sha2`. Algorithm errors implement
`std::error::Error`; no tensor framework or inference engine is required.

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

## Probability and history contract

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

## Detection

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

## Algorithm and format

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

## Development

```bash
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run --example synthid
```

The example samples synthetic probability distributions without loading a model.
Tests cover tournament math, deterministic hashing, repeated contexts, detection,
wrong keys, invalid public inputs, and preservation of the expected distribution.
