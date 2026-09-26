# Keyed formats and implementation choices

The original SynthID format is unchanged and is documented in the README. The
six new schemes use separate domains ending in a zero byte, exported as each
module's `HASH_DOMAIN`. Their PRNG is local to this crate, not PyTorch or NumPy.
For interoperability, reproduce this format and the configured parameters exactly.

## Shared byte stream

Start a SHA-256 prefix with `domain || key[32] || parameters`. Encode every
parameter as an unsigned little-endian 64-bit integer. Domains and parameters:

| Module | Domain (before the terminating zero) | Parameters in order |
| --- | --- | --- |
| `kgw` | `llm-watermarking-kgw-v1` | vocabulary size, context width |
| `unigram` | `llm-watermarking-unigram-v1` | vocabulary size |
| `mpac` | `llm-watermarking-mpac-v1` | vocabulary size, context width, payload length, radix |
| `exponential` | `llm-watermarking-exponential-v1` | vocabulary size, sequence length |
| `inverse_transform` | `llm-watermarking-inverse-transform-v1` | vocabulary size, sequence length |
| `semstamp` | `llm-watermarking-semstamp-v1` | embedding dimension, hyperplane count |

For each PRNG block, hash the prefix (including any additional scheme input),
one tag byte, and a little-endian 64-bit counter starting at zero. Split the
32-byte digest into four little-endian 64-bit words and consume them in order.
Increment the counter for the next block. Independent calls start at counter zero.

Convert a word `x` into an open-interval uniform using
`((x >> 12) + 0.5) / 2^52`, evaluated in f64. Both endpoints stay strictly inside
`(0, 1)` and avoid infinite logarithms. To sample below an integer bound `b`, reject
words less than `2^64 mod b`, then take `x mod b`.

Permutations use tag 0 and descending Fisher-Yates: start with `[0, ..., N-1]`,
and for `i=N-1` through 1 swap position `i` with an unbiased draw below `i+1`.
Fixed-size favored sets take the first `floor(fraction*N)` entries.

## Scheme inputs

- KGW appends its final `context_width` token IDs as little-endian u32 values to
  the prefix before permuting. Unigram permutes the unextended prefix once.
- MPAC uses the same context encoding. Tag 1 draws a message position below
  `payload_len`; a separate tag-0 stream permutes the vocabulary. Consecutive
  equal-sized chunks are colorlists in symbol order. Leftover tokens are uncolored.
- Exponential appends `LE64(position % sequence_len) || LE32(candidate_token)`;
  tag 2 supplies that candidate's first uniform. Sampling minimizes `-ln(U)/p`.
  Detection uses the paper's nonnegative theoretical `-ln(U)` cost, rather than
  its experimental negative `ln(1-U)` score. Lower costs indicate a better match.
- Inverse transform permutes the unextended prefix once. For uniforms it appends
  `LE64(position % sequence_len)` and uses tag 1. Sampling takes the first occupied
  CDF interval strictly above `U * total_mass` in permuted order. Detection uses
  `abs(U - rank/(vocab_size-1))`, with zero-based rank.
- SemStamp uses tag 3 on the unextended prefix for hyperplanes. For each coordinate
  draw two uniforms and compute `sqrt(-2*ln(u1))*cos(2*pi*u2)`; discard the sine
  partner. Normalize each normal vector. Signature bit `i` is one iff its dot
  product with normal `i` is positive. Append the previous signature as LE32 and
  use the tag-0 permutation over all `2^num_hyperplanes` regions to select the next
  valid set. Margin checks use absolute cosine, including equality at the margin.

Partition fractions, bias, margin, retry count, and detector deduplication options
do not seed permutations. Changing them changes selection or scoring, not the
underlying ordering. MPAC payload values likewise do not seed position allocation.
SemStamp uses floating-point transcendental operations; embeddings exactly at a
hyperplane or margin boundary may behave differently across math-library/platform
implementations. Preserve the encoder and numeric environment for reproducibility.

## Detection choices

The added schemes return raw evidence without a universal decision threshold.
KGW and MPAC exclude repeated context-plus-token n-grams by default; SemStamp
excludes repeated signature transitions. Unigram defaults to counting all tokens,
as in the paper. Prompt-only occurrences do not populate deduplication history.

MPAC exposes erasures and vote counts instead of silently resolving ties. It
deliberately does not apply an ordinary binomial z-score to an inferred message:
maximizing over possible messages biases the null distribution. Known-message
scoring is separate; blind-message confidence calibration and error-correcting
codes are left to the host.

Sampling alignment implements the paper's window/offset search and simple
Levenshtein recurrence. Each token window and cyclic key window has `block_size`
items. Gap costs initialize both boundaries, and substitution costs are the
scheme-specific costs above. The returned score is total cost divided by that
block size, including gaps; it is not the mean over only matched tokens. Ties
choose the earliest text window, then earliest key offset. A zero gap penalty is
allowed but can erase all evidence by aligning everything to gaps.

No analytic p-values are reported after alignment search. A host randomization
test should draw independent null keys and rerun the identical search and
parameters for each, comparing their minimum costs to the observed minimum.
With `B` null samples, `(1 + count(null_cost <= observed_cost)) / (B + 1)` handles
ties conservatively; independence and matching null distributions are essential.

The SemStamp retry helper returns the last candidate on maxout with an explicit
`accepted=false`. It never treats failed generation or invalid embeddings as a
successful fallback. Segmentation, sentence encoding, encoder fine-tuning, and
actual text paraphrase evaluation are host responsibilities.
