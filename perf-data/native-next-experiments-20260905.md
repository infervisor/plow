# Native H100 experiments after the main merge

Work continues on `perf/h100-native`. Main's merge remains `0125acf`.
The next GPU window starts after the active Gemma12 vLLM reference group
finishes and cleans up. Existing benchmark executables and scripts are
reused; the queue manifest and live status are under
`/tmp/plow-model-support-checks/native-next-window-20260905/`.

## FP8 producer coordinates

`PGM90_FP8_TMA_ISSUE_CURSOR=1` is a default-off build experiment in the
uniform native FP8 TMA body. It advances a K/tile cursor and calculates
tile coordinates only at tile transitions. Stage count, barriers, WGMMA
ordering, descriptors and output arithmetic remain unchanged.

SASS confirms that ordinary K iterations skip tile-remap divisions and
remove the K-index division. Matched primitive builds retain 137 registers;
matched generic-role builds retain 180. All four have zero spills/stack,
the role retains ABI1 and a 99,376-byte dynamic arena, and rebuilt control
instruction streams match the previous frozen controls.

CPU gates: 23,100 producer address/barrier trace cases match, and six
preprocessor cases pass. These are not GPU numerical or performance gates.
Frozen source, commands, SASS and hashes live in
`/tmp/plow-model-support-checks/fp8-refill-coordinate-cache/`.

The GPU queue runs two alternating-order primitive comparisons at
M=1024/4096/8192. Every case must be bitwise exact, finite, and pass
activation quantization and output-canary checks. Block checks depend on
all primitive gates passing; candidate block timing depends on all block
checks passing. Proposed advancement bar: at least 5% QKV improvement at
each M, with no repeatable regression above 3% on another projection.
The flag stays off until those measurements justify advancement.

## Packet-derived per-slot KV descriptors

Serialized prefill now binds rank-3 KV TMA descriptors to the same slot as
its K/V pointers. The loader validates packet recipe geometry and extents,
creates immutable 256-byte descriptor pairs for additional slots, and
retains them for the engine lifetime. B1 creates no additional descriptors.
This adds no model configuration or per-token work.

Four focused CPU tests cover B1/B2/B16 addressing, malformed recipes,
geometry conflicts and overflow. All ten existing capacity packets pass
validation. The complete CUDA/HSA runtime suite passes 384 tests, with one
actual-packet test ignored unless explicitly supplied its fixture. The
new live-KV GPU test compiles. Multi-slot GPU parity is still required;
live allocation continues to reject batch > 1 KV tensormaps.

## Measurement order and scope

1. Warmed baseline block sweeps: coarse/isolated/generic-role packets,
   layers 0/3, contexts 1K/4K/8K/32K, two reversed-order repeats. Each uses
   15 prefill measurements, 100 decode measurements, and existing warmups.
2. FP8 primitive gate, paired block-output gate, then warmed candidate
   block sweeps. Isolated block timing is diagnostic kernel timing;
   chunked hidden states and decoded tokens are not model-quality evidence.
3. Existing VMM remap lifecycle test, then live-versus-flat full-model
   logits/tokens at 8K, 32K, padded 8193 tokens, and a repeated 8K prompt.
   Both prefix-cache flags are off; attachment/cache counters must stay zero.
4. Separate Gemma31 BF16 vLLM startup/32K-C1 reference at MBNT2048,
   preserving the original dtype, APC, concurrency cap and 512-output-token
   contract. Its full-cache plus sliding-cache admission bound is estimated
   at 8.918 GiB, versus 18.293 GiB at MBNT8192; startup must verify capacity.

The original vLLM reference driver resumes after queue cleanup. Failures
remain recorded and dependent candidates are skipped. No new native
performance win is established by these CPU checks or queued jobs.
