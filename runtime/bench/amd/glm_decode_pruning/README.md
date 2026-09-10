# GLM decode inventory pruning on MI300X

The FP8 decode dispatch omitted inventory guards on six cases. The fix uses
the existing `PLOW_DECODE_INVENTORY_PRUNE` / `PLOW_HAS_*` convention. Generic
build defaults are unchanged.

[mi300x.json](mi300x.json) records the build commands, source and object hashes,
preprocessor checks, resource metadata, campaign recipe, and serving results.
The specialized image uses the generated packet header, enables inventory
pruning, and disables both Gemma MoE families after checking that no program
in the packet uses Gemma opcodes. This is a combined specialization comparison;
it does not isolate the effect of the six guards.

The generic build before and after the fix has identical `.text`, `.rodata`,
and `.data` bytes. The specialized image reduces `.text` from 3,140,288 to
1,635,264 bytes. Register, scratch, LDS, and spill metadata remain unchanged.
Code size includes exported device functions and does not measure executed
work or establish an occupancy improvement. Preprocessor checks retain the
required FP8 MLA case, remove five unused cases, and restore all six when
their inventory bits are enabled.

One exclusive eight-MI300X lease ran both arms without concurrent builds or
other GPU work. Each arm used 18 retrieval checks at concurrency 20, followed
by 20 random requests with input 70,000, output 700, range ratio 0.14, seed 0,
and concurrency 20. Both use the same FP8-KV B20 packet and runtime, local
selection and native decode GEMM enabled, native fold and narrow tiers
disabled, and no speculative decoding. Only the main decode image differs.

| Metric | Specialized | Configured, unpruned | Change |
| --- | ---: | ---: | ---: |
| Output tokens/s | 42.361 | 41.771 | +1.41% |
| Mean TPOT (ms) | 287.014 | 283.755 | +1.15% |
| P99 TPOT (ms) | 442.493 | 417.246 | +6.05% |
| Mean TTFT (ms) | 110,884.851 | 111,587.012 | −0.63% |

Both arms passed 18/18 retrieval checks and completed 20/20 requests with no
failures. Input and output length arrays match exactly (1,414,538 input and
13,795 output tokens); 6/20 generated texts match exactly. This single pair
does not establish repeatability or broad output-quality equivalence.
Throughput improved slightly while TPOT worsened, so specialization remains
opt-in. These results do not establish parity with the 100-request H200 run.
