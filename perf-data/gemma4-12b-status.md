# Gemma 4 12B benchmark target

The requested target is now `google/gemma-4-12B-it`, pinned to
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. BF16/FP8 text-generation comparison
against vLLM through 16K input and concurrency 128 is pending. Checkpoint download
has started. Reports named `gemma3-*` measure Gemma 3 and do not qualify this model.

The shared compact terminal now accepts an uncapped norm/head/argmax tail and
paired head tensor maps, using mapless GEMM after gathering selected rows. It
also permits preallocated KV. Existing capped tails remain supported. The CUDA,
HSA and CPU exec test run passed 327 tests with 25 ignored; the CUDA release build
passed. GPU qualification of this terminal extension remains pending.
