# Fixed-order MXFP4 MoE prefill combine

This isolated gfx950 experiment keeps the production `f32[T,topk,H]` part
buffer and sums slots in interpreter order. It changes only the mapping from a
flat element grid to one workgroup per token, avoiding division in the hot loop
and allowing a resident-grid sweep.

`plowrt` can route a pure compatible segment to the candidate when packet
generation enables `PLOW_MOE_COMBINE_LEAN=1`. The structural contract is
model-independent: nonzero runtime H/T, topk 16, materialized f32 parts, and no
banded, part16, or deterministic-accumulator encoding. A missing object falls
back to the interpreter; a present object must pass the ABI/resource markers.

A materialization-free expert-sorted down path cannot preserve the
interpreter's f32 association: atomic arrival changes the order, while
token-major down compute loses expert weight reuse. A later two-phase resident
Down+Combine kernel can remove the launch boundary, but exact slot order still
requires the part buffer.

```sh
nix develop -c runtime/bench/amd/lean_moe_combine_ref/build.sh /tmp/moe-combine
nix develop -c perf-data/tools/gpulease -n 1 moe-combine \
  runtime/bench/amd/lean_moe_combine_ref/run_gate.sh /tmp/moe-combine
```
