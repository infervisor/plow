# AITER Opus attention subset

This directory contains the dependency-free source subset needed by Plow's
standalone gfx950 bf16 materialized-attention object.

- Upstream: `ROCm/aiter`
- Revision checked: `10b192f5b5bda90f2af33ceae7a6c2f416bfc674`
- License: MIT (`LICENSE`)
- Selection: `D_QK=192`, `D_V=128`, gfx950, wave64; no model-name gate

The selected sources retain the upstream implementation. Plow enables one
guarded integration adaptation: batch-mode workgroups are decoded from a flat
1D grid because plowrt's device API is 1D. The arithmetic preserves upstream's
q-block-fastest `(q-block, head, batch)` ordering. Plow's entry point and ABI
markers live separately in `runtime/amd/mla_materialized_opus.hip`.
