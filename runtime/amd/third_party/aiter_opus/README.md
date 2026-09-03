# AITER Opus attention subset

This directory contains the dependency-free source subset needed by Plow's
standalone gfx950 bf16 materialized-attention object.

- Upstream: `ROCm/aiter`
- Revision checked: `10b192f5b5bda90f2af33ceae7a6c2f416bfc674`
- License: MIT (`LICENSE`)
- Selection: `D_QK=192`, `D_V=128`, gfx950, wave64; no model-name gate

The three source files are copied unchanged. Plow's entry point and ABI markers
live separately in `runtime/amd/mla_materialized_opus.hip`.
