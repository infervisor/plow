# GLM-5.3 MXFP4 preparation

User-authorized separate preparation/download track, 2026-09-23.
Owner: glm53_mxfp4_preparation agent. Original FP8 checkpoint/campaign untouched.

## Selected checkpoint

- Repository: https://huggingface.co/amd/GLM-5.3-Quark-MXFP4-AttnFP8
- Revision: `4992911b6f4e4d20947573ac7e04ff7d18b32288`.
- Destination: `/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b`.
- Original source declared by AMD: `zai-org/GLM-5.3`; source revision not declared.
- Architecture/config identical to local original outside quantization_config:
  GlmMoeDsaForCausalLM / glm_moe_dsa, 78 layers + MTP, 256 routed experts.
- 141 safetensors shards; 118629 indexed tensors; 409520061992 shard-file bytes;
  actual tensor payload409505191616 bytes (index total_size includes headers);
  all 154 repository files total 409552735587 bytes. Disk free before download: 2.2TiB.
- MXFP4 E2M1 weights and dynamic activations, groups32, E8M0 power-of-two scales,
  half-even rounding. Quark native real_quantized/reorder layout.
- HTTP safetensors header checks: layer3 expert0 gate packed U8 [2048,3072],
  scale U8 [2048,192], logical [2048,6144]. Attention q_a_proj F8_E4M3,
  router/head BF16, FP32 scales. Attention stays FP8; this is mixed precision.
- GLM-5.3 LICENSE is present and permits local use/conversion. Preserve license.
- Chat template differs from original pinned aca966e4 (AMD SHA256
  69bb3ab52067898e2466b855407636de559568947f367945842aabcb7fcc1705 vs original
  3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c): AMD lacks
  later None-content guard and loop breaks. Preserve upstream artifact and record
  exact prompt/template if running any future paired quality/performance tests.
- Alternatives examined: RedHatAI/GLM-5.3-MXFP4 (compressed-tensors, broadly
  quantized Linear layers); OneNexus/GLM-5.3-MXFP4 (BF16-derived, SGLang,
  targeted SmoothQuant refinements). No need to perform a new lossy conversion.

## Runtime boundary

AMD card targets MI350/MI355 + vLLM and references a nightly plus PR link
https://github.com/vllm-project/vllm/pull/54556 (displayed as #54566 in the card).
CPU-only source inspection of pinned vLLM0.29 Docker digest
`sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1`
finds QuarkW8A8Fp8PerBlock and QuarkOCP_MX_MoEMethod W4A4 backend selection.
This establishes implementation presence, not a successful serving run. No GPU
process launched. Queue/doctor required before any eventual runtime qualification.
No Plow MXFP4 preparation/support claimed. No speed/quality claims.

Confirmed loader compatibility gap in this exact image: Quark block-FP8 creates
`weight_scale_inv`, while the checkpoint stores `weight_scale`, and the installed
DeepSeek/GLM loader lacks the alias/remap. Upstream PR54556 is OPEN and fixes this
exact issue. PR54566 is unrelated DeepSeek vision support. Serving will need an
isolated compatible runtime/loader before any GPU qualification. The downloaded
checkpoint will remain upstream-exact; do not edit/repack its scales to conceal
the loader difference.

## Completion gates

1. Download exact pinned revision using existing `hf download` in CPU-only Docker.
2. `hf cache verify --revision ... --local-dir ... --fail-on-missing-files --json`.
3. Validate every local safetensors header, complete shard inventory, quantized
   shapes/dtypes, unquantized exceptions, original nonquantization config/tokenizer.
4. Record exact commands and integrity artifacts. Keep runtime limitations explicit.

## Representative numerical inspection (CPU only)

`representative_numerics.py` / `representative-numerics.json` in artifact directory
PASS in pinned Docker without GPU device access. Expert layer10/id0 gate/up/down
weights reconstructed using E2M1 low/even, high/odd nibble order and E8M0 groups32;
all finite, no reserved255 scale codes. Relative-L2 weight reconstruction errors
against local original FP8 dequantized with its FP32 128x128 scales:
gate .112674974, up .112837858, down .112711631. Max absolute error .015625.
These are weight errors, NOT output/logit/quality errors or serving qualification.
All28 FP8 attention weights/scales in downloaded shard1 are bit-identical to the
local original (scale-name mapping only), providing direct partial provenance
evidence in addition to matching architecture config.

## Download command

Started in exec session42764; CPU-only container
`plow-glm53-mxfp4-download-4992911b`, no GPU devices passed. Log:
`/opt/models/glm53-mxfp4-preparation-20260923/download.log`.
Started2026-09-23 06:14:26UTC, PID2895371; 19GiB at06:18:09UTC.
Official pinned remote metadata (all141 shards have LFS SHA256) saved as
`/opt/models/glm53-mxfp4-preparation-20260923/remote-manifest.json`.
Offline audit tool in the same artifact directory, `audit_checkpoint.py`, checks
all headers/layouts/shapes/dtypes against the full original without editing it.
Initial complete shard1 SHA256 matches official LFS digest
`b2f786de2d3f9da9f02a8934f5e32a6a96c8aa2272ef69e446d4ebdea7bb50c8`.
Session25182 waits for download exit0 then automatically runs full `hf cache
verify` and layout audit. Outputs `verify.json`, `verify.log`, `layout-audit.json`
in the preparation artifact directory. Final completion results follow.

## Completed 2026-09-23 07:14UTC

- Download session42764 exited0; all141 shards present.
- Official `hf cache verify` passed all154 repository files at pinned4992911b:
  all141 shard SHA256 values match official LFS digests; no missing/mismatched files.
  `verify.json` contains checked154. 311 extra local files are exclusively HF
  `.cache/huggingface/` download metadata (checked), not extra model tensors.
- All154 remote file sizes also match the saved pinned manifest; total409552735587B.
- Full layout audit PASS: 118629 tensors, 58605 packed MXFP4 weight/scale pairs,
  dtype counts U8=117210, F8_E4M3=439, BF16=465, F32=515. All indexed names,
  shards, nonoverlapping payload spans, shape-derived lengths, logical weight
  dimensions, groups32 scale shapes, and original exception shapes/dtypes checked.
- Config outside quantization identical; tokenizer/tokenizer config/generation
  config/LICENSE byte-identical to original. Separate chat-template mismatch above.
- First audit assertion assumed standard payload-only index total_size and failed.
  Inspection proved AMD's field equals the sum of entire shard file lengths,
  including14870376 header bytes. Audit now explicitly checks and records that
  convention. No checkpoint bytes altered; all official integrity hashes passed.
- Final reports: `verify.json`, `verify.log`, `layout-audit.json`,
  `representative-numerics.json`, `remote-manifest.json` under
  `/opt/models/glm53-mxfp4-preparation-20260923`.
- Checkpoint preparation/download complete. Serving NOT qualified: pinnedvLLM0.29
  needs the documented block-FP8 scale-name loader compatibility fix; PlowMXFP4
  integration not implemented or implied. No GPU process, original-precision
  campaign change, publication, runtime default change, or original overwrite.

```bash
nix develop --command bash -lc 'sudo -n docker run --rm \
  --name plow-glm53-mxfp4-download-4992911b --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf-home -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -v /opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b:/checkpoint \
  --entrypoint hf \
  vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1 \
  download amd/GLM-5.3-Quark-MXFP4-AttnFP8 \
  --revision 4992911b6f4e4d20947573ac7e04ff7d18b32288 \
  --local-dir /checkpoint --max-workers 8 \
  > /opt/models/glm53-mxfp4-preparation-20260923/download.log 2>&1'
```
