# K3 hybrid phase-asset emission gate

## Decision

Phase-object and expert-parallel experiments use the production `devblob`
emitter. They do not use the scheduled-packet HF planner: that bridge has no
executable lowering for `Conv1dDepthwise`, `LinearAttention`, `SituGlu`, or
`BlockResidual`. Extending its uniform transformer plan would duplicate the
already-validated hybrid emitter and risk producing a plausible wrong graph.

The production hybrid emitter is now default-on. `K3_FULL=0` retains the old
capability report explicitly. The graph-derived phase and EP rewrites contain
no model-name predicate; they consume packet opcodes, dependencies, tensor
placement, and routed-MoE geometry.

Every device-blob manifest now records its input contract as token IDs, text
only, with vision unsupported. This makes the scope of using the text tower
from a multimodal checkpoint explicit in the artifact.

## Refusal reproduced

The wrong artifact path was reproduced against the real checkpoint:

```text
nix develop --command cargo run -q -p plowc -- \
  --hf-dir /home/shaswot/models/Kimi-K3 --emit packets \
  --gpu mi350 --num-gpus 8 --parallel tp --batch 1 --seq 128 \
  --phase decode --out /tmp/k3-hybrid-emitter-repro

invalid dimension: kimi_k3 carries a vision_config ... Strip
`vision_config` ... to compile the text tower explicitly.
```

After that explicit text-only conversion, the same path reaches the separate
`build_full_model_plan` refusal. That refusal remains intentional: the
scheduled CPU/simulator bridge cannot execute the hybrid operators listed
above. It is not evidence that production K3 emission is unavailable.

## Exact full-model asset gate

This emitted all 93 layers without `K3_FULL=1`:

```text
nix develop --command env \
  PLOW_K3_LAYERS=all K3_PREFILL=128 PLOW_MXFP4=1 \
  PLOW_PHASE_OBJECTS=1 PLOW_MOE_PREFILL_EP=1 \
  cargo run -q -p plowc -- \
  --hf-dir /home/shaswot/models/Kimi-K3 --emit devblob \
  --arch gfx950 --gpu mi350 --max-ctx 128 --n-cu 256 \
  --num-gpus 8 --parallel tp --block 0 \
  --out /tmp/k3-hybrid-phase-full93 \
  --lean-verify-devblob=false --lean-oracle-devblob=false
```

`--block 0` only suppresses the expensive advisory egg exploration; the K3
production emitter takes its layer set from `PLOW_K3_LAYERS` and emitted the
complete model.

| item | emitted value |
|---|---:|
| layers | 93 = 69 KDA + 24 MLA |
| tensors | 5,939 |
| prefill instructions / segments / phases | 2,764 / 925 / 649 |
| decode instructions / segments / phases | 2,165 / 49 / 49 |
| routed-MoE EP8 boundaries | 92 |
| dense GEMM selections | 1,275 / 1,275 measured |
| packet size | 42 MiB |

The EP manifest declares `E=896`, full intermediate width 3072, balanced
contiguous whole-expert ownership, and the four specialist objects. Its
resource contract is wave64, zero private segment, fail closed. The audited
decode-safe companion allocation is 241,076,011,008 bytes/rank, so this asset
is valid experiment input but still correctly refuses production load without
the capacity acknowledgement.

Artifact hashes from the initial full-model gate:

```text
model.pkt     800049b1a65f20ceb688e345fc8097f1b12b0761a08a021305b19c3442c738ac
build.json    2454567988ebbaf7eb586cda4e53c54b542458037ca48fc9ad58de1cd3cfbc0e
plow_config.h 5c20414e70c2f0f1dacbd66cca5679d5d00293ff11bb9038e77ba0a114d65f77
```
