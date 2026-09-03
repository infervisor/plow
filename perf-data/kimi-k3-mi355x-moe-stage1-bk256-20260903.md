# Lean MXFP4 MoE stage 1 BK256 — MI355X TP8

Date: 2026-09-03

## Contract and route audit

- The standalone object owns only structurally eligible `MoeGroupGluPf` packets:
  local intermediate 384, latent input 3584, 896 experts, MXFP4, SiTU activation, and the exact
  eight operands produced by a top-16 BM64 align boundary.
- It calls the same `d_moe_group_pf_a4w4<true>` implementation as the interpreter with BK256.
  The epilogue applies SiTU to gate and up, writes MXFP4 payload plus E8M0 scales in sorted-row
  order, and skips align padding through `row_partidx`. The stage-2 input contract is unchanged.
- Selection is shape/capability based and defaults on after the exact network qualification.
  `PLOW_MOE_STAGE1_LEAN=0` opts out; unsupported packets stay on the interpreter.
- The original generic full-model emitter wired this knob, but K3's separate full-model builder
  did not. Consequently the knob emitted a byte-identical K3 packet and P5 could not run. The K3
  prefill builder now applies the same generic stage-1 segmentation rule; a regression test pins
  that path.

The shipping geometry is wave64, 512 threads (eight waves), not the four-wave geometry in the
initial experiment plan. The object has 190 VGPR, 0 AGPR, 90 SGPR, occupancy 2, zero private
bytes, and zero VGPR/SGPR spills. Its 119,808-byte dynamic-LDS reservation is exact for the BK256
double-buffered A/B tiles, scale tiles, and fused-GLU bridge. The build rejects total registers
above 192, occupancy below 2, wave32, private memory, spills, a changed symbol, or a missing ABI
marker. Object SHA-256: `78178651a9867c9bf30efc09e298e25ffc3fcff9e113b299faa6f6f5911ce5b3`.

## Isolated gate

The existing exact T8192 screen measured 2.496 ms for the interpreter control and 2.202 ms for
BK256, an 11.8% improvement. Payload and scale output passed the exact oracle, and the object
passed the zero-private/zero-spill gate. This misses P5's `MoeGroupGluPf <= 90 ms` network target:
2.202 ms x 92 layers projects to about 203 ms, with only about 27 ms projected gain.

## TP8 8192→1 network gate

Three uncontended, order-alternated control/candidate folds used BF16 KV, one 8192-token chunk,
TP8, exact greedy output, the same checkpoint/object directory, all 7,650 measured dense tiles,
and a passing Lean ordering certificate. The candidate contains exactly 92 singleton stage-1
segments at the 8192 rung; the control contains zero. Pairing hash is `0x3d1373f347b64fde`
in both.

| fold | control TTFT | BK256 TTFT | paired gain | trace-chain gain |
|---:|---:|---:|---:|---:|
| 1 | 2688.316 ms | 2547.844 ms | 140.471 ms (5.23%) | 139.737 ms |
| 2, candidate first | 2689.552 ms | 2546.656 ms | 142.897 ms (5.31%) | 142.186 ms |
| 3, control first | 2687.645 ms | 2547.557 ms | 140.088 ms (5.21%) | 141.567 ms |
| mean | 2688.504 ms | 2547.352 ms | **141.152 ms (5.25%)** | **141.164 ms** |

All six arms completed one request with zero failures, token 6896, checksum
`fnv1a64:7d749e3b002fafa7`, complete non-overflowed diagnostics, and all eight ranks completing
prefill. Paired-gain sample standard deviation is 1.523 ms.

The traced interpreter stage-1 body is 354.7--355.4 ms. Standalone launches do not currently
emit raw packet-trace records, so the candidate's 223.1--224.1 ms residual contains their
execution. Across the folds the remaining visible body is stable to a few milliseconds; endpoint
and trace-chain mean gains differ by 0.012 ms. This supports an approximately 221--224 ms
standalone stage path and 131--134 ms direct stage-path saving, rather than an unrelated
network-wide shift.

The extra code object is 38,320 bytes per rank (299 KiB of file payload over TP8). The route adds
no persistent device buffer or workspace. During a launch it reserves 119,808 bytes of dynamic
LDS per workgroup and uses 190 VGPR/90 SGPR at occupancy 2, with zero private memory and spills.

Packet SHA-256:

- control: `dc95d77cd15ae5c1d5556ce96a81c6dfd984a50e4887994f288838b23a8afdf9`
- candidate: `91828acbfc619c292f43e20444127cbe0795d9ad79eccf8dcb8d2a2a61b17e75`

Trace SHA-256, control/candidate by fold:

- fold 1: `308de1b87a1bbb536f57c5a4004f7fc18249f747b1b90cb397409dc4f2748a89` /
  `7171df7c6fe2e4ef747ad83f1a5b3f94203d5a1c7eaaff61339da58c03cc598d`
- fold 2: `d8101e64341055d815fbb1cec7434e1a1036597bf253ce8db4c5d1a1d8ca0c02` /
  `e8cfc76d67f8fb0cc5151d1edfb1debe96629b6bb62b4434b432a75ccbe97a90`
- fold 3: `a567982e3257906d026b3eb95415b77cbe113787c0fd7e359e4c90183fffd178` /
  `31b9af4905177a34573e0ddc88036c4eb5a6940f2e4d1f5b0562ce6b33bf27d1`

Raw artifacts: `/tmp/k3-p5-{control,candidate}{,-r2,-r3}.{json,log,trace}`.

## Qualification status

- Exact A4W4 stage-1/stage-2 layout contract: pass.
- Resource gate: pass, with corrected production resource facts above.
- K3 full-model knob reachability: fixed and regression-tested.
- Isolated performance target: miss.
- Full-network exactness and performance: pass, mean -141.152 ms (-5.25%).
- The declared <=90 ms stage-category target remains unmet; the inferred standalone path is
  about 221--224 ms.
- Default-on gate: pass on the material, replicated network gain; the exact shape/resource route
  remains guarded and has an explicit opt-out.
