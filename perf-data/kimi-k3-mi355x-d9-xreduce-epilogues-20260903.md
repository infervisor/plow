# Decode XReduce epilogue screen (MI355X, 2026-09-03)

Status: rejected after the isolated TP8 gate; no packet, runtime, or interpreter changes.

## Structural census

The current T=1 counter graph has 278 one-shot XReduce packets:

- 186 have one AttnRes consumer. Of these, 178 materialize an exact prefix add where the
  XReduce output is one of AttnRes `t6/t7` (92 in `t6`, 86 in `t7`); eight have no add.
- 92 have two GEMV consumers using the same `routed_expert_norm`. XReduce uses seven WGs and
  each GEMV uses 224 WGs, so the norm can be computed once only after all seven reduction slices
  publish. This is a graph/tensor/shape contract, not a model-name rule.

The AttnRes body itself is not eligible: XReduce uses 14 WGs and AttnRes uses one. Folding the
whole consumer would require the D6 gang-admission/grid-rendezvous primitive.

## Isolated gate

The TP8 harness calls the production reduction and ordering primitives. The prefix candidate
writes `bf16(residual + bf16(rank_ordered_sum))` directly. The norm candidate preserves the
rounded reduction, has seven WGs publish completion through a device counter, and runs the exact
RMSNorm order once after all slices arrive. The control runs two norms, matching the two current
GEMV consumers. All comparisons are bit-exact.

| total producer + consumer body | control | fused | delta |
|---|---:|---:|---:|
| prefix add, 4000-iteration hot | 1.004 us | 0.999 us | -0.005 us |
| routed norm, 4000-iteration hot | 1.567 us | 1.464 us | -0.103 us |
| prefix add, one cold iteration | 1.480 us | 1.448 us | -0.032 us |
| routed norm, one cold iteration | 2.076 us | 1.888 us | -0.188 us |

The hot projection is only `178*0.005 + 92*0.103 = 10.37 us`, or 0.0104 ms/token. Even the
single-iteration projection is about 0.023 ms/token. The approximately 15.4 us production cost
per XReduce includes the persistent-interpreter packet handoff, cache maintenance, and cold
protocol path; these same-packet epilogues do not remove that floor.

## Resource gate

| isolated object | VGPR | SGPR | VGPR spill | SGPR spill | private | wave |
|---|---:|---:|---:|---:|---:|---:|
| prefix control | 17 | 62 | 0 | 0 | 0 B | 64 |
| prefix fused | 14 | 60 | 0 | 0 | 0 B | 64 |
| norm control | 68 | 80 | 0 | 0 | 0 B | 64 |
| norm fused | 66 | 80 | 0 | 0 | 0 B | 64 |

The arms pass the isolated resource gate but miss the performance gate by roughly two orders of
magnitude. Adding packet fields, graph rewrites, and a new rendezvous for this saving is not
justified. D9 is closed; D6 must attack the packet/protocol floor itself.

Artifacts: `/tmp/k3-d9-bench/resources.log`, `/tmp/k3-d9-bench/run-4000.log`, and
`/tmp/k3-d9-bench/debug-cleanenv2.log`. Two earlier setup attempts died in the host dynamic loader
because nix and system glibc/ROCm libraries were mixed; neither launched a kernel and neither is
included above. Rebuilding the host with the repository's clean `/usr/bin/env -i` contract fixed
the harness.
