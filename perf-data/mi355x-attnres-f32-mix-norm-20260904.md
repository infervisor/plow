# MI355X AttnRes f32-mix output-norm screen (2026-09-04)

Decision: **rejected**. Keep the production BF16-seam exact fusion and do not run TP8.

The candidate matches the pinned-vLLM arithmetic order: prefix plus delta rounds to BF16; source
scores and the residual mix accumulate in f32; output RMSNorm consumes the unrounded f32 mix and
has an epsilon distinct from the score epsilon. Eligibility is operation/consumer semantics plus
`HID <= 8192` and vector alignment, not a model name. The static template is unreachable from the
production dispatcher and changes no default.

## Reproducibility

The isolated harness is `runtime/bench/amd/attnres_f32_mix_norm`. It records the complete
prefix/delta/rounded-prefix/ring/two-score-factor/output-gain contract. The deterministic input
hashes were:

| payload | FNV-1a64 |
|---|---|
| prefix | `8ac6df52dc14844d` |
| delta | `7978cd753f4c003f` |
| rounded prefix | `297157d5e4e7b741` |
| residual ring | `1659d70809b8a82a` |
| residual norm factor | `19aff55509cb98a8` |
| residual projection factor | `294813c830495b2a` |
| output norm gain | `8045d83cf287b899` |

State: `H=7168`, capacity 8, no ring write, score epsilon `1e-5`, output epsilon `1e-5`,
`output_norm_input=mixed-f32`, wave64. Real captures use the SHA256-sealed residual seam in
`scripts/mla_boundary_abi.py`; this screen did not weaken or replace that gate.

The run held GPU 0 through `/tmp/gpulease`; the concurrent EP probe held GPU 5. The preceding
TP8 showdown released all devices before either probe acquired one.

## Quality

Candidate GPU output was compared as BF16 against a CPU transcription of the pinned-vLLM order.
Three adjacent candidate executions were byte-identical, so the isolated adjacent-repeat floor is
zero.

| live ring rows | current BF16-seam relL2 | f32-mix relL2 | f32-mix max abs | exact BF16 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 100.00% |
| 1 | 2.913e-3 | 4.759e-9 | 4.17233e-7 | 99.99% |
| 4 | 2.751e-3 | 0 | 0 | 100.00% |
| 8 | 2.869e-3 | 2.718e-9 | 2.38419e-7 | 99.99% |

This establishes the intended semantic improvement. It does not by itself qualify a real captured
MLA seam; the existing full SHA256 contract remains required if this route is revisited.

## Performance and resources

| boundary | current | f32 mix | gain |
|---|---:|---:|---:|
| T1, nb=0 | 4.128 us | 3.394 us | 17.78% |
| T1, nb=1 | 5.053 us | 4.365 us | 13.61% |
| T1, nb=4 | 7.310 us | 6.608 us | 9.61% |
| T1, nb=8 | 10.627 us | 9.761 us | 8.15% |
| T8192, nb=8, 256 WGs | 0.503024 ms | 0.466553 ms | 7.25% |

The prefill row is the median of the middle pair from four alternating folds. It is below the 10%
boundary gate. The deep-ring decode sites also miss the gate, so the early-ring wins do not justify
a whole-network run.

Code-object metadata:

| object | VGPR | SGPR | VGPR spill | SGPR spill | private | wave |
|---|---:|---:|---:|---:|---:|---:|
| T1 current | 134 | 106 | 0 | 6 | 0 B | 64 |
| T1 f32 mix | 132 | 106 | 0 | 4 | 0 B | 64 |
| T8192 current | 129 | 106 | 0 | 41 | 0 B | 64 |
| T8192 f32 mix | 131 | 106 | 0 | 42 | 0 B | 64 |

Both objects retain the interpreter's 147,464-byte LDS envelope, so LDS already fixes occupancy at
one workgroup per CU. A forced exact-width one-shot specialization was also screened and spilled
160 B/thread at T1 and 240 B/thread at T8192; a separate segment object does not rescue this route.

## Conclusion

The semantic correction is real but saves only 7.25% at the prefill boundary and 8.15% at the
deepest decode boundary, while retaining scalar spills. The prior norm-fold regression is not
repeated—the candidate is faster—but it fails the explicit promotion margin. No production knob,
dispatcher branch, object recipe, or TP8 run was added.
