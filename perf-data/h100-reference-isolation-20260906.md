# Reference isolation

Gemma independent BF16 eager vLLM oracle validates10full rows with5bitexact repeats, exact histories128..132 and262144vocabulary entries. Only checkpoint-declared suppressed IDs258882/258883 are excluded from numerical metrics. Control argmax matches5/5; WG64 matches4/5. Candidate centered error is lower on the prefill row but higher on all four decode histories; raw error is higher on allfive. Neither this comparison nor local operator passes relax the failed native1% full-model gate. The control itself has5.37–13.65% centered error versus this independent reference; matching argmax alone is not proof of numerical parity.

Qwen corrected recurrence replay reconstructs the captured automatic-CP reference exactly. Native adapter output/state/persistent state remain exact against the nonCP library path for both beta arms. The CP/nonCP and beta perturbations are diagnostic differences, not accepted full-model equivalence.

Projection reconstruction still fails, including the compiled native-quant route: QKV relativeL2 .004058 and BA .004081. Same-input native/CUDA quant and same-operand GEMM controls pass. This isolates an unobserved upstream/reference boundary; it does not demonstrate a production kernel defect or qualify independent model quality. Original failures, corrected route evidence and fixed thresholds are preserved.

E3 packet numerical qualification is separate and passed384row comparisons plus12scratch/cast guards. Its two unfiltered sanitizer runs returned99 withtwoAPI500records each; exact host-callsite classification is pending. Timing stays held until memory qualification.
