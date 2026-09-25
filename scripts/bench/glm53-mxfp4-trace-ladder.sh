#!/usr/bin/env bash
# Rank-0 device traces of one GLM-5.3 MXFP4 TP8 packet at every prefill and decode rung, for
# per-op measured-vs-roofline (scripts/campaign/op_roofline.py). One 8-GPU queue job:
#   gpuq.py submit glm53-trace 8 glm53-mxfp4-trace-ladder.sh PLOWRT PACKET OBJ CKPT PROMPTS OUT
# PROMPTS holds p<T>.ids (comma-separated ids, >= T tokens). Instrumented: diagnostic timing.
set -uo pipefail
rt=${1:?plowrt}; pkt=${2:?packet dir}; obj=${3:?objects}; ckpt=${4:?checkpoint}
prompts=${5:?prompt dir}; out=$(realpath -m -- "${6:?out}")
mkdir -p "$out"
export PLOW_TP_NO_AUDIT=0 PLOW_TP_AGREE_EVERY=1 PLOW_PREFIX_CACHE=0
bench() {
    local tag=$1; shift
    PLOW_TRACE_RAW="$out/$tag.trace" "$rt" amd-bench --blob "$pkt/model.pkt" --hsaco "$obj" \
        --checkpoint "$ckpt" --tp 8 "$@" > "$out/$tag.log" 2>&1
    echo "$tag rc=$?" | tee -a "$out/status.txt"
}
for t in ${PREFILL_RUNGS:-1024 2048 4096 8192 16384}; do
    n=$t
    [ "$t" -ge 16384 ] && n=$((t - 8))  # leave room for the decode steps under max_ctx
    python3 -c "import sys; ids=open(sys.argv[1]).read().split(','); open(sys.argv[2],'w').write(','.join(ids[:int(sys.argv[3])]))" \
        "$prompts/p16384.ids" "$out/prompt-$t.ids" "$n"
    bench "prefill-T$t" --prompt "@$out/prompt-$t.ids" --steps 2
done
for m in ${DECODE_RUNGS:-2 4 8 16 32 64 128}; do
    bench "decode-M$m" --prompt "@$prompts/p1024.ids" --steps 4 --batched --active-batch "$m"
done
