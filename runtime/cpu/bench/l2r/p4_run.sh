#!/usr/bin/env bash
# p4_run.sh <outdir> <tag> <refdir> <path A|B> <batch> [ENV=VAL ...]: one l2r_layer run for the P4 KV matrix (P5: <batch> = L2R_BATCH).
#   path B: one KV copy (KV stays wherever the previous step left it). path A: enough copies that the KV the run
#   rotates through is >= L2R_COLD_MIB (default 1536 MiB, 3.2x the 480 MiB L3), so every step reads it from DRAM.
#   Steps scale with the KV bytes per step so that a run is a few seconds. Output: <outdir>/<tag>.json.
O=$1 tag=$2 ref=$3 path=$4 seqs=$5; shift 5  # seqs = L2R_BATCH rows (L2R_ROWS in the env args)
B=${L2R_BIN:-/tmp/g4c/l2r/l2r_layer}
mkdir -p "$O"
kvmib=$(python3 -c "
import json; m = json.load(open('$ref/meta.json'))
print(max(1, round(2 * m['kv_heads'] * (m['cache_len'] + 1) * m['head_dim'] * 2 * $seqs / 2**20)))")
copies=1
[ "$path" = A ] && copies=$(( (${L2R_COLD_MIB:-1536} + kvmib - 1) / kvmib ))
steps=5000
[ $kvmib -gt 64 ] && steps=2000
[ $kvmib -gt 512 ] && steps=600
[ $kvmib -gt 2048 ] && steps=300
steps=${P4_STEPS:-$steps}  # explicit step count (long stability runs)
env L2R_BCAST=repnt L2R_BATCH=$seqs L2R_KV_COPIES=$copies "$@" $B $ref $steps > "$O/$tag.json" 2> "$O/$tag.err"
echo "$tag kv=${kvmib}MiB copies=$copies steps=$steps rc=$?"
