#!/usr/bin/env bash
# Materialized MLA prefill (P5) gate on TP8: candidate = the served emit + PLOW_MLA_MATERIALIZED_PREFILL=1.
# Every stage is one command the lease owner runs from the repo root; nothing here is default-on.
#
#   bundle  <dir> [emit env...]            emit + gfx950 objects + pinned binaries (showdown_bundle.sh, any worktree)
#   prompts <out>                          fixed prompt set (300/1024/8192/8400/9000 GSM8K tokens) via the pinned vLLM image
#   dump    <arm> <bundle> <out>           TP8 amd-bench per prompt: greedy tokens + rank-0 logits, one manifest per arm
#   tokens  <out>                          (1) continuation exactness probe: first greedy divergence, candidate vs control
#   cases   <out>                          union of both arms' teacher-forced histories, x2 for the vLLM repeat floor
#   oracle  <out>                          (2a) pinned vLLM 0.28 dense logits for those cases (TP8, in the container)
#   compare <out>                          (2b) logit_quality_compare.py + the Q0 verdict for both arms
#   gsm8k   <arm> <bundle> <port>          (3) scripts/bench_gsm8k.sh n=200 on one arm
#   ttft    <cand> <ctl> <out>             (4) 3 alternating 8192->1 folds + one 8192->256 pair, checksums included
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
model=${K3_MODEL:-/home/shaswot/models/Kimi-K3}
ckpt=${K3_CKPT:-/tmp/k3-farm.dvzmZN}
steps=${Q0_STEPS:-16}
export GPU_LEASE_TIMEOUT=${GPU_LEASE_TIMEOUT:-14400}
export GPU_LEASE_DIR=${GPU_LEASE_DIR:-/tmp/gpulease}
image="vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032"
docker_base="docker run --rm --network host --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined --ipc=host --shm-size=32g -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_ROCM_USE_AITER=1 -v $model:/model_weights:ro -v $repo/scripts:/plow-scripts:ro"
cd "$repo"

stage=${1:?stage}; shift
case "$stage" in
bundle)
  B=${1:?bundle dir}; shift
  mkdir -p "$B/bin" "$B/hsaco"
  verify=${PLOW_VERIFY_BIN:-$repo/lean-plow/.lake/build/bin/plow_verify}
  [ -x "$verify" ] || verify=/home/lava/plow/.claude/worktrees/d1-moe-decode-rule/lean-plow/.lake/build/bin/plow_verify
  nix develop -c env PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix K3_FULL=1 PLOW_VERIFY_BIN="$verify" "$@" \
    ./target/release/plowc --hf-dir "$model" --emit devblob --arch gfx950 --gpu mi350 --num-gpus 8 --parallel tp \
    --max-ctx 16384 --n-cu 256 --out "$B/assets" >"$B/emit.log" 2>&1
  ln -sfn "$model" "$B/assets/checkpoint"; ln -sfn "$model/tokenizer.json" "$B/assets/tokenizer.json"
  echo "lean: $(jq -c '.lean | {verified, oracle, reason}' "$B/assets/build.json")"
  echo "tiles: $(jq -c '.tuning' "$B/assets/build.json"); pairing $(jq -r .pairing.hash "$B/assets/build.json"); sha $(sha256sum "$B/assets/model.pkt" | cut -c1-16)"
  grep -o "tunedb: .*ANALYTICAL MODEL" "$B/emit.log" | head -1 || true
  nix develop -c bash -c 'exec cmake "$@" -DPLOW_HSACO_HIPCC="$PLOW_HIPCC" -DPLOW_HSACO_BUNDLER="$PLOW_BUNDLER"' bash \
    -S runtime -B "$B/cmake" -DPLOW_GFX950_HSACO=ON -DPLOW_HSACO_ARCH=gfx950 \
    -DPLOW_HSACO_CONFIG="$B/assets/plow_config.h" -DPLOW_HSACO_DIR="$B/hsaco" \
    -DPLOW_HSACO_DECODE_INVENTORY_PRUNE=ON -DPLOW_HSACO_DECODE_MLA_SEGMENTS=ON -DPLOW_HSACO_KDA_KEY_FACTOR=OFF \
    -DPLOW_HSACO_MOE_DECODE_GROUPED=ON >"$B/config.log" 2>&1
  nix develop -c cmake --build "$B/cmake" --target gfx950_hsaco -j 24 >"$B/build.log" 2>&1
  echo "objects: $(ls "$B/hsaco" | grep -c '\.elf$') (materialized: $(ls "$B/hsaco" | grep -c mla_materializ))"
  cp -f target/release/plowrt "$B/bin/plowrt"; cp -f target/release/plowc "$B/bin/plowc"
  git rev-parse HEAD > "$B/source-head.txt"
  echo BUNDLE_DONE
  ;;
prompts)
  out=${1:?out dir}; mkdir -p "$out"
  gsm=${GSM8K:-$HOME/.cache/plow/gsm8k/test.jsonl}
  [ -s "$gsm" ] || { echo "no GSM8K test split at $gsm (scripts/bench_gsm8k.sh fetches it)"; exit 2; }
  # The quantize shell carries transformers + tiktoken (K3's tokenizer); the default shell has neither.
  nix develop .#quantize -c env HF_HOME="$out/hf-home" python3 scripts/k3_q0_oracle.py prompts --tokenizer "$model" --gsm8k "$gsm" \
    --lengths "${Q0_LENGTHS:-300,1024,8192,8400,9000}" --output "$out/prompts.json"
  ;;
dump)
  arm=${1:?arm}; B=${2:?bundle}; out=${3:?out dir}
  mkdir -p "$out/$arm"
  python3 - "$out/prompts.json" "$out/$arm" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for p in d["prompts"]:
    open(f"{sys.argv[2]}/{p['id']}.ids", "w").write(",".join(map(str, p["ids"])))
PY
  perf-data/tools/gpulease -n 8 "q0-$arm" bash -c "
    set -e
    for f in $out/$arm/*.ids; do
      id=\$(basename \"\$f\" .ids)
      nix develop -c env RUST_LOG=warn $B/bin/plowrt --amd-tp-no-audit amd-bench --blob $B/assets/model.pkt --hsaco $B/hsaco \
        --checkpoint $ckpt --tp 8 --steps $steps --prompt \"\$(cat \"\$f\")\" --dump-logits $out/$arm/\$id >$out/$arm/\$id.stdout 2>$out/$arm/\$id.stderr
      python3 scripts/plow_logit_manifest.py --name $arm-\$id --prompt \"\$f\" --stdout $out/$arm/\$id.stdout \
        --logits-dir $out/$arm/\$id --output $out/$arm/\$id.manifest.json
      echo \"\$id: \$(grep -o 'prefill:.*' $out/$arm/\$id.stdout | cut -c1-80)\"
    done"
  python3 scripts/k3_q0_oracle.py merge --name "$arm" --output "$out/$arm.json" "$out/$arm"/*.manifest.json
  ;;
tokens)
  out=${1:?out dir}
  python3 scripts/k3_q0_oracle.py tokens --left "$out/${CAND:-materialized}.json" --right "$out/${CTL:-absorbed}.json"
  ;;
cases)
  out=${1:?out dir}
  python3 scripts/k3_q0_oracle.py cases --repeats 2 --output "$out/cases.json" "$out/${CAND:-materialized}.json" "$out/${CTL:-absorbed}.json"
  ;;
oracle)
  out=${1:?out dir}
  sg docker -c "perf-data/tools/gpulease -n 8 q0-vllm $docker_base -v $out:$out --entrypoint python3 $image \
    /plow-scripts/vllm_logit_oracle.py --model /model_weights --cases $out/cases.json --output $out/vllm --tp 8 \
    --trust-remote-code --max-num-batched-tokens 4096"
  ;;
compare)
  out=${1:?out dir}
  # numpy lives in the quantize shell only.
  nix develop .#quantize -c python3 scripts/logit_quality_compare.py --reference "$out/vllm/manifest.json" \
    --candidate "$out/${CAND:-materialized}.json" --candidate "$out/${CTL:-absorbed}.json" \
    --repeat-floor-multiplier 2 --output "$out/q0-report.json" | tail -20
  python3 scripts/k3_q0_oracle.py verdict --report "$out/q0-report.json"
  ;;
gsm8k)
  arm=${1:?arm}; B=${2:?bundle}; port=${3:?port}
  ln -sfn "$B/hsaco" "$B/assets/hsaco"
  N=${N:-200} PLOWRT_BIN="$B/bin/plowrt" PLOW_CHECKPOINT="$ckpt" PLOW_HSACO="$B/hsaco" \
    perf-data/tools/gpulease -n 8 "gsm8k-$arm" scripts/bench_gsm8k.sh "$B/assets" "$port" auto
  ;;
ttft)
  cand=${1:?candidate bundle}; ctl=${2:?control bundle}; out=${3:?out dir}; mkdir -p "$out"
  bench() { local B=$1 tag=$2 outlen=$3
    perf-data/tools/gpulease -n 8 "mlamat-$tag" nix develop -c env RUST_LOG=info \
      "$B/bin/plowrt" --rt-checkpoint "$ckpt" --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
      bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
      --concurrency 1 --requests 3 --warmup-requests 1 --output-len "$outlen" >"$out/bench-$tag.log" 2>&1
    echo "[$(date +%T)] rc=$? $tag: $(python3 docs/k3-mi355x-20260904/scripts/bench_fields.py "$out/bench-$tag.log")"
  }
  bench "$cand" m1 1; bench "$ctl" c1 1; bench "$ctl" c2 1; bench "$cand" m2 1; bench "$cand" m3 1; bench "$ctl" c3 1
  bench "$cand" m256 256; bench "$ctl" c256 256
  echo TTFT_GATE_DONE
  ;;
*) echo "unknown stage $stage"; exit 2 ;;
esac
