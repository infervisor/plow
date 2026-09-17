#!/usr/bin/env bash
# plowbench — the shared serve-and-measure harness. SOURCE this; do not execute it.
#
#   source "$(git rev-parse --show-toplevel)/scripts/bench/plowbench.sh"
#
# WHY THIS EXISTS. Every campaign probe re-implemented the same six things — preflight, port
# choice, server launch, readiness poll, `vllm bench serve` invocation, result parsing — and each
# re-implementation reintroduced a bug that had already been paid for once. The failures this file
# encodes are all real, each one having cost at least one leased GPU run:
#
#   * `--result-dir X --result-filename bench.json` does NOT always write `X/bench.json`; it can
#     write `X/main/bench.json`. A summary that looks in one place prints "(missing)" after a
#     20-minute run. See pb_result.
#   * `build_gfx942.sh` does not emit the pinned vendor `.co` kernels. An object dir built from
#     scratch loads fine right up to the first MLA decode segment, then dies with
#     "mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co: No such file or directory". Three GPU slots.
#     See pb_check_objects.
#   * `target/release/plowrt` is SHARED. A concurrent `cargo build -p plowrt` with no features
#     replaces it mid-run with a binary that has no `hsa`, and it does NOT fail — it serves from
#     the CPU reference interpreter, i.e. fluent garbage, ready in 2 s instead of a 12 s weight
#     upload. See pb_check_plowrt.
#   * `PLOW_PREFILL_SEG_TIMING=1` silently disables segment-major and forces an all-rank drain per
#     segment (3.7x inflation). One rung-width experiment was scored and written up before anyone
#     noticed one arm had it set and the other did not. See pb_hazard_env.
#   * Benching several rungs against one warmed server and then comparing the numbers to a
#     single-rung run is invalid: the same 8192 rung read 952 vs 586 depending on position.
#     pb_bench stamps arm order into the result so a later reader can see it.
#
# Everything here is CPU-only except pb_serve_start. Nothing here leases a GPU — go through the
# queue (`submit.sh`) for that, as the campaign rules require.

[ -n "${PB_SOURCED:-}" ] && return 0
PB_SOURCED=1

PB_FAIL=0
PB_WARN=0
pb_ok()   { printf '  ok    %s\n' "$*"; }
pb_warn() { printf '  WARN  %s\n' "$*"; PB_WARN=$((PB_WARN + 1)); }
pb_bad()  { printf '  FAIL  %s\n' "$*"; PB_FAIL=$((PB_FAIL + 1)); }
pb_info() { printf '        %s\n' "$*"; }

# ---------------------------------------------------------------- environment

# The pinned vendor code objects. `build_gfx942.sh` compiles the plow kernels and the adapters but
# NOT these: they are prebuilt AITER binaries copied in. A set missing them looks complete.
PB_VENDOR_CO="fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co
fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co
fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_psx_64x256.co
mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co
mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co"

# Adapters that load the above, plus the ported Tensile set. An object dir without glm_lt_*.elf
# serves until the first GemmLtPf segment.
PB_REQUIRED_ELF="dsa_tp_adapter_gfx942.elf
glm_fold_adapter.elf
glm_fold_lt_gfx942.elf
glm_lt_gfx942.elf"

# Env vars that CHANGE WHAT IS MEASURED and are easy to leave set from a previous probe.
# Each entry: NAME:why.
PB_HAZARD_ENV="PLOW_PREFILL_SEG_TIMING:disables segment-major, ~3.7x inflation, not a latency number
PLOW_TUNEDB:selects measured GEMM tiles; measured tiles are SLOWER at every rung (5-8 ms)
PLOW_HSACO_LOWRUNG:swaps in a different object tier for the narrow rungs
PLOW_AMD_DECODE_MIN_RUNG:changes which decode program runs; default 8 is the measured optimum
PLOW_GLM_ROWBAND:the row-band serve gate; arms must agree on it
PLOW_TICK_LOG:adds per-tick logging; fine for instrumented arms, not for a clean number"

pb_require_nix() {
    if [ -z "${ROCM_PATH:-}" ]; then
        pb_bad "not inside 'nix develop' (ROCM_PATH unset) — build and serve tasks need it"
        return 1
    fi
    pb_ok "nix dev shell, ROCM_PATH=$ROCM_PATH"
}

# Refuse a run whose environment silently redefines the measurement.
# PB_ALLOW_HAZARD="PLOW_TICK_LOG PLOW_GLM_ROWBAND" to opt in deliberately.
pb_hazard_env() {
    local allow=" ${PB_ALLOW_HAZARD:-} "
    local line name why
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        name=${line%%:*}; why=${line#*:}
        [ -n "${!name:-}" ] || continue
        case "$allow" in
            *" $name "*) pb_info "$name=${!name} (declared via PB_ALLOW_HAZARD)" ;;
            *) pb_warn "$name=${!name} is set — $why" ;;
        esac
    done <<< "$PB_HAZARD_ENV"
}

pb_check_plowrt() {
    local rt="${1:?plowrt path}"
    [ -x "$rt" ] || { pb_bad "no plowrt at $rt"; return 1; }
    # A binary without the hsa feature serves from the CPU interpreter and never says so.
    if ! grep -aq 'dec_inflight_enq' "$rt"; then
        pb_warn "$rt has no TICK instrument — probably an old or feature-less build"
    fi
    case "$rt" in
        */target/release/plowrt)
            pb_warn "using the SHARED target/release/plowrt — a concurrent 'cargo build -p plowrt'"
            pb_info "can replace it mid-run with a no-hsa binary that serves fluent garbage."
            pb_info "For anything you will publish a number from, copy it first:"
            pb_info "  cp $rt /workspace/\$USER-plowrt && export PLOWRT_BIN=/workspace/\$USER-plowrt" ;;
    esac
    pb_ok "plowrt $rt"
}

pb_check_assets() {
    local assets="${1:?assets dir}" want="${2:-}"
    [ -e "$assets/model.pkt" ] || { pb_bad "no model.pkt in $assets"; return 1; }
    local got; got=$(sha256sum < "$assets/model.pkt" | cut -c1-16)
    if [ -n "$want" ] && [ "$got" != "$want" ]; then
        pb_bad "packet is $got, expected $want — wrong assets dir"
        return 1
    fi
    [ -e "$assets/build.json" ] || pb_warn "no build.json beside the packet (compile not reproducible)"
    pb_ok "packet $assets/model.pkt sha=$got"
}

# The check that pays for this whole file.
pb_check_objects() {
    local dir="${1:?object dir}" miss=0 f
    [ -d "$dir" ] || { pb_bad "no object dir $dir"; return 1; }
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -e "$dir/$f" ] || { pb_bad "object dir lacks pinned vendor kernel $f"; miss=$((miss + 1)); }
    done <<< "$PB_VENDOR_CO"
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -e "$dir/$f" ] || { pb_bad "object dir lacks $f"; miss=$((miss + 1)); }
    done <<< "$PB_REQUIRED_ELF"
    if [ "$miss" -gt 0 ]; then
        pb_info "build_gfx942.sh does NOT emit the vendor .co kernels. Copy them from a known-good"
        pb_info "serving set; they are byte-identical across sets built from the same vendor drop."
        return 1
    fi
    # Verify what IS present against the manifest, if one shipped with the set.
    if [ -e "$dir/MANIFEST.sha256" ]; then
        local bad
        bad=$( cd "$dir" && sha256sum -c --ignore-missing --quiet MANIFEST.sha256 2>&1 | head -3 )
        if [ -n "$bad" ]; then
            pb_bad "object hashes disagree with MANIFEST.sha256:"
            printf '        %s\n' "$bad"
            return 1
        fi
        pb_ok "objects in $dir ($(ls "$dir" | wc -l) files, manifest verified)"
    else
        pb_warn "no MANIFEST.sha256 in $dir — contents unverified"
    fi
}

pb_check_vllm() {
    local v="${PB_VLLM:-/app/plow/build-gemma31/vllm-python}"
    [ -x "$v" ] || { pb_bad "no vLLM client at $v (set PB_VLLM)"; return 1; }
    local lib="${PB_VLLM_ROCM_LIB:-/opt/rocm/core-7.14/lib}"
    [ -d "$lib" ] || { pb_bad "VLLM_ROCM_LIB dir missing: $lib"; return 1; }
    pb_ok "vLLM client $v (ROCm lib $lib)"
}

# ---------------------------------------------------------------- serving

# Hand-picked ports collide when two probes overlap. Ask the kernel instead.
pb_free_port() {
    python3 - <<'PYEOF'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PYEOF
}

# pb_serve_start <plowrt> <assets> <objdir> <port> <logfile> [timeout-s]
# setsid so the whole server tree can be torn down by process group: `nix develop -c` execs a
# shell that forks plowrt, so killing the pid we waited on can leave the real server holding cards.
pb_serve_start() {
    local rt="$1" assets="$2" objdir="$3" port="$4" log="$5" tmo="${6:-5400}"
    PLOW_HSACO="$objdir" setsid timeout -s TERM "$tmo" \
        "$rt" serve --assets "$assets" --port "$port" > "$log" 2>&1 &
    PB_SERVER_PID=$!
    PB_SERVER_PORT=$port
    PB_SERVER_LOG=$log
}

# pb_serve_wait [seconds] — returns 1 and prints the tail if the server died or never answered.
pb_serve_wait() {
    local secs="${1:-900}" i
    for i in $(seq 1 "$secs"); do
        curl -fsS "http://127.0.0.1:$PB_SERVER_PORT/v1/models" > /dev/null 2>&1 && return 0
        kill -0 "$PB_SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    echo "NOT READY — server did not answer in ${secs}s. last 8 lines:"
    tail -8 "$PB_SERVER_LOG" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-170 | sed 's/^/   /'
    return 1
}

pb_model_id() {
    curl -fsS "http://127.0.0.1:$PB_SERVER_PORT/v1/models" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}

pb_serve_stop() {
    [ -n "${PB_SERVER_PID:-}" ] || return 0
    kill -TERM -- "-$PB_SERVER_PID" 2>/dev/null || kill -TERM "$PB_SERVER_PID" 2>/dev/null
    local i
    for i in $(seq 1 120); do kill -0 "$PB_SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$PB_SERVER_PID" 2>/dev/null || kill -KILL "$PB_SERVER_PID" 2>/dev/null
    PB_SERVER_PID=
}

# ---------------------------------------------------------------- benching

# pb_bench <resdir> <tag> <model> <conc> <nprompts> <isl> <osl> [extra args...]
# One canonical invocation. PB_ARM_N records the order arms ran in, because position against a
# warming server moves the number and a later reader must be able to see that.
PB_ARM_N=0
pb_bench() {
    local res="$1" tag="$2" model="$3" conc="$4" np="$5" isl="$6" osl="$7"; shift 7
    local v="${PB_VLLM:-/app/plow/build-gemma31/vllm-python}"
    local lib="${PB_VLLM_ROCM_LIB:-/opt/rocm/core-7.14/lib}"
    local tokz="${PB_TOKENIZER:-zai-org/GLM-5.3}"
    PB_ARM_N=$((PB_ARM_N + 1))
    mkdir -p "$res"
    echo "$PB_ARM_N" > "$res/$tag.armorder"
    timeout -s TERM "${PB_BENCH_TIMEOUT:-3000}" env VLLM_ROCM_LIB="$lib" "$v" \
        -m vllm.entrypoints.cli.main bench serve \
        --backend vllm --host 127.0.0.1 --port "$PB_SERVER_PORT" --model "$model" \
        --tokenizer "$tokz" --trust-remote-code --dataset-name random \
        --seed "${PB_SEED:-8193}" --num-prompts "$np" \
        --random-input-len "$isl" --random-output-len "$osl" --random-range-ratio 0 \
        --max-concurrency "$conc" --request-rate "${PB_RATE:-inf}" --ignore-eos \
        --percentile-metrics ttft,tpot,itl --save-result --save-detailed \
        --result-dir "$res/$tag" --result-filename bench.json \
        "$@" > "$res/$tag.bench.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || echo "   arm $tag: bench exited $rc (see $res/$tag.bench.log)"
    return $rc
}

# pb_result <resdir> <tag> — print the result JSON path, wherever the client decided to put it.
pb_result() {
    local res="$1" tag="$2" p
    for p in "$res/$tag/bench.json" "$res/$tag/main/bench.json"; do
        [ -e "$p" ] && { echo "$p"; return 0; }
    done
    p=$(find "$res/$tag" -name '*.json' -type f 2>/dev/null | head -1)
    [ -n "$p" ] && { echo "$p"; return 0; }
    return 1
}
