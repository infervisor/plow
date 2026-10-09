#!/usr/bin/env bash
# plow-voice.sh: install, check and run the plow H100 voice kit (plowrt + model bundles).
# Adapted from scripts/asr/nvidia/l4_asr_deploy.sh (the L4 /opt/plow-asr deploy) for the
# multi-model H100 kit. Every path is relative to the kit this script sits in.
#
#   deploy/plow-voice.sh hostcheck [config]          GPU, driver, glibc, disk, port, python
#   deploy/plow-voice.sh preflight [--full]          kit pairing hashes (--full: every file in SHA256SUMS)
#   sudo deploy/plow-voice.sh install [options]      copy the kit, write the config, install + start the unit
#        --prefix DIR   install root (default /opt/plow-voice; the kit goes to DIR/<kit name>, DIR/current links it)
#        --config FILE  config path (default /etc/plow-voice/plow-voice.conf; kept if it exists)
#        --user NAME    service user (default plow-voice, created as a system user if missing)
#        --no-systemd   copy + config only; start it with `plow-voice.sh run`
#        --no-start     install the unit but do not start it
#   deploy/plow-voice.sh run [config]                foreground server (the unit's ExecStart)
#   deploy/plow-voice.sh print-cmd [config]          the plowrt command `run` would exec
#   deploy/plow-voice.sh wait-ready [config] [secs]  poll /health until 200 (default 900 s)
#
# Config: KEY=VALUE lines (deploy/plow-voice.conf.sample documents every key). Default path:
# $PLOW_VOICE_CONFIG, else /etc/plow-voice/plow-voice.conf, else the kit's sample.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=$(cd "$HERE/.." && pwd)
DEFAULT_CONFIG=/etc/plow-voice/plow-voice.conf

die() { echo "plow-voice: $*" >&2; exit 2; }
note() { echo "plow-voice: $*" >&2; }

config_path() {
    local c=${1:-${PLOW_VOICE_CONFIG:-}}
    [ -n "$c" ] || { [ -r "$DEFAULT_CONFIG" ] && c=$DEFAULT_CONFIG || c=$HERE/plow-voice.conf.sample; }
    [ -r "$c" ] || die "config $c not readable"
    echo "$c"
}

# KEY=VALUE parser (no shell evaluation; optional surrounding quotes stripped). Sets CFG_<KEY>.
load_kv() {
    local file=$1 prefix=$2 line key val
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%$'\r'}
        [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
        [[ $line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || die "$file: bad line: $line"
        key=${BASH_REMATCH[1]}; val=${BASH_REMATCH[2]}
        val=${val%"${val##*[![:space:]]}"}
        if [[ $val =~ ^\"(.*)\"$ || $val =~ ^\'(.*)\'$ ]]; then val=${BASH_REMATCH[1]}; fi
        printf -v "${prefix}${key}" '%s' "$val"
    done < "$file"
}

load_config() {
    local cfg; cfg=$(config_path "${1:-}")
    CFG_PROFILE=voice-core CFG_BIND=127.0.0.1 CFG_PORT=8000 CFG_API_KEYS= CFG_MODELS= CFG_LIVE_CTX= \
        CFG_EXTRA_ARGS= CFG_LOG_LEVEL=info CFG_LOG_FILE= CFG_DRAIN_TIMEOUT_MS=30000 \
        CFG_ASR_REQUEST_TIMEOUT_MS=120000 CFG_SESSION_TTL_MS=60000 CFG_HTTP_MAX_CONNECTIONS=4096 \
        CFG_CUDA_VISIBLE_DEVICES= CFG_LIBCUDA=
    load_kv "$cfg" CFG_
    local prof=$KIT/deploy/profiles/$CFG_PROFILE.profile
    [ -r "$prof" ] || die "profile '$CFG_PROFILE' not found ($(ls "$KIT/deploy/profiles" | sed 's/\.profile$//' | tr '\n' ' '))"
    PROF_MODELS= PROF_LIVE_CTX= PROF_ENV= PROF_ARGS=
    load_kv "$prof" PROF_
    MODELS=${CFG_MODELS:-$PROF_MODELS}
    LIVE_CTX=${CFG_LIVE_CTX:-$PROF_LIVE_CTX}
    [ -n "$MODELS" ] || die "profile $CFG_PROFILE lists no MODELS"
    CONFIG_FILE=$cfg
}

model_dir() {
    local m=$1
    for d in "$KIT/models/$m" "$KIT/experimental/$m"; do [ -d "$d" ] && { echo "$d"; return; }; done
    die "model '$m' is not in this kit ($(ls "$KIT/models" | tr '\n' ' '))"
}

build_cmd() {
    CMD=("$KIT/plowrt/plowrt" serve)
    local m d
    for m in $MODELS; do
        d=$(model_dir "$m")
        if [ -f "$d/nemotron.pkt" ]; then
            CMD+=(--asr-packet "$m=$d/nemotron.pkt,tokenizer=$d/tokenizer.q8_0.gguf,backend=cuda")
        elif [ -f "$d/silero_vad.pkt" ]; then
            CMD+=(--asr-vad-packet "$d/silero_vad.pkt")
        else
            CMD+=(--assets "$d")
        fi
    done
    CMD+=(--bind "$CFG_BIND" --port "$CFG_PORT" --session-ttl-ms "$CFG_SESSION_TTL_MS"
          --asr-request-timeout-ms "$CFG_ASR_REQUEST_TIMEOUT_MS" --http-max-connections "$CFG_HTTP_MAX_CONNECTIONS")
    # shellcheck disable=SC2206
    CMD+=($PROF_ARGS $CFG_EXTRA_ARGS)
    # A fatal device fault exits 1 (after <= 5 s of drain) so systemd restarts the server.
    ENVS=(PATH=/usr/bin:/bin HOME="${HOME:-/}" RUST_LOG="$CFG_LOG_LEVEL" PLOW_DRAIN_TIMEOUT_MS="$CFG_DRAIN_TIMEOUT_MS"
          PLOW_EXIT_ON_ENGINE_DEATH=1)
    # The packets' compiler receipts are re-checked at load by the verifier they were issued with.
    [ -x "$KIT/plowrt/plow_verify" ] && ENVS+=(PLOW_VERIFY_BIN="$KIT/plowrt/plow_verify")
    [ -n "$LIVE_CTX" ] && ENVS+=(PLOW_LIVE_CTX_MODELS="$LIVE_CTX")
    [ -n "$CFG_API_KEYS" ] && ENVS+=(PLOW_API_KEYS="$CFG_API_KEYS")
    [ -n "$CFG_CUDA_VISIBLE_DEVICES" ] && ENVS+=(CUDA_VISIBLE_DEVICES="$CFG_CUDA_VISIBLE_DEVICES")
    [ -n "$CFG_LIBCUDA" ] && ENVS+=(PLOW_LIBCUDA="$CFG_LIBCUDA")
    # shellcheck disable=SC2206
    [ -n "$PROF_ENV" ] && ENVS+=($PROF_ENV)
    return 0
}

redact() { sed -E 's/(PLOW_API_KEYS=)[^ ]*/\1<redacted>/'; }

cmd_print() {
    load_config "${1:-}"; build_cmd
    echo "env -i ${ENVS[*]} ${CMD[*]}" | redact
}

cmd_run() {
    load_config "${1:-}"; build_cmd
    note "config $CONFIG_FILE, profile $CFG_PROFILE: $MODELS"
    if [ -n "$CFG_LOG_FILE" ]; then
        mkdir -p "$(dirname "$CFG_LOG_FILE")"
        exec >>"$CFG_LOG_FILE" 2>&1
    fi
    echo "plow-voice: exec env -i ${ENVS[*]} ${CMD[*]}" | redact
    exec env -i "${ENVS[@]}" "${CMD[@]}"
}

cmd_wait_ready() {
    load_config "${1:-}"
    local secs=${2:-900} host=$CFG_BIND i
    [ "$host" = 0.0.0.0 ] && host=127.0.0.1
    [ "$host" = "::" ] && host=::1
    [[ $host == *:* ]] && host="[$host]"
    for ((i = 0; i < secs; i++)); do
        if curl -fsS -o /dev/null --max-time 2 "http://$host:$CFG_PORT/health" 2>/dev/null; then
            note "ready on $host:$CFG_PORT after ${i}s"; return 0
        fi
        sleep 1
    done
    die "not ready after ${secs}s (journalctl -u plow-voice, or the run log)"
}

# ---------------------------------------------------------------- checks
FAILS=0 WARNS=0
ok() { echo "  ok    $*"; }
warn() { echo "  WARN  $*"; WARNS=$((WARNS + 1)); }
fail() { echo "  FAIL  $*"; FAILS=$((FAILS + 1)); }

vercmp_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

cmd_hostcheck() {
    echo "host check (kit $KIT)"
    local cfg_port=8000 cfg_bind=127.0.0.1
    if cfg=$(config_path "${1:-}" 2>/dev/null); then
        load_config "$cfg"; cfg_port=$CFG_PORT; cfg_bind=$CFG_BIND
    fi
    [ "$(uname -m)" = x86_64 ] && ok "x86_64" || fail "arch $(uname -m): the kit is x86_64 only"
    local glibc; glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
    local need; need=$(sed -n 's/.*"min_glibc": *"GLIBC_\([0-9.]*\)".*/\1/p' "$KIT/plowrt/BUILD.json" 2>/dev/null | head -1)
    need=${need:-2.34}
    if [ -n "$glibc" ] && vercmp_ge "$glibc" "$need"; then ok "glibc $glibc (>= $need)"; else fail "glibc ${glibc:-unknown} < $need required by plowrt"; fi
    if ! command -v nvidia-smi >/dev/null; then
        fail "nvidia-smi not found: install the NVIDIA driver"
    else
        local q; q=$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null) || q=
        [ -n "$q" ] || fail "nvidia-smi cannot query a GPU"
        local idx name mem drv h100=0
        while IFS=, read -r idx name mem drv; do
            name=$(echo "$name" | xargs); mem=$(echo "$mem" | xargs); drv=$(echo "$drv" | xargs)
            if [[ $name == *H100* ]] && [ "${mem:-0}" -ge 79000 ]; then
                ok "GPU $idx: $name, $mem MiB"; h100=1
            else
                warn "GPU $idx: $name, $mem MiB (the bundles target H100 80GB, sm_90a, 132 SMs)"
            fi
            if vercmp_ge "$drv" 575; then ok "driver $drv (tested 595.91.07)"
            elif vercmp_ge "$drv" 525.60; then warn "driver $drv: CUDA 12 minor-version compatibility only; tested 595.91.07, recommended >= 575"
            else fail "driver $drv < 525.60 (CUDA 12.x needs >= 525.60)"; fi
        done <<<"$q"
        [ $h100 = 1 ] || fail "no H100 80GB visible"
        local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | xargs)
        [ "${used:-0}" -lt 2048 ] && ok "GPU 0 free (${used} MiB used)" || warn "GPU 0 already has ${used} MiB in use; the voice profiles need the whole 80 GB"
        local procs; procs=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | head -3)
        [ -z "$procs" ] || warn "GPU processes running: $(echo "$procs" | tr '\n' ';')"
    fi
    if ldconfig -p 2>/dev/null | grep -q 'libcuda.so.1'; then ok "libcuda.so.1 registered with the loader"
    else warn "libcuda.so.1 not in the ldconfig cache; set LIBCUDA=/path/libcuda.so.1 in the config"; fi
    local need_gb; need_gb=$(du -s --apparent-size -BG "$KIT" 2>/dev/null | cut -f1 | tr -d G)
    local free_gb; free_gb=$(df -BG --output=avail /opt 2>/dev/null | tail -1 | tr -d ' G')
    [ -n "$need_gb" ] && ok "kit size ${need_gb} GB; /opt has ${free_gb:-?} GB free (install copies the kit)"
    [ -n "$free_gb" ] && [ -n "$need_gb" ] && [ "$free_gb" -lt "$need_gb" ] && warn "/opt has less free space than the kit; pass --prefix on a larger filesystem"
    local mem_gb; mem_gb=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
    [ "$mem_gb" -ge 64 ] && ok "RAM ${mem_gb} GB" || warn "RAM ${mem_gb} GB: checkpoints are mmap'd and page-cached (>= 64 GB recommended)"
    if (exec 3<>"/dev/tcp/127.0.0.1/$cfg_port") 2>/dev/null; then
        warn "port $cfg_port is already in use on 127.0.0.1"
    else ok "port $cfg_port free (bind $cfg_bind)"; fi
    command -v curl >/dev/null && ok "curl" || fail "curl not found (readiness checks use it)"
    command -v systemctl >/dev/null && ok "systemd $(systemctl --version | head -1 | awk '{print $2}')" || warn "no systemd: use 'plow-voice.sh run' under your supervisor"
    if command -v python3 >/dev/null; then
        local pv; pv=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
        vercmp_ge "$pv" 3.9 && ok "python $pv (clients/eval/perf only; the server needs none)" || warn "python $pv < 3.9 (clients only)"
    else warn "python3 not found (clients/eval/perf only; the server needs none)"; fi
    echo "hostcheck: $FAILS failed, $WARNS warnings"
    [ $FAILS = 0 ]
}

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

cmd_preflight() {
    echo "preflight (kit $KIT)"
    [ -f "$KIT/PAIRING.txt" ] || fail "PAIRING.txt missing"
    local want got
    want=$(awk '$2=="plowrt"{print $1}' "$KIT/plowrt/plowrt.sha256")
    got=$(sha_of "$KIT/plowrt/plowrt")
    [ "$want" = "$got" ] && ok "plowrt ${got:0:12}" || fail "plowrt sha256 $got != plowrt.sha256 $want"
    want=$(awk '$2=="libcublasLt.so.12"{print $1}' "$KIT/plowrt/plowrt.sha256")
    got=$(sha_of "$KIT/plowrt/libcublasLt.so.12")
    [ "$want" = "$got" ] && ok "libcublasLt.so.12 ${got:0:12}" || fail "libcublasLt.so.12 sha256 mismatch"
    want=$(awk '$2=="plow_verify"{print $1}' "$KIT/plowrt/plowrt.sha256")
    if [ -n "$want" ]; then
        got=$(sha_of "$KIT/plowrt/plow_verify")
        [ "$want" = "$got" ] && ok "plow_verify ${got:0:12}" || fail "plow_verify sha256 mismatch"
    fi
    # Pairing: every packet qualified with this plowrt (PAIRING.txt), by sha256.
    local m f sha
    want=$(sed -n 's/^# .*qualified with plowrt \([0-9a-f]*\).*/\1/p' "$KIT/PAIRING.txt" 2>/dev/null)
    [ "$want" = "$(sha_of "$KIT/plowrt/plowrt")" ] && ok "PAIRING.txt names this plowrt" || fail "PAIRING.txt is for plowrt '${want:0:12}'"
    while IFS=' ' read -r m f sha; do
        [ -n "$m" ] && [ "${m:0:1}" != "#" ] || continue
        if [ ! -f "$KIT/$f" ]; then fail "$m: $f missing"; continue; fi
        got=$(sha_of "$KIT/$f")
        [ "$got" = "$sha" ] && ok "$m: $(basename "$f") ${sha:0:12}" || fail "$m: $f sha256 $got != paired $sha"
    done < "$KIT/PAIRING.txt"
    if [ "${1:-}" = --full ]; then
        echo "  verifying every file in SHA256SUMS (a few minutes)..."
        if (cd "$KIT" && sha256sum --quiet -c SHA256SUMS); then ok "SHA256SUMS: all files match"
        else fail "SHA256SUMS: mismatches above"; fi
    else
        echo "  (quick mode: pass --full to verify every file against SHA256SUMS)"
    fi
    echo "preflight: $FAILS failed, $WARNS warnings"
    [ $FAILS = 0 ]
}

# ---------------------------------------------------------------- install
cmd_install() {
    local prefix=/opt/plow-voice config=$DEFAULT_CONFIG user=plow-voice systemd=1 start=1
    while [ $# -gt 0 ]; do
        case $1 in
            --prefix) prefix=$2; shift 2 ;;
            --config) config=$2; shift 2 ;;
            --user) user=$2; shift 2 ;;
            --no-systemd) systemd=0; shift ;;
            --no-start) start=0; shift ;;
            *) die "install: unknown option $1" ;;
        esac
    done
    [ "$(id -u)" = 0 ] || die "install needs root (sudo)"
    ( cmd_hostcheck ) || die "host check failed; not installing"
    ( cmd_preflight ) || die "preflight failed; not installing"
    local name; name=$(basename "$KIT")
    local dest=$prefix/$name
    id "$user" >/dev/null 2>&1 || useradd --system --no-create-home --home-dir "$prefix" --shell /usr/sbin/nologin "$user"
    if [ $systemd = 1 ] && systemctl is-active --quiet plow-voice.service; then
        note "stopping plow-voice.service (graceful drain)"
        systemctl stop plow-voice.service
    fi
    mkdir -p "$prefix"
    if [ "$(realpath "$KIT")" != "$(realpath -m "$dest")" ]; then
        rm -rf "$dest.new"
        note "copying the kit to $dest"
        cp -a "$KIT" "$dest.new"
        chmod -R u+w "$dest.new"
        rm -rf "$dest"
        mv "$dest.new" "$dest"
        chmod -R a-w "$dest"
    fi
    ln -sfn "$dest" "$prefix/current"
    mkdir -p "$(dirname "$config")" /var/log/plow-voice
    chown "$user": /var/log/plow-voice
    if [ ! -f "$config" ]; then
        install -m 640 -o root -g "$(id -gn "$user")" "$dest/deploy/plow-voice.conf.sample" "$config"
        note "wrote $config (edit PROFILE, BIND, PORT, API_KEYS; then systemctl restart plow-voice)"
    else
        note "keeping existing $config"
    fi
    if [ $systemd = 0 ]; then
        note "installed to $dest; run: $prefix/current/deploy/plow-voice.sh run $config"
        return 0
    fi
    sed -e "s#@PREFIX@#$prefix#g" -e "s#@CONFIG@#$config#g" -e "s#@USER@#$user#g" \
        "$dest/deploy/plow-voice.service" > /etc/systemd/system/plow-voice.service
    chmod 644 /etc/systemd/system/plow-voice.service
    systemctl daemon-reload
    systemctl enable plow-voice.service
    if [ $start = 1 ]; then
        note "starting plow-voice.service (loads the models; up to 15 min on a cold page cache)"
        systemctl restart plow-voice.service
        systemctl --no-pager --lines=0 status plow-voice.service || true
    fi
}

case ${1:-} in
    hostcheck) shift; cmd_hostcheck "$@" ;;
    preflight) shift; cmd_preflight "$@" ;;
    install) shift; cmd_install "$@" ;;
    run) shift; cmd_run "$@" ;;
    print-cmd) shift; cmd_print "$@" ;;
    wait-ready) shift; cmd_wait_ready "$@" ;;
    -h|--help|help|"") sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown command $1 (try --help)" ;;
esac
