#!/usr/bin/env bash
# bench_ib.sh — build (once) and run the pinned inference-benchmarker binary.
#
# The huggingface/inference-benchmarker rev below is the campaign's benchmark
# harness (B2 concurrency sweeps). It MUST match the rev pinned in
# tools/bench/Cargo.toml (and therefore Cargo.lock); the binary is built from
# that same rev with `cargo install --git … --rev …` into target/tools so the
# runs use the tool's own binary directly.
#
# The tool's dep tree needs openssl (reqwest native-tls). The nix dev shell is
# deliberately openssl-free, so we point openssl-sys at the system libssl
# (headers in /usr/include/openssl, lib in /usr/lib/x86_64-linux-gnu) — this
# only affects the benchmark tool, never the workspace artifacts.
#
# Usage: perf-data/bench_ib.sh [inference-benchmarker args…]
set -euo pipefail

IB_REV=bad4f947ef5f34ef264d2451439ab90cf7cbd65c
IB_GIT=https://github.com/huggingface/inference-benchmarker

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/target/tools"
BIN="$TOOLS/bin/inference-benchmarker"
STAMP="$TOOLS/.inference-benchmarker.rev"

if [ ! -x "$BIN" ] || [ "$(cat "$STAMP" 2>/dev/null || true)" != "$IB_REV" ]; then
    echo "building inference-benchmarker @ $IB_REV -> $BIN" >&2
    # Debian splits the openssl headers (opensslconf.h lives in the arch
    # include dir), and a global -I would poison other crates' C builds with
    # system glibc headers, so merge the two openssl header dirs into a
    # private include dir and hand only that to openssl-sys.
    if [ -z "${OPENSSL_INCLUDE_DIR:-}" ]; then
        OPENSSL_INCLUDE_DIR="$TOOLS/openssl-include"
        mkdir -p "$OPENSSL_INCLUDE_DIR/openssl"
        ln -sf /usr/include/openssl/* "$OPENSSL_INCLUDE_DIR/openssl/"
        ln -sf /usr/include/x86_64-linux-gnu/openssl/* "$OPENSSL_INCLUDE_DIR/openssl/"
    fi
    OPENSSL_NO_VENDOR=1 \
    OPENSSL_LIB_DIR="${OPENSSL_LIB_DIR:-/usr/lib/x86_64-linux-gnu}" \
    OPENSSL_INCLUDE_DIR="$OPENSSL_INCLUDE_DIR" \
        cargo install --git "$IB_GIT" --rev "$IB_REV" --root "$TOOLS" \
        --force inference-benchmarker >&2
    echo "$IB_REV" > "$STAMP"
fi

# The binary is linked in the nix dev shell against the system libssl, which
# is not on the nix loader's default search path. Expose ONLY libssl/libcrypto
# (a whole /usr/lib/x86_64-linux-gnu on LD_LIBRARY_PATH would shadow the nix
# glibc and crash with GLIBC_PRIVATE symbol errors).
mkdir -p "$TOOLS/lib"
ln -sf /usr/lib/x86_64-linux-gnu/libssl.so.3 /usr/lib/x86_64-linux-gnu/libcrypto.so.3 "$TOOLS/lib/"
export LD_LIBRARY_PATH="$TOOLS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$BIN" "$@"
