#!/usr/bin/env bash
# rtx12_build.sh — rebuild plowrt with the nix rust toolchain binaries directly
# (NOT `nix develop`: no cmake/CUDA env, so no glibc/CUDA clash), reusing the
# shared target dir to keep the incremental build small (disk is tight). The nix
# rustc wrapper auto-injects the same glibc/gcc rpath as the committed binary.
set -euo pipefail
export PATH=/nix/store/z480b23kymbmrijrl49246mrzyphli15-cargo-1.95.0/bin:/nix/store/wnhmqix7bippbbzasj29qiyb422g9asg-rustc-wrapper-1.95.0/bin:/usr/bin:/bin:/usr/local/bin
export CARGO_TARGET_DIR=/root/plow/target
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "cargo: $(cargo --version)"
echo "rustc: $(rustc --version)"
cargo build --release -p plowrt --features cuda,hf-tokenizer "$@"
