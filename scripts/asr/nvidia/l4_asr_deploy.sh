#!/usr/bin/env bash
# Install the L4 ASR set into /opt/plow-asr and (re)start plow-asr.service. <stage> holds the
# three recipe builds:
#   <stage>/qwen3-asr          campaign.py build recipes/infervisor/qwen3-asr/sm89-l4-tp1.toml
#   <stage>/qwen3-asr-0.6b     campaign.py build recipes/infervisor/qwen3-asr-0.6b/sm89-l4-tp1.toml
#   <stage>/nemotron-3.5-asr   scripts/asr/nvidia/nemotron_l4_build.sh
#
#   scripts/asr/nvidia/l4_asr_deploy.sh <stage> <plowrt binary> [libcublasLt.so.13]
set -euo pipefail
stage=$(realpath "$1")
bin=$(realpath "$2")
cublaslt=${3:-}
dest=${PLOW_ASR_ROOT:-/opt/plow-asr}
here=$(cd "$(dirname "$0")" && pwd)

for d in "$stage/qwen3-asr/assets" "$stage/qwen3-asr-0.6b/assets"; do
  [ -f "$d/model.pkt" ] && [ -f "$d/encoder.pkt" ] && [ -f "$d/interp_sm89_speech.cubin" ] || { echo "$d incomplete" >&2; exit 2; }
done
for f in nemotron.pkt interp_sm89_speech.cubin tokenizer.q8_0.gguf; do
  [ -f "$stage/nemotron-3.5-asr/$f" ] || { echo "$stage/nemotron-3.5-asr/$f missing" >&2; exit 2; }
done

sudo systemctl stop plow-asr.service 2>/dev/null || true
sudo mkdir -p "$dest/bin" "$dest/lib" "$dest/models"
sudo chown -R "$(id -un)": "$dest"
install -m 755 "$bin" "$dest/bin/plowrt"
[ -n "$cublaslt" ] && install -m 755 "$cublaslt" "$dest/lib/libcublasLt.so.13"
[ -f "$dest/lib/libcublasLt.so.13" ] || { echo "$dest/lib/libcublasLt.so.13 missing: pass it as the 3rd argument" >&2; exit 2; }
for m in qwen3-asr qwen3-asr-0.6b; do
  rm -rf "$dest/models/$m.new"
  # -L: the build links `checkpoint` to the HF dir; production gets its own copy.
  cp -aL "$stage/$m/assets" "$dest/models/$m.new"
done
rm -rf "$dest/models/nemotron-3.5-asr.new"
mkdir "$dest/models/nemotron-3.5-asr.new"
cp -a "$stage/nemotron-3.5-asr"/{nemotron.pkt,interp_sm89_speech.cubin,tokenizer.q8_0.gguf,commit,sha256} "$dest/models/nemotron-3.5-asr.new/"
for m in qwen3-asr qwen3-asr-0.6b nemotron-3.5-asr; do
  rm -rf "$dest/models/$m"
  mv "$dest/models/$m.new" "$dest/models/$m"
done

sudo install -m 644 "$here/plow-asr.service" /etc/systemd/system/plow-asr.service
sudo systemctl daemon-reload
sudo systemctl enable plow-asr.service
sudo systemctl restart plow-asr.service
