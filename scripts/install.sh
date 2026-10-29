#!/bin/sh
# Install the released `plowrt` binary from the registry.
#
#   curl -fsSL https://pdist.infervisor.ai/install.sh | sh
#
# Reads `v1/runtime/latest` (or PLOWRT_VERSION), then that release's
# `release.json` for this machine's platform triple, downloads the tarball,
# verifies its sha256 against the `.sha256` the build wrote, and installs
# `plowrt` into PLOWRT_INSTALL_DIR (default: /usr/local/bin when writable,
# else ~/.local/bin). POSIX sh; needs curl or wget, tar, and sha256sum or
# shasum. Published to the bucket root by `plowrt push-runtime`.
set -eu

REGISTRY="${PLOW_REGISTRY:-https://pdist.infervisor.ai}"
case "$REGISTRY" in http://*|https://*|file://*) ;; *) REGISTRY="https://$REGISTRY" ;; esac
REGISTRY="${REGISTRY%/}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install: $*"; exit 1; }

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1"
  elif command -v wget >/dev/null 2>&1; then wget -qO- "$1"
  else die "need curl or wget"; fi
}
fetch_to() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$2" "$1"
  else wget -qO "$2" "$1"; fi
}
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else die "need sha256sum or shasum"; fi
}
# The one string value of key $1 in a flat JSON object on stdin.
json_str() { tr -d '\n' | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p"; }

os=$(uname -s); arch=$(uname -m)
case "$os/$arch" in
  Linux/x86_64)  triple=x86_64-unknown-linux-gnu ;;
  Linux/aarch64) triple=aarch64-unknown-linux-gnu ;;
  Darwin/arm64)  triple=aarch64-apple-darwin ;;
  Darwin/x86_64) triple=x86_64-apple-darwin ;;
  *) die "no plowrt release for $os/$arch" ;;
esac

version="${PLOWRT_VERSION:-}"
if [ -z "$version" ]; then
  version=$(fetch "$REGISTRY/v1/runtime/latest" | tr -d '[:space:]') || die "cannot read $REGISTRY/v1/runtime/latest"
  [ -n "$version" ] || die "$REGISTRY/v1/runtime/latest is empty"
fi
base="$REGISTRY/v1/runtime/$version/$triple"
release=$(fetch "$base/release.json") || die "no release $version for $triple at $base"
file=$(printf '%s' "$release" | json_str file)
[ -n "$file" ] || die "release.json names no file"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/plowrt-install.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM
say "plowrt $version ($triple)"
fetch_to "$base/$file" "$tmp/$file"
fetch_to "$base/$file.sha256" "$tmp/$file.sha256"
want=$(cut -d' ' -f1 < "$tmp/$file.sha256")
have=$(sha256_of "$tmp/$file")
[ "$want" = "$have" ] || die "sha256 mismatch for $file: expected $want, got $have"

tar -xzf "$tmp/$file" -C "$tmp"
bin=$(find "$tmp" -type f -name plowrt | head -n 1)
[ -n "$bin" ] || die "$file holds no plowrt binary"

dir="${PLOWRT_INSTALL_DIR:-}"
if [ -z "$dir" ]; then
  if [ -w /usr/local/bin ]; then dir=/usr/local/bin; else dir="$HOME/.local/bin"; fi
fi
mkdir -p "$dir"
install -m 0755 "$bin" "$dir/plowrt.tmp.$$" && mv -f "$dir/plowrt.tmp.$$" "$dir/plowrt"
say "installed $dir/plowrt"
case ":$PATH:" in *":$dir:"*) ;; *) say "note: $dir is not on PATH" ;; esac
say "next: plowrt load <model>    # e.g. plowrt load qwen3-asr-0.6b"
