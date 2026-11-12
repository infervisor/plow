#!/usr/bin/env bash
# Say whether main holds anything a new plowrt/plowc release would carry.
#
#   release-needed.sh                  # prints "true" or "false"
#   release-needed.sh --paths          # prints the watched paths, one per line
#   release-needed.sh --watch a b …    # watch these paths instead (self-test)
#
# A release is needed when a watched path differs between the last release and
# HEAD. Watched: every workspace crate `plowrt` or `plowc` is built from (from
# `cargo tree`), the C sources plowrt's build script compiles, the lock files,
# the flake, and `install.sh`. A change to docs, recipes, perf data, CI or a
# crate neither binary links releases nothing.
#
# The last release is the newest `vX.Y.Z` tag. CI tags its own bump commit, so
# the source that release was built from is that commit's parent. The tag is
# pushed before the upload, so a tag whose version the registry
# (`PLOW_REGISTRY`) does not list is an upload that failed: a release is
# needed then too, and takes the next version.
set -euo pipefail

# Not crates, but part of what a release is built from or publishes.
always=(Cargo.toml Cargo.lock flake.nix flake.lock .cargo runtime include scripts/install.sh)

# Workspace directories of the path crates plowrt and plowc depend on.
crate_paths() {
  local root tree
  root="$(git rev-parse --show-toplevel)/"
  tree="$(cargo tree --locked -p plowrt -p plowc --all-features -e normal,build --prefix none)"
  sed -n 's|.*(\(/[^)]*\)).*|\1|p' <<<"$tree" | sort -u | sed "s|^$root||"
}

if [ "${1:-}" = "--watch" ]; then
  shift
  paths=("$@")
else
  mapfile -t crates < <(crate_paths)
  [ "${#crates[@]}" -ge 2 ] || { echo "cargo tree named no workspace crate" >&2; exit 1; }
  paths=("${always[@]}" "${crates[@]}")
fi
if [ "${1:-}" = "--paths" ]; then
  printf '%s\n' "${paths[@]}"
  exit 0
fi

tag="$(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1 || true)"
if [ -z "$tag" ]; then
  echo "no release yet" >&2
  echo true
  exit 0
fi
base="$(git rev-parse "$tag^{commit}")"
if [ "$(git show -s --format=%s "$base")" = "plow: $tag [skip ci]" ]; then
  base="$(git rev-parse "$base^")"
fi
echo "last release $tag, built from ${base:0:12}" >&2

# An unreadable registry (or one with no release yet) is not a reason to skip:
# publishing is create-only, so trying again is safe.
registry="${PLOW_REGISTRY:-https://pdist.infervisor.ai}"
index="$(curl -fsL --max-time 30 "${registry%/}/v1/plow-runtime/index.json" || true)"
if ! grep -Eq "\"semver\": *\"${tag#v}\"" <<<"$index"; then
  echo "$tag is tagged but the registry does not list it" >&2
  echo true
  exit 0
fi

changed="$(git diff --name-only "$base" HEAD -- "${paths[@]}")"
if [ -z "$changed" ]; then
  echo "nothing a release is built from changed" >&2
  echo false
else
  head -n 20 <<<"$changed" | sed 's/^/changed: /' >&2
  echo true
fi
