#!/usr/bin/env bash
# Print the version main's next plowrt/plowc release gets, or nothing when
# there is nothing to release.
#
#   next-release.sh               # "0.1.5", or empty
#   next-release.sh --self-test   # check this script against throwaway repos
#
# A release is needed when something plowrt or plowc is built from changed
# since the last release: a workspace crate either binary links (from `cargo
# tree`), runtime/ and include/ (the C sources plowrt's build script
# compiles), the lock files, the flake, or scripts/install.sh. Docs, recipes,
# perf data, CI and crates neither binary links release nothing.
#
# The last release is the newest vX.Y.Z tag. CI tags its own bump commit
# ("plow: vX.Y.Z [skip ci]"), so that release was built from the commit's
# parent. CI pushes the tag before it uploads, so a tag whose version the
# registry (PLOW_REGISTRY) does not list is a failed upload: it is released
# again under the next version.
#
# The version is one patch past the last release, or Cargo's workspace
# version when someone raised it past that (`nix run .#bump -- X.Y.0`).
set -euo pipefail

# Not crates, but part of what a release is built from or publishes.
ALWAYS=(Cargo.toml Cargo.lock flake.nix flake.lock .cargo runtime include scripts/install.sh)

watched() {
  if [ -n "${NEXT_RELEASE_WATCH:-}" ]; then
    tr ' ' '\n' <<<"$NEXT_RELEASE_WATCH"
    return
  fi
  local root
  root="$(git rev-parse --show-toplevel)/"
  printf '%s\n' "${ALWAYS[@]}"
  cargo tree --locked -p plowrt -p plowc --all-features -e normal,build --prefix none \
    | sed -n 's|.*(\(/[^)]*\)).*|\1|p' | sort -u | sed "s|^$root||"
}

next_release() {
  local cargo tag base index changed
  cargo="$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n1)"
  [[ "$cargo" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "invalid workspace version '$cargo'" >&2; return 1; }
  tag="$(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1 || true)"
  if [ -z "$tag" ]; then
    echo "no release yet" >&2
    echo "$cargo"
    return
  fi
  base="$(git rev-parse "$tag^{commit}")"
  if [ "$(git show -s --format=%s "$base")" = "plow: $tag [skip ci]" ]; then
    base="$(git rev-parse "$base^")"
  fi
  echo "last release $tag, built from ${base:0:12}" >&2

  # An unreadable registry is not a reason to skip: publishing is create-only.
  index="$(curl -fsL --max-time 30 "${PLOW_REGISTRY:-https://pdist.infervisor.ai}/v1/plow-runtime/index.json" || true)"
  if grep -Eq "\"semver\": *\"${tag#v}\"" <<<"$index"; then
    mapfile -t paths < <(watched)
    changed="$(git diff --name-only "$base" HEAD -- "${paths[@]}")"
    if [ -z "$changed" ]; then
      echo "nothing plowrt or plowc is built from changed: no release" >&2
      return
    fi
    head -n 20 <<<"$changed" | sed 's/^/changed: /' >&2
  else
    echo "$tag is tagged but the registry does not list it" >&2
  fi

  local last="${tag#v}" major minor patch
  if [ "$cargo" != "$last" ] && [ "$(printf '%s\n' "$cargo" "$last" | sort -V | tail -n1)" = "$cargo" ]; then
    echo "$cargo"
  else
    IFS=. read -r major minor patch <<<"$last"
    echo "$major.$minor.$((patch + 1))"
  fi
}

self_test() {
  local script
  script="$(realpath "$0")"
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  local work="$WORK"
  export PLOW_REGISTRY="file://$work/registry" NEXT_RELEASE_WATCH="crates/plowrt Cargo.lock"
  mkdir "$work/repo" && cd "$work/repo"
  git init -q && git config user.name t && git config user.email t@example.invalid
  printf '[workspace.package]\nversion = "0.1.0"\n\n[profile.release]\n' > Cargo.toml
  git add Cargo.toml && git commit -qm init

  commit() { mkdir -p "$(dirname "$1")"; echo "$RANDOM$RANDOM" >> "$1"; git add -A; git commit -qm "${2:-change $1}"; }
  listed() { # the registry's index, listing these versions
    mkdir -p "$work/registry/v1/plow-runtime"
    local rows="" v
    for v in "$@"; do rows="$rows{\"semver\": \"$v\"},"; done
    echo "{\"releases\": [${rows%,}]}" > "$work/registry/v1/plow-runtime/index.json"
  }
  expect() { # want, why
    local got
    got="$(bash "$script" 2>/dev/null)"
    [ "$got" = "$1" ] || { echo "next-release self-test: $2: got '$got', want '$1'" >&2; exit 1; }
  }
  release() { # tag a CI-style bump commit off main
    git checkout -q --detach
    sed -i "s/^version = .*/version = \"$1\"/" Cargo.toml; git commit -q --allow-empty -am "plow: v$1 [skip ci]"
    git tag "v$1"; git checkout -q -
  }

  commit crates/plowrt/src/main.rs
  expect 0.1.0 "no release yet: Cargo's version"
  release 0.1.0
  expect 0.1.1 "tagged but never uploaded"
  listed 0.1.0
  expect "" "nothing changed since the release"
  commit docs/guide.md; commit crates/other/src/lib.rs
  git tag rc1-20260101; git tag inferos-v9.9.9
  expect "" "docs, an unlinked crate and other tags release nothing"
  commit crates/plowrt/src/main.rs
  expect 0.1.1 "a runtime source change"
  release 0.1.1; release 0.1.2; release 0.1.9; release 0.1.10; listed 0.1.10
  commit Cargo.lock
  expect 0.1.11 "numeric, not lexical, tag order"
  sed -i 's/^version = .*/version = "0.2.0"/' Cargo.toml; git commit -qam "bump to 0.2.0"
  expect 0.2.0 "Cargo raised by hand is used as is"
  echo "next-release self-test: ok"
}

if [ "${1:-}" = "--self-test" ]; then self_test; else next_release; fi
