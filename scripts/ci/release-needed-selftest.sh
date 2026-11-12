#!/usr/bin/env bash
# release-needed.sh against a throwaway repository.
set -euo pipefail
script="$(cd "$(dirname "$0")" && pwd)/release-needed.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir "$work/repo"
cd "$work/repo"
git init -q
git config user.name t
git config user.email t@example.invalid

# The registry's index, listing the versions given.
export PLOW_REGISTRY="file://$work/registry"
index() {
  mkdir -p "$work/registry/v1/plow-runtime"
  local rows="" v
  for v in "$@"; do rows="$rows{\"semver\": \"$v\"},"; done
  echo "{\"releases\":[${rows%,}]}" > "$work/registry/v1/plow-runtime/index.json"
}

commit() { mkdir -p "$(dirname "$1")"; echo "$RANDOM$RANDOM" >> "$1"; git add -A; git commit -qm "${2:-change $1}"; }
expect() { # $1 = want, $2 = why
  local got
  got="$(bash "$script" --watch crates/plowrt Cargo.lock 2>/dev/null)"
  [ "$got" = "$1" ] || { echo "$2: got $got, want $1" >&2; exit 1; }
}

commit crates/plowrt/src/main.rs
expect true "no release yet"

# A release the way CI makes it: the tag is on a bump commit off main.
git checkout -q --detach
commit Cargo.lock "plow: v0.1.0 [skip ci]"
git tag v0.1.0
git checkout -q -
expect true "tagged, but the registry has no index"
index 0.0.9
expect true "tagged, but the upload never reached the registry"
index 0.0.9 0.1.0
expect false "nothing changed since the release (the bump itself is not a change)"

commit docs/guide.md
commit crates/other/src/lib.rs
git tag rc1-20260101
expect false "docs and an unrelated crate release nothing"

commit crates/plowrt/src/main.rs
expect true "a runtime source change"

# A release tagged by hand on main itself.
git tag v0.1.1
index 0.1.0 0.1.1
expect false "released at this commit"
commit Cargo.lock
expect true "a lock file change"
echo "release-needed selftest: ok"
