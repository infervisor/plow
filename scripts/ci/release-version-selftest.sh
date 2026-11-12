#!/usr/bin/env bash
# release-version.sh against throwaway repositories.
set -euo pipefail
script="$(cd "$(dirname "$0")" && pwd)/release-version.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

expect() { # $1 = want, $2 = Cargo version, rest = tags
  local want="$1" cargo="$2" dir got
  shift 2
  dir="$(mktemp -d -p "$work")"
  git -C "$dir" init -q
  printf '[workspace.package]\nversion = "%s"\n\n[profile.release]\n' "$cargo" > "$dir/Cargo.toml"
  git -C "$dir" add Cargo.toml
  git -C "$dir" -c user.name=t -c user.email=t@example.invalid commit -qm init
  for tag in "$@"; do git -C "$dir" tag "$tag"; done
  got="$(cd "$dir" && bash "$script" 2>/dev/null)"
  [ "$got" = "$want" ] || { echo "cargo $cargo, tags [$*]: got $got, want $want" >&2; exit 1; }
}

expect 0.1.0 0.1.0                              # first release: Cargo's version
expect 0.1.4 0.1.3 v0.1.3                       # released already: one patch on
expect 0.1.11 0.1.3 v0.1.9 v0.1.10              # numeric, not lexical, order
expect 0.2.0 0.2.0 v0.1.9                       # Cargo raised by hand: used as is
expect 0.1.10 0.1.2 v0.1.9                      # Cargo behind the tags (bumps live on tags)
expect 0.1.3 0.1.2 v0.1.2 rc8-freeze-20261103 v0.1.2-rc1 inferos-v9.9.9   # other tags ignored
echo "release-version selftest: ok"
