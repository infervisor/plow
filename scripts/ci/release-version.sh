#!/usr/bin/env bash
# Print the version the next release gets.
#
# The newest `vX.Y.Z` tag is the last release. Cargo's workspace version wins
# when someone raised it past that tag by hand (`nix run .#bump -- X.Y.0`, for
# a minor or major release); otherwise the next release is one patch on.
# Release tags point at CI's bump commits, which are not on main.
set -euo pipefail
current="$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n1)"
part='(0|[1-9][0-9]{0,8})'
semver="$part\.$part\.$part"
[[ "$current" =~ ^$semver$ ]] || { echo "invalid workspace version '$current'" >&2; exit 1; }

last=""
while IFS= read -r tag; do
  [[ "$tag" =~ ^v($semver)$ ]] || continue
  version="${tag#v}"
  if [ -z "$last" ] || [ "$(printf '%s\n' "$version" "$last" | sort -V | tail -n1)" = "$version" ]; then
    last="$version"
  fi
done < <(git tag --list 'v*')

if [ -z "$last" ] || { [ "$current" != "$last" ] && [ "$(printf '%s\n' "$current" "$last" | sort -V | tail -n1)" = "$current" ]; }; then
  next="$current"
else
  IFS=. read -r major minor patch <<<"$last"
  next="$major.$minor.$((patch + 1))"
fi
echo "workspace $current; last release ${last:-none} -> $next" >&2
printf '%s\n' "$next"
