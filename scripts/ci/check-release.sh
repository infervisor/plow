#!/usr/bin/env bash
# Check `nix build .#plowrt-release*` outputs before they are published.
#
#   check-release.sh <semver> <release dir>...
#
# Each directory holds the tarball, its `.sha256` and `release.json`. Checked:
# the version is `<semver>+<commit>` from a clean tree, the digest matches, the
# tarball holds `plowrt` and `plowc` for the triple it names, neither loads
# through /nix/store, and a binary this machine can run reports that version.
set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: check-release.sh <semver> <release dir>..." >&2; exit 1; }
semver="$1"
shift
die() { echo "check-release: $*" >&2; exit 1; }
# One string field of the flake's release.json.
field() { sed -n "s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p" "$2" | head -n1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for dir in "$@"; do
  version="$(field version "$dir/release.json")"
  triple="$(field triple "$dir/release.json")"
  file="$(field file "$dir/release.json")"
  [ -n "$version" ] && [ -n "$triple" ] && [ -n "$file" ] || die "$dir/release.json is incomplete"
  case "$version" in
    *dirty* | *+unknown) die "$dir: version $version is not a clean build" ;;
    "$semver"+*) ;;
    *) die "$dir: version $version is not a build of $semver" ;;
  esac
  (cd "$dir" && sha256sum --check --quiet "$file.sha256") || die "$dir/$file does not match its .sha256"

  case "${triple%%-*}" in
    x86_64) machine=62 ;;
    aarch64) machine=183 ;;
    *) die "$dir: unknown arch in triple $triple" ;;
  esac
  rm -rf "$work/unpacked" && mkdir "$work/unpacked"
  tar -xzf "$dir/$file" -C "$work/unpacked" --strip-components=1
  for binary in plowrt plowc; do
    path="$work/unpacked/$binary"
    [ -x "$path" ] || die "$file holds no $binary"
    # ELF e_machine: two bytes at offset 18.
    [ "$(head -c 4 "$path" | od -An -c | tr -d ' ')" = '177ELF' ] || die "$file: $binary is not an ELF binary"
    [ "$(od -An -tu2 -j18 -N2 "$path" | tr -d ' ')" = "$machine" ] || die "$file: $binary is not a ${triple%%-*} binary"
    if readelf -l "$path" | grep -q 'interpreter: /nix/store'; then
      die "$file: $binary loads through /nix/store and runs on no other host"
    fi
    if [ "${triple%%-*}" = "$(uname -m)" ]; then
      said="$("$path" --version)"
      want="$version"
      [ "$binary" = plowrt ] || want="$semver"
      [ "${said##* }" = "$want" ] || die "$file: $binary --version says '$said', not $want"
    fi
  done
  echo "ok $file (plowrt, plowc; sha256 $(cut -c1-12 "$dir/$file.sha256"))"
done
