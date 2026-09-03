#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d /tmp/bringup-tuning-profile-selftest.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

expect_reject() {
  if "$HERE/bringup_tuning_profile.py" "$1" 1 >"$TMP/rejected.out" 2>"$TMP/rejected.err"; then
    echo "required tuning accepted invalid provenance: $1" >&2
    exit 1
  fi
  grep -q "tuned profile required" "$TMP/rejected.err"
}

missing="$TMP/missing.json"
[ "$("$HERE/bringup_tuning_profile.py" "$missing" 0)" = $'0\t<missing>\tanalytical-fallback' ]
expect_reject "$missing"

missing_profile="$TMP/missing-profile.json"
printf '%s\n' '{"schema":1}' >"$missing_profile"
[ "$("$HERE/bringup_tuning_profile.py" "$missing_profile" 0)" = $'0\t<missing>\tanalytical-fallback' ]
expect_reject "$missing_profile"

analytical="$TMP/analytical.json"
printf '%s\n' '{"schema":1,"tuning":{"tile_measured":0,"tile_source":"analytical"}}' >"$analytical"
[ "$("$HERE/bringup_tuning_profile.py" "$analytical" 0)" = $'0\tanalytical\tanalytical-fallback' ]
expect_reject "$analytical"

false_measured="$TMP/false-measured.json"
printf '%s\n' '{"schema":1,"tuning":{"tile_measured":3,"tile_source":"analytical cold start"}}' >"$false_measured"
expect_reject "$false_measured"

empty_source="$TMP/empty-source.json"
printf '%s\n' '{"schema":1,"tuning":{"tile_measured":3,"tile_source":""}}' >"$empty_source"
expect_reject "$empty_source"

measured="$TMP/measured.json"
printf '%s\n' '{"schema":1,"tuning":{"tile_measured":3,"tile_source":"measured"}}' >"$measured"
[ "$("$HERE/bringup_tuning_profile.py" "$measured" 0)" = $'3\tmeasured\tmeasured' ]
[ "$("$HERE/bringup_tuning_profile.py" "$measured" 1)" = $'3\tmeasured\tmeasured' ]

grep -q 'bringup_tuning_profile.py' "$HERE/bringup_gate.sh"
grep -q 'bringup_tuning_profile.py' "$HERE/bringup_showdown.sh"
grep -q 'tuning_profile=%s' "$HERE/bringup_gate.sh"
grep -q 'tuning_profile=%s' "$HERE/bringup_showdown.sh"
echo "bringup tuning profile selftest: PASS"
