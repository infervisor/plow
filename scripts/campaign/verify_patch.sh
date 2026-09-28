#!/usr/bin/env bash
# verify_patch.sh <patch> [base-ref] — verify a patch on the STAGED tree it would produce, never on a
# (shared, possibly dirty) worktree: apply it to base-ref (default HEAD) in a private index, export
# `git write-tree` with `git archive`, and build + test that tree. Prints one summary table.
#
# Steps (each skipped when the patch touches nothing it covers; VERIFY_ALL=1 runs every step):
#   apply     git apply --check against base-ref (whitespace errors reported)
#   build     cargo build --release -p plowc -p plowrt --features plowrt/cuda --bins --examples
#   tests     cargo build --release --tests -p plowrt --features cuda (compiles, does not run)
#   plowrt    cargo test -p plowrt --features cuda,hsa --lib
#   knob      cargo test -p devgen --lib knob; cargo test -p plowrt --features cuda,hsa --lib knob
#   asset     cargo test -p plow-asset; cargo test -p packet
#   scripts   python3 -m py_compile / bash -n / TOML parse of every touched .py / .sh / .toml
# Env: CARGO_TARGET_DIR (give each agent its own; default <tmp>/target), VERIFY_STEPS (subset,
# e.g. "apply scripts knob"), VERIFY_KEEP=1 (keep the temp tree), PLOW_CAMPAIGN_NO_NIX=1 (run cargo
# in the current shell instead of `nix develop`), VERIFY_TIMEOUT (per step, 3600 s).
set -u
PATCH=$(realpath "${1:?usage: verify_patch.sh <patch> [base-ref]}")
BASE=${2:-HEAD}
REPO=$(git rev-parse --show-toplevel)
T=$(mktemp -d "${TMPDIR:-/tmp}/verify-patch.XXXXXX")
[ "${VERIFY_KEEP:-0}" = 1 ] || trap 'rm -rf "$T"' EXIT
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$T/target}
STEPS=${VERIFY_STEPS:-apply build tests plowrt knob asset scripts}
declare -a SUM
res() { SUM+=("$(printf '%-8s %-5s %6ss  %s' "$1" "$2" "$3" "$4")"); }
want() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
cargo_run() {
    if [ "${PLOW_CAMPAIGN_NO_NIX:-0}" = 1 ]; then (cd "$T/src" && timeout "${VERIFY_TIMEOUT:-3600}" "$@")
    else (cd "$T/src" && timeout "${VERIFY_TIMEOUT:-3600}" nix develop --command "$@"); fi
}
step() { # name cmd...
    local name=$1 t0=$SECONDS rc; shift
    "$@" > "$T/$name.log" 2>&1; rc=$?
    if [ $rc = 0 ]; then res "$name" ok $((SECONDS - t0)) ""
    else res "$name" FAIL $((SECONDS - t0)) "$(grep -m1 -E '^error|FAILED|Error|panicked' "$T/$name.log" | cut -c1-100)"
         cp "$T/$name.log" "${PATCH%.patch}.$name.log"; FAILED=1; fi
}
FAILED=0
export GIT_INDEX_FILE=$T/index
git -C "$REPO" read-tree "$BASE" || { echo "bad base $BASE"; exit 2; }
if ! git -C "$REPO" apply --cached --whitespace=warn "$PATCH" > "$T/apply.log" 2>&1; then
    cat "$T/apply.log"; echo "RESULT: patch does not apply to $BASE ($(git -C "$REPO" rev-parse --short "$BASE"))"; exit 1
fi
res apply ok 0 "$(grep -c 'trailing whitespace\|space before tab' "$T/apply.log") whitespace warning(s)"
TREE=$(git -C "$REPO" write-tree)
unset GIT_INDEX_FILE
mkdir -p "$T/src"
git -C "$REPO" archive "$TREE" | tar -x -C "$T/src"
# Untracked but needed by builds (lean-plow verifier, perf-data tools) are not in the tree: link them.
for d in lean-plow/.lake perf-data; do [ -e "$REPO/$d" ] && [ ! -e "$T/src/$d" ] && ln -s "$REPO/$d" "$T/src/$d"; done
FILES=$(git -C "$REPO" apply --numstat "$PATCH" | cut -f3)
touches() { echo "$FILES" | grep -qE "$1" || [ "${VERIFY_ALL:-0}" = 1 ]; }
RUST='^(crates|runtime|Cargo\.(toml|lock)|build\.rs)'
if touches "$RUST"; then
    want build && step build cargo_run cargo build --release -p plowc -p plowrt --features plowrt/cuda --bins --examples
    want tests && step tests cargo_run cargo build --release --tests -p plowrt --features cuda
    want plowrt && step plowrt cargo_run cargo test --release -p plowrt --features cuda,hsa --lib
    want knob && step knob cargo_run sh -c 'cargo test -p devgen --lib knob && cargo test -p plowrt --features cuda,hsa --lib knob'
    want asset && step asset cargo_run sh -c 'cargo test -p plow-asset && cargo test -p packet'
else
    res cargo skip 0 "no crates/ runtime/ Cargo changes (VERIFY_ALL=1 to force)"
fi
if want scripts; then
    t0=$SECONDS; bad=""; n=0
    while IFS= read -r f; do
        [ -e "$T/src/$f" ] || continue
        case "$f" in
            *.py) n=$((n+1)); python3 -m py_compile "$T/src/$f" 2>>"$T/scripts.log" || bad="$bad $f" ;;
            *.sh) n=$((n+1)); bash -n "$T/src/$f" 2>>"$T/scripts.log" || bad="$bad $f" ;;
            *.toml) n=$((n+1)); python3 -c 'import sys,tomllib; tomllib.load(open(sys.argv[1],"rb"))' "$T/src/$f" 2>>"$T/scripts.log" || bad="$bad $f" ;;
        esac
    done <<< "$FILES"
    if [ -z "$bad" ]; then res scripts ok $((SECONDS - t0)) "$n file(s)"; else res scripts FAIL $((SECONDS - t0)) "$bad"; cp "$T/scripts.log" "${PATCH%.patch}.scripts.log"; FAILED=1; fi
fi
echo "verify_patch $(basename "$PATCH") on $BASE ($(git -C "$REPO" rev-parse --short "$BASE")), tree $TREE"
echo "$FILES" | sed 's/^/  touched  /'
printf '%s\n' "${SUM[@]}"
[ "$FAILED" = 0 ] && echo "RESULT: pass" || { echo "RESULT: FAIL (logs beside the patch)"; exit 1; }
