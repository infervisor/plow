#!/usr/bin/env bash
# nx.sh <cmd...> : run <cmd> in the plow flake's dev env (nix print-dev-env of the worktree, captured
# in devenv.sh) without proot/bwrap: userns is restricted on this boot, and nix develop nested under
# the proot fallback cannot stat /nix/store.
source /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/devenv.sh > /dev/null 2>&1
[ -n "${NIX_BUILD_TOP:-}" ] && rmdir "$NIX_BUILD_TOP" 2>/dev/null
unset TMPDIR TMP TEMP TEMPDIR NIX_BUILD_TOP
exec "$@"
