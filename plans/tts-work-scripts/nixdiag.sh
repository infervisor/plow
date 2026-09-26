#!/usr/bin/env bash
export PATH=/nix/var/nix/profiles/default/bin:$PATH
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
D=$(nix eval --raw .#devShells.x86_64-linux.default.drvPath)
echo "drv=$D"
nix build --dry-run "$D^*" 2>&1 | head -n 40
