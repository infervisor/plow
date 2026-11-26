#!/usr/bin/env bash
# main_pytests.sh: the two AMD-object python tests on the origin/main scratch worktree.
cd /home/ifv/plow-scratch/consolidate/main-wt || exit 1
NP_RUNTIME=bwrap nix develop --command bash -c '
python3 scripts/tests/test_gfx950_objects.py 2>&1 | tail -3
python3 scripts/tests/test_recipe_objects_env.py 2>&1 | tail -3'
