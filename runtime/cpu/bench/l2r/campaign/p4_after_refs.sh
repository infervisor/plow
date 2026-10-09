#!/usr/bin/env bash
until grep -q REFS_DONE /tmp/g4c/l2r/gen_refs_p4.log; do sleep 20; done
bash /tmp/g4c/l2r/p4_tile.sh
