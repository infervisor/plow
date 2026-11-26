#!/usr/bin/env bash
# wait_suite.sh <gpuq id> <commit>: wait for the suite job, print its verdict lines.
J=/tmp/plow-gpuq/$1.json
until grep -qE '"state": "(done|failed|cancelled|timeout)"' "$J"; do sleep 60; done
grep -o '"state": "[a-z]*"' "$J"
grep -E "==|rc=|FAILED|panicked|SUITE-DONE" "/home/ifv/plow-scratch/consolidate/suite-$2.log"
