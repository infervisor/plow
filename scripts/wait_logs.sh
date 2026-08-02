#!/usr/bin/env bash
# Block until at least $1 vllm bench logs matching $2 exist.
N="${1:?count}"; PAT="${2:?glob}"
while [ "$(ls $PAT 2>/dev/null | wc -l)" -lt "$N" ]; do sleep 20; done
echo "have $(ls $PAT | wc -l) logs"
