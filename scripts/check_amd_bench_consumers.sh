#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY=${PLOW_AMD_BENCH_CONSUMERS_REGISTRY:-"$ROOT/scripts/amd-bench-consumers.tsv"}
CHECK_REL="scripts/$(basename "${BASH_SOURCE[0]}")"

declare -A classified
declare -A legacy_performance=(
  [scripts/glm52_glusplit_run.sh]=1
  [scripts/glm52_linfp8_run.sh]=1
  [scripts/glm52_linfp8_stacked_run.sh]=1
  [scripts/k3_block_sweep.sh]=1
  [scripts/sweep_batch_ceiling.sh]=1
  [scripts/walk_b16_ab.sh]=1
)
declare -A migrated_batch_gates=(
  [scripts/gate_batched.sh]=1
  [scripts/k3_batch_gate.sh]=1
)
while IFS=$'\t' read -r class path binding disposition; do
  [[ -z "$class" || "$class" == \#* ]] && continue
  case "$class" in
    performance|correctness|trace-dump|TP-audit|prefill-sweep|synthetic-probe) ;;
    *) echo "FAIL: invalid amd-bench class '$class' for $path" >&2; exit 1 ;;
  esac
  case "$binding" in
    checkpoint|synthetic) ;;
    *) echo "FAIL: invalid amd-bench binding '$binding' for $path" >&2; exit 1 ;;
  esac
  [[ -n "$path" && -n "$disposition" ]] || {
    echo "FAIL: incomplete amd-bench registry row for '$path'" >&2; exit 1;
  }
  [[ -z "${classified[$path]+x}" ]] || {
    echo "FAIL: duplicate amd-bench registry row for $path" >&2; exit 1;
  }
  if [[ "$class" == performance && -z "${legacy_performance[$path]+x}" ]]; then
    echo "FAIL: new amd-bench performance consumers are forbidden: $path" >&2
    echo "Use plowrt bench/serve, or classify a genuinely diagnostic consumer by its surface." >&2
    exit 1
  fi
  [[ -f "$ROOT/$path" ]] || {
    echo "FAIL: classified amd-bench consumer does not exist: $path" >&2; exit 1;
  }
  if [[ "$binding" == synthetic ]]; then
    awk '$0 !~ /^[[:space:]]*#/ && /amd-probe/ { found=1 } END { exit !found }' "$ROOT/$path" || {
      echo "FAIL: synthetic consumer must use the distinct amd-probe command: $path" >&2; exit 1;
    }
    awk '$0 !~ /^[[:space:]]*#/ && /amd-bench/ { found=1 } END { exit found }' "$ROOT/$path" || {
      echo "FAIL: synthetic consumer still invokes amd-bench: $path" >&2; exit 1;
    }
  else
    rg -q -- '--checkpoint' "$ROOT/$path" || {
      echo "FAIL: checkpoint-bound amd-bench consumer lacks --checkpoint: $path" >&2; exit 1;
    }
  fi
  classified[$path]="$class"
done < "$REGISTRY"

declare -A observed
while IFS= read -r file; do
  rel="${file#"$ROOT/"}"
  [[ "$rel" == "$CHECK_REL" ]] && continue
  if awk '$0 !~ /^[[:space:]]*#/ && /amd-(bench|probe)/ { found=1 } END { exit !found }' "$file"; then
    observed[$rel]=1
    [[ -n "${classified[$rel]+x}" ]] || {
      echo "FAIL: unclassified active amd-bench consumer: $rel" >&2
      echo "Add it to scripts/amd-bench-consumers.tsv or migrate it to plowrt bench/serve." >&2
      exit 1
    }
  fi
done < <(rg --files "$ROOT/scripts" -g '*.sh')

for path in "${!classified[@]}"; do
  [[ -n "${observed[$path]+x}" ]] || {
    echo "FAIL: stale amd-bench classification (no active invocation): $path" >&2
    exit 1
  }
done

for path in "${!legacy_performance[@]}"; do
  [[ "${classified[$path]-}" == performance ]] || {
    echo "FAIL: frozen legacy performance entry is absent or reclassified: $path" >&2
    exit 1
  }
done

for path in "${!migrated_batch_gates[@]}"; do
  for flag in bench --prompt-rows --token-audit --engine-diagnostics; do
    rg -q -- "$flag" "$ROOT/$path" || {
      echo "FAIL: migrated production batch gate lost '$flag': $path" >&2
      exit 1
    }
  done
  awk '$0 !~ /^[[:space:]]*#/ && /amd-bench/ { found=1 } END { exit found }' "$ROOT/$path" || {
    echo "FAIL: migrated production batch gate regressed to amd-bench: $path" >&2
    exit 1
  }
done

"$ROOT/scripts/batch_gates_selftest.sh"

echo "PASS: ${#observed[@]} active AMD direct-runner script consumers are classified"
