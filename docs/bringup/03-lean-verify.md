# Stage 3 — Formal Verification & Bundle Qualification

Target-independent stage. Verifies mathematical invariants and safety contracts across the emitted model artifacts using Lean 4.

* **Goal:** Every required checkpoint returns `ok`. `plowc qualify` verifies all bundle obligations without unapproved gaps.
* **Design reference:** [`docs/arch/08-formal-verification.md`](../arch/08-formal-verification.md).

---

## 1. Checkpoint Verification Endpoints

All checkpoints dispatch through `Plow.CLI.Dispatch.endpoints` (`lean-plow/Plow/CLI/Dispatch.lean`):

| Checkpoint | Verified Property | Lean Proof Root |
|---|---|---|
| **A** | Rewrite rules expand to their unfused equivalent | `RewriteBody.check_sound`, `Rewrite.rule_*` |
| **B** | GEMM tiling covers iteration space without gap or overlap | `TilePartition.check_sound` |
| **C** | SRAM buffer hand-offs remain within physical page budget | `Sram.occupancy_le_of_temporal_fit` |
| **D** | Counter schedules enforce dependencies; tensor effects conflict-free | `Verify.verifyDependencies_sound`, `Effects.checkConflicts_sound` |
| **E** | Binary wire encoding round-trips without loss | `Wire.decodeProgram_encodeProgram` |
| **F** | Buffer allocation addresses are disjoint and valid | `Verify.verifyAddressMap_sound_strict` |
| **G** | Staged GEMV operands fit LDS memory limits | `LdsFit.fits_of_check_ok` |
| **K** | Hardware and runtime knob consistency | `Knobs.checkK_sound` |
| **S** | Knob modification scope is bounded to declared rungs/routes | `Knobs.Scope.checkS_sound` |
| **P** | Performance floor ledger preservation | `Knobs.Ledger.checkP_sound` |
| **R** | Selected GEMM and tile policies | `MeasuredPolicy.check_sound` |
| **L** | Padded MLA tensor layout mappings | `MlaLayout.address_in_bounds` |
| `media_geometry.v1` | Speech and multimodal pipeline buffers fit declared bounds | `MediaGeometry.check_sound` |
| `kv_ring.v1` | Packed prefill spans do not overwrite active sliding-window rows | `KvRing.launch_safe` |
| `vmm_trace.v1` | Virtual memory operations stay within allocations and disjoint | `KvRing.traceOk_sound` |
| `speech_fusion.v1` | Speech prologue and normalization sites satisfy preconditions | `SpeechFusion.check_sound` |
| `page_envelope.v1` | Paged attention memory loader reads only valid request pages | `PageEnvelope.route_safe` |

*Axiom Gate:* `lake exe proof_audit` guarantees proof roots depend only on standard Lean axioms `{propext, Classical.choice, Quot.sound}`.

---

## 2. Devblob Call Paths

During `--emit devblob`, `plowc` automatically invokes checkpoints:
* **Checkpoint A:** Rule catalog and expansion verification.
* **Checkpoint D (Coarse & Effects):** Validates program dependency DAG and logical tensor access effects.
* **Checkpoint G:** Staged LDS fit verification.
* **Checkpoint K:** Knob consistency verification (requires `PLOW_VERIFY_BIN`).
* **Checkpoint R / L:** Measured policies and padded MLA layout mappings where applicable.

---

## 3. Bundle Qualification Protocol

`plow_asset::certificates` aggregates proofs into a `Qualification` manifest; `plowc qualify` verifies them against the approved verifier binary:

```bash
# Replay all qualification obligations for an emitted asset directory
nix develop --command plowc qualify <assets-dir> [--format json]

# Strict qualification during campaign build
nix develop --command python3 scripts/campaign/campaign.py qualify --assets <assets-dir> --strict
```

### Verifier Identity
`lean-plow/approved-verifiers.json` tracks approved combinations of the `plow_verify` binary hash and Lean source digest. When modifying Lean proofs:
1. Rebuild the verifier: `cd lean-plow && lake build plow_verify proof_audit`.
2. Run correctness CI: `scripts/lean_correctness_ci.sh`.
3. Obtain approval before committing new verifier hashes.

---

## 4. Execution & CI Commands

Execute all checks inside `nix develop`:

```bash
# 1. Build Lean verifier and verify proof audit
nix develop --command bash -c 'cd lean-plow && lake build plow_verify proof_audit && lake exe proof_audit'

# 2. Run Lean correctness test suite
nix develop --command scripts/lean_correctness_ci.sh

# 3. Emit devblob with strict Lean qualification
nix develop --command plowc --hf-dir <checkpoint> --emit devblob --lean-qualify strict --out <scratch>/emit

# 4. Verify qualification receipts
nix develop --command plowc qualify <scratch>/emit
```

Debug dumps: Set `PLOW_VERIFY_DUMP=<dir>` to capture failing verification payloads for offline replay:
```bash
plow_verify < payload.json
```

---

## 5. Gate Checklist

Before advancing to Stage 4:
- [ ] `lake build` completes without errors or warnings (`warningAsError := true`).
- [ ] `lake exe proof_audit` passes with zero unapproved axioms.
- [ ] `scripts/lean_correctness_ci.sh` passes completely.
- [ ] Devblob emission completes with `ok` across all checkpoints (A, D, G, K, R, L).
- [ ] `plowc qualify --strict` verifies all bundle receipts without missing obligations.
- [ ] No `sorry` keywords or vacuous proofs exist in modified Lean modules.

---

## 6. Common Pitfalls

* **Unset `PLOW_VERIFY_BIN`:** Emits without `PLOW_VERIFY_BIN` fail at Checkpoint K with spawn errors. Do not bypass using `--no-knob-verify`.
* **Lean binary environment:** The verifier binary requires the Nix ELF interpreter. Running outside `nix develop` triggers silent startup exits.
* **Recursive stack overflow:** Checkpoint D/F uses `FastCheckD` for deep execution graphs; do not fall back to unrolled reference recursion.
