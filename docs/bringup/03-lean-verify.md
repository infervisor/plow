# Stage 3 — Formal Verification & Bundle Qualification

Target-independent formal stage (CPU-only). Verifies compiler transformations, schedule legality, and memory safety against machine-checked Lean 4 theorems.

* **Input:** Generated packet artifacts and compiler IR from Stages 1 and 2.
* **Output:** Machine-checked verification certificate confirming 0 proof gaps, validated against `approved-verifiers.json`.
* **Architecture Reference:** [`docs/arch/08-formal-verification.md`](../arch/08-formal-verification.md).

---

## 1. Verification Architecture

Plow enforces formal verification across two complementary pipelines:
1. **Ahead-of-Time (AOT) Proofs (`lean-plow/LeanPlow/`):** Lake-built Lean 4 library proving mathematical equivalence of rewrite rules, schedule validity, and tensor layout soundness.
2. **Just-in-Time (JIT) Semantic Checks (`crates/lean_verify/`):** Dynamic verification passes executed during packet emission that evaluate schedule legality, address bounds, and race-freedom on concrete models.

---

## 2. Formal Checkpoints

The compilation pipeline must satisfy all applicable checkpoints:

| Checkpoint | Scope | Verified Invariant |
|---|---|---|
| **Checkpoint A** | Term Rewrites | Every e-graph rewrite rule preserves denotational semantics (`LeanPlow.Rewrites.soundRules`). |
| **Checkpoint B** | Schedule Legality | Execution graph is a DAG; all read-after-write, write-after-read, and write-after-write hazards are synchronized. |
| **Checkpoint C** | Address Bounds | Buffer spans $[base, base + size)$ never overflow allocated VMM reservation windows. |
| **Checkpoint D** | Tensor Geometry | Stride equations and multidimensional index projections are injective (no overlapping alias collisions). |
| **Checkpoint E** | KV Ring Headroom | Sliding window ring pointers remain strictly within allocated physical ring buffer envelopes. |
| **Checkpoint F** | Kernel Launch Geometry | Grid and block launch bounds do not exceed hardware limits recorded in `hwspec`. |
| **Checkpoint G** | Weight Offsets | Weight segment table byte offsets align to 16-byte boundaries and match manifest length. |
| **Checkpoint H** | Stream Invariants | Async barrier dependencies between compute and memory streams are acyclic and complete. |
| **Checkpoint I** | CUDAGraph Capture Envelope | Dynamically bound addresses within graph captures reside in pinned VMM space. |
| **Checkpoint J** | MoE Token Routing | Expert dispatch permutation maps conserve token count: $\sum \text{tokens\_per\_expert} = K \cdot N_{\text{tokens}}$. |
| **Checkpoint K** | Quantization Descaling | Scale tensor dimensions match block/channel granularities of quantized weight matrices. |
| **Checkpoint L** | Cross-Layer Ties | Shared weight tensors are marked immutable across all referring execution stages. |

### Domain Extension Proofs
- `media_geometry.v1`: Injective bounding of multidimensional spatial/temporal video patches.
- `kv_ring.v1`: Bounded circular buffer wrapping with zero overwrite of active context tokens.
- `vmm_trace.v1`: Virtual memory physical commit sequences remain within device physical limits.
- `speech_fusion.v1`: Audio spectrogram frame chunking preserves sequential temporal ordering.
- `page_envelope.v1`: Dynamic paged KV block allocation never intersects static weight memory.

---

## 3. Verifier Integrity & `approved-verifiers.json`

Verification results are validated against [`lean-plow/approved-verifiers.json`](../../lean-plow/approved-verifiers.json):
* Each verifier binary has its cryptographic SHA-256 hash pinned in `approved-verifiers.json`.
* `plowc qualify` inspects the bundle and fails if any verifier is unregistered or modified.
* No proof waivers or skipped checkpoints are permitted in production recipes.

---

## 4. Verification Workflow

### Step 1: Build & Verify Lean 4 Theorems
```bash
cd lean-plow && lake build
```
Confirms all AOT mathematical theorems compile without `sorry` or axiom violations.

### Step 2: Run JIT Verification Test Suite
```bash
cargo test -p lean_verify
```
Runs property-based tests over schedule generation, address bounding, and hazard checkers.

### Step 3: Qualify Packet Bundle
```bash
nix develop --command plowc qualify \
  --bundle <scratch>/assets \
  --approved lean-plow/approved-verifiers.json
```
Exits with code 0 only when all checkpoints evaluate to `ok` and verifier hashes match.

---

## 5. Machine-Checked Gate Checklist

Before advancing to Stage 4:
- [ ] `cd lean-plow && lake build` succeeds with 0 errors and 0 warnings.
- [ ] `cargo test -p lean_verify` passes with 100% success rate.
- [ ] Checkpoints A through L evaluate to `ok` on the candidate model packet.
- [ ] `plowc qualify` reports 0 unapproved verifiers and 0 unproven theorems.
- [ ] Weight offsets, VMM reservations, and KV ring bounds are mathematically confirmed valid.
