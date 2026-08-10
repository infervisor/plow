# Agent Prompt — Stage 3: Formal Verification with Lean

You are executing **Stage 3** of the model bringup playbook: proving the
compiled schedule correct with the Lean 4 verifier before Stage 4 (kernel
tuning). Read [`docs/bringup/03-lean-verify.md`](../../docs/bringup/03-lean-verify.md)
and [`docs/arch/08-formal-verification.md`](../../docs/arch/08-formal-verification.md)
first; they are authoritative and were recently corrected against the code.

## Your objective

Get every compiled bucket of the new model to a clean `ok` from all seven
checkpoints (A–G), with a clean `lake build` and **no `sorry` and no vacuous
proofs**. A rejection is a real correctness bug in the compiler output, not a
warning to suppress.

## Ground rules

- Trust the code over any prose. Verify symbols and commands before relying on
  them.
- Do not weaken a proof to make it pass. No `sorry`, no `by trivial` on a
  statement that should be a real equality, no restating a term as itself, no
  proving `True`.
- A skipped checkpoint is **not** a passed checkpoint. Only a genuinely
  unrunnable binary (`VerifyError::is_binary_unusable`) may be skipped, and it
  must be reported as a skip.
- Keep changes minimal and scoped to what the new model requires. Do not
  refactor the Lean project or touch unrelated proofs.

## Procedure

1. **Environment.** Use `nix develop` so `elan`/`lake` and a runnable
   `plow_verify` are on hand. Confirm the toolchain matches
   `lean-plow/lean-toolchain` (`leanprover/lean4:v4.15.0`).

2. **Build the proofs + CLI.**
   ```bash
   cd lean-plow && lake build
   ```
   This must complete with no errors and no `sorry`. The `Plow` library is a
   default target, so a failing theorem fails the build. Confirm the binary
   exists at `lean-plow/.lake/build/bin/plow_verify`.

3. **Sanity-check the binary** with a known-good request (see the doc's example
   for Checkpoint B). Expect `{"ok":true,...}` and exit 0.

4. **Check for a new Checkpoint-A obligation.** Diff Stage 2's
   `crates/rewrite/src/egl/rules.egg` `; rule:` annotations against
   `Plow.Rewrite.soundRules`:
   ```bash
   grep '; rule:' crates/rewrite/src/egl/rules.egg
   ```
   Every annotation must have a matching `soundRules` entry and a `rule_*`
   theorem. If Stage 2 added a rule not yet covered, discharge it (step 5).
   Otherwise, no new theorems are needed — the universal lemmas already cover
   new tiles/schedules/allocations/frames/staged-GEMVs.

5. **Discharge a new fusion rule soundly** (only if step 4 found one), following
   the `Plow.Rewrite` module contract in `lean-plow/Plow/Rewrite.lean`:
   1. Add a fused variant to the `Op` inductive.
   2. Extend `expand` so the fused op unfolds to its exact unfused composition.
   3. Write `theorem rule_<name> ... : expand (fused …) = <unfused composition>`
      and prove it — it should close by `rfl`. If `rfl` does not work, the
      `expand` definition and the intended semantics disagree; fix `expand`, do
      not weaken the theorem.
   4. Add the exact rule-name string to `soundRules`.
   Re-run `lake build`. The proof must genuinely equate the fused op to its
   unfused meaning.

6. **Run the gate over the model.**
   ```bash
   cargo build --features lean-verify
   export PLOW_VERIFY_BIN="$PWD/lean-plow/.lake/build/bin/plow_verify"
   plowc <model> --lean-verify
   ```
   This runs the verifier once per bucket (schedule path checks D/F; the devblob
   path also runs G). Watch the logs: an accepted bucket logs `accepted`; a
   rejection is fatal (`PlowcError::LeanVerify`).

7. **Triage any rejection.** A rejection means the compiler produced a
   provably-wrong schedule. Read the `reason` (it names the first violating
   candidate/hand-off/byte/instance). Fix the **compiler**, not the proof. Use
   `PLOW_VERIFY_DUMP=<dir>` to capture the exact request and replay it by hand
   against `plow_verify` for a minimal repro.

## Verification gate before Stage 4

Do not advance to Stage 4 (kernel tuning) until all of:

- `cd lean-plow && lake build` is clean, no `sorry`.
- `grep -rn 'sorry' lean-plow --include='*.lean'` finds nothing outside
  comments; any new `rule_*` closes by `rfl` and is a real equality.
- `soundRules` ⇔ `; rule:` annotations ⇔ `rule_*` theorems: one-to-one, no
  orphans either way.
- Every bucket of the model passes: all seven checkpoints emit `ok` under
  `plowc --lean-verify`, with no skipped checkpoint (unless the binary was
  genuinely unusable and you reported it).

## When to stop and ask

- A checkpoint rejects and the fix is in the compiler/scheduler, not this stage
  — report the rejection and its `reason`; do not paper over it in Lean.
- A new fusion rule's `rule_*` theorem will not close by `rfl` and the correct
  `expand` definition is unclear — stop; a forced proof here is a correctness
  hole.
- `lake build` fails for reasons outside a new-rule addition (toolchain drift,
  library break) — report it rather than editing unrelated proofs.
- The only way to get green is to weaken a proof or skip a runnable checkpoint —
  stop and ask.

## Report format

- Changed: files/theorems touched (1–3 bullets).
- Verified: `lake build` result; checkpoints run and their verdicts; new-rule
  coverage status.
- Notes: any rejection triaged to the compiler, skips, or blockers.
