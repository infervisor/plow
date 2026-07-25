import Lake
open Lake DSL

package «plow» where
  -- Strict release-mode build by default; the universal lemmas are all proven
  -- so `sorry` is not expected. Callers relying on the verifier can compile
  -- against the `Plow` library or invoke the `plow_verify` CLI binary.
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

@[default_target]
lean_lib «Plow» where
  -- Universal lemmas + executable verifier live under Plow/.
  roots := #[`Plow]

/-- JSON-IPC CLI: reads a request from stdin, dispatches by checkpoint,
    writes a JSON certificate to stdout. Used by the Rust `lean_verify`
    crate to run per-instance checks during compilation. -/
@[default_target]
lean_exe plow_verify where
  root := `Main

lean_exe bench where
  root := `Bench
