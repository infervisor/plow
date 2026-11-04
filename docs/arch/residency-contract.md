# Placement and residency contract

`serve/placement.rs` is a pure registration planner. Groups are nonempty and
disjoint, instance keys are unique, and ranks of one logical model use distinct
groups. Automatic assignments fit individually and use the group's TP width or
one device for TP1. An explicit pin requires the exact group width and first
ordinal. Pins deliberately bypass the individual byte-fit test. Registration
may oversubscribe; it is not a claim that every registered model can be resident.
The policy load counter uses u128 so oversubscribed u64 requirements do not wrap.

`serve/residency_contract.rs` supplies the CPU-only arithmetic and return gate
used by the manager. The complete footprint includes weights, compiled KV,
activations/tables, sidecar packets, overhead and minimum admitted KV. A load
also leaves the transient reserve. Saturated accounting (`u64::MAX`) is refused;
overflowed free-plus-pool credit is zero. Existing resident KV claims and planned
headroom both constrain admission. Metadata and allocator accounting must include
all allocations; concurrent external GPU users can still invalidate a snapshot.

Eviction first hides the mux from routing, but retains its internal handle until
completion is acknowledged. A dropped acknowledgement is not a completion
fence. Cancellation preserves the retiring handle so the next eviction can
resume draining. A successful completion marker persists after the channel
closes. It is set only when no live slots or waiting work remain; subsequent
queued messages are rejected or acknowledged, never dispatched to the device.
This depends on an engine tick's completion contract actually fencing its device
work. No CPU proof establishes driver or kernel behavior.

The local `ReleaseGate` permits one engine removal after acknowledgement. Across
separate eviction calls, once-only ownership comes from the manager switch mutex
and `AppState::remove_gpu_engine` taking the map entry; the local gate does not
replace either. Allocation reclamation still depends on Rust ownership and the
allocator's drop semantics, including independently owned sidecars. A retained
external engine Arc can delay reclamation and is logged.

For automatic residency, `ensure_resident_locked` calls `clear_partial` before
its resident fast path. A retiring engine has no routable mux, so this resumes
eviction before reloading. An explicitly unloading/unloaded model preserves the
operator's intent and requires an explicit load to rearm admission.

Inferos CI pins this production revision and file hashes, compiles the pure Rust
modules, checks their real outputs with Lean, and audits the proof dependencies.
The proofs establish acceptance-checker soundness and arithmetic/trace safety;
the exercised Rust fixtures are not an unbounded proof of Rust, GPU execution,
allocator internals, or the quality of placement. Lean is not a runtime dependency.
