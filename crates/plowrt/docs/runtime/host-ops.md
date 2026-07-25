# §P Sampling & tokenization as counter-gated host packets

Sampling (and tokenization) are **nodes in the packet stream**, not a
post-schedule step: a packet that waits on the logits producer's counter, runs
the instant logits are ready — unblocked from the tile packets — executed by a
dedicated **HostExecutor** (or, when the compiler chooses, an SM).

## Why it fits the packet design

The ABI already had `ResourceKind::{Host,Sm}`, `TaskKind::Host`, and counter
gating. This adds the missing pieces:

- **Opcodes** (`crates/packet`): `FAMILY_TOKEN = 7`, `SAMPLE = 0x0700`,
  `TOKENIZE = 0x0701`, and a compact `#[repr(C)] TokenBody { in_slot, out_slot,
  kind, vocab, arg }` (16 B) behind `Body::Token`. Placement-agnostic: the same
  body drives host and GPU execution.
- **Compiler-flag gated, decode-only** (`crates/plowc`): `--emit-sample` appends
  the SAMPLE packet to **decode** buckets only (prefill has no per-token draw);
  `--emit-tokenize` prepends a TOKENIZE packet to **prefill**. Both off by
  default, alongside the existing `--counter-elim`/`--prefetch`/… passes.
- **Runtime** (`crates/plowrt`): `exec/host.rs::HostExecutor` runs the host op
  when its packet fires.

## Compiler decides host *or* GPU

Sample is an ordinary op; **placement picks the resource**. On `Host` it becomes
a `SAMPLE` host packet run by the CPU sampler; on `Sm` it's a compute packet with
the *same body* resolved to a GPU argmax/softmax-sample kernel — exactly how GEMM
variants resolve per backend. Sampling on-GPU is a different placement, not a
separate path. (The `--net` path emits the host placement; the GPU kernel is a
stub.)

## How the gating works

`inject_sample_packet` ([plowc/src/lib.rs](../../../plowc/src/lib.rs)) adds a
fresh counter; **every terminal instruction** (no successors — the logits/output
stores) increments it, and the SAMPLE packet waits on it. So SAMPLE fires the
instant the whole output stage — including logits — is done, concurrently with
nothing left to block it. Tokenize is symmetric at the head: root compute
instructions gate on a TOKENIZE counter, so the first matmul can't start until
`tokens` exist (without serializing independent weight DMAs).

## Runtime execution — the observer seam

The elegant fit: `HostExecutor` is a `StepObserver`
([exec/host.rs](../../../plowrt/src/exec/host.rs)). The interpreter
(`device/cpu.rs::run_streams`) calls `on_fire` the moment a packet fires and
**before** its successors increment. So the HostExecutor runs the SAMPLE's work
(argmax / top-k / top-p via `text::sample`) in `on_fire`, writes the token, and
the interpreter then bumps the successor that unblocks the consumer. No new
execution path — one walk drives host and device packets. Per-request params
(temperature, top-p, seed) come from the API request through the indirection
table, so the cached packet stays static.

## A general host-op pattern

`FAMILY_TOKEN` + `Body::Token{kind}` and the HostExecutor generalize beyond
sampling: the same counter-gated-host-packet mechanism carries **multi-model
pipeline hand-off** (a host packet gating on model A's counter to kick model B),
**host post-processing** (detokenize, JSON/grammar finalize, guards), and
coordination. New host steps are new `kind`s dispatched by the HostExecutor.

## Verified

- `packet`: `Body::Token` round-trips; `TokenBody` is 16 B, 4-aligned (size test).
- `plowc`: `--emit-sample` on a decode `--net` bucket emits a `SAMPLE` host packet
  whose `wait` counter has real producers; the schedule simulates deadlock-free
  and SAMPLE starts after its producers (`net_e2e_sim.rs`).
- `plowrt`: a gated SAMPLE packet fires on the HostExecutor after its producer,
  writes the greedy-argmax token, and bumps its successor; removing the producer's
  increment deadlocks and the SAMPLE never runs (`host_ops.rs`).

## CLI

```sh
# emit a host SAMPLE at the decode tail
plowc --net model.json --phase decode --emit-sample --out ./out
# inspect it — the per-packet log shows SAMPLE gated on the output counter
plowrt simulate --assets ./out
```
