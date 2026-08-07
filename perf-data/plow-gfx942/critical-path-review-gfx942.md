# Runtime Critical Path Review, re-measured on gfx942 (MI300X)

Every number in the review itself is **GLM-5.2 TP8** (22.7 ms/token, host 677 µs
= 2.96%). GLM-5.2 does not load on this box — ~700B, 181 GB/card at TP4 — so
none of it is reproducible here and none of it was assumed. This file is the
re-measurement on the model that DOES run: **Gemma-4-12B fp8, single GPU**.

Harness, identical for every row below:

```
PLOW_FP8_DIR=/workspace/models/gemma-4-12B-it-fp8 ROCR_VISIBLE_DEVICES=0 \
  ./target/release/plowrt amd-bench \
  --blob       /workspace/assets/gfx942/g12b-default/model.pkt \
  --hsaco      /workspace/assets/gfx942/hsaco-cprv \
  --checkpoint /workspace/assets/gfx942/g12b-fp8/checkpoint \
  --steps 128 --ctx 4096
```

Objects: `PLOW_OCC4=1 JOBS=16 scripts/build_gfx942.sh`, built from this branch.
Binary: `cargo build --release -p plowrt --features hsa`, run INSIDE `nix develop`.
`rocm-smi --showuse` read 0% before every arm.

Run-to-run reproducibility measured, not assumed: 3 back-to-back stock reps at
`--steps 48` gave 12.189 / 12.181 / 12.198 ms — **sd 0.07%, spread 0.14%**.
The 0.4% figure quoted for this box is conservative; 0.15% is what it actually
does at `--steps 128`.

---

## 1. The host/GPU split — the number the whole exercise turns on

`PLOW_DSTEP_LOG=1`. The single-GPU decode path had **no `dstep` instrumentation
at all** (only `exec::amd_tp` and the serve mux were wired), so this had to be
added before it could be answered — `AmdEngine::run` / `decode_step` and the
`amd-bench` step loop, in this branch.

Steady-state window, n=64 tokens, ctx 4096+:

| phase | µs/token | % of token |
|---|---|---|
| `pre  decode_prepare` (kvrow patch + 2 scalars) | 31.7 | 0.27% |
| `pre  rearm` (counters + GQ cursor) | 56.2 | 0.48% |
| `pre  enqueue` (1 AQL launch) | 0.6 | 0.01% |
| **`GPU  drain`** | **11 532** | **99.1%** |
| `post read_sampled` (4 B D2H) | 10.2 | 0.09% |
| **HOST TOTAL** | **98.7** | **0.85%** |

**The host phase is 0.85% of a Gemma-4-12B single-GPU decode token, not 2.96%.**
The review's guess that it would be "roughly double" on this model is wrong in
the other direction: it is a THIRD of the TP8 share, because the TP8 host phase
is dominated by per-rank work (8× seed, 8× prepare, 8× rearm, `zero_xctr`) that
a single GPU simply does not have.

So the total available prize from removing ALL host work on this model is
**98.7 µs = 0.85%**, of which `rearm` is 57%.

Two facts that are not visible from the code and that change what `rearm` IS:

* the decode program has **`n_counter` = 15 550**, so one counter bank is
  **1 943 KiB**. `rearm` is not "zeroing a few gates" — it pushes **~2 MB of
  zeros host→device, per token**. 1.99 MB / 56 µs ≈ 36 GB/s, i.e. it is
  PCIe-**bandwidth**-bound, not latency-bound.
* the count is that large because the blob is L2-placed: `hier_base` reserves
  `3 · n_inst · l2_domains` counters of maintenance scratch at the tail.

## 2. Option A — counter/cursor double-buffer. SHIPPED.

`PLOW_CTR_DBUF`, default ON, `=0` reverts.

The review's §3 prices two ways to get the synchronous `rearm` off the critical
path and concludes Option A (two banks, host clears the stale one during GPU
execution: no kernel change, nothing added to the GPU critical path) is
"strictly better" than Option B (kernel prologue self-clear: ~2 µs ON the GPU
critical path, plus a kernel/ABI change). Its own priority table then lists B as
P1 and A as P5. **§3's reasoning is right and the table is wrong** — A is what
is implemented here, and on this box the choice is not close: B would add its
2 µs to the 11.5 ms drain while removing 56 µs of host, and A removes the same
56 µs while adding nothing to the drain.

Shape: `d_ctr` and `AmdGq::d_cursor` are allocated `2 ×` and zeroed once at
load; `kernarg` addresses `base + bank·span`; `run` **enqueues first, then
clears the stale bank, then drains**, so the clear's blocking SDMA round trips
overlap the megakernel. `memcpy_htod_pinned` is `hsa_amd_memory_async_copy` +
`hsa_signal_wait` — synchronous to the host, but on the SDMA engine, which the
persistent cooperative megakernel does not contend for.

**A/B, one binary, interleaved, 8 reps per arm, `--steps 128`:**

| rep | OFF (single bank, sync) | ON (double bank, overlapped) |
|---|---|---|
| 1 | 11.903 | 11.807 |
| 2 | 11.890 | 11.825 |
| 3 | 11.884 | 11.810 |
| 4 | 11.868 | 11.818 |
| 5 | 11.866 | 11.777 |
| 6 | 11.828 | 11.765 |
| 7 | 11.855 | 11.798 |
| 8 | 11.870 | 11.802 |
| **mean** | **11.8705** | **11.7978** |

**−72.8 µs/token, −0.61%.** The arms do not overlap at all: the slowest ON rep
(11.825) is faster than the fastest OFF rep (11.828). Against a 0.15% spread
this is ~4× the noise floor.

Where it went, from `PLOW_DSTEP_LOG=1` on the same two arms:

| phase | OFF | ON |
|---|---|---|
| `rearm` | 56.2 (in FRONT of the launch) | 73.3 (BEHIND it, overlapped) |
| `GPU drain` | 11 534 | 11 475 |
| TOKEN | 11 634 | 11 591 |

The drain shrinks by ~57 µs — exactly the `rearm` that no longer precedes it.
`rearm` itself gets *slower* (56 → 73 µs) because the SDMA copy now runs while
304 CUs are hammering HBM; that is the price of the overlap and it is paid out
of a window the host was going to spend blocked in `drain` anyway.

Costs:

* **VRAM +2.3 MiB** for this model (decode 1.90 MiB + three prefill programs
  ~0.35 MiB). Logged at `RUST_LOG=plowrt::exec::amd=debug` as `counter banks`.
* **TP is untouched.** `rearm_prog` still clears the CURRENT bank synchronously
  and `AmdTpGroup` never flips, so a TP rank is byte-for-byte on the old path.
  Prefill (`run_segmented`) likewise.

**It survives the serve path, which is the one that ships.** `amd-bench` is a
tight host loop; serving adds a mux tick, detokenisation and an SSE frame per
token, any of which could have absorbed the win. One server per arm, 3 timed
256-token greedy completions each after an untimed warm request, blocks run
`0 1 0 1` then `1 0 1 0` so clock drift cannot alias onto the arm:

| block | OFF | ON |
|---|---|---|
| 1 | 11.687 / 11.740 / 11.749 | 11.714 / 11.724 / 11.705 |
| 2 | 11.776 / 11.770 / 11.785 | 11.700 / 11.704 / 11.720 |
| 3 (reversed) | 11.773 / 11.794 / 11.792 | 11.700 / 11.705 / 11.704 |
| 4 (reversed) | 11.779 / 11.779 / 11.786 | 11.700 / 11.708 / 11.714 |
| **mean of 12** | **11.7675** | **11.7082** |

**−59.3 µs, −0.50%** over all 24 requests. The ON arm is also far steadier: its
12 reps span 0.014 ms, the OFF arm's span 0.107 ms. The single OFF block that
overlaps ON is block 1 — the first server started in the session, before the
card had settled; dropping it gives OFF 11.7815 and **−73.3 µs, −0.62%**, which
is the `amd-bench` number to three digits. Blocks 2–4 separate completely.

**Correctness gate — the serve path, not `amd-bench`'s `last id`:**
`plowrt serve --assets /workspace/assets/gfx942/g12b-cprv --port 8099`,
`/v1/chat/completions`:

* "What is the capital of France? Answer in one word." → **`Paris`**, both arms.
* 300-token greedy generation, `temperature: 0`: **byte-identical between
  `PLOW_CTR_DBUF=1` and `PLOW_CTR_DBUF=0`** (1359 chars). Open-ended prose is
  fluent and correct (Rayleigh scattering).

## 3. P3 — coalesce the two decode scalar H2Ds. NOT APPLICABLE. Not shipped.

The review proposes fusing the `pos` and `kvlen` `memcpy_htod_pinned` calls in
`decode_prepare_batched` into one packed copy. That requires the two tensors to
be device-adjacent **at the batch stride**. Measured layout on this blob
(`RUST_LOG=plowrt::exec::amd=debug`, `decode scalar tensors`):

```
in.pos   = 0x7fe34fc10000 + 65536
in.kvlen = 0x7fe34fc20000 + 4
```

They ARE adjacent — but `in.pos` is **65 536 B, not `batch·4`**. It is sized by
`max_ctx` (16 384 positions), because PREFILL writes a whole position vector
through it; `max_ctx` is in fact *derived* from that tensor's size in
`AmdEngine::load`. So a single packed copy would have to push 64 KiB of padding
between the two live scalars, and would write junk over `in.pos[1..16384]` —
storage prefill owns.

The prize if it worked at all: one `hsa_amd_memory_async_copy` round trip. That
round trip is measured at **~10 µs** (`post read_sampled` is exactly one 4 B
copy and costs 10.1 µs; `decode_prepare`'s 31.7 µs is three copies). 10 µs is
**0.085% of the token — 6× below the 0.15% run-to-run spread**, i.e.
unmeasurable end-to-end on this box even if the layout allowed it.

Recorded as a negative so it is not re-bought: **P3 is layout-illegal here and
worth 0.085% if it were legal.**

## 4. Not verifiable on this box

* **P0 — KDA conv → state-step fusion.** Gemma-4 has no KDA layers. GLM-5.2 and
  Kimi-K3 do not load here. Cannot be measured; not attempted.
* **P4 — device-side `xctr` swap.** TP-only, and TP does not work on this box
  for the models present. Not attempted.
* **P2 — `fuse_norm_gemv` default ON.** It is default-off because of an
  UNRESOLVED end-to-end divergence. Flipping it needs a bit-exactness proof
  that is a larger task than the review's whole priority table. Not attempted;
  the flag is untouched.

## 5. What is left on the host after this

`decode_prepare` 31.7 + `enqueue` 0.6 + `read_sampled` 10.2 = **42.5 µs, 0.36%
of the token**, and `read_sampled` is a post-drain readback the client is
waiting on regardless. There is no second Option A here. The interesting number
this exercise surfaced is not the host phase at all: it is the **fixed ~10 µs
floor of a `memcpy_htod_pinned` / `memcpy_dtoh_pinned` round trip** (signal
create → `hsa_amd_memory_async_copy` → blocking `hsa_signal_wait` → destroy).
A 4 B copy costs 10.1 µs; `rearm`'s 1.99 MB copy costs ~46 µs of the 56 —
**500 000× the bytes for 4.6× the time**. Every "just push one more scalar"
therefore costs 0.085% of a token whatever its size, and the only copies worth
attacking are the ones that can be moved off the critical path entirely, which
is what §2 did to the biggest of them.
