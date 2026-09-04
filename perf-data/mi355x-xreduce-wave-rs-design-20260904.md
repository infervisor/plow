# MI355X XReduce wave-per-peer reduce-scatter experiment — 2026-09-04

## Scope

`PLOW_XR_WAVE_RS=1` is a default-off experiment in
`d_xreduce_twoshot_mega`. It changes phase 1 only. Both cross-GPU rendezvous,
phase 2, folded-gather and residual semantics, packet ABI, and dispatch geometry
remain unchanged.

For an aligned TP8 slice, waves 0 through 7 each load one logical peer. Every lane
loads eight adjacent BF16 values with one 16-byte transaction into an 8 KiB LDS
tile. After a workgroup barrier, wave 0 accumulates each value in strict logical
rank 0 through 7 FP32 order and writes the same BF16 boundary as the scalar body.
The workgroup still owns the same 512-element tile and advances by
`nblk * 512`. Non-TP8, misaligned, or partial-tile slices use the existing scalar
body.

This isolates AITER's wave-per-peer/LDS idea from its 80-workgroup dispatch and
block-local synchronization. The 2026-09-04 grid sweep showed that applying the
80-workgroup choice alone severely regresses Plow's scalar schedule; both arms in
this experiment therefore retain 256 workgroups.

## Specialist segment

Inlining the arm into the primary prefill interpreter was rejected: the primary
object moved from 8 to 12 VGPR spills, 74 to 76 SGPR spills, 1,348 to 1,364 bytes
of private storage, and 147,504 to 155,696 bytes of LDS. Instead, opt-in packet
emission isolates every `XReduceTwoShot` into a pure opcode segment marked with
`SE_XR_WAVE_RS`. Plowrt accepts that marker only on a pure segment and routes it
to the packet-paired `interp_xreduce{_gq}.elf`; a missing object, missing marker,
pairing mismatch, mixed segment, private segment, or LDS above 16 KiB is fatal.
Older runtimes ignore the spare stream flag and execute the isolated segment on
the ordinary interpreter.

The primary interpreter does not receive `PLOW_XR_WAVE_RS`. Its machine-code
section is byte-identical to the pre-experiment control
(`sha256(.text) = 83ff4baf95c0ca2986d6bd46340980476988894ee2f02fdf53a6ab3ec80a95b5cb`)
and its resource envelope is unchanged. The complete ELF differs in 84 bytes of
non-code AMD metadata encoding after the preprocessor-excluded source was added;
decoded metadata, section layout, and disassembly are identical.

Static gfx950 GQ specialist result:

| ELF bytes | VGPR | SGPR | LDS | private | VGPR spills | SGPR spills | occupancy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19,928 | 82 | 104 | 8,200 B | 0 B | 0 | 0 | 5 |

The exact T8192 graph has 2,971 instructions, 278 eligible calls, and 624 current
ordered segments (4,992 per-XCD queue windows). Reclassifying only those calls
produces 1,088 ordered segments (8,704 windows): **+464 cooperative launches**.
At the measured 6.538 us segment-rendezvous cost, the exact launch overhead
projection is 3.03 ms TTFT. This cost must be included in the full-network
acceptance decision; an isolated kernel win alone is insufficient.

## Gates

1. Build control and candidate with `PLOW_XR_AGG=1`; candidate additionally sets
   `PLOW_XR_WAVE_RS=1`.
2. Static candidate object: wave64, WG512, at least 8 KiB LDS, zero private
   segment, and zero SGPR/VGPR spills.
3. Eight-rank exact full-vector parity with no timeout for T8192/H3584 plain,
   T8192/H7168 plain, and T8192/H7168 folded gather.
4. Three alternating samples per arm. Require improvement in both plain shapes;
   folded gather must not materially regress.
5. Before a network run, require the primary interpreter's identical `.text` and
   resource envelope, and require zero spills/private memory in the specialist.
6. Include the exact added segment launches in the full-network acceptance gate.

No model identity participates in eligibility or runtime dispatch.

## Result: rejected

The exclusive TP8 gate used 256 workgroups, 100 iterations, three alternating
repetitions per arm and shape. All 18 runs reported `parity=PASS`, `timeout=no`,
and `bad=0`.

| T8192 shape | calls | scalar median | wave-RS median | delta |
|---|---:|---:|---:|---:|
| H3584 plain | 92 | 38.581 us | 39.293 us | +1.85% |
| H7168 plain | 94 | 73.479 us | 76.632 us | +4.29% |
| H7168 folded gather | 92 | 97.117 us | 99.588 us | +2.54% |

The 92/94/92 weighted kernel projection is 19.391 ms scalar versus 19.980 ms
wave-RS: +0.589 ms (+3.04%). Adding the measured +464 launch cost projects a
total **+3.623 ms TTFT regression**. The full-network GPU gate was therefore
not run. `PLOW_XR_WAVE_RS` remains default-off and is rejected as a performance
arm; the specialist segment remains as the spill-safe vehicle for a distinct
synchronization experiment.

Artifacts:

- scalar ELF SHA256 `b302232464af0310bc0518bca412f4461b07d7f897d85e97b84c3c26d8cf08e7`
- wave-RS ELF SHA256 `0c67eda7d96b0ef296acabaf14de349eeb0bf0ffb5f3f2bc40adbe767e13b405`
- marked packet SHA256 `dc4222301636dc841708e20639b103de8c9fb0e323b616d74116ca3e54188d1a`
- raw runs `/tmp/xr-wave-rs-isolated/gpu-20260904`
- emit and lease logs `/tmp/xr-wave-rs-isolated/{emit-full.log,gpu-lease.log}`

## Next isolated experiment

Keep the scalar phase-1 body and change only the phase-1-to-phase-2 handoff.
Give each producer workgroup a monotonically increasing ready word in its own
rank's peer-visible scratch. After its phase-1 stores and block barrier, the
workgroup publishes one system-release ready value. The corresponding consumer
workgroup polls the same ready word on each rank, takes one system acquire, and
copies only the elements produced by that workgroup. This removes the global
`nranks*nblk` rendezvous and makes readiness block-local without changing FP32
reduction order, all-gather ownership, or the initial full-partial gate.

First implement this only in the standalone harness behind a new flag. Use
per-XCD-grouped ready storage to avoid different L2 partitions writing the same
cache line; never infer the physical XCD from block index in production. Gate on
the same three exact shapes, then account for the additional ready-word reset
bytes before adapting it to the specialist object and packet allocator.
