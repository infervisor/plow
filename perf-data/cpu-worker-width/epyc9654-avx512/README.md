# Worker width vs packet executors — EPYC 9654, AVX-512

The automatic CPU worker width ignored the packet's executor count, so a dense model on a
wide host spawned far more workers than the packet could ever give work to. Capping the
automatic width at `--n-cu` measured **8.3x faster decode** and **1.55x faster TTFT**.

## The defect

`cu_map` deals cus `0..n_cu` over the pool, so a worker with index past `n_cu` owns nothing
— in every program, permanently. It is not idle-for-this-program, it is unusable for the
whole model. Each run it still wakes on the control ring, finds an empty stream, returns,
and then spins `--cpu-spin-us` (2000 µs default) before parking. A healthy decode step here
is ~400 µs, so each surplus worker burns roughly five steps' worth of spin per gap, on the
cores the working set needs.

The old width rule never looked at `n_cu`: it picked physical cores for MoE and *logical*
CPUs for a dense single-row decode. On this host that is 384 workers against a 144-cu
packet — 240 of them unusable.

## Measured

Gemma-4 31B dense bf16, 57.18 GiB, `n_cu = 144`, `--prompt-lens 512 --decode 6`, AVX-512,
means of 3 runs:

| workers | TTFT | decode step | idle workers |
| ---: | ---: | ---: | ---: |
| 384 (all logical — the old automatic width) | 20499 ms | 3459 ms | 240 |
| 192 (all physical) | 16235 ms | 1196 ms | 48 |
| **144 (`n_cu` — the new automatic width)** | **14078 ms** | **415 ms** | **0** |

The penalty tracks the idle count, not the thread count: 240 idle is 8.3x on decode, 48 idle
is 2.9x, 0 is the baseline. Weight bandwidth over the same runs goes 52 → 164 GB/s.

Confirmed after the fix, with the width left automatic (`--cpu-threads 0`): 144 workers
selected, 13245 ms TTFT and 418 ms per step — i.e. it reproduces the explicit-144 arm.

## The rule

Auto width = min(model-preferred topology width, `n_cu`), floor 1.

Two exemptions, both deliberate:

* **An explicit `--cpu-threads` is honoured as given**, above `n_cu` included. The surplus is
  the caller's to spend; the flag exists for hosts that disagree with the built-in rule.
* **`--cpu-global-queue` is exempt.** Under the global queue every worker claims from the
  shared per-(segment, domain) window rather than owning a static stream, so workers past
  `n_cu` do useful work instead of idling. The cap would be a real loss there.

## Relation to the per-program narrowing that was rejected earlier

`exec::cpu::engine` already carries a note explaining that narrowing the pool *per program*
measured worse — an idle worker polls on the SMT sibling of a busy core and that tax exceeded
the gain. This cap is not that. A per-program narrowing leaves a worker idle for one program
and busy for another, so there is a width tradeoff to lose; a worker past `n_cu` is idle for
every program, so there is none. The two coexist: the width is still chosen per model, and
then capped.

## Host

EPYC 9654 96-Core, dual socket, 192 physical / 384 logical cpus, 8 NUMA nodes, 2.2 TiB RAM.
AVX-512 F/BW/VL/BF16/VNNI, no AMX.

## Reproducing

```sh
for t in 0 192 144; do
  cargo run --release -p plowrt --no-default-features --features cpu \
    --example cpu_bench -- <bundle>/model.pkt <checkpoint> \
    --prompt-lens 512 --decode 6 --isa avx512 --threads "$t"
done
```

`--threads 0` is the automatic width; the `engine:` line reports what it chose.
