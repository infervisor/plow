# Packet-domain NUMA placement vs `cu % nodes` — EPYC 9654, AVX-512

Following the packet's L2 locality domains when placing CPU executors measured
**1.5x slower** than the `cu % nodes` round-robin it was meant to improve on, and
was never faster at any node count tried. It ships **off** (`--cpu-l2-place`,
`PLOW_CPU_L2_PLACE`), and `node_plan` declines the losing shape even when it is on.

## Why the idea does not pay here

An L2 domain says which slices share a **GPU** cache. It says nothing about which
weights they touch, and CPU model tensors are interleaved across every node anyway,
so grouping cus by domain creates no memory locality on this engine.

It does cost the round-robin's balance, because nothing makes domains equal in cost.
Under the blocked map (`domain = cu / sms_per_partition`) the low-numbered cus carry
every op that is sliced narrowly, so grouping them puts a third of the packet on one
node. With `n_cu = 132` over 8 domains of 18 SMs, the last domain also gets a ragged
6 cus instead of 18:

```
cus per domain: [18, 18, 18, 18, 18, 18, 18, 6]
```

Work per node, in stream entries — the unit the static walk executes, and the
quantity the makespan follows (`cargo run --example l2_probe -- model.pkt 8`):

| program | placed max/min | round-robin max/min | busiest node, placed vs RR |
| --- | ---: | ---: | ---: |
| T=128 prefill | 3.79x | 1.04x | 14131 vs 13086 |
| T=512 prefill | 3.01x | 1.06x | 14131 vs 13328 |
| T=1024 prefill | 3.01x | 1.06x | 14131 vs 13328 |
| T=1 decode | 4.08x | 1.12x | 7377 vs 5676 |

The busiest-node ratio (1.06–1.30x) sets the sign but under-predicts the size of the
end-to-end loss, so the balance metric is a guard, not a model of the cost.

The guard applies that test **per program**, not to their sum. Programs are
alternatives — a prefill bucket or the decode program is chosen per dispatch — so each
one's own busiest node is its own makespan. Summing first is unsound: two programs at
200/2 and 2/200 across a pair of nodes total 202/202 and look perfectly balanced,
while each on its own is a 100x spread. `node_plan` takes one work row per program and
requires every row to hold.

## Measured

Interleaved A/B (`place=1`, `place=0`, repeated) on one blob, so drift and page-cache
state hit both arms equally.

8 nodes, 192 threads, prompt 128, batch 1:

| rep | placed TTFT ms | RR TTFT ms | placed step ms | RR step ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11364.0 | 7535.8 | 1939.6 | 1253.1 |
| 2 | 11416.5 | 7141.8 | 1872.2 | 1275.9 |
| 3 | 11237.8 | 7012.8 | 1841.0 | 1243.5 |
| **mean** | **11339.4** | **7230.1** | **1884.3** | **1257.5** |

**TTFT 1.57x slower, decode step 1.50x slower.** The three repeats span 1.6% (placed)
and 7.2% (RR), so the gap is far outside run-to-run noise.

2 nodes (`--numa 0,1`), 48 threads:

| rep | placed TTFT ms | RR TTFT ms | placed step ms | RR step ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11916.7 | 9686.5 | 1087.1 | 846.5 |
| 2 | 9999.5 | 9356.0 | 851.5 | 842.7 |

Smaller and noisier — 1.15x TTFT, 1.14x step on the means — but still never faster.

After the guard landed, both settings take the round-robin and become one population
(TTFT 6361–8666 ms, step 1100–1520 ms across four runs straddling both flags), which
is the ±20% noise band this bench actually has at 192 threads.

## Can a change rescue it? No — balance was not the whole cause

The obvious objection to the above is that the losses are load imbalance, not the idea, so
fix the imbalance and the idea should pay. It does not.

`n_cu = 132` does not divide by the 18 SMs per partition, which left domain 7 with 6 cus
instead of 18. Recompiling at `--n-cu 144` removes exactly that:

| | n_cu=132 | n_cu=144 |
| --- | ---: | ---: |
| emitter skew, prefill | 66.8–75.5% | **0.3%** |
| placed vs round-robin peak, T=512 | 14131 vs 13328 (6.0% worse) | 14131 vs 14111 (**0.14% worse**) |
| placed vs round-robin peak, T=1 decode | 7377 vs 5676 (30% worse) | 7377 vs 5977 (23% worse) |

Prefill is now balanced to 0.14%, which makes it a fair test of locality alone. Forcing the
guard open (a temporary local patch, not a shipped flag) and A/B-ing at prompt 512, 192
threads:

| rep | placed TTFT ms | RR TTFT ms | placed step ms | RR step ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18250.0 | 15286.5 | 1161.0 | 1136.7 |
| 2 | 18582.7 | 15042.0 | 1094.6 | 1121.4 |
| 3 | 18999.7 | 14786.8 | 1138.8 | 1151.0 |
| **mean** | **18610.8** | **15038.4** | **1131.5** | **1136.4** |

**Prefill is still 1.24x slower with the balance objection removed.** Decode became neutral
(1.00x), which is consistent: at `n_cu = 144` no node is left mostly idle, so the earlier
1.50x decode loss was the 6-cu node, and what remains in prefill is not imbalance at all.

The reason total balance does not rescue it is that **per-node work balance is not per-op
concurrency**. Ops do not all span the full width — in the T=512 program `PLOW_DOP_GEMM` is
291 instructions averaging 89.6 of 144 slices:

```
     8 PLOW_DOP_GEMM      insts=291   slices=26064     # 89.6 slices per instruction
     7 PLOW_DOP_SOFTCAP   insts=1     slices=64
    18 PLOW_DOP_ARGMAX_FIN insts=1    slices=1
```

A contiguous domain map confines a k-slice op to `ceil(k / 18)` nodes, so a 90-slice GEMM
runs on 5 of 8 nodes and the other 3 idle through it; the round-robin spreads the same 90
slices over all 8. Summed over a program those totals even out — hence the 0.14% — while
every barrier still waits on a narrower machine. No domain-to-node assignment can fix that,
because the narrowing is in the domain map's contiguity in cu index, not in the assignment.

Nor is there locality to win it back. Model tensors of at least 256 KiB are `mbind`
interleaved across every node, so a weight read is ~1/8 local wherever the reading thread
sits — placement cannot change it. Making the idea pay would need the weights placed per
node to match, which is NUMA tensor parallelism and a different design (see the note at the
end of the runtime doc).

## What was NOT measured

* A blob under the AMD round-robin domain map. It reduces to `cu % nodes` exactly when
  the domain count equals the node count, so an 8-XCD blob on this 8-node host is the
  identical mapping and cannot show a difference; a node count other than 8 would be
  needed, and a gfx950 blob does not load on the CPU backend (`KV-row site 689 exceeds
  decode program 6's 676 instructions`). Note the round-robin map would also spread each
  op across nodes, which is the property the blocked map lacks — so it is the shape most
  likely to come out neutral, not the shape most likely to win.
* The global-queue path (`PLOW_CPU_GQ=1`), which is itself off by default.
* Any host that is not this one.

## Reproducing

Compile a placed, CPU-loadable blob (H100 is the default `--gpu` and has an 8-GPC L2
partitioning; `PLOW_L2_PLACE=1` forces placement on a non-CDNA target):

```sh
PLOW_L2_PLACE=1 cargo run --release -p plowc --bin plowc -- \
  --hf-dir <gemma4-checkpoint> --gpu h100 \
  --batch 1,4 --seq 128,512 --max-ctx 2048 --phase both --emit devblob \
  --out target/l2-place-on
```

Inspect what it carries, what each node would run under either mapping, and whether
the runtime accepts the plan (it reports `CANDIDATE` and `ACCEPTED` separately, and
calls the engine's own `cu_domains`/`node_plan` so it cannot drift from the decision
the runtime actually makes):

```sh
cargo run --release -p plowrt --no-default-features --features cpu \
  --example l2_probe -- target/l2-place-on/model.pkt 8
```

On this blob it prints `ACCEPTED: no` — every one of the four programs has a busier
peak node under the plan than under the round-robin.

A/B the two placements:

```sh
for place in 1 0; do
  PLOW_CPU_L2_PLACE=$place cargo run --release -p plowrt \
    --no-default-features --features cpu --example cpu_bench -- \
    target/l2-place-on/model.pkt <checkpoint> \
    --prompt-lens 128 --decode 8 --isa avx512 --threads 192
done
```

## Host

EPYC 9654 96-Core, dual socket, 192 physical / 384 logical cpus, 8 NUMA nodes
(24 cores each), 2.2 TiB RAM. AVX-512 F/BW/VL/BF16/VNNI, no AMX.
Model: Gemma-4 31B dense bf16, 60 layers, hidden 5376, 57.18 GiB of weights,
`n_cu = 132`, 8 L2 domains, blocked map.
