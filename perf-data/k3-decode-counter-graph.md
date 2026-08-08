# Kimi-K3 decode: static analysis of the counter gate graph

Source: `plowrt disasm --program 1 --counters /home/lava/models/k3_base/model.pkt`
Asset: 93 layers (69 KDA + 24 MLA), TP8, fp8 KV, mxfp4 experts, `--max-ctx 32768`, `n_cu=256`.
Everything below is STATIC — read off the emitted blob, no GPU, so it is unaffected by machine
contention. It answers "what must each packet update, and what must it wait on".

## The protocol, per packet

Every packet owns exactly ONE counter and signals exactly that one.

    signals 0 counters:     1 packet    (#2458 ArgmaxFin — its counter is dead, see below)
    signals 1 counter :  2458 packets

The wait side is where the fan-in lives:

    waits on 0 counters:     1 packet    (#0 Embed, the root)
    waits on 1 counter :  1995 packets   81.1%
    waits on 2 counters:   370 packets   15.0%
    waits on 3 counters:    69 packets    2.8%
    waits on 4 counters:    24 packets    1.0%
    ------------------------------------------
    3038 dependency edges over 2459 packets

A counter's `threshold` is its producer's block count: the producer is complete when all `b` of its
workgroups have bumped. A consumer polls that counter from every one of ITS OWN `b` workgroups.

## Traffic per token

    bumps  (signal ops) 324,940   = sum of `blocks` over packets
    polls  (wait   ops) 454,942   = sum of CONSUMER `blocks` over edges
    ---------------------------------------------------------------
    total               779,882 counter operations per token

**The wait side is 1.4x the signal side.** This matters because cost is not symmetric: the release
RMW is a posted, non-returning `global_atomic_add` measured at 0.07 us per workgroup, whereas the
`buffer_wbl2` / `buffer_inv` cache maintenance around it is per-L2 and serialises (see
the design notes, grid sweep 32/16/8/4/1 WG per XCD -> 13.20/4.35/2.73/1.85/1.05 us).

## Poll cost is set by the CONSUMER's width, not the producer's

This is the central structural finding and it is counter-intuitive.

    producer width class   counters   polls caused
    narrow  b<=2                671    185,184   44.1%
    mid     b<=64               488     44,069   10.5%
    wide    b>64               1300    190,361   45.4%

671 narrow counters — 27% of them — cause 44% of all polling. A `b=1` packet bumps once, but if it
feeds a `b=256` consumer, that consumer polls it with all 256 workgroups.

**770 edges have a consumer at least 64x wider than the producer.** The recurring shapes:

    AttnRes(b=1)       -> Gemv(b=256)               x165   42,240 polls
    RmsNorm(b=1)       -> Gemv(b=256)               x140   35,840 polls
    AttnRes(b=1)       -> GemvGlu(b=256)             x92   23,552 polls
    MoeRouterTopk(b=1) -> MoeGroupGluFp8Blk(b=256)   x92   23,552 polls
    AttnRes(b=1)       -> Gemv(b=224)                x92   20,608 polls
    AttnRes(b=1)       -> GemvQkvg(b=256)            x69   17,664 polls
    AttnRes(b=1)       -> Gemv(b=128)                x69    8,832 polls
    MlaOutGate(b=1)    -> Gemv(b=256)                x24    6,144 polls

The worst single counters are all the same shape — an `AttnRes` at `b=1` with fan-out 4:

    ctr 68 thr=1 by #68 AttnRes fan=4 polls=832
      -> #69 Gemv(b=256), #70 Gemv(b=256), #71 Gemv(b=64), #80 Gemv(b=256)

By opcode, counting bumps + polls it causes:

    opcode                pkts     bumps     polls   % of traffic
    Gemv                   816   175,662    71,956      33.3%
    AttnRes                187        187   115,758      15.6%   <- 187 bumps, 115k polls
    GemvQkvg                69    17,664    35,328       7.1%
    MoeGroupGluFp8Blk       92    23,552    23,552       6.3%
    GemvGlu                 92    23,552    23,552       6.3%
    RmsNorm                116        116    35,840       4.8%   <- 116 bumps, 36k polls
    KdaGatedNorm            69    17,664    17,664       4.7%
    KdaConv3                69    17,664    13,248       4.2%
    KdaStateStepG           69    13,248    17,664       4.2%
    XReduce                278     3,248    24,008       3.7%
    MoeGroupDownFp8Blk      92    23,552       644       3.2%
    MoeRouterTopk           92         92    23,552       3.2%

`AttnRes` and `RmsNorm` are the clearest cases: 303 packets contributing 303 bumps but 151,598
polls, i.e. 19% of all counter traffic from 12% of the packets, purely because they are `b=1` gates
in front of `b=256` consumers.

## The graph is a chain, and that sets a hard floor

    packets        2459
    counters       2459   (one per packet; 1 dead)
    critical path  1739
    mean width     1.41 packets per level
    liveness       max 5 concurrent, p50 2, p99 5

Only 5 counters are ever live at once and the median is 2. There is almost no independent work to
overlap — the token is a 1739-deep dependency chain with an average of 1.41 packets per level. Per
layer that is 26.4 packets, 18.7 deep.

Multiply the chain depth by the measured per-packet protocol cost:

    per-packet cost                      chain floor (1739 packets)
    5.72 us  (empty b=256 packet, real trace)     9.95 ms/token
    13.20 us (ctr_convergence baseline)          22.95 ms/token
    3.76 us  (both fences dropped)                6.54 ms/token
    3.46 us  (HIER2, per-XCD leader)              6.02 ms/token

**The chain alone spends the entire 10.0 ms budget at today's per-packet cost, before a single byte
of weight is read.** Even with HIER2 landed, the graph structure still costs 6.02 ms of the 10.0 ms
target. Reaching 100 tok/s requires attacking BOTH terms:

    to fit protocol in 2.0 ms at 5.72 us/packet -> chain must be <= 349 packets
    to fit protocol in 2.0 ms at 3.46 us/packet -> chain must be <= 578 packets

against 1739 today.

## Free wins already visible in the graph

1. **69 redundant edges -> 52,992 removable polls (11.6% of all polling).** These are edges implied
   by transitivity: e.g. `2 -> 8` is redundant given the path `[2, 6, 7, 8]`. Each removal saves 256
   polls. The disassembler already computes the transitive reduction (`edges_tr` 2969, `polls_tr`
   401,950) — the emitter is simply not applying it. No semantic change, no kernel work.

2. **1 dead counter** (#2458 `ArgmaxFin`, 1 wasted bump). Trivial, listed for completeness.

## What this says to do, in order

1. **Apply the transitive reduction in the emitter.** 11.6% of polls, mechanical, zero risk.
2. **Fuse the narrow gates into their consumers.** `AttnRes`/`RmsNorm`/`MoeRouterTopk` at `b=1` in
   front of `b=256` consumers are 19% of counter traffic AND they lengthen the chain by one level
   each. Fusing removes the packet, the edge and the chain level together — this is the same lever
   that already bought 278 packets and ~0.97 ms via norm fusion. `AttnRes` alone is 187 chain levels.
3. **Only then chase per-packet cost** (HIER2). It is worth ~2.3x on the floor, but on a 1739-deep
   chain that still leaves 6.02 ms.

Note that (2) is worth more than it looks: shortening the chain scales the WHOLE protocol term,
whereas (1) and (3) scale only parts of it.

---

## FOLLOW-UP: what (1) and (2) actually did

(1) **is landed.** The emitter applies the transitive reduction — it reports `69 ... 207 wait
entries removed` at emit time, and `disasm --counters` now finds 0 remaining redundant edges.

(2) **is IMPLEMENTED for `RmsNorm` only, and is OFF by default pending a numerical issue**, and the numbers above oversold it in one specific way. See
`perf-data/k3-narrow-gate-fusion.md`. In short:

* **The chain-level claim held exactly.** Folding all 116 `RmsNorm` gates into their GEMV
  consumers took the decode program 2459 -> 2343 packets, 2969 -> 2853 edges, and the critical
  path **1831 -> 1715**. One deleted packet, one deleted edge, one deleted level, as claimed.
* **The counter-traffic claim did NOT hold.** "19% of counter traffic" implies fusing recovers
  that polling. It does not: polls fell 400,110 -> 399,994, i.e. **0.03%**, only the deleted
  packets' own waits. Fusing a narrow gate into a wide consumer REDIRECTS its polls rather than
  removing them — the consumer polled the `b=1` norm with 256 workgroups and now polls the norm's
  producer with 256 workgroups. Poll traffic is a function of consumer width and edge count, and
  this fusion changes neither. Argue for these fusions on chain depth alone.
* **`AttnRes` is a LOSS and was not attempted.** Its fan-out is 3 or 4 (not 1), and its mix spans
  `nb+1` rows of 7168 against a GEMV that stages one — 8.9 GB/token of extra reads to buy 1.07 ms.
  The 187 levels in the paragraph above are not available at this price.
* **`MoeRouterTopk` is fan=2, not fan=1.** The table above lists only `MoeGroupGluFp8Blk`;
  `MoeGroupDownFp8Blk` reads `route_tab` too. Still open, still worth 92 levels.
