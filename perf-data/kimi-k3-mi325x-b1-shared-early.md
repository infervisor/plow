# Kimi-K3 B1 early shared-expert screen

Status: rejected.

The candidate moved each B1 `GemvGlu` shared-expert packet before the routed
TopK chain. The shared-down projection retained its routed `XReduce`
dependency, so TP slot-A ownership and all arithmetic were unchanged.

The control and candidate have 2,274 decode instructions, critical-path depth
1,715, identical tensor tables, identical prefill programs, and identical
sorted decode-instruction multisets. Only 92 `GemvGlu` packets move.

## Result

The five-layer TP8 block screen ranked the candidate correctly:

| arm | 2,048-step samples, ms/token |
|---|---|
| control | 4.212, 4.212 |
| early shared | 4.127, 4.138 |

The full `plowrt serve` gate used the vLLM 0.27 client, C1, three measured
requests, random input 32, output 256, and one warmup:

| arm | mean TPOT | output tok/s |
|---|---:|---:|
| control | 48.832 ms | 20.081 |
| early shared | 48.517 ms | 20.208 |

The 0.315 ms improvement is only 0.64%, far below the multi-millisecond
promotion threshold and the remaining 28.5 ms gap to 50 tok/s. All three
generated texts are byte-identical across arms; both complete 768/768 output
tokens with zero failures, empty errors, and a clean compact TP audit.

Packet SHA256:

- control: `66eb9409ea5f928bbd2a68359eb85a659e16b15ddc5e50e695d56fd53861d43c`
- candidate: `7a386b392d157690457e17130cd8e7b4107c3dc2ec49d4aee7d5a3665865960f`

Evidence:

- `/tmp/k3-b1-shared-early-serve/{ctl,cand}/result.json`
- `/tmp/k3-b1-shared-early-serve/{ctl,cand}/server.log`
- `/tmp/k3-b1-shared-early-long-{ctl,cand}-*.log`

The candidate emit flag was removed after measurement.
