# Kimi K3 MI325X shared-down slot-D experiment

Status: rejected.

The candidate gives the shared-expert down projection an independent fourth TP
peer partial slot. Batched decode can issue shared gate/up, SiTU, and shared
down before routed TopK; the final collective reduces slot D while gathering
the routed partial from slot C. Arithmetic, instruction count, weights, and
decode objects are unchanged.

## Result

Matched `plowrt serve` TP8, vLLM 0.27 client, C32/N32, input 32, output 512,
one warmup, seed 0:

| arm | output tok/s | mean TPOT | median TPOT | median TTFT |
|---|---:|---:|---:|---:|
| control | 118.975 | 246.528 ms | 247.155 ms | 7733.972 ms |
| slot D | 117.543 | 249.681 ms | 250.321 ms | 7752.056 ms |

Slot D regresses output throughput 1.20% and mean TPOT 1.28%. DSTEP samples
place control GPU drain near 231 ms/step and the candidate near 234--236
ms/step. The extra overlap does not repay the larger peer region/protocol.

Both arms completed 32/32 requests and exactly 16,384 output tokens. Their 32
generated-text arrays are byte-identical. Errors are empty, compact TP audit is
clean, and the post-run lease audit reports no GPU process.

## Provenance

- control packet SHA256: `96f3ea10bd1d04e889ef4cb32836d4190c832366567a8caf2f04c9a49750ab6a`
- candidate packet SHA256: `857011c83bc6e0bded8f7d9c4e240a677e24ea63f3e356701c98b931c403cec2`
- runtime SHA256: `2856e9827e5c69480c14c456366e24ef0108f254108f09aa59844c8dae53b0ef`
- result JSON: `/tmp/k3-slotd-ab/{control,candidate}/out512-seed0.json`
- server logs: `/tmp/k3-slotd-ab/{control,candidate}/server.log`

The candidate code was removed after measurement.
