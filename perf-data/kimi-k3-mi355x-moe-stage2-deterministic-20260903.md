# Deterministic lean MXFP4 MoE stage 2 — MI355X TP8

Date: 2026-09-03

## Route

- Opt-in packet emission isolates only a structurally eligible
  `MoeGroupDownPf`; `MoeCombinePf` remains in the next interpreter segment.
- The lean wave64 kernel writes each row to its fixed f32
  `part[row_partidx]` slot. There is no output clear and no atomic epilogue.
- Runtime selection is structural and capability-based. It does not inspect a
  model name. A missing object or missing shuffled weight/scale companion falls
  back to the ordinary interpreter route.
- The production route remains default-off pending three order-alternated,
  byte-exact full-network folds.

## Focused gate

Hardware: one MI355X XCD-complex GPU, ROCm 7.14 Nix toolchain.

| Shape | Exact oracle | Forward | Reverse |
|---|---:|---:|---:|
| T1024, H3584, I384, E896, top-16 | bad=0, max abs=0 | 0.243242 ms | 0.243322 ms |
| T8192, H3584, I384, E896, top-16 | bad=0, max abs=0 | 0.800607 ms | 0.800646 ms |

Object resource gate: wave64, workgroup 256, 98 VGPR, 40 SGPR, 4,352 B
dynamic LDS, zero private bytes, zero SGPR/VGPR spills, occupancy 4. The build
rejects more than 100 VGPR. Executable `.text` SHA-256:
`0c697ba09401e11c0f10fa7bf47e3eaf7d289f689ee98329867dbe1737016644`.

## TP8 8192→1 A/B fold

Both arms used the same packet, checkpoint, random 8,192-token prompt, runtime
binary, interpreter objects, concurrency 1, no warmup, and one output token.
The only A/B difference was presence of the stage-2 lean object. Prompt token
array SHA-256 was
`dd51e931308683300f372862a682d56a332e0e3e8d71cd70ef06c34739362dcd`
in both arms.

| Arm | TTFT | Output token | Output checksum |
|---|---:|---:|---|
| Lean deterministic scatter | 2,352.872307 ms | 6896 | `fnv1a64:7d749e3b002fafa7` |
| Interpreter fallback control | 2,987.964661 ms | 6896 | `fnv1a64:7d749e3b002fafa7` |

Delta: -635.092354 ms, -21.255%. Both processes completed one request with zero
failures. This is one byte-exact fold, not sufficient evidence to enable the
route by default.

Exact command shape:

```text
plowrt --rt-checkpoint /tmp/k3-farm.dvzmZN bench \
  --assets /tmp/plow-k3-moe2det-16k.EJaJu9 \
  --random-input-len 8192 --output-len 1 \
  --requests 1 --warmup-requests 0 --concurrency 1 \
  --parity-report --engine-diagnostics
```

Raw evidence:

- `/tmp/k3-moe2det-8192-candidate.{json,log}`
- `/tmp/k3-moe2det-8192-control.{json,log}`

## Memory cost

The loader-derived down-weight and padded-scale companions add 667,942,912 B
per rank = 637.0 MiB = 0.622 GiB. TP8 aggregate is 5,343,543,296 B = 4.976 GiB.
The original row-major tables stay resident for decode and fallback.

## Qualification status

- Focused T1024 exact/resource gate: pass.
- Focused T8192 exact/performance gate (≤1.2 ms/layer): pass.
- TP8 8192→1 byte-exact network fold: 1/3 pass.
- Default-on gate: pending two more order-alternated folds and regenerated tuned
  packet data for publication-quality comparisons.
