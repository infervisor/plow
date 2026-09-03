# Deterministic lean MXFP4 MoE stage 2 — MI355X TP8

Date: 2026-09-03

## Route

- Default-on packet emission isolates only a structurally eligible
  `MoeGroupDownPf`; `MoeCombinePf` remains in the next interpreter segment.
- The lean wave64 kernel writes each row to its fixed f32
  `part[row_partidx]` slot. There is no output clear and no atomic epilogue.
- Runtime selection is structural and capability-based. It does not inspect a
  model name. A missing object or missing shuffled weight/scale companion falls
  back to the ordinary interpreter route.
- `PLOW_MOE_STAGE2_LEAN=0` is the explicit rollback. Unsupported packets remain
  on the interpreter route.

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

## TP8 FP8-KV 8192→1 A/B fold

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
route by default. The asset manifest has `features.fp8_kv=true`; this result must
not be presented as a BF16-KV comparison against the vLLM baseline. A matched
BF16-KV integration gate is tracked separately.

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

## TP8 BF16-KV 8192→1 qualification

Three uncontended, order-alternated folds used the same checkpoint, runtime,
BF16-KV precision, analytical dense-tile choices, stage-1 route, and runtime
switches. The only packet difference was `PLOW_MOE_STAGE2_LEAN=0/1`. Current
source changes invalidate the older TuneDB digest, so both arms deliberately
used the same 7,650 analytical choices; no measured/analytical profiles were
mixed. Both manifests passed the Lean ordering certificate and oracle.

| fold / order | interpreter control | deterministic stage 2 | paired gain | trace-chain gain |
|---|---:|---:|---:|---:|
| 1, candidate -> control | 2513.490 ms | 1871.814 ms | 641.676 ms (25.53%) | 641.904 ms |
| 2, control -> candidate | 2516.652 ms | 1873.411 ms | 643.240 ms (25.56%) | 643.210 ms |
| 3, candidate -> control | 2515.745 ms | 1874.075 ms | 641.670 ms (25.51%) | 641.379 ms |
| mean | 2515.296 ms | 1873.100 ms | **642.195 ms (25.53%)** | **642.164 ms** |

All six arms completed one request with zero failures, token 6896, checksum
`fnv1a64:7d749e3b002fafa7`, complete non-overflowed diagnostics, and all eight
ranks completing prefill. Paired-gain sample standard deviation is 0.905 ms.

The control traces attribute 700.981/701.008/701.195 ms to
`MoeGroupDownPf`. Standalone launches do not emit raw trace records, so the
candidate residual grows from 222.2--223.5 ms to 294.6--295.7 ms. The
approximately 72 ms residual increase plus the removed 701 ms interpreter body
attributes roughly 629 ms directly to stage 2. The remaining trace shift closes
the full 642.164 ms mean chain gain, within 0.031 ms of the endpoint mean.

Packet SHA-256:

- control: `ce17d7cfd8dbcc7cdf216e74f7e63b6ae77600a46ef72c3f8eae95a9e0f8f07f`
- candidate: `ed35dfc3008e8147d2b4c172c112a84119994503ebda839f90d31c10deb801b8`

Trace SHA-256, candidate/control by fold:

- fold 1: `d763ce0b487e3e327494d70bb4b4b3dad451d966f5b5a2ad42c16343f1919db1` /
  `76c0a855ccd38adfed5bc65e60c9c5931a6a550163a883fc127ab1059fd07372`
- fold 2: `827335462ba1e9a41ecb57a2c968072ff04305eae060fcc505c52f2953a2971f` /
  `d9aab3b4d2e34bde4b312f1a4a9312f4eb1ad57c0cfda856d64402f7adf16dd4`
- fold 3: `79b4b82019822490e833f06c0eafcc618bb22df0b919df85cbe5316c5502a8be` /
  `faa0d767644489136a10e107e4b7875d8fbcc9ea408ded6b256c5ebcdf85158d`

Raw evidence: `/tmp/k3-p1-{candidate,control}-r{1,2,3}.{json,log,trace}`.

## Memory cost

The loader-derived down-weight and padded-scale companions add 667,942,912 B
per rank = 637.0 MiB = 0.622 GiB. TP8 aggregate is 5,343,543,296 B = 4.976 GiB.
The original row-major tables stay resident for decode and fallback.

## Qualification status

- Focused T1024 exact/resource gate: pass.
- Focused T8192 exact/performance gate (≤1.2 ms/layer): pass.
- TP8 BF16-KV 8192→1 byte-exact network gate: 3/3 pass, mean -642.195 ms
  (-25.53%).
- Default-on gate: pass. The route remains structural/capability-gated with
  `PLOW_MOE_STAGE2_LEAN=0` as rollback.
