# Kimi-K3 TP8 atomic grouped-MoE prefill screen (MI355X)

Decision: **NOT PROMOTED**. The A4W4 atomic arm is now memory-safe and passes focused
correctness/coherence gates, but three production-path pairs show no TPOT win and a repeatable
TTFT regression. Keep the serial scatter/combine decomposition as the default.

## Cell

- Source: current `codex/amd-agent-harness` worktree, ROCm 7.14.0, gfx950, 8×MI355X.
- Model: text tower from `/home/lava/plow/build-amd/k3-gfx950-farm`; BF16 KV, MXFP4 experts.
- Packet: TP8, max context 32768, prefill rungs 128/512/1024/2048/4096/8192, decode rung 1.
- Runtime: default global queue with L2-domain placement; input 8192, output 32,
  concurrency 1, seed 0, warmup 1, measured requests 3.
- Pair order: control→atomic, atomic→control, control→atomic. One exclusive eight-GPU lease.

Common benchmark command (substitute the asset and log paths below):

```sh
nix develop --command ./target/release/plowrt bench \
  --assets ASSET --random-input-len 8192 --concurrency 1 \
  --requests 3 --warmup-requests 1 --output-len 32
```

Emitter axes were identical except `PLOW_MOE_PF_ATOMIC=0/1`; both used
`K3_FULL=1 PLOW_DECODE_BATCH=1 PLOW_DECODE_BATCH_LADDER=1 PLOW_MOE_PF_DET=0
PLOW_MXFP4=1 PLOW_FP8_KV=0 PLOW_L2_PLACE=1`. Objects used
`PLOW_HSACO_GQ=ON PLOW_L2_PLACE_DISPATCH=ON` and the matching atomic axis.

## Results

| Pair | Order | Arm | TTFT mean ms | TTFT min–max ms | TPOT mean ms | TPOT min–max ms |
|---:|:---:|:---|---:|---:|---:|---:|
| 1 | C→A | control | 3883.369 | 3881.843–3885.073 | 63.4361 | 63.3957–63.4724 |
| 1 | C→A | atomic | 3913.104 | 3913.092–3913.114 | 63.2016 | 63.1805–63.2304 |
| 2 | A→C | atomic | 3913.281 | 3912.075–3915.335 | 63.1820 | 63.1572–63.1978 |
| 2 | A→C | control | 3884.076 | 3882.543–3885.351 | 63.1531 | 63.0936–63.2167 |
| 3 | C→A | control | 3882.728 | 3880.782–3884.686 | 63.2045 | 63.1388–63.2446 |
| 3 | C→A | atomic | 3912.050 | 3911.385–3913.349 | 63.3126 | 63.2789–63.3459 |

Across pair means: control TTFT 3883.391 ms vs atomic 3912.812 ms
(atomic **+29.421 ms, +0.758%**); control TPOT 63.2646 ms vs atomic 63.2321 ms
(atomic -0.0325 ms, -0.051%, noise-sized and directionally inconsistent by pair).

Logs:

- Control: `/tmp/plow-k3-b1-control-gq.ms6fLa/pair{1,2,3}-control.log`
- Atomic: `/tmp/plow-k3-b1-atomic-gq.VtiCOr/pair{1,2,3}-atomic.log`
- Driver: `/tmp/plow-k3-b1-paired-run-driver.log`

## Artifacts

| Artifact | SHA-256 |
|:---|:---|
| control `model.pkt` | `7571792ac8a757fe52dda60443b0efa5e869d21a6915012b26b47275d7bcb926` |
| control `build.json` | `16a9c17d832170b145673696c6ea1e23b41aa77ec93a0894fab7e761b7ae3a05` |
| control A4W4 GQ object | `ff9da369f2837f3784c551701334cd62b5f18ae6d1ad9c61e6bf9e7449e9c825` |
| atomic `model.pkt` | `7d2940bb168ba4624894263e48f1ebb82d3cafe967d672e5f1824f10dd55ce76` |
| atomic `build.json` | `9606c4b74aa88e35275a22ee35a3189f404f4dabf0e0307a242d09a928e2fc53` |
| atomic A4W4 GQ object | `360da9e8589c511c34e6afd075207ba154b4723312ad9e5f5a935e12c19a0aa8` |

## Gates and disclosure

- Cross-XCD f32 atomic probe: all 8 XCDs observed; 3 cold and 6 hot exact-count rounds passed,
  zero bad cells. Log: `/tmp/plow-atomic-coherence/gfx950-coherence.log`.
- Focused A4W4 oracle: scatter checked 14,208 values with zero unwritten; atomic accumulator
  worst relative error 0; deterministic f64 fixed-point accumulator zero mismatches. Log:
  `/tmp/moe_pf_a4w4_accum-oracles.log`.
- The original A4W4 wrapper ignored `atom_ksh`/`det_ksh` and overran the compact accumulator.
  Both native gfx950 and CDNA3-simulated DOWN epilogues now implement the accumulator modes.
- `plowrt bench` output checksums directly establish the expected f32 arrival-order
  nondeterminism. Control repeated `fnv1a64:168b899240c1bf02` in all three cells. Atomic produced
  three different values: `fnv1a64:62cb59406e2ccefc`, `fnv1a64:b9a1dd06bef0f665`, and
  `fnv1a64:85899b3e451548bd`; none match control. The deterministic arm has focused oracle
  coverage but was not performance-screened here.
