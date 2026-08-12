# Kimi K3 TP8 MLA decode split sweep on MI325X

Date: 2026-08-11. GPU: 8x MI325X (`gfx942`, 304 CUs/rank). Toolchain:
Nix ROCm 7.14. Client: vLLM 0.27 `bench serve`, OpenAI chat, C1/N1,
128 output tokens, one warmup, seed 0, greedy, exact compact TP audit.

## Decision

Adopt `PLOW_K3_NS=64` as the K3 default. It is tied with ns16 at short
context, wins from 4K through 128K, and brackets its U-shaped optimum at
ns128. A context selector is not justified: its largest observed short-context
benefit would be about 0.14 ms and is inside run noise.

`nsplit` changes only the packet decomposition of `FlashMlaDecodeFp8` and
`MlaMergeFold`. All four arms reuse identical weights and HSACO. The split
boundaries change floating-point reduction association, so cross-arm text is
not required to be byte-identical; all-rank agreement and model quality are.

## Artifacts

| arm | packet SHA256 | workgroups/MLA layer |
|---|---|---:|
| ns16 | `87c781bf21476425d663249e9cb5bd7e904353d047b96b3ce123dcba2bdca9f7` | 48 |
| ns32 | `eb20c93418f497350b8bbebfe9592222a7dad0e47556cc363fb6d51631729e7f` | 96 |
| ns64 | `31ab1493cddcc3a93fe4b102ebbbfc2e42f34882353037ff47c3368a9eb11b6f` | 192 |
| ns128 | `9b5571e5529dbe1156212ea803acc94a27a189f8b2c6621ec0b6b31731e17a53` | 384 |

Frozen runtime SHA256:
`eea7729ee46ad18c3c17849d60a32db6220cd9f0a410e299fa8c5913eb3220ff`.
Decode GQ object SHA256:
`7604d152dcfcdf428ab222e10f8d12ec2d6ac655790a9e4cfd676a897c440f97`.
Flash GQ object SHA256:
`1e2a3848542c9520108772b9f74678b34bf1ab64931e5646a7de16fcfb820fa7`.

Disassembly comparison found 2,459 decode instructions in every packet and
exactly 48 changed instructions: 24 flash-decode packets and 24 merge packets.
Tensor names and model weights are identical; only partial scratch extents grow.

## Results

Matched 128K sweep:

| nsplit | TPOT (ms) | output tok/s | vs ns16 |
|---:|---:|---:|---:|
| 16 | 81.400 | 12.285 | control |
| 32 | 67.417 | 14.833 | -17.18% |
| 64 | **60.569** | **16.510** | **-25.59%** |
| 128 | 60.683 | 16.479 | -25.45% |

The ns64 context curve is 53.521/54.314/54.414/54.598/55.143/56.902/
60.569 ms TPOT at actual input lengths 149/4,117/8,213/16,021/32,021/
64,021/128,021. Matched ns16 controls are 53.381 ms at 149 and 54.453 ms
at 4,117: the short differences are noise, while ns64 is already ahead at 4K.

TTFT is unchanged, as expected for a decode-only packet axis. Host time remains
about 3.1 ms; the improvement is in GPU drain.

## Correctness and quality

All served cells completed 1/1 requests, produced exactly 128 output tokens,
reported no request error or in-band error, and passed exact compact TP counter
and all-rank agreement checks. Post-run GPU audits were clean.

The adoption gate used the first 200 GSM8K test questions, 8-shot chain of
thought, greedy decoding, and a 320-token cap:

- Paris coherence: PASS.
- Exact match: **197/200 = 0.9850**.
- Request errors: **0**.
- Median/mean latency: 5.86/6.35 seconds; total 1,271 seconds.
- Test/train SHA256:
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` /
  `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.

This exactly matches the established K3 B1 score and clears the >=196/200
adoption bar. Evidence: `/tmp/k3-ns64-gsm8k-client.log` (SHA256
`6177fb3299ccdc330d7dcb6f214e3a96d6eb3e50e6d9697f60429f9ea167e13c`)
and `/tmp/gsm8k_serve_8055.log` (SHA256
`df4e43ca8e01ecbd1d4fd975998833d3f374ee6355e9f726e3194c481d9bb2ff`).

## Reproduction

Emit arms by changing only `PLOW_K3_NS` in the canonical TP8 command. Serve
with `PLOW_MLA_PF_V2=1`, `PLOW_L2_PLACE_DISPATCH=1`,
`PLOW_TP_AUDIT_COMPACT=1`, `PLOW_CTR_DBUF=1`, and
`PLOW_STATE_CLEAR_DEVICE=1`. Run the pinned client with C1/N1, one warmup,
`--ignore-eos`, temperature 0, and the desired random input length.

For broad screening, use `K3_FULL=1 PLOW_K3_LAYERS=single:3` as documented in the Stage 4
bringup playbook. The recipe was smoke-tested: it emits one MLA layer, zero KDA
layers, 104 tensors, and 36 decode instructions. Only the finalists receive
whole-model serve and GSM8K gates.
