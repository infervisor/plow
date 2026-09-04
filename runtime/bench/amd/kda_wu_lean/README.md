# KDA lean Wu screen

Isolated gfx950 screen for the dense BT64 chunk-KDA W/U factor build at `D=V=128` with the
q-precomputed operand (`d_kda_chunk_wu_bt64(q_precompute = true)`). It compares the interpreter
Wu body as the prefill interpreter runs it (eight waves, one workgroup per CU) with the four-wave
lean body in `runtime/amd/op_kda_wu_lean.h`:

- `control_wg512_g256`: the shipping body; every wave re-derives its 16x64 operand tile from
  exec-masked scalar loads for each of the 64 (row block, column tile) items of a (chunk, head).
- `key_factor_object_wg256_g256`: the rejected standalone key-factor Wu object body
  (`kda_chunk_key_factor_wu.hip`): the same body at four waves plus the key-factor pair pass.
- `lean_g{256,512,768,items}`: one workgroup per (chunk, head); bf16(Ainv) and the transposed
  `beta k exp2(g)` / `beta v` operands are staged in LDS once from coalesced loads, then each wave
  forms its 16 output tiles with the products in swapped A/B roles (8-byte row stores). The q
  pre-scale rides the same g loads. Grid = persistent workgroups.
- `lean_keys_g{768,items}`: the same body also emitting the carry's scaled-key hi/lo pair
  `k exp2(g_last - g)` (the `kda_chunk_key_factor` formula) from the same k/g loads.

The oracle requires bit equality of W, U, and the pre-scaled q against the control and of the key
pair against a reference precompute kernel. `MODE` selects the inputs: 0 structured, 1 LCG
uniform, 2 adversarial (NaN/Inf/denormal/RNE-tie sprinkles in every operand, gates spanning exp2
overflow and underflow, zero rows). `run.sh` rejects private memory or register spills before
acquiring one GPU.

Run inside `nix develop` (`T=8191` exercises the 63-row tail chunk):

```sh
runtime/bench/amd/kda_wu_lean/run.sh                       # T=8192 H=12, 21 samples, MODE=0
T=8191 SAMPLES=3 MODE=2 runtime/bench/amd/kda_wu_lean/run.sh
```

## MI355X result (2026-09-04)

Exact shape `T8192,H12,D128,V128,BT64` (1536 items), 21 order-rotated samples, one GPU:

| arm | median | vs control | gfx950 resources |
| --- | ---: | ---: | --- |
| control WG512 grid 256 | 0.553 ms | -- | VGPR 64, SGPR 52, no spill |
| key-factor object WG256 grid 256 | 1.114 ms | +101% (0.50x) | VGPR 64, SGPR 66 |
| lean grid 256 | 0.101 ms | 5.50x | VGPR 168, SGPR 44, occupancy 3, LDS 46,080 B |
| lean grid 512 | 0.068 ms | 8.19x | same object |
| lean grid 768 | 0.056 ms | 9.86x | same object |
| lean grid 1536 (one item each) | 0.059 ms | 9.44x | same object |
| lean + keys grid 768 | 0.093 ms | 5.94x | VGPR 230, SGPR 48, occupancy 2, LDS 46,080 B |
| lean + keys grid 1536 | 0.078 ms | 7.08x | same object |

Every arm matched all 12,582,912 W, U, and q elements bit-for-bit, and both key arms matched the
reference pair; the same held for MODE 1 and MODE 2 at T=8192 and T=8191 and for T=100, H=3
(two chunks, 36-row tail). No arm uses scratch or spills.

The key-factor object arm explains the rejected 09-04 pair screen (+41 ms TTFT): at four waves the
interpreter body is 0.56 ms/layer slower than at eight (x69 layers = +39 ms), independent of the
carry. The lean body is a different mechanism (operands built once, LDS-fed MFMA) and is faster
at four waves than the interpreter is at eight.
