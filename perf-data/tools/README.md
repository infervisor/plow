# Benchmark + analysis harnesses

- `gpulease` — advisory GPU lease for concurrent agents on one card. Wrap the
  RUN, not the build: `gpulease <label> <cmd>`. It also AUDITS for foreign
  compute processes and **exits 76 if the GPU was contended**, because a
  contended run silently invalidates timings. It cannot stop a process that
  never calls it.
- `diskbench.c`, `coldtest.c`, `pipe2.cu` — I/O ceilings. Measured on this box:
  **NVMe 3.71 GB/s** (flat across 1-16 threads — a volume limit, not queue
  depth), **PCIe H2D 28.44 GB/s** pinned. Disk binds the weight loader, not
  PCIe, which is why GDS/cuFile is worthless here even where available.
  `cudaDeviceScheduleBlockingSync` was the single largest tuning effect (+74%).
- `expo_hist.py`, `codec_roundtrip.py`, `tile_sweep.py` — SplitZip ratio
  analysis. bf16 **1.3313x** at top-16/4b on Qwen3-4B (beats the paper's 1.324x),
  **zero losing tiles of 443,520**. fp8 e4m3 at top-16 is **0.9988x — a net
  loss**, structurally: all 16 exponents occur, so a 4-bit code is the identity.
  Tile size is NOT a ratio lever (0.54% across an 8x range).
- `analyze_tiles.py`, `sweep_f.py`, `sweep_knee.py`, `negctrl.py`, `e2e_gate.py`
  — per-tile fp8 sensitivity. **Refuted per-tile codec selection**: per-tile
  relL2 is mean 0.02644 / std 0.00034 (1.3% CV) with no structure by layer,
  projection, or position; best compile-time predictor is Spearman 0.103 *with
  the wrong sign*; random selection beat sorted at f=0.25 and f=0.50.

## Model / GPU bring-up

The runbook is `docs/bringup/` — gpulease discipline, the token-identity gate,
baseline rules, the attribution ladder, the probe law (standalone probes
overstate — in-model only), roofline sanity, and the campaign write-up format.
Measured detail behind it: `perf-data/gemma12b-gh200-prefill-campaign.md`
(sm_90a prefill, canonical config, refuted ideas) and
`perf-data/gemma4-12b-sm120-serving.md` (sm_120a decode/serving).

- `bringup_gate.sh` / `bringup_bench.sh` / `bringup_showdown.sh` /
  `bringup_ceiling.py` — the parameterized scripts those stages drive.
  `bringup_showdown.sh` alternates complete Plow/vLLM server lifetimes for at
  least three rounds and drives both through the same raw `/v1/completions`
  client. Run the whole command under `gpulease -n "$TP"`; it requires frozen
  Plow artifacts plus frozen vLLM artifacts or explicit immutable image/model
  identities. `bringup_bench_selftest.sh` checks workload maps and exact-count
  refusal without a GPU.
  A pinned container command is passed whole, for example
  `VLLM_CLIENT_COMMAND_ARGV="docker run --rm --network host --entrypoint vllm IMAGE@sha256:..."`
  and likewise `VLLM_SERVER_COMMAND_ARGV` (with the required GPU and mount
  arguments). The harness appends `bench serve` or `serve`; it never inserts a
  host-side executable after the image.
