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
