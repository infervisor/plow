# The Bringup Target — Hardware Parameter Block

Stages 1–3 are target-independent (CPU-only). From Stage 4 onward, every roofline percentage, block latency, recipe entry, and campaign benchmark is a statement about physical hardware.

The parameter block below must be populated once at the beginning of Stage 4 and passed to every subsequent stage and agent prompt. **Any row with an unassigned value is a blocker, not a default.**

---

## 1. Parameter Block Specification

| Parameter | Meaning | Exact Extraction Command | Verification Contract |
|---|---|---|---|
| `$VENDOR` | Hardware vendor: `nvidia`, `amd`, `apple`, or `cpu` | `plowc --list-gpus` | Must match `IsaLevel::vendor()` (`crates/hwspec/src/isa.rs`) |
| `$ISA` | Target architecture flag passed to `--arch` | `plowc --list-gpus` | Exact flag: `sm_89`, `sm_90a`, `sm_100a`, `sm_120a`, `gfx942`, `gfx950`, `gfx1201`, `metal3`, `cpu` |
| `$GPU` | Canonical part name/alias passed to `--gpu` | `plowc --list-gpus` | Must be an exact match from the supported GPU registry |
| `$NCU` | Number of compute units / SMs targeted | `plowc --list-gpus` | Default `0` uses full physical count from `hwspec`; non-zero under-subscribes |
| `$NGPU` / `$PARALLEL` | GPU rank count and parallelism mode | `--num-gpus $NGPU --parallel tp` | TP is compile-time; DP/PD are serve-time (`plowrt serve --dp`, `--pd`) |
| `$MAXCTX` | Maximum sequence context length | Checkpoint config `max_position_embeddings` | Must hold `input + output` tokens of every evaluated campaign row |
| `$TOOLCHAIN` | Compiler version (`nvcc` or `hipcc`) | `nvcc --version` or `hipcc --version` inside `nix develop` | Object compilation requires exact toolchain alignment |
| `$BUILD` | Device object build invocation | See [Device Objects Per ISA](#2-device-objects-per-isa) table below | Verified executable script producing device objects |
| `$FEATURES` | `plowrt` Cargo feature flag | `cuda` (NVIDIA), `hsa` (AMD), `metal` (Apple), `cpu` (CPU) | Required cargo feature when building `plowrt` |
| `$BW_BOUND` | Empirical HBM read/write streaming bandwidth (GB/s) | Measured via calibration probe (see §3 below) | Real floor from `perf-data/probes/<hw_fingerprint>.json`; never datasheet |
| `$COMPUTE_CEIL` | Empirical sustained Tensor Core compute peak (TFLOPS) | Measured via calibration probe (see §3 below) | Real floor from `perf-data/probes/<hw_fingerprint>.json`; never datasheet |
| `$RESULTS` | Scratch directory for raw benchmark output | Outside Git repo (e.g. `/tmp/plow-campaign/<campaign>`) | Repo keeps only `docs/bringup/results/<campaign>/comparison.{md,csv}` |

---

## 2. Device Objects Per ISA

| `$ISA` | Target Part Example | `$BUILD` Command | Build Output Artifact |
|---|---|---|---|
| `sm_90a` | NVIDIA H100 / H800 / H200 | `scripts/build_sm90a_cubin.sh` | `.cubin` objects in asset bundle |
| `sm_120a` | NVIDIA Blackwell | `scripts/build_sm120_cubin.sh` | `.cubin` objects in asset bundle |
| `sm_89` | NVIDIA L4 / L40S | `plowc --emit devblob+cubin --arch sm_89` | Embedded CUBIN in packet |
| `gfx942` | AMD MI300X / MI300A | `scripts/build_gfx942.sh` | `.hsaco` objects in asset bundle |
| `gfx950` | AMD MI350X | `scripts/build_gfx950.sh` | `.hsaco` objects in asset bundle |
| `gfx1201` | AMD RDNA4 (Strix / Navi) | `PLOW_HIP_ARCH=gfx1201 scripts/build_gfx942.sh` | `.hsaco` objects in asset bundle |
| `metal3` | Apple Silicon (M-series) | Built-in shader pipeline | Shaders compile at runtime from packet |
| `cpu` | Intel Xeon / AMD EPYC | Built with `cargo build -p plowrt --features cpu` | C++ interpreter (`runtime/cpu/`) |

A recipe build (`python3 scripts/campaign/campaign.py build <recipe> --out <dir>`) automatically executes the recipe's declared object step.

---

## 3. Empirical Calibration: Establishing Floor Denominators

Datasheet peaks (e.g., "3.35 TB/s HBM3" or "989 TFLOPS BF16") are theoretical limits that cannot be sustained under real memory controllers, clock governors, and bus contention. **All roofline percentages reported in Plow must use empirically calibrated denominators.**

### Step 1: Run Empirical Calibration Under GPU Queue
```bash
# NVIDIA
python3 scripts/bench/gpuq.py submit calibrate 1 \
  python3 scripts/campaign/calibrated_roofline.py --gpu "$GPU" --arch "$ISA" \
  --out "perf-data/probes/${GPU}_${ISA}.json"

# AMD
python3 scripts/bench/gpuq.py submit calibrate 1 \
  python3 scripts/campaign/calibrated_roofline.py --gpu "$GPU" --arch "$ISA" \
  --out "perf-data/probes/${GPU}_${ISA}.json"
```

### Step 2: Probe Output Schema (`perf-data/probes/<hw_fingerprint>.json`)
```json
{
  "fingerprint": "sm90_h100_sxm5_80gb",
  "empirical_hbm_bw_gbps": 3052.4,
  "empirical_l2_bw_gbps": 11840.0,
  "empirical_bf16_tensor_tflops": 892.1,
  "empirical_fp8_tensor_tflops": 1740.5,
  "dispatch_latency_ns": 1820
}
```

The resulting JSON is consumed directly by:
* `python3 scripts/campaign/block_roofline.py --ceilings <probe.json>`
* `python3 scripts/campaign/certify_kernel.py --input <sweep.json>`
* `python3 scripts/campaign/campaign.py block-ab --ceilings <probe.json>`

---

## 4. Per-ISA Architectural Authority

Architectural divergence has three distinct owners ([`docs/arch/14-amd-arch-divergence.md`](../arch/14-amd-arch-divergence.md) §3):

| Layer | Authority Owns | Implementation File |
|---|---|---|
| **Instruction Primitives** | Builtin opcodes (MFMA shapes, FP8 encodings, MX conversions) | `runtime/amd/amd_arch.h` or `runtime/nvidia/dev_isa.cuh` |
| **Geometry & Tile Budget** | LDS / Shared memory bytes, GEMM tile shapes, stage buffers | `crates/hwspec/src/isa.rs` (`IsaLevel::geometry()`) |
| **Build Profile** | Compiler flags, optimization levels, target architecture defines | Per-ISA build script (`scripts/build_<isa>.sh`) |

`IsaLevel::geometry()` returns `Some` for AMD architectures. For NVIDIA, GEMM tiles are probed directly from the compiled cubin via `plowc tune inventory`, never guessed from a static lookup table.

---

## 5. Non-Negotiable Target Rules

1. **Parameter Immutability in Commands:** Never hardcode device names or architectures in scripts or documentation. Always write `--gpu $GPU --arch $ISA --num-gpus $NGPU`.
2. **Hardware Re-Anchoring:** Performance measured on one part does not transfer to another, even under the identical `$ISA` (e.g. MI300X 192GB vs MI300A 128GB). Always recalibrate ceilings when changing physical machines.
3. **Capacity vs Code Emission:** Two parts sharing an `$ISA` may differ in memory capacity (`crates/hwspec/src/amd/mi300.rs`). Memory capacity dictates maximum concurrency and batch size; it does not change emitted packet bytecode.
4. **Target Integrity:** The `$ISA` recorded in `build.json` via `--arch` selects what `--emit devblob+cubin` builds. Emitting for a mismatched `$ISA` produces invalid bytecode and immediate driver launch failures.
