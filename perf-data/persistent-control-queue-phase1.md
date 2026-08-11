# Persistent control queue Phase 1: dispatch-floor stop

**Decision: STOP.** On one MI325X, a capacity-1 resident command took **88.309 us median** versus
**45.629 us** for a fresh cooperative launch and synchronization. Saving was **-42.680 us**, well
below the required **+25 us**. Cooperative occupancy was valid, so this is the performance stop,
not an invalid-probe refusal.

## Scope

This is a synthetic one-rung bound on outer dispatch only. Both arms use 304 workgroups, 512
threads, one block/CU, the same synthetic body, and a grid barrier before completion. The resident
arm adds the minimum leader-poll and grid-broadcast path. It does not execute a K3 packet program
and makes no K3 latency claim. A negative outer-dispatch bound is sufficient to reject production
control-queue integration.

| 10,000 commands | median us | mean us | p90 us |
|---|---:|---:|---:|
| cooperative launch + synchronize | 45.629 | 45.930 | 49.309 |
| resident publish + completion | 88.309 | 88.127 | 104.299 |

The device timestamp window was 25.040 us for the baseline body and 44.840 us for resident pickup
plus the same body, putting the required pickup barrier at about 19.800 us. The remaining host
residuals were 20.589 and 43.469 us respectively. The measured binary's raw JSON called these
fields `device_body_us` and `control_us`; the source now names them `device_window_us` and
`host_residual_us` to reflect the actual timestamp boundaries. The total A/B and STOP verdict are
unchanged.

## Exact result and provenance

```json
{"schema":"plow.control-floor.v1","scope":"synthetic-one-rung","device":"AMD Instinct MI325X","samples":10000,"blocks":304,"threads":512,"blocks_per_cu":1,"max_resident_blocks_per_cu":4,"realtime_ticks_per_us":100.0,"baseline_us":{"median":45.629,"mean":45.930,"p90":49.309},"resident_us":{"median":88.309,"mean":88.127,"p90":104.299},"device_body_us":{"baseline_median":25.040,"resident_median":44.840},"control_us":{"baseline_inferred":20.589,"resident_inferred":43.469},"saving_us":-42.680,"stop_threshold_us":25.000,"verdict":"STOP"}
```

- Date: 2026-08-11 UTC.
- Repository revision at measurement: `7322ec076cc072109e617c1dd6a9e0b13d9e87b3` plus the three
  uncommitted probe files recorded below.
- ROCm: HIP 7.14.60850, AMD clang 23.0.0git
  (`46fcb339fb61119b337f973c7ca9e710a319fdd0`, patched).
- Nix hipcc:
  `/nix/store/9i3g77yxafyrsiphzmpljmq6j5xj4imx-rocm-therock-gfx94X-dcgpu-7.14.0/bin/hipcc`.
- Measured binary: `/tmp/plow-control-floor-gfx942`, SHA-256
  `2d2c71195749654ceea6ebfc5be49ae91277da821f5c9b95086091afe792ba56`.
- Raw JSON: `/tmp/plow-control-floor-result.json`, SHA-256
  `52c63c1dd2f4fccc265f48e44594be790ee7aef2661eec69b046f84d1656a8f0`.
- Post-run `gpulease --audit`: `GPU: no foreign compute procs`.

Exact measured build and run:

```bash
nix develop -c bash -lc '"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -std=c++17 \
  -Wall -Wextra -Werror runtime/bench/dispatch/control_queue_dispatch_floor.hip \
  -o /tmp/plow-control-floor-gfx942'
nix develop -c bash -lc 'perf-data/harness/gpulease -n 1 control-floor \
  /tmp/plow-control-floor-gfx942 10000 > /tmp/plow-control-floor-result.json'
```

The probe exits 4 for a threshold STOP. Current CMake reproduction targets are default-off behind
`PLOW_BENCH`:

```bash
nix develop --command cmake -S runtime -B build-control-floor \
  -DPLOW_ROCM=ON -DPLOW_HIP_ARCH=gfx942 -DPLOW_BENCH=ON
nix develop --command cmake --build build-control-floor \
  --target control_queue_probe_test control_queue_dispatch_floor_gfx942
nix develop --command ctest --test-dir build-control-floor -R control_queue_probe_abi \
  --output-on-failure
```

Static gfx942 metadata for the measured geometry: baseline 10 VGPR / 24 SGPR; resident 12 VGPR /
34 SGPR; both use zero LDS, scratch, and spills. Runtime occupancy reported four resident blocks
per CU; the probe deliberately uses one.
