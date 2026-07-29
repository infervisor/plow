{
  description = "plow — inference compiler/runtime (Rust workspace)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      linuxSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = nixpkgs.lib.genAttrs systems;
      forLinux = nixpkgs.lib.genAttrs linuxSystems;
      pkgsFor = system: import nixpkgs { inherit system; };

      cargoLock = { lockFile = ./Cargo.lock; };
    in
    {
      packages = forLinux (system:
        let
          pkgs = pkgsFor system;
          src = ./.;
        in
        rec {
          default = plowrt;

          # --- compiler -------------------------------------------------------
          # Pure Rust, thanks to the rustls TLS stack (the workspace pins hf-hub
          # with `default-features = false`, so no native-tls → no openssl-sys).
          # Nothing here needs pkg-config or a system library.
          plowc = pkgs.rustPlatform.buildRustPackage {
            pname = "plowc";
            version = "0.1.0";
            inherit src cargoLock;

            cargoBuildFlags = [ "--package" "plowc" ];
            cargoTestFlags = [ "--package" "plowc" ];

            meta = {
              description = "plow compiler: model/network → packet streams for a hardware spec";
              mainProgram = "plowc";
              platforms = nixpkgs.lib.platforms.linux;
            };
          };

          # --- runtime (one binary, CPU + GPU) ---------------------------------
          # A single artifact that serves CPU and GPU. Both vendor backends are
          # compiled in, but neither is *linked*: `device::select` `dlopen`s
          # libcuda.so.1 / libamdhip64.so at startup and falls back to the CPU
          # reference backend when neither loads. So this builds on a machine with
          # no GPU toolchain, runs on a machine with no GPU driver, and lights up
          # the GPU when one is present — and one process can drive NVIDIA and AMD
          # side by side (`cu*` and `hip*` share no symbol names).
          #
          # It is therefore *dynamically* linked, and cannot be `crt-static`: a
          # static musl binary cannot `dlopen` at all (musl stubs it out), which
          # would make the GPU path dead code on every host. What it links is only
          # libc/libm/libdl/libgcc_s — every Rust crate is still static, and there
          # is no GPU link-time dependency. Speed comes from the release profile
          # (fat LTO, one codegen unit, panic=abort) — see the workspace Cargo.toml.
          plowrt = pkgs.rustPlatform.buildRustPackage {
            pname = "plowrt";
            version = "0.1.0";
            inherit src cargoLock;

            buildFeatures = [ "cuda" "rocm" ];
            cargoBuildFlags = [ "--package" "plowrt" ];
            cargoTestFlags = [ "--package" "plowrt" ];

            meta = {
              description = "plow host runtime — CPU + GPU in one binary, drivers dlopen'd";
              mainProgram = "plowrt";
              platforms = nixpkgs.lib.platforms.linux;
            };
          };

          # --- C runtime core (CPU) --------------------------------------------
          # libplow_runtime.a: packet decode, dispatch table, interpreter, memmap,
          # and the CPU golden kernels that serve as the correctness oracle. Built
          # PIC so the GPU plugins can absorb it. Runs the ctest suite on build.
          plow-runtime = pkgs.stdenv.mkDerivation {
            pname = "plow-runtime";
            version = "0.1.0";
            src = ./runtime;

            nativeBuildInputs = [ pkgs.cmake ];
            doCheck = true;
            checkPhase = ''
              runHook preCheck
              ctest --output-on-failure
              runHook postCheck
            '';

            meta = {
              description = "plow C runtime core + CPU golden kernels";
              platforms = nixpkgs.lib.platforms.linux;
            };
          };
        });

      # `nix flake check` on Linux: build the compiler, the static runtime, and the
      # C core (whose ctest suite runs as part of its build).
      checks = forLinux (system:
        let p = self.packages.${system}; in {
          inherit (p) plowc plowrt plow-runtime;
        });

      devShells = forAll (system:
        let pkgs = pkgsFor system; in {
          default = pkgs.mkShell {
            name = "plow-dev";
            packages = [
              pkgs.cargo
              pkgs.rustc
              pkgs.rustfmt
              pkgs.clippy
              pkgs.rust-analyzer
              # The C runtime (runtime/) builds with CMake.
              pkgs.cmake
              # Lean 4 toolchain manager — installs the version pinned by
              # lean-plow/lean-toolchain on first `lake` invocation.
              pkgs.elan
              # ROCr's own shared-library dependencies, from nix rather than
              # the system. `plowrt --features hsa` dlopens
              # /opt/rocm/lib/libhsa-runtime64.so, which needs libelf, libdrm,
              # libnuma and libz. Putting /usr/lib/x86_64-linux-gnu on
              # LD_LIBRARY_PATH to satisfy them does NOT work: the nix-built
              # binary then resolves libc.so.6 to the system glibc 2.35 and
              # dies with `GLIBC_2.39 not found`. Supplying them from nix keeps
              # exactly one glibc in the process.
              pkgs.elfutils
              pkgs.libdrm
              pkgs.numactl
              pkgs.zlib
            ];
            shellHook = ''
              echo "plow dev shell — $(cargo --version)"

              if command -v elan >/dev/null && [ -f lean-plow/lean-toolchain ]; then
                echo "lean toolchain: $(cat lean-plow/lean-toolchain)"
              fi

              # The AMD GPU path. ROCm itself stays SYSTEM-installed (the code
              # objects are built by the system hipcc); only its library search
              # path is wired up here. /opt/amdgpu carries libdrm_amdgpu, which
              # ROCr loads for the kernel driver. Deliberately does NOT include
              # /usr/lib/x86_64-linux-gnu — see the package list above.
              # mkShell wires nix packages for BUILD time only; a dlopen at run
              # time reads LD_LIBRARY_PATH, so the nix lib dirs go on it
              # explicitly. Without this, ROCr's libelf is simply not found and
              # `--features hsa` falls back to the CPU backend — which looks
              # like "no GPU on this box" rather than a missing library.
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
                pkgs.elfutils pkgs.libdrm pkgs.numactl pkgs.zlib
                # ROCr is C++; libstdc++ must come from the SAME toolchain as
                # the rest of the process, not from /usr.
                pkgs.stdenv.cc.cc.lib
              ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              for d in /opt/rocm/lib /opt/amdgpu/lib/x86_64-linux-gnu; do
                [ -d "$d" ] && export LD_LIBRARY_PATH="''${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$d"
              done
              if [ -e /dev/kfd ]; then
                echo "amd gpu: $(ls /dev/dri/renderD* 2>/dev/null | wc -l) render node(s), ROCm $(cat /opt/rocm/.info/version 2>/dev/null || echo '?')"
              fi
            '';
          };

          # Weight quantization only. SEPARATE from `default` on purpose.
          #
          # `perf-data/harness/quantize_fp8.py` needs torch for
          # `torch.float8_e4m3fn`: that dtype's RTN+saturate rounding is the
          # reference the emitted fp8 twins are DEFINED against, so
          # approximating it would silently shift every fp8 weight.
          #
          # But torch drags triton, tensorboard and a multi-GB closure. Putting
          # it in `default` made every OTHER task in this repo — a cargo build,
          # a cmake run, a test — re-evaluate the devshell and block on that
          # fetch. Measured: it cost concurrent work several dead build cycles
          # for a dependency none of it uses. A shell is a per-task tool, not a
          # place to accumulate everything anyone needs.
          #
          #   nix develop .#quantize --command python3 \
          #       perf-data/harness/quantize_fp8.py <hf-dir>
          #
          # From nix rather than pip because PyPI is unreachable from this box
          # (`pip download numpy` and `curl https://pypi.org/simple/` both time
          # out, while huggingface.co returns 200 — the block is selective).
          # cache.nixos.org is reachable and torch is fully cached there: 302
          # MiB fetched, zero paths built.
          quantize = pkgs.mkShell {
            name = "plow-quantize";
            packages = [
              (pkgs.python3.withPackages (ps: [
                ps.torch
                ps.safetensors
                ps.numpy
                # scripts/glm52_prep_full.py reads the checkpoint's config.json
                # through `transformers.AutoConfig`. GLM-5.2's full-model weight
                # repack is a ~750 GB CPU/IO job and this shell is where it runs;
                # without this it dies on `No module named 'transformers'` after
                # the flake has already been evaluated.
                ps.transformers
              ]))
            ];
            shellHook = ''
              echo "plow quantize shell — $(python3 -c 'import torch; print("torch", torch.__version__)')"
            '';
          };
        });
    };
}
