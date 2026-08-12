{
  description = "plow — inference compiler/runtime (Rust workspace)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAll = nixpkgs.lib.genAttrs systems;
      # allowUnfree is for the CUDA toolchain only (nvcc's EULA); ROCm is free.
      # The insecure-package allowance is exactly the optional vllm baseline
      # shell (nixpkgs flags this release; it never enters a plow build).
      pkgsFor = system: import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          permittedInsecurePackages = [ "python3.13-vllm-0.27.0" ];
        };
      };

      cargoLock = {
        lockFile = ./Cargo.lock;
        # tools/bench's optional `ib` feature pins a git dependency; importCargoLock
        # needs its hash spelled out even though plowc/plowrt never compile it.
        outputHashes = {
          "inference-benchmarker-1.1.0" = "sha256-Tr/urmjuYQXH50kxCn05DOLfv4JTYv+mjsBv5EhQXAE=";
        };
      };

      # GPU toolchains, from nix. This is what makes the build self-contained:
      # no /opt/rocm, no /usr/local/cuda anywhere in a `nix build`.
      #
      # VERSION NOTE: ROCm here is 7.14 (clang-23) — the toolchain the branch's
      # kernel measurements were taken on (scripts/build_gfx942.sh header).
      # AMD publishes 7.14.0 as a stable relocatable TheRock SDK. The gfx94X
      # family package carries the MI300/MI325 kernel packs. Its compiler,
      # device-lib bitcode and ROCr can also compile Plow gfx950 code objects;
      # only vendor math-library kernel packs are family-specific, and Plow
      # links none of those into its interpreter.
      rocmVersion = "7.14.0";
      gpuToolsFor = pkgs: rec {
        cuda = pkgs.cudaPackages;
        rocm = pkgs.stdenv.mkDerivation {
          pname = "rocm-therock-gfx94X-dcgpu";
          version = rocmVersion;
          src = pkgs.fetchurl {
            url = "https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx94X-dcgpu-${rocmVersion}.tar.gz";
            sha256 = "sha256-MuFtyn+EQKCKjWNqan2wA0xhUY8y6pFTR7mNn1UZmww=";
          };
          dontUnpack = true;
          # autoPatchelf sets the nix ELF interpreter and resolves the few
          # NEEDED libs the SDK does not bundle; everything else already
          # carries $ORIGIN rpaths. IgnoreMissing: the tree ships hundreds of
          # test/bench binaries whose extra deps plow never runs.
          nativeBuildInputs = [ pkgs.autoPatchelfHook ];
          buildInputs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.zstd
            pkgs.ncurses
            pkgs.libxml2
            pkgs.libdrm
            pkgs.numactl
            pkgs.elfutils
            pkgs.expat
          ];
          autoPatchelfIgnoreMissingDeps = true;
          dontStrip = true;
          installPhase = ''
            runHook preInstall
            mkdir -p $out
            tar -xzf $src -C $out --strip-components=1

            # AMD's clang expects a host gcc under /usr for libstdc++ and libc
            # headers; the nix sandbox has no /usr, and the HIP runtime wrapper
            # includes <cstdlib> in every device pass. Clang loads
            # <basename>.cfg from the invoked binary's directory (symlink dir
            # counts), so baking the paths here makes hipcc/clang work for any
            # caller with no environment setup — and pins ONE glibc, the same
            # one everything else in the flake uses.
            gccdir=$(echo ${pkgs.gcc-unwrapped}/lib/gcc/*/*)
            for dir in $out/bin $out/lib/llvm/bin; do
              for c in clang clang++ clang-23 clang-cpp amdclang amdclang++ amdclang-cpp; do
                [ -e "$dir/$c" ] || continue
                # -idirafter, NOT -isystem: libstdc++'s <cmath> reaches libc via
                # #include_next, which only searches directories AFTER the C++
                # headers — an -isystem libc lands before them and math.h is
                # "not found" the moment no /usr/include is there to mask it.
                printf -- '--gcc-install-dir=%s\n-idirafter %s\n' \
                  "$gccdir" "${pkgs.glibc.dev}/include" > "$dir/$c.cfg"
              done
            done
            runHook postInstall
          '';
          meta = {
            description = "ROCm ${rocmVersion} SDK (TheRock stable, clang-23)";
            platforms = [ "x86_64-linux" ];
          };
        };
        # Classic /opt/rocm layout: bundler and llvm-readelf share lib/llvm/bin
        # (hipcc_hsaco.sh finds readelf next to the bundler via dirname).
        rocmLlvmBin = "${rocm}/lib/llvm/bin";
        # Explicit for the same reason as ever: a clang that cannot find the
        # bitcode dies with "cannot find ROCm device library" on --genco.
        hipDeviceLibPath = "${rocm}/lib/llvm/amdgcn/bitcode";
        # Classic single-prefix layout: bin/nvcc, bin/cuobjdump, include/ — the
        # shape the scripts' `$(dirname $NVCC)` discovery expects.
        cudatoolkit = cuda.cudatoolkit;
        # The PATH the scripts' `env -i` nvcc calls run under (PLOW_NVCC_PATH in
        # runtime/cmake/nvcc_cubin.sh and scripts/build_sm90a_cubin.sh): the nix
        # sandbox has no /usr/bin, so nvcc's host compiler must come from here.
        nvccPath = pkgs.lib.makeBinPath [
          cudatoolkit
          cuda.backendStdenv.cc
          pkgs.coreutils
          pkgs.gnugrep
          pkgs.gnused
          pkgs.bash
        ];
        # -ccbin: nix's nvcc has no default host compiler. -I: nvcc resolves its
        # TOP through the bin/nvcc symlink to the real cuda_nvcc package, whose
        # include/ lacks cuda_runtime.h — the merged toolkit's include has it.
        nvccCcbin = "-ccbin ${cuda.backendStdenv.cc}/bin -I ${cudatoolkit}/include";
      };
    in
    {
      packages = forAll (system:
        let
          pkgs = pkgsFor system;
          lib = pkgs.lib;
          isLinux = pkgs.stdenv.isLinux;
          # The GPU toolchains only exist here: the ROCm SDK tarball is a
          # linux-x86_64 binary distribution.
          isGpuHost = system == "x86_64-linux";
          src = ./.;
          gpu = gpuToolsFor pkgs;

          # One cmake configure per served-object family, driving the canonical
          # tables in runtime/CMakeLists.txt (never the per-object scripts: the
          # tables are the single definition of each object's define set, and
          # this file must not become another copy that drifts).
          mkHsaco = arch: extraFlags: pkgs.stdenv.mkDerivation {
            pname = "plow-interp-${arch}";
            version = "0.1.0";
            src = ./runtime;
            nativeBuildInputs = [ pkgs.cmake ];
            cmakeFlags = [
              "-DPLOW_GFX950_HSACO=ON"
              "-DPLOW_HSACO_ARCH=${arch}"
              "-DPLOW_HSACO_HIPCC=${gpu.rocm}/bin/hipcc"
              "-DPLOW_HSACO_BUNDLER=${gpu.rocmLlvmBin}/clang-offload-bundler"
              "-DPLOW_HSACO_DIR=${placeholder "out"}/hsaco/${arch}"
            ] ++ extraFlags;
            env.HIP_DEVICE_LIB_PATH = gpu.hipDeviceLibPath;
            buildPhase = ''
              runHook preBuild
              cmake --build . --target gfx950_hsaco -j "$NIX_BUILD_CORES"
              runHook postBuild
            '';
            dontInstall = true;
            meta = {
              description = "plow ${arch} persistent-interpreter code objects (.elf hsaco)";
              platforms = lib.platforms.linux;
            };
          };

          mkNvCubin = tag: flags: pkgs.stdenv.mkDerivation {
            pname = "plow-interp-${tag}";
            version = "0.1.0";
            src = ./runtime;
            nativeBuildInputs = [ pkgs.cmake ];
            cmakeFlags = [
              "-DPLOW_CUBIN_NVCC=${gpu.cudatoolkit}/bin/nvcc"
              "-DPLOW_CUBIN_DIR=${placeholder "out"}/cubin"
            ] ++ flags;
            env = {
              PLOW_NVCC_PATH = gpu.nvccPath;
              NVCC_PREPEND_FLAGS = gpu.nvccCcbin;
            };
            buildPhase = ''
              runHook preBuild
              cmake --build . --target nv_cubins -j "$NIX_BUILD_CORES"
              runHook postBuild
            '';
            dontInstall = true;
            meta = {
              description = "plow ${tag} interpreter cubins `plowrt serve` loads";
              platforms = lib.platforms.linux;
            };
          };
        in
        rec {
          default = plowrt;

          # --- compiler -------------------------------------------------------
          # Pure Rust, thanks to the rustls TLS stack (the workspace pins hf-hub
          # with `default-features = false`, so no native-tls → no openssl-sys).
          # Nothing here needs pkg-config or a system library. Builds on darwin
          # too: macOS is a supported plowc host.
          plowc = pkgs.rustPlatform.buildRustPackage {
            pname = "plowc";
            version = "0.1.0";
            inherit src cargoLock;

            cargoBuildFlags = [ "--package" "plowc" ];
            cargoTestFlags = [ "--package" "plowc" ];

            meta = {
              description = "plow compiler: model/network → packet streams for a hardware spec";
              mainProgram = "plowc";
              platforms = nixpkgs.lib.platforms.unix;
            };
          };

          # --- runtime (one binary, CPU + GPU) ---------------------------------
          # On Linux: a single artifact that serves CPU and GPU. Both vendor
          # backends are compiled in, but neither is *linked*: `device::select`
          # `dlopen`s libcuda.so.1 / libamdhip64.so at startup and falls back to
          # the CPU reference backend when neither loads. So this builds on a
          # machine with no GPU toolchain, runs on a machine with no GPU driver,
          # and lights up the GPU when one is present — and one process can drive
          # NVIDIA and AMD side by side (`cu*` and `hip*` share no symbol names).
          #
          # It is therefore *dynamically* linked, and cannot be `crt-static`: a
          # static musl binary cannot `dlopen` at all (musl stubs it out), which
          # would make the GPU path dead code on every host. What it links is only
          # libc/libm/libdl/libgcc_s — every Rust crate is still static, and there
          # is no GPU link-time dependency. Speed comes from the release profile
          # (fat LTO, one codegen unit, panic=abort) — see the workspace Cargo.toml.
          #
          # On darwin: CPU reference backend only. There is no CUDA or ROCm on
          # macOS, so the GPU features stay off and the binary serves the CPU path.
          plowrt = pkgs.rustPlatform.buildRustPackage {
            pname = "plowrt";
            version = "0.1.0";
            inherit src cargoLock;

            # `hsa`, not "rocm": the AMD backend is direct ROCr (dlopen of
            # libhsa-runtime64), no HIP runtime involved.
            buildFeatures = lib.optionals isLinux [ "cuda" "hsa" ];
            cargoBuildFlags = [ "--package" "plowrt" ];
            cargoTestFlags = [ "--package" "plowrt" ];

            meta = {
              description = "plow host runtime — CPU + GPU in one binary on Linux (drivers dlopen'd), CPU-only on darwin";
              mainProgram = "plowrt";
              platforms = nixpkgs.lib.platforms.unix;
            };
          };
        }
        // lib.optionalAttrs isLinux {
          # --- C runtime core (CPU) --------------------------------------------
          # libplow_runtime.a: packet decode, dispatch table, interpreter, memmap,
          # and the CPU golden kernels that serve as the correctness oracle. Built
          # PIC so the GPU plugins can absorb it. Runs the ctest suite on build.
          plow-runtime = pkgs.stdenv.mkDerivation {
            pname = "plow-runtime";
            version = "0.1.0";
            # The whole tree, not ./runtime: the C core includes ../include/packet.h
            # (the ABI header the Rust side shares), which a runtime-only src cuts off.
            inherit src;
            preConfigure = "cd runtime";

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
        }
        // lib.optionalAttrs isGpuHost (rec {
          # --- served interpreter objects, one package per arch -----------------
          # ROCm: the full hsaco table (runtime/CMakeLists.txt PLOW_GFX950_HSACO).
          # gfx942 carries the same MXFP4/K3 object rows as the canonical CMake
          # table. The CDNA3 bodies and register cliffs are gated by that table.
          plow-interp-gfx950 = mkHsaco "gfx950" [ ];
          plow-interp-gfx942 = mkHsaco "gfx942" [ ];
          # CUDA: the served cubin tables (PLOW_SM120_CUBIN / PLOW_SM90A_CUBIN),
          # fp8-KV twins included, built the fast-prefill way.
          plow-interp-sm120a = mkNvCubin "sm120a" [
            "-DPLOW_SM120_CUBIN=ON"
            "-DPLOW_SM120_CUBIN_FP8KV=ON"
            "-DPLOW_FP8_KV_FASTPF=ON"
          ];
          plow-interp-sm90a = mkNvCubin "sm90a" [
            "-DPLOW_SM90A_CUBIN=ON"
            "-DPLOW_SM120_CUBIN_FP8KV=ON"
            "-DPLOW_FP8_KV_FASTPF=ON"
          ];
          # Every flavour under one root: hsaco/<gfx-arch>/*.elf + cubin/*.cubin.
          plow-interp = pkgs.symlinkJoin {
            name = "plow-interp";
            paths = [
              plow-interp-gfx942
              plow-interp-gfx950
              plow-interp-sm90a
              plow-interp-sm120a
            ];
          };
        }));

      # `nix flake check`: compiler + runtime everywhere; the C core (whose ctest
      # suite runs as part of its build) on Linux. The interp flavours are `nix
      # build` targets, not checks — they pull the multi-GB GPU toolchains.
      checks = forAll (system:
        let
          p = self.packages.${system};
          pkgs = pkgsFor system;
        in
        { inherit (p) plowc plowrt; }
        // nixpkgs.lib.optionalAttrs pkgs.stdenv.isLinux { inherit (p) plow-runtime; });

      devShells = forAll (system:
        let
          pkgs = pkgsFor system;
          gpu = gpuToolsFor pkgs;
          # nixpkgs' ROCm stack is x86_64-linux-only; the GPU dev wiring goes
          # with it. aarch64-linux gets the plain Rust/CMake shell.
          isGpuHost = system == "x86_64-linux";
        in
        {
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
            ] ++ pkgs.lib.optionals isGpuHost [
              # GPU dev requirements (x86_64-linux only; darwin builds plowc and
              # the CPU plowrt, nothing else): hipcc + nvcc from nix, so kernel
              # and interpreter builds need no /opt/rocm or /usr/local/cuda.
              # `rocm` is the full 7.14 SDK: bin/hipcc, bin/rocminfo, lib/llvm.
              gpu.rocm
              gpu.cudatoolkit
              # ROCr's own shared-library dependencies, from nix rather than
              # the system. `plowrt --features hsa` dlopens
              # libhsa-runtime64.so, which needs libelf, libdrm,
              # libnuma and libz. Putting /usr/lib/x86_64-linux-gnu on
              # LD_LIBRARY_PATH to satisfy them does NOT work: the nix-built
              # binary then resolves libc.so.6 to the system glibc 2.35 and
              # dies with `GLIBC_2.39 not found`. Supplying them from nix keeps
              # exactly one glibc in the process. Linux-only: elfutils, libdrm
              # and numactl do not build on darwin, and the GPU path does not
              # exist there anyway.
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

            '' + pkgs.lib.optionalString isGpuHost ''
              # The GPU toolchains, from nix (ROCm ${rocmVersion},
              # CUDA ${gpu.cudatoolkit.version}). The PLOW_* variables are the
              # override knobs every kernel build script already honors, so
              # scripts/build_gfx942.sh, build_gfx950.sh, build_sm90a_cubin.sh
              # and the cmake served-object targets all pick the nix toolchain
              # with no flags. A system toolchain under /opt can still be
              # selected per-invocation by overriding these.
              export ROCM_PATH=${gpu.rocm}
              export HIP_PATH=${gpu.rocm}
              export CUDA_PATH=${gpu.cudatoolkit}
              export HIP_DEVICE_LIB_PATH=${gpu.hipDeviceLibPath}
              export PLOW_HIPCC=${gpu.rocm}/bin/hipcc
              export PLOW_BUNDLER=${gpu.rocmLlvmBin}/clang-offload-bundler
              export PLOW_READELF=${gpu.rocmLlvmBin}/llvm-readelf
              export PLOW_TOOLCHAIN_LABEL=rocm-${rocmVersion}-nix
              # build_gfx950.sh's host `chat` harness: pair the nix cc with the
              # nix ROCm libs (the system gcc cannot link the nix libhsa — its
              # symbol versions are nix-glibc's).
              export PLOW_HOST_CC=cc
              export LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
              export PLOW_NVCC=${gpu.cudatoolkit}/bin/nvcc
              export PLOW_NVCC_PATH=${gpu.nvccPath}
              export NVCC_PREPEND_FLAGS='${gpu.nvccCcbin}'

              # mkShell wires nix packages for BUILD time only; a dlopen at run
              # time reads LD_LIBRARY_PATH, so the nix lib dirs go on it
              # explicitly. Deliberately does NOT include
              # /usr/lib/x86_64-linux-gnu — see the package list above.
              # Without this, ROCr's libelf is simply not found and
              # `--features hsa` falls back to the CPU backend — which looks
              # like "no GPU on this box" rather than a missing library.
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
                pkgs.elfutils pkgs.libdrm pkgs.numactl pkgs.zlib
                gpu.rocm
                # ROCr is C++; libstdc++ must come from the SAME toolchain as
                # the rest of the process, not from /usr.
                pkgs.stdenv.cc.cc.lib
              ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              if [ -e /dev/kfd ]; then
                echo "amd gpu: $(ls /dev/dri/renderD* 2>/dev/null | wc -l) render node(s), ROCm ${rocmVersion} (nix)"
              fi
            '';
          };

        }
        # Weight quantization: everywhere except legacy x86_64-darwin, where
        # nixpkgs' torch/arrow stack is marked broken and even *evaluating* the
        # shell fails `nix flake check --all-systems`.
        // pkgs.lib.optionalAttrs (system != "x86_64-darwin") {
          # Weight quantization only. SEPARATE from `default` on purpose.
          #
          # `perf-data/tools/quantize_fp8.py` needs torch for
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
          #       perf-data/tools/quantize_fp8.py <hf-dir>
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
                # Kimi-K3 ships a tiktoken model; its tokenizer conversion
                # verifies every sampled id against the reference decoder.
                ps.tiktoken
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
        }
        # vllm is OPTIONAL: a comparison/baseline tool, not a build input, and
        # its closure is torch-sized. Same per-task-shell rule as `quantize`:
        #   nix develop .#vllm
        // pkgs.lib.optionalAttrs isGpuHost {
          vllm = let
            xgrammarClient = pkgs.python3Packages.xgrammar.overridePythonAttrs (old: {
              version = "0.2.3";
              src = pkgs.fetchFromGitHub {
                owner = "mlc-ai";
                repo = "xgrammar";
                tag = "v0.2.3";
                fetchSubmodules = true;
                hash = "sha256-bznSz1fOCCGFR3NsuXm5eWo7EXrvBrFavEllC5+vDHM=";
              };
              patches = [ ];
              doCheck = false;
              build-system = old.build-system ++ [ pkgs.python3Packages.apache-tvm-ffi ];
              dependencies = old.dependencies ++ [
                pkgs.python3Packages.apache-tvm-ffi
                pkgs.python3Packages.typing-extensions
              ];
            });
            # `vllm bench serve` is a pure HTTP client. Building vLLM's CPU
            # execution extension is unnecessary here and currently fails
            # against nixpkgs' newer Torch API. Keep the upstream Python client
            # and its declared dependencies, but omit only the server extension.
            vllmClient = pkgs.python3Packages.vllm.overridePythonAttrs (old: {
              version = "0.27.0";
              src = pkgs.fetchFromGitHub {
                owner = "vllm-project";
                repo = "vllm";
                rev = "v0.27.0";
                hash = "sha256-ksKzkMDZGbqramOydhW49DU+p4lBteaAvvTKEjHfEAs=";
              };
              patches = [ ];
              postPatch = "";
              pyproject = null;
              format = "other";
              dontBuild = true;
              doCheck = false;
              pythonImportsCheck = [ ];
              dependencies = map
                (dep: if (dep.pname or "") == "xgrammar" then xgrammarClient else dep)
                old.dependencies;
              installPhase = ''
                runHook preInstall
                site="$out/${pkgs.python3.sitePackages}"
                mkdir -p "$site" "$site/vllm-0.27.0.dist-info" "$out/bin"
                cp -r vllm "$site/"
                cat > "$site/vllm/_version.py" <<'EOF'
                __version__ = "0.27.0"
                __version_tuple__ = (0, 27, 0)
                EOF
                cat > "$site/vllm-0.27.0.dist-info/METADATA" <<'EOF'
                Metadata-Version: 2.1
                Name: vllm
                Version: 0.27.0
                EOF
                cat > "$out/bin/vllm" <<'EOF'
                #!/usr/bin/env python3
                from vllm.entrypoints.cli.main import main
                main()
                EOF
                chmod +x "$out/bin/vllm"
                runHook postInstall
              '';
            });
          in pkgs.mkShell {
            name = "plow-vllm-client";
            packages = [
              (pkgs.python3.withPackages (_: [ vllmClient ]))
              pkgs.jq
              pkgs.curl
            ];
            shellHook = ''
              echo "plow vllm client shell — $(python3 -c 'import vllm; print(vllm.__version__)')"
            '';
          };
        });
    };
}
