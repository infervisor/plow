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
            ];
            shellHook = ''
              echo "plow dev shell — $(cargo --version)"

              if command -v elan >/dev/null && [ -f lean-plow/lean-toolchain ]; then
                echo "lean toolchain: $(cat lean-plow/lean-toolchain)"
              fi
            '';
          };
        });
    };
}
