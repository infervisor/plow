#!/bin/sh
# Build the MLA correctness gate outside nix (the Rust oracle is built separately, inside).
set -e
R="$(cd "$(dirname "$0")/.." && pwd)/runtime"
OUT="${1:-/tmp/mlab}"
PATH=/opt/rocm/bin:/usr/bin:/bin
export PATH
unset LD_LIBRARY_PATH
BUN=$(ls -1 /opt/rocm/lib/llvm/bin/clang-offload-bundler /opt/rocm/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)
INC="-I$R/amd -I$R/common"
mkdir -p "$OUT"
cd "$OUT"
hipcc --offload-arch=gfx950 -O3 -w --genco "$R/amd/test_kernels.hip" -o tk.co $INC
"$BUN" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input=tk.co --output=test_kernels.elf
cp /tmp/mlab_ref ./mla_ref
./mla_ref fixture.bin
/usr/bin/gcc -O2 -std=gnu11 -o mla_test "$R/tests/mla_gfx950_test.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
echo "built in $OUT"
