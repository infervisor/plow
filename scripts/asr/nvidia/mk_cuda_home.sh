#!/usr/bin/env bash
# Build a CUDA_HOME overlay on a venv's pip nvidia/cu13 wheel (for vLLM/flashinfer JIT on hosts without a CUDA toolkit).
set -eu
VENV=${1:-/root/asr-work/venv}
DST=${2:-/root/asr-work/cuda_home}
CU=$VENV/lib/python3.12/site-packages/nvidia/cu13
mkdir -p "$DST/bin" "$DST/lib64/stubs"
for f in "$CU"/bin/*; do ln -sfn "$f" "$DST/bin/"; done
rm -f "$DST/bin/nvcc"
# pip nvcc 13.x ships 13.0 runtime headers; CCCL's version check rejects that pairing.
printf '#!/usr/bin/env bash\nexec %s/bin/nvcc -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK "$@"\n' "$CU" > "$DST/bin/nvcc"
chmod +x "$DST/bin/nvcc"
for d in include lib nvvm; do ln -sfn "$CU/$d" "$DST/$d"; done
for f in "$CU"/lib/*.so* "$CU"/lib/*.a; do
  b=$(basename "$f"); ln -sfn "$f" "$DST/lib64/$b"; ln -sfn "$f" "$DST/lib64/${b%%.so.*}.so" 2>/dev/null || true
done
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$DST/lib64/stubs/libcuda.so"
"$DST/bin/nvcc" --version | tail -2
