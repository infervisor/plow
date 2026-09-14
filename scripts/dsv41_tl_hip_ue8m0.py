"""Private copy of tilelang's templates + the UE8M0 scale type HIP lacks.

Kept OUT of the shared venv on purpose: /app/plow/.venv-vllm028 is used by
other campaigns, and tilelang honours TL_TEMPLATE_PATH if it is already set
(tilelang/env.py:495), so a private copy needs no mutation of shared state.
"""
import os
import shutil

SRC = "/app/plow/.venv-vllm028/lib/python3.12/site-packages/tilelang/src"
DST = "/workspace/dsv41-tl-templates/src"
ANCHOR = "// Note: E8M0 types are not supported in current HIP version"

IMPL = r"""
// --- UE8M0, added for DeepSeek-V4.1-Flash on ROCm -------------------------
// An 8-bit unsigned exponent: value = 2^(data - 127), no sign, no mantissa.
// This is the MXFP scale format the V4.1 checkpoint stores every block scale
// in (config `scale_fmt: ue8m0`; each *.scale tensor is F8_E8M0). CUDA gets it
// from CUTLASS as float_ue8m0_t lowering to cvt.rp.satfinite.ue8m0x2.f32; the
// HIP backend has none, so tilelang emits `fp8_e8_t` and every act_quant and
// quantized GEMM fails to compile with "unknown type name 'fp8_e8_t'".
//
// No hardware support is needed. An fp32 is 2^(e-127) * 1.m, so the encoded
// byte IS the fp32 biased exponent; decoding puts it back in the exponent
// field with a zero mantissa. That is what the CUTLASS path does too.
struct fp8_e8_t {
  unsigned char data;
  __device__ fp8_e8_t() {}
  __device__ explicit fp8_e8_t(unsigned char raw) : data(raw) {}
  __device__ fp8_e8_t(float val) {
    // Round toward +infinity, matching CUDA's cvt.rp: anything that is not
    // already an exact power of two takes the next exponent up, so a block
    // scale never underestimates the block's magnitude.
    if (!(val > 0.0f)) { // non-positive and NaN have no UE8M0 encoding
      data = 0;
      return;
    }
    unsigned int b = __float_as_uint(val);
    unsigned int e = (b >> 23) & 0xFFu;
    unsigned int m = b & 0x7FFFFFu;
    e += (m != 0u) ? 1u : 0u;
    data = (unsigned char)(e > 255u ? 255u : e);
  }
  __device__ operator float() const {
    return __uint_as_float(((unsigned int)data) << 23);
  }
};
// -------------------------------------------------------------------------
"""


def main() -> None:
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    shutil.copytree(SRC, DST)
    print("copied ->", DST)

    h = os.path.join(DST, "tl_templates/hip/hip_fp8.h")
    text = open(h).read()
    if "struct fp8_e8_t" in text:
        print("already defined, nothing to do")
        return
    assert ANCHOR in text, "anchor comment not found; header layout changed"
    # Insert BEFORE the anchor's commented-out aliases, so the type is declared
    # ahead of every later use in this header.
    text = text.replace(ANCHOR, IMPL + ANCHOR, 1)
    open(h, "w").write(text)
    print("inserted fp8_e8_t into", h)


main()
