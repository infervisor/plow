// Empirically DERIVE the per-lane fragment layout of
// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 on sm_120a.
//
// Method: pairwise one-hot probe. For every A-fragment slot (lane la, byte ia)
// and every B-fragment slot (lane lb, byte ib), run an mma where A has a single
// 1.0 at that slot and B has a single 1.0 at its slot, everything else zero.
// The product is nonzero iff the two slots share the same k. When nonzero, it
// lands in exactly one accumulator slot (lane lc, reg ic), which tells us the
// (i,j) that the pair maps to. From this incidence table the full layout is
// reconstructed on the host with zero assumptions.
//
// A frag: 4 x b32 per lane = 16 e4m3 values -> 32 lanes * 16 = 512 slots (M*K=16*32)
// B frag: 2 x b32 per lane =  8 e4m3 values -> 32 lanes *  8 = 256 slots (N*K=8*32)
// C frag: 4 x f32 per lane                  -> 32 lanes *  4 = 128 slots (M*N=16*8)

#include <cstdio>
#include <cstdint>

#define A_SLOTS 512
#define B_SLOTS 256

// out[pair] = lane_c*4 + reg_c, or -1 if the product was zero.
// pair index = a_slot * B_SLOTS + b_slot
__global__ void probe(int16_t *out) {
  int pair = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  int lane = threadIdx.x % 32;
  if (pair >= A_SLOTS * B_SLOTS) return;

  int a_slot = pair / B_SLOTS;
  int b_slot = pair % B_SLOTS;
  int la = a_slot / 16, ia = a_slot % 16;  // lane, byte-index within 16 bytes
  int lb = b_slot / 8,  ib = b_slot % 8;

  uint32_t a[4] = {0, 0, 0, 0};
  uint32_t b[2] = {0, 0};
  float d[4] = {0.f, 0.f, 0.f, 0.f};

  // e4m3 (E4M3FN) encoding of +1.0 is 0x38.
  if (lane == la) a[ia / 4] = 0x38u << (8 * (ia % 4));
  if (lane == lb) b[ib / 4] = 0x38u << (8 * (ib % 4));

  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

  int found = -1;
  for (int r = 0; r < 4; ++r)
    if (d[r] != 0.f) found = lane * 4 + r;

  // Reduce across the warp: at most one lane has a hit.
  for (int off = 16; off; off >>= 1) {
    int o = __shfl_down_sync(0xffffffffu, found, off);
    if (o >= 0) found = o;
  }
  if (lane == 0) out[pair] = (int16_t)found;
}

int main() {
  int n = A_SLOTS * B_SLOTS;
  int16_t *d_out, *h_out = (int16_t *)malloc(n * sizeof(int16_t));
  cudaMalloc(&d_out, n * sizeof(int16_t));
  int warps_per_block = 8;
  probe<<<(n + warps_per_block - 1) / warps_per_block, warps_per_block * 32>>>(d_out);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(h_out, d_out, n * sizeof(int16_t), cudaMemcpyDeviceToHost);
  FILE *f = fopen("/workspace/fp8poc/incidence.bin", "wb");
  fwrite(h_out, sizeof(int16_t), n, f);
  fclose(f);
  long hits = 0;
  for (int i = 0; i < n; ++i) if (h_out[i] >= 0) hits++;
  printf("probe done: %ld hits out of %d pairs\n", hits, n);
  return 0;
}
