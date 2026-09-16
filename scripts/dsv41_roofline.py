"""Per-op roofline for one V4.1 layer at T=8192, TP8, gfx942.

Floors are the GENEROUS ones: compute at full MFMA peak, traffic at full HBM peak, whichever
binds. No op can beat its own floor, so the SUM of floors is a hard lower bound on the layer --
and 40x it is a hard lower bound on the model.
"""
T, TP = 8192, 8
HID, NH_L, DLAT, DROPE, TOPK = 5120, 8, 512, 64, 512
MOE_INTER, K_EFF, N_EXP = 2304, 7, 385
HBM = 5.3e12          # B/s
MFMA_BF16 = 653e12    # MAC/s  (1307 TFLOP/s)
MFMA_FP8 = 1306e12    # MAC/s
VALU_F32 = 40.9e12    # FMA/s  (304 CU * 64 lanes * 2.1 GHz)
XGMI = 400e9          # B/s per GPU, effective

def us(x): return x * 1e6

rows = []
def add(name, meas_us, floor_s, note):
    rows.append((name, meas_us, us(floor_s), note))

# FLASH_GATHER: QK over d_latent + PV over d_latent, per (token, head, key).
# THE FLOOR IS THE VECTOR ALU, NOT THE MATRIX CORE, and this line used to say MFMA_BF16 -- which
# priced the layer's biggest op against an engine it provably cannot use. doc 12.59: the written
# MFMA body (PLOW_FA_GATHER_MFMA) was ranked on hardware and is 2.6x SLOWER, because its work item
# is one query token and TP8 leaves n_head=8 on a 16-wide MFMA M-dimension. The shipped body does
# every score and every PV MAC on the VALU by construction, so VALU_F32 is its bound.
# This is not a detail: it raises the model's floor from 91 ms to ~125 ms for 40 layers.
mac = T * NH_L * TOPK * (DLAT + DROPE + DLAT)
add("FLASH_GATHER_PREFILL", 2955, mac / VALU_F32, f"{mac/1e9:.1f} G MAC, VALU (MFMA ranked 2.6x slower)")

# GEMM_FP8_MX: the five projections. N,K per rank.
proj = [(1280, HID), (NH_L*(DLAT+DROPE), 1280), (DLAT+DROPE, HID),
        (1024, NH_L*DLAT), (HID, 1024)]
mac = sum(T * n * k for n, k in proj)
add("GEMM_FP8_MX", 2788, mac / MFMA_BF16, f"{mac/1e9:.0f} G MAC, w8a16 so bf16 peak")

# MoE pair: k_eff slots per token, gate+up+down over moe_inter/TP.
imoe = MOE_INTER // TP
mac = T * K_EFF * (2 * HID * imoe + imoe * HID)
add("MOE pair (GLU+DOWN)", 1562, mac / MFMA_FP8, f"{mac/1e9:.0f} G MAC, fp4/fp8 peak")

# XREDUCE2: two all-reduces of [T,HID] bf16; ring moves 2(N-1)/N of the payload.
payload = T * HID * 2
add("XREDUCE2", 1673, 2 * (2 * (TP-1) / TP) * payload / XGMI, "2 all-reduce, ring, xGMI")

# GEMV_F32: mHC mix, [T, mix=24] over K = hc_mult*hidden.
mac = T * 24 * (4 * HID)
traf = T * (4 * HID) * 2
add("GEMV_F32", 1060, max(mac / VALU_F32, traf / HBM), "f32 VALU vs x traffic")

# Streaming ops: bytes moved / HBM. hc_mult=4 residual is 4*hidden wide.
add("HYPER_CONN PRE+POST", 1089, (2*(T*4*HID*2) + 2*(T*4*HID*2 + T*HID*2)) / HBM, "read+write 4x residual")
add("COMPRESS_ROPE_QUANT", 378, 3 * (T * (DLAT+DROPE) * 2 * 2) / HBM, "3 packets, read+write")
add("RMSNORM", 298, 4 * (T * HID * 2 * 2) / HBM, "4 packets, read+write")
add("MOE_COMBINE_PF", 316, (T*K_EFF*HID*4 + T*HID*2) / HBM, "read k partials, write hidden")
add("FLASH_MLA_PREFILL", 407, (T*128*(DLAT+DROPE)*2) / MFMA_BF16 * 0 + (T*128*(DLAT+DROPE)*2)/HBM, "window=128 band")

meas_tot = sum(r[1] for r in rows)
floor_tot = sum(r[2] for r in rows)
print(f"{'op':<24}{'measured':>10}{'floor':>10}{'x off':>8}  note")
for n, m, f, note in rows:
    print(f"{n:<24}{m:>9.0f}{f:>10.0f}{m/f:>8.1f}  {note}")
print(f"{'-'*24}{'':->10}{'':->10}")
print(f"{'SUM (these ops)':<24}{meas_tot:>9.0f}{floor_tot:>10.0f}{meas_tot/floor_tot:>8.1f}")
LAYER = 14900.0
other = LAYER - meas_tot
print(f"{'other ~10 ops':<24}{other:>9.0f}{'':>10}")
print(f"{'LAYER':<24}{LAYER:>9.0f}")
print()
print(f"40-layer measured      : {LAYER*40/1000:.0f} ms")
print(f"40-layer FLOOR (these) : {floor_tot*40/1000:.0f} ms   <- hard lower bound, ignores the other ops entirely")
print(f"target                 : 90 ms")
print(f"target / floor         : {90000/(floor_tot*40):.2f}x")
