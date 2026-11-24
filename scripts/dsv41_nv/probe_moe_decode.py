# V4.1 routed-expert MoE at decode (op_moe_decode_v41.cuh) on the prefill role's operands, vs a torch
# reference of the role's numerics (GluEpiS / DownEpiS), plus HBM-cold timing.
# Usage: python probe_moe_decode.py <packet_ops_test.cubin>  (under gpulease)
import sys, ctypes, torch
sys.path.insert(0, "/root/plow/.claude/worktrees/dsv41-on-main/scripts/dsv41_nv")
from cudrv import Cubin, i32
P = Cubin(sys.argv[1])
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
E, K6, H, I, LIM, BM = 384, 6, 5120, 576, 10.0, 64
UNUSED = 0xFFFFFFFF
COPIES = 8
SMEM = (4 * E + 15) // 16 * 16 + 8 * (32 + 3 * 8192)  # moe_decode_v41_smem_bytes(E, 8 warps)
g = torch.Generator(device="cpu").manual_seed(0)
# physical experts: fp4 bytes (all 16 codes), ue8m0 near 2^-6
w13 = torch.randint(0, 256, (E, 2, I, H // 2), generator=g, dtype=torch.uint8).to(dev)
s13 = torch.randint(119, 124, (E, 2, I, H // 32), generator=g, dtype=torch.uint8).to(dev)
w2 = torch.randint(0, 256, (E, H, I // 2), generator=g, dtype=torch.uint8).to(dev)
s2 = torch.randint(119, 124, (E, H, I // 32), generator=g, dtype=torch.uint8).to(dev)
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
EXP_BYTES = 2 * I * H // 2 + 2 * I * H // 32 + H * I // 2 + H * I // 32


def tables(copies):
    wt, st = [], []
    for c in range(copies):
        for e in range(E):
            p = (e + 97 * c) % E
            wt += [w13[p, 0].data_ptr(), w13[p, 1].data_ptr(), w2[p].data_ptr()]
            st += [s13[p, 0].data_ptr(), s13[p, 1].data_ptr(), s2[p].data_ptr()]
    return (torch.tensor(wt, dtype=torch.int64, device=dev), torch.tensor(st, dtype=torch.int64, device=dev))


def deq(w, s):
    n, kh = w.shape
    v = torch.stack([LUT[(w & 15).long()], LUT[(w >> 4).long()]], -1).view(n, kh * 2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, 1)


def fq(x):  # act_quant(x, 32, ue8m0) as a bf16 fake quant
    r, k = x.shape
    xb = x.float().view(r, k // 32, 32)
    t = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    s = torch.exp2(torch.ceil(torch.log2(t)))
    return ((xb / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).view(r, k).bfloat16()


def case(T, seed):
    gg = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.stack([torch.randperm(E, generator=gg)[:K6] for _ in range(T)])
    wt = torch.rand(T, K6, generator=gg) * 0.5 + 0.05
    cnt = torch.bincount(idx.flatten(), minlength=E)
    padded = (cnt + BM - 1) // BM * BM
    rowoff = torch.cumsum(padded, 0) - padded
    rows = int(padded.sum())
    tok = torch.full((rows,), UNUSED, dtype=torch.int64)
    part = torch.full((rows,), UNUSED, dtype=torch.int64)
    gate = torch.zeros(rows)
    fill = rowoff.clone()
    for s in range(T * K6):
        e = int(idx.flatten()[s])
        at = int(fill[e])
        fill[e] += 1
        tok[at], part[at], gate[at] = s // K6, s, wt.flatten()[s]
    tilep = torch.zeros(E + 1, dtype=torch.int64)
    tilep[1:] = torch.cumsum(padded // BM, 0)
    meta = torch.cat([rowoff, cnt, tilep]).to(torch.int32).to(dev)
    x = fq(torch.randn(T, H, generator=gg).to(dev))
    return dict(T=T, cnt=cnt, rowoff=rowoff, rows=rows, meta=meta, x=x,
                tok=tok.to(torch.int32).to(dev), part=part.to(torch.int32).to(dev), gate=gate.to(dev))


def ref(c):
    fu = torch.zeros(c["rows"], I, device=dev, dtype=torch.bfloat16)
    part = torch.zeros(c["T"] * K6, H, device=dev)
    for e in range(E):
        n = int(c["cnt"][e])
        if n == 0:
            continue
        r0 = int(c["rowoff"][e])
        xs = c["x"][c["tok"][r0:r0 + n].long()].float()
        gv = (xs @ deq(w13[e, 0], s13[e, 0]).T).bfloat16().float().clamp(max=LIM)
        uv = (xs @ deq(w13[e, 1], s13[e, 1]).T).bfloat16().float().clamp(-LIM, LIM)
        h = (c["gate"][r0:r0 + n, None] * (gv / (1 + torch.exp(-gv)) * uv)).bfloat16()
        fu[r0:r0 + n] = fq(h)
        part[c["part"][r0:r0 + n].long()] = (fu[r0:r0 + n].float() @ deq(w2[e], s2[e]).T).bfloat16().float()
    return fu, part


ok = True
wt1, st1 = tables(1)
wtc, stc = tables(COPIES)
SCR = 2 * 132 * 8 * 512  # moe_glu_decode_v41_scratch_floats(132, 8) per copy
CTR = 132 * 8  # moe_glu_decode_v41_ctrs(132, 8)
scr = torch.zeros(COPIES * SCR, device=dev)
ctr = torch.zeros(COPIES * CTR, dtype=torch.int32, device=dev)
for T in (1, 4, 16, 64):
    c = case(T, T)
    fu = torch.zeros(c["rows"], I, device=dev, dtype=torch.bfloat16)
    part = torch.zeros(T * K6, H, device=dev)
    glu = lambda wt, st, reps, copies: P.launch("t_moe_glu_decode", (132,), (256,), [fu, c["x"], wt, st, c["meta"], c["tok"], c["gate"], i32(I), i32(H),
                                                                                     i32(E), i32(4), ctypes.c_float(LIM), scr, i32(SCR), ctr, i32(CTR),
                                                                                     i32(reps), i32(copies)], smem=SMEM)
    down = lambda wt, st, reps, copies: P.launch("t_moe_down_decode", (132,), (256,), [part, fu, wt, st, c["meta"], c["part"], ctypes.c_void_p(0),
                                                                                       i32(H), i32(I), i32(E), i32(reps), i32(copies)], smem=SMEM)
    glu(wt1, st1, 1, 1)
    down(wt1, st1, 1, 1)
    torch.cuda.synchronize()
    fr, pr = ref(c)
    live = torch.zeros(c["rows"], dtype=torch.bool, device=dev)
    for e in range(E):
        live[int(c["rowoff"][e]):int(c["rowoff"][e]) + int(c["cnt"][e])] = True
    rg = ((fu[live].float() - fr[live].float()).norm() / fr[live].float().norm()).item()
    mis = (fu[live] != fr[live]).float().mean().item()
    rd = ((part - pr).norm() / pr.norm()).item()
    good = rg < 1e-2 and rd < 3e-3 and torch.isfinite(part).all().item()
    ok &= good
    reps = 20
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    torch.cuda._sleep(int(4e6))
    ev[0].record(); glu(wtc, stc, reps, COPIES); ev[1].record()
    torch.cuda._sleep(int(4e6))
    ev[2].record(); down(wtc, stc, reps, COPIES); ev[3].record()
    torch.cuda.synchronize()
    ug, ud = ev[0].elapsed_time(ev[1]) * 1e3 / reps, ev[2].elapsed_time(ev[3]) * 1e3 / reps
    na = int((c["cnt"] > 0).sum())
    gb = na * EXP_BYTES
    gbg, gbd = na * (2 * I * H // 2 + 2 * I * H // 32), na * (H * I // 2 + H * I // 32)
    print(f"[{'PASS' if good else 'FAIL'}] B={T:2d} experts={na:3d} glu rel={rg:.1e} mis={mis:.2%} down rel={rd:.1e} | "
          f"GLU {ug:7.1f} us {gbg / ug / 1e3:5.0f} GB/s  DOWN {ud:7.1f} us {gbd / ud / 1e3:5.0f} GB/s  total {ug + ud:7.1f} us "
          f"{gb / (ug + ud) / 1e3:5.0f} GB/s", flush=True)
print("ALL PASS" if ok else "SOME FAIL")
