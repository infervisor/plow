# Sparse attention decode (op_sparse_attn_decode.cuh) vs a torch reference + HBM-cold timing.
# Usage: python probe_sparse_attn_decode.py <packet_ops_test.cubin>  (under gpulease)
import sys, math, torch, ctypes
sys.path.insert(0, "/root/plow/.claude/worktrees/dsv41-on-main/scripts/dsv41_nv")
from cudrv import Cubin, i32
P = Cubin(sys.argv[1])
dev = "cuda"
D, W, CAP = 512, 128, 2048
Q_B, KV_B = 16 * D * 2, 32 * D * 2
SMEM = Q_B + 4 * KV_B + 2 * 16 * 33 * 4 + 16 * 40 * 2 + 3 * 16 * 4 + 2048 * 4
scale = D ** -0.5
torch.manual_seed(0)


def ref(q, ring, cmp, idx, pos, sink):
    B, H, _ = q.shape
    out = torch.empty(B, H, D, device=dev)
    for b in range(B):
        nwin = min(int(pos[b]) + 1, W)
        rows = [ring[b, :nwin].float()]
        if idx is not None:
            j = idx[b][idx[b] >= 0].long()
            rows.append(cmp[b, j].float())
        kv = torch.cat(rows, 0)
        s = q[b].float() @ kv.T * scale
        m = torch.maximum(s.amax(1), torch.tensor(-1e30, device=dev))
        p = torch.exp(s - m[:, None])
        den = p.sum(1) + torch.exp(sink - m)
        out[b] = (p.bfloat16().float() @ kv) / den[:, None]
    return out


def scratch_bytes(B, groups, nsplit):
    return B * groups * nsplit * (16 * D + 32) * 4


ok = True
for H in (16, 64):
    for B in (1, 4, 16, 64):
        for topk in (512, 0):
            groups = H // 16
            q = torch.randn(B, H, D, device=dev).bfloat16()
            ring = torch.randn(B, W, D, device=dev).bfloat16()
            cmp = torch.randn(B, CAP, D, device=dev).bfloat16() if topk else None
            pos = torch.randint(1000, 60000, (B,), device=dev, dtype=torch.int32)
            pos[0] = 37  # ring still filling
            idx = None
            if topk:
                idx = torch.stack([torch.randperm(CAP, device=dev)[:topk] for _ in range(B)]).int()
                idx[0, 100:] = -1  # short context: few compressed rows
                idx[:, 7] = -1
            sink = torch.randn(H, device=dev)
            for nsplit in sorted({1, 2, max(1, min(20, math.ceil(132 / (B * groups)))), max(1, min(20, math.ceil(264 / (B * groups)))), 8}):
                o = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)
                copies = min(8, max(1, math.ceil(200e6 / (B * (W + CAP if topk else W) * D * 2))))
                ringr = ring.repeat(copies, 1, 1)
                cmpr = cmp.repeat(copies, 1, 1) if topk else None
                scr = torch.zeros(copies * scratch_bytes(B, groups, nsplit) // 4 + 16, device=dev)
                nul = ctypes.c_void_p(0)
                args = lambda reps: [o, q, ringr, cmpr if topk else nul, idx if topk else nul, pos, sink, scr, i32(B), i32(H), i32(W), i32(CAP),
                                     i32(topk), i32(nsplit), ctypes.c_float(scale), i32(reps), i32(copies)]
                margs = lambda reps: [o, scr, sink, i32(B), i32(H), i32(nsplit), i32(reps), i32(copies)]
                P.launch("t_sparse_attn_decode", (132,), (256,), args(1), smem=SMEM)
                if nsplit > 1:
                    P.launch("t_sparse_attn_merge", (132,), (256,), margs(1))
                torch.cuda.synchronize()
                r = ref(q, ring, cmp, idx, pos, sink)
                rel = ((o.float() - r).norm() / r.norm()).item()
                reps = 40
                e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                e[0].record(); P.launch("t_sparse_attn_decode", (132,), (256,), args(reps), smem=SMEM); e[1].record()
                if nsplit > 1:
                    P.launch("t_sparse_attn_merge", (132,), (256,), margs(reps))
                e[2].record(); torch.cuda.synchronize()
                us = e[0].elapsed_time(e[1]) * 1e3 / reps
                mus = e[1].elapsed_time(e[2]) * 1e3 / reps
                rows = sum(min(int(p) + 1, W) for p in pos.tolist()) + (int((idx >= 0).sum()) if topk else 0)
                gbs = (rows * D * 2 + 2 * B * H * D * 2) / (us + mus) / 1e3
                good = rel < 1e-2 and torch.isfinite(o).all().item()
                ok &= good
                print(f"[{'PASS' if good else 'FAIL'}] H={H} B={B:2d} topk={topk:3d} nsplit={nsplit:2d} rel={rel:.2e} attn {us:6.2f} + merge {mus:5.2f} us {gbs:6.0f} GB/s", flush=True)
print("ALL PASS" if ok else "SOME FAIL")
