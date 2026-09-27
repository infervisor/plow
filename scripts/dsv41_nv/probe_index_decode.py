# V4.1 indexer decode (op_index_decode.cuh) vs a torch reference + HBM-cold timing.
# Usage: python probe_index_decode.py <packet_ops_test.cubin>  (under gpulease)
import sys, math, ctypes, torch
sys.path.insert(0, "/root/plow/.claude/worktrees/dsv41-on-main/scripts/dsv41_nv")
from cudrv import Cubin, i32
P = Cubin(sys.argv[1])
dev = "cuda"
DI, TOPK = 128, 512
torch.manual_seed(0)


def ref_idx(score, lens):
    out = torch.full((len(lens), TOPK), -1, dtype=torch.int32, device=dev)
    for b, n in enumerate(lens):
        if n <= TOPK:
            out[b, :n] = torch.arange(n, device=dev, dtype=torch.int32)
            continue
        order = torch.sort(-score[b, :n], stable=True).indices[:TOPK]  # ties: lowest index first
        out[b] = order.sort().values.int()
    return out


ok = True
for HI in (8, 32):
    for B in (1, 4, 16, 64):
        for L in (400, 2048, 8192, 16500):
            cap = (L + 63) // 64 * 64
            q = (torch.randn(B, HI, DI, device=dev) * 0.3).bfloat16()
            w = (torch.randn(B, HI, device=dev) * 0.5).bfloat16()
            WS = 0.1
            k = torch.randn(B, cap, DI, device=dev).bfloat16()
            pos = torch.full((B,), L - 1, dtype=torch.int32, device=dev)
            if B > 1:
                pos[0] = min(L, 300) - 1  # one short slot: take-all path
            lens = [int(p) + 1 for p in pos.tolist()]
            copies = min(64, max(1, math.ceil(200e6 / k.numel() / 2)))
            kr = k.repeat(copies, 1, 1)
            score = torch.zeros(B, cap, device=dev)
            idx = torch.empty(B, TOPK, dtype=torch.int32, device=dev)
            sargs = lambda reps: [score, q, w, ctypes.c_float(WS), kr, pos, i32(B), i32(HI), i32(cap), i32(1), i32(reps), i32(copies)]
            sw = 400 + cap + cap // 4
            P.launch("t_index_score_decode", (132,), (256,), sargs(1))
            P.launch("t_index_select_decode", (132,), (256,), [idx, score, pos, i32(B), i32(cap), i32(1), i32(1), i32(sw)], smem=sw * 4)
            torch.cuda.synchronize()
            ref_s = (torch.relu(torch.einsum("bhd,btd->bht", q.float(), k.float())) * (w.float() * WS)[:, :, None]).sum(1)
            srel = 0.0
            for b in range(B):
                if lens[b] > TOPK:
                    r = ref_s[b, :lens[b]]
                    srel = max(srel, ((score[b, :lens[b]] - r).norm() / r.norm()).item())
            idx_ok = torch.equal(idx, ref_idx(score, lens))
            # uncached select path (keys re-read from global each pass)
            idx2 = torch.empty_like(idx)
            P.launch("t_index_select_decode", (132,), (256,), [idx2, score, pos, i32(B), i32(cap), i32(1), i32(1), i32(1024)], smem=4096)
            torch.cuda.synchronize()
            idx_ok &= torch.equal(idx, idx2)
            reps = 30
            e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            torch.cuda._sleep(int(4e6))  # GPU stays busy while Python enqueues: events time the kernels, not the launch path
            e[0].record(); P.launch("t_index_score_decode", (132,), (256,), sargs(reps)); e[1].record()
            P.launch("t_index_select_decode", (132,), (256,), [idx, score, pos, i32(B), i32(cap), i32(1), i32(reps), i32(sw)], smem=sw * 4)
            e[2].record(); torch.cuda.synchronize()
            sus = e[0].elapsed_time(e[1]) * 1e3 / reps
            selus = e[1].elapsed_time(e[2]) * 1e3 / reps
            kbytes = sum(n for n in lens if n > TOPK) * DI * 2
            gbs = kbytes / sus / 1e3 if kbytes else 0
            good = srel < 1e-3 and idx_ok
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] HI={HI:2d} B={B:2d} len={L:5d} score_rel={srel:.1e} idx_ok={idx_ok} "
                  f"score {sus:7.2f} us ({gbs:5.0f} GB/s) select {selus:6.2f} us", flush=True)
# selection alone on tie-heavy scores (few distinct values), both smem and global paths
for L, nv in ((3000, 7), (16500, 50), (16500, 1)):
    B, cap = 4, (L + 63) // 64 * 64
    score = torch.randint(0, nv, (B, cap), device=dev).float() - nv / 2
    pos = torch.full((B,), L - 1, dtype=torch.int32, device=dev)
    lens = [L] * B
    for sw in (400 + cap, 1024):
        idx = torch.empty(B, TOPK, dtype=torch.int32, device=dev)
        P.launch("t_index_select_decode", (132,), (256,), [idx, score, pos, i32(B), i32(cap), i32(1), i32(1), i32(sw)], smem=sw * 4)
        torch.cuda.synchronize()
        good = torch.equal(idx, ref_idx(score, lens))
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] ties len={L} distinct={nv} smem_words={sw}", flush=True)
print("ALL PASS" if ok else "SOME FAIL")
