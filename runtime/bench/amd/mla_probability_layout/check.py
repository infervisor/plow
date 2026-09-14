import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timing', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert 'gfx942' in torch.cuda.get_device_properties(0).gcnArchName
    ptr, uint = ctypes.c_void_p, ctypes.c_uint
    libs = {name: ctypes.CDLL(str(path.resolve())) for name, path in (('off', args.off), ('on', args.on))}
    for lib in libs.values():
        lib.plow_attention.argtypes = [ptr] * 9 + [uint] * 7 + [ctypes.c_float, ptr]
        lib.plow_union.argtypes = [ptr] * 4 + [uint] * 3 + [ptr]

    def launch(fn, tensors, *scalars):
        status = fn(*(t.data_ptr() for t in tensors), *scalars, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f'HIP launch status {status}')

    def guarded(shape):
        n = math.prod(shape)
        backing = torch.full((n + 256,), 12345.0, device='cuda')
        return backing, backing[128:-128].view(shape)

    cases = []
    shapes = [(1, 1, 1, 0), (7, 65, 7, 0), (33, 129, 2, 31),
              (65, 513, 1, 0), (129, 2049, 3, 0), (128, 4097, 1, 0)]
    if args.timing:
        shapes = [(128, 8193, 1, 0), (512, 16385, 1, 0)]
    for rows, length, splits, window in shapes:
        ctx, cap = ((length + 255) // 256) * 256, 16384
        qa = torch.randn(rows, 8, 512, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(rows, 8, 64, dtype=torch.bfloat16, device='cuda')
        ck = torch.randn(ctx, 512, dtype=torch.bfloat16, device='cuda')
        kr = torch.randn(ctx, 64, dtype=torch.bfloat16, device='cuda')
        cs = ck.float().abs().amax(-1).clamp_min(1e-6) / 448
        rs = kr.float().abs().amax(-1).clamp_min(1e-6) / 448
        ck8 = (ck.float() / cs[:, None]).to(torch.float8_e4m3fn)
        kr8 = (kr.float() / rs[:, None]).to(torch.float8_e4m3fn)
        scales = torch.cat((cs, rs))
        scales[length:ctx] = float('nan')
        scales[ctx + length:] = float('nan')
        ck8.view(torch.uint8)[length:] = 0x7f
        kr8.view(torch.uint8)[length:] = 0x7f
        lengths = torch.tensor([length], dtype=torch.int32, device='cuda')
        idx = torch.zeros(rows, 2048, dtype=torch.int32, device='cuda')
        live = length - rows + torch.arange(rows, device='cuda') + 1
        for q in range(rows):
            n = int(live[q])
            step = next(s for s in range(37, 101) if math.gcd(n, s) == 1)
            idx[q] = (torch.arange(2048, device='cuda') * step + q * 313) % n
        tiles = (rows + 7) // 8
        header = (tiles * 4 + 255) // 256 * 256
        uni = torch.empty(header + tiles * cap * 12, dtype=torch.uint8, device='cuda')
        mask = torch.empty(min(tiles, 304) * ctx, dtype=torch.int64, device='cuda')
        launch(libs['off'].plow_union, [uni, mask, idx, lengths], rows, ctx, cap)
        sampled = sorted(set((0, min(7, rows - 1), rows // 2, rows - 1)))
        for mode in range(4):
            gather, fp8 = bool(mode & 2), bool(mode & 1)
            ns = 1 if gather else splits
            for rope8 in ((0, 1) if fp8 else (0,)):
                tensors = []
                for _ in libs:
                    bo, out = guarded((rows, 8, ns, 512))
                    bm, ml = guarded((rows, 8, ns, 2))
                    tensors.append((bo, bm, out, ml))
                calls = {}
                for (name, lib), (bo, bm, out, ml) in zip(libs.items(), tensors):
                    values = [out, ml, qa, qr, ck8 if fp8 else ck, kr8 if rope8 else kr, lengths, uni, scales]
                    calls[name] = lambda lib=lib, values=values: launch(lib.plow_attention, values,
                        rows, ctx, cap, ns, window, rope8, mode, 576**-0.5)
                    for _ in range(3):
                        out.fill_(float('nan'))
                        ml.fill_(float('nan'))
                        calls[name]()
                        torch.cuda.synchronize()
                        assert torch.isfinite(out).all(), (name, rows, length, mode, rope8,
                            int(torch.isnan(out).sum()), int(torch.isinf(out).sum()), ml.flatten()[:16].tolist())
                        assert torch.isfinite(ml[..., 1]).all(), (name, rows, length, mode, rope8, 'ml')
                        assert all((b[:128] == 12345).all() and (b[-128:] == 12345).all() for b in (bo, bm))
                assert torch.equal(tensors[0][2], tensors[1][2]), (rows, mode, rope8, 'out')
                assert torch.equal(tensors[0][3], tensors[1][3]), (rows, mode, rope8, 'ml')
                out, ml = tensors[1][2:]
                m, l = ml[..., 0], ml[..., 1]
                factors = torch.where(l > 0, torch.exp2(m - m.max(-1, keepdim=True).values), 0)
                got = (out * factors[..., None]).sum(2) / (l * factors).sum(2)[..., None]
                refs = []
                for q in sampled:
                    n = int(live[q])
                    selected = idx[q, :min(2048, n)].long().unique() if gather else torch.arange(
                        max(0, n - window) if window else 0, n, device='cuda')
                    values = ck8.view(torch.uint8)[selected].view(torch.float8_e4m3fn).float() * cs[selected, None] if fp8 else ck[selected].float()
                    rope = kr8.view(torch.uint8)[selected].view(torch.float8_e4m3fn).float() * rs[selected, None] if rope8 else kr[selected].float()
                    scores = (qa[q].float() @ values.T + qr[q].float() @ rope.T) * 576**-0.5
                    refs.append(scores.softmax(-1) @ values)
                ref = torch.stack(refs)
                err = ((got[sampled] - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                assert err < .01, (rows, length, mode, rope8, err)
                record = dict(rows=rows, length=length, splits=ns, window=window,
                              mode=mode, rope8=rope8, relative_l2=err, exact_schedule_match=True)
                if args.timing:
                    graphs = {}
                    for name, call in calls.items():
                        call()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            call()
                        graphs[name] = graph
                    samples = {name: [] for name in libs}
                    for repetition in range(12):
                        order = ('off', 'on') if repetition % 2 == 0 else ('on', 'off')
                        for name in order:
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            begin.record()
                            graphs[name].replay()
                            end.record()
                            end.synchronize()
                            samples[name].append(begin.elapsed_time(end))
                    record['samples_ms'] = samples
                    record['median_ms'] = {k: statistics.median(v) for k, v in samples.items()}
                cases.append(record)
                print(json.dumps(record), flush=True)
    args.out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        hip=torch.version.hip, seed=73, timing=args.timing, cases=cases,
        library_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in (('off', args.off), ('on', args.on))},
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main()
