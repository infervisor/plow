import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import torch


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    assert 'gfx942' in torch.cuda.get_device_properties(0).gcnArchName
    lib = ctypes.CDLL(str(args.library.resolve()))
    fn = lib.plow_glm_gemv
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 7 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    shapes = json.loads((args.capture / 'shapes.json').read_text())
    records = []

    def read(file, rows, cols, exact):
        raw = bytearray((args.capture / file).read_bytes())
        values = torch.frombuffer(raw, dtype=torch.bfloat16)
        assert values.numel() >= rows * cols
        if exact:
            assert values.numel() == rows * cols
        return values[:rows * cols].reshape(rows, cols).clone()

    arms = {'valu': 0, 'mfma': 1, 'selective': 2}

    def launch(c, x, w, m, n, k, mm, arm, groups, reps=1):
        rc = fn(c.data_ptr(), x.data_ptr(), w.data_ptr(), m, n, k, mm,
                arms[arm], groups, reps, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc

    def error(actual, reference):
        actual = actual.cpu().double()
        assert bool(torch.isfinite(actual).all())
        diff = actual - reference
        rel = float(torch.linalg.vector_norm(diff) /
                    torch.linalg.vector_norm(reference).clamp_min(1e-30))
        assert rel < 0.01, rel
        return dict(relative_l2=rel, max_abs=float(diff.abs().max()))

    def timing(call, reps):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        samples = []
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / reps)
        return samples

    for shape in shapes:
        n, k, groups = shape['n'], shape['k'], shape['groups']
        wh = read(shape['weight_file'], n, k, True)
        xh = read(shape['activation_file'], 8, k, False)
        assert bool(torch.isfinite(wh).all()) and bool(torch.isfinite(xh).all())
        reference = xh.double() @ wh.double().T
        w = wh.cuda()
        row0 = None
        for mm in (2, 4, 8):
            for m in sorted({1, mm - 1, mm}):
                x = xh[:mm].cuda()
                x[m:] = float('nan')
                output = {}
                checks = {}
                for arm in arms:
                    buf = torch.full((mm * n + 256,), 0x6B6B, dtype=torch.int16, device='cuda')
                    c = buf[128:128 + mm * n].view(torch.bfloat16).reshape(mm, n)
                    launch(c, x, w, m, n, k, mm, arm, groups)
                    torch.cuda.synchronize()
                    assert bool((buf[:128] == 0x6B6B).all())
                    assert bool((buf[128 + m * n:] == 0x6B6B).all()), (shape, mm, m, arm)
                    output[arm] = c[:m].clone()
                    checks[arm] = error(c[:m], reference[:m])
                delta = error(output['mfma'], output['valu'].cpu().double())
                expected = output['mfma' if k <= 2048 else 'valu']
                assert torch.equal(output['selective'].view(torch.int16), expected.view(torch.int16))
                if m == 1:
                    bits = output['mfma'].view(torch.int16).cpu()
                    if row0 is not None:
                        assert torch.equal(row0, bits), (shape, mm, 'cross-rung MFMA row')
                    row0 = bits
                else:
                    assert torch.equal(row0, output['mfma'][:1].view(torch.int16).cpu())
                if m == mm:
                    reverse = x.flip(0).contiguous()
                    c = torch.empty_like(output['mfma'])
                    launch(c, reverse, w, m, n, k, mm, 'mfma', groups)
                    torch.cuda.synchronize()
                    assert torch.equal(c.view(torch.int16), output['mfma'].flip(0).view(torch.int16))
                records.append(dict(shape=shape, mm=mm, m=m, oracle=checks,
                                    mfma_vs_valu=delta, guard_and_inactive_rows=True))

        # Stream fresh slabs within each launch; repeated small GLM weights fit in L2.
        reps = max(1, (3 << 30) // (n * k * 2))
        stream_weights = w.repeat(reps, 1)
        x = xh.cuda()
        for mm in (2, 4, 8):
            c = torch.empty((mm, n), dtype=torch.bfloat16, device='cuda')
            samples = {arm: [] for arm in arms}
            for arm in ('valu', 'mfma', 'selective', 'selective', 'mfma', 'valu'):
                samples[arm] += timing(
                    lambda: launch(c, x, stream_weights, mm, n, k, mm,
                                   arm, groups, reps), reps)
            records.append(dict(shape=shape, mm=mm, m=mm, stream_reps=reps,
                                stream_bytes=reps * n * k * 2, samples_us=samples,
                                median_us={arm: statistics.median(v) for arm, v in samples.items()}))
        del stream_weights, w
        args.out.write_text(json.dumps(dict(records=records), indent=2) + '\n')
        print(json.dumps(dict(weight=shape['weight'], checked=True)), flush=True)

    capture_files = {shape[key] for shape in shapes for key in ('weight_file', 'activation_file')}
    report = dict(schema='plow.glm53.gemv-mfma.primitive.v1', records=records,
                  gates=dict(relative_l2_limit=0.01, oracle='CPU FP64 matmul, all live outputs',
                             selective_dispatch='production d_gemv_t, GV_MFMA4_MAXK=2048; exact MFMA below/equal cutoff and exact VALU above',
                             guards='256 bytes before/after and all inactive output rows',
                             inactive_activation_rows='NaN',
                             mfma_row_independence='bitwise within and across MM=2/4/8; row permutation',
                             numerical_cases=sum('oracle' in r for r in records)),
                  library_sha256=sha(args.library),
                  capture_sha256={file: sha(args.capture / file) for file in sorted(capture_files)},
                  source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('kernels.hip')]},
                  limitations=['Primitive timings do not predict full-interpreter scheduling or resource pressure.',
                               'MFMA changes VALU reduction order; B1 production still uses VALU.',
                               'One captured layer/rank; serving and model quality require separate qualification.'])
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
