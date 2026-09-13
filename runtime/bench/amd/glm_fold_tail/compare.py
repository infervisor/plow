import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import torch


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    assert 'gfx942' in torch.cuda.get_device_properties(0).gcnArchName
    functions = {}
    for arm in ('control', 'candidate'):
        fn = ctypes.CDLL(str(args.root / f'fold-{arm}.so')).plow_fold
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 3 + [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        functions[arm] = fn

    def read(name, dtype):
        data = bytearray((args.root / f'{name}.prefill.bin').read_bytes())
        return torch.frombuffer(data, dtype=dtype).clone()

    weight = read('weight', torch.bfloat16).reshape(8, 512, 256).cuda()
    captured = read('opart', torch.float32)[:8192 * 8 * 512].reshape(8192, 8, 1, 512)
    captured_ml = read('mlpart', torch.float32)[:8192 * 8 * 2].reshape(8192, 8, 1, 2)
    expected = read('oat', torch.bfloat16)[:8191 * 8 * 256].reshape(8191, 8, 256)
    lengths = [1, 7, 8, 9, 127, 128, 129, 303, 304, 305, 311, 312,
               511, 512, 513, 2047, 2048, 2049, 4463, 4464, 4465, 8191, 8192]
    records = []
    for splits in (1, 2, 7):
        if splits == 1:
            partial = captured.cuda()
            ml = captured_ml.cuda()
        else:
            partial = torch.randn((8192, 8, splits, 512), device='cuda')
            ml = torch.randn((8192, 8, splits, 2), device='cuda')
            ml[..., 1] = ml[..., 1].abs() + 1
            ml[::3, :, -1, 0] = float('-inf')
            ml[::3, :, -1, 1] = 0
            partial[::3, :, -1] = 0
        for rows in lengths:
            if splits == 1 and rows == 8192:
                continue  # capture contains only 8191 live rows
            active_partial = partial[:rows].clone()
            active_ml = ml[:rows].clone()
            outputs = {}
            buffers = {}
            calls = {}
            for arm, fn in functions.items():
                guard = torch.full((8192 * 8 * 256 + 256,), 0x6B6B,
                                   dtype=torch.int16, device='cuda')
                out = guard[128:128 + rows * 8 * 256].view(torch.bfloat16)
                def call(fn=fn, out=out):
                    rc = fn(out.data_ptr(), active_partial.data_ptr(), active_ml.data_ptr(),
                            weight.data_ptr(), rows, 8, splits,
                            torch.cuda.current_stream().cuda_stream)
                    assert rc == 0, rc
                call()
                torch.cuda.synchronize()
                assert bool((guard[:128] == 0x6B6B).all())
                assert bool((guard[128 + rows * 8 * 256:] == 0x6B6B).all())
                assert bool(torch.isfinite(out).all())
                outputs[arm] = out.clone()
                buffers[arm] = guard
                calls[arm] = call
            differences = int((outputs['control'].view(torch.int16) !=
                               outputs['candidate'].view(torch.int16)).sum())
            capture_differences = None
            if splits == 1:
                capture_differences = int((outputs['control'].view(torch.int16).cpu() !=
                                           expected[:rows].flatten().view(torch.int16)).sum())
            timings = {arm: [] for arm in functions}
            if rows >= 2047:
                for arm in ('control', 'candidate', 'candidate', 'control'):
                    calls[arm]()
                    for _ in range(5):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(5):
                            calls[arm]()
                        end.record()
                        end.synchronize()
                        timings[arm].append(start.elapsed_time(end) * 1000 / 5)
            row = dict(rows=rows, splits=splits, differences=differences,
                       control_vs_capture_differences=capture_differences,
                       guards=True, samples_us=timings,
                       median_us={arm: statistics.median(t) for arm, t in timings.items() if t})
            records.append(row)
            print(json.dumps(row), flush=True)
            del buffers, outputs, calls, guard, out, active_partial, active_ml
        del partial, ml
    report = dict(records=records, source_sha256={p.name: sha(p) for p in
                  [Path(__file__), Path(__file__).with_name('kernels.hip')]},
                  library_sha256={arm: sha(args.root / f'fold-{arm}.so') for arm in functions},
                  capture_sha256={name: sha(args.root / f'{name}.prefill.bin')
                                  for name in ('weight', 'opart', 'mlpart', 'oat')})
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    assert all(r['differences'] == 0 for r in records), 'candidate differs from control'
    assert all(r['control_vs_capture_differences'] in (None, 0) for r in records), 'control differs from capture'


if __name__ == '__main__':
    main()
