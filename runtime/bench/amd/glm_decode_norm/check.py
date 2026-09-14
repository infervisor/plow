import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--library', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
lib = ctypes.CDLL(str(args.library.resolve()))
fn = lib.plow_norm_rows
fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint] * 4 + [ctypes.c_void_p]
fn.restype = ctypes.c_int
torch.manual_seed(7281)
stream = torch.cuda.current_stream().cuda_stream
guard = 256
records = []
for rows in (1, 2, 3, 4, 7, 8, 16, 20):
    for feat in (2048, 6144):
        a = torch.randn(rows, feat, device='cuda', dtype=torch.bfloat16)
        b = torch.randn_like(a) * 2
        weight = torch.randn(feat, device='cuda', dtype=torch.bfloat16)
        for add, alias, weighted in ((0, False, True), (0, False, False),
                                      (1, False, True), (1, True, True), (1, True, False)):
            outputs = []
            for groups in (1, rows):
                n = rows * feat
                out = torch.full((n + guard * 2,), 91, device='cuda', dtype=torch.bfloat16)
                resid = torch.full_like(out, 93)
                if alias:
                    resid[guard:-guard].copy_(a.flatten())
                src = resid[guard:-guard] if alias else a
                gamma = weight.data_ptr() if weighted else 0
                status = fn(out[guard:-guard].data_ptr(), resid[guard:-guard].data_ptr(),
                            src.data_ptr(), b.data_ptr(), gamma, rows, feat, groups, add, stream)
                assert status == 0, status
                torch.cuda.synchronize()
                assert bool((out[:guard] == 91).all() and (out[-guard:] == 91).all())
                assert bool((resid[:guard] == 93).all() and (resid[-guard:] == 93).all())
                if not add:
                    assert bool((resid == 93).all())
                outputs.append((out[guard:-guard].clone(), resid[guard:-guard].clone()))
            for serial, parallel in zip(outputs[0], outputs[1]):
                assert torch.equal(serial.view(torch.int16), parallel.view(torch.int16)), (rows, feat, add, alias, weighted)
            value = a.float() + b.float() if add else a.float()
            reference = value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + 1e-5)
            if weighted:
                reference *= weight.float()
            actual = outputs[1][0].view(rows, feat).float()
            rel_l2 = ((actual - reference).square().sum() / reference.square().sum()).sqrt().item()
            assert rel_l2 < 0.005, (rows, feat, rel_l2)
            if add:
                assert torch.equal(outputs[1][1].view(rows, feat), value.bfloat16())
            records.append(dict(rows=rows, feat=feat, add=bool(add), residual_alias=alias,
                                weighted=weighted, exact_serial_parallel=True, relative_l2=rel_l2))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result = dict(cases=records, passed=len(records), gpu=torch.cuda.get_device_name(0),
              library_sha256=sha(args.library), script_sha256=sha(Path(__file__)),
              scope='Existing RMSNorm/AddNorm bodies: one workgroup versus one per row. Exact BF16 outputs, in-place residual, optional gamma, ragged row counts and 512-byte guards. Numerical check only; no timing.')
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(passed=len(records), max_relative_l2=max(r['relative_l2'] for r in records))))
