import argparse
import ctypes
import hashlib
import json
import statistics
import struct
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--capture', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--selected', type=Path, required=True)
parser.add_argument('--object', type=Path, required=True)
args = parser.parse_args()
root, capture = args.root, args.capture
torch.set_num_threads(8)
torch.manual_seed(42)
torch.cuda.set_device(0)
assert 'gfx942' in torch.cuda.get_device_properties(0).gcnArchName
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def bind(path, name, types):
    fn = getattr(ctypes.CDLL(str(path)), name)
    fn.argtypes = types
    fn.restype = ctypes.c_int
    return fn


ptr = ctypes.c_void_p
ui = ctypes.c_uint
selection = json.loads(args.selected.read_text())
assert hashlib.sha256(args.object.read_bytes()).hexdigest() == selection['object_sha256']
hip = ctypes.CDLL('libamdhip64.so')
hip.hipModuleLoad.argtypes = [ctypes.POINTER(ptr), ctypes.c_char_p]
hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ptr), ptr, ctypes.c_char_p]
hip.hipModuleLaunchKernel.argtypes = [ptr] + [ui] * 7 + [ptr] * 3
module = ptr()
assert hip.hipModuleLoad(ctypes.byref(module), str(args.object).encode()) == 0
kernels = {}
for choice in selection['records']:
    fn = ptr()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), module, choice['name'].encode()) == 0
    kernels[choice['rows']] = (fn, choice)

fold = bind(capture / 'fold-candidate.so', 'plow_fold', [ptr] * 4 + [ui] * 3 + [ptr])
norm = bind(root / 'helpers.so', 'plow_normalize', [ptr] * 3 + [ui] * 3 + [ptr])
mfma = bind(root / 'helpers.so', 'plow_fold_mfma', [ptr] * 4 + [ui] * 3 + [ptr])
bf16x2 = bind(root / 'helpers.so', 'plow_fold_bf16x2', [ptr] * 4 + [ui] * 3 + [ptr])
convert = bind(root / 'helpers.so', 'plow_convert', [ptr] * 2 + [ui] * 2 + [ptr])


def read(name, dtype):
    return torch.frombuffer(bytearray((capture / f'{name}.prefill.bin').read_bytes()), dtype=dtype).clone()


weight_cpu = read('weight', torch.bfloat16).reshape(8, 512, 256)
weight = weight_cpu.cuda()
weight_float = weight.float()
partial_cpu = read('opart', torch.float32).reshape(8192, 8, 1, 512)
ml_cpu = read('mlpart', torch.float32).reshape(8192, 8, 1, 2)
expected = read('oat', torch.bfloat16)[:8191 * 8 * 256].reshape(8191, 8, 256)
flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device='cuda')
records = []


def timing(call, out, scratch):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    reference = out.clone()
    for _ in range(3):
        out.fill_(float('nan'))
        for tensor in scratch:
            tensor.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, reference), 'graph replay did not reproduce direct output'
    result = {'graph_poison_reuses': 3}
    for mode in ['warm', 'cold']:
        samples = []
        for _ in range(11):
            if mode == 'cold':
                flush.fill_(17)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        result[mode + '_us'] = statistics.median(samples)
        result[mode + '_samples_us'] = samples
    return result


for splits, rows in [(1, r) for r in [1, 7, 9, 128, 512, 2048, 4463, 4464, 8191]] + [(s, r) for s in [2, 7] for r in [128, 2048, 8191]]:
    if splits == 1:
        p_cpu = partial_cpu[:rows].clone()
        m_cpu = ml_cpu[:rows].clone()
    else:
        p_cpu = torch.randn(rows, 8, splits, 512)
        m_cpu = torch.randn(rows, 8, splits, 2)
        m_cpu[..., 1] = m_cpu[..., 1].abs() + 1
        m_cpu[::3, :, -1, 0] = float('-inf')
        m_cpu[::3, :, -1, 1] = 0
        p_cpu[::3, :, -1] = 0
        m_cpu[::17, :, :, 0] = float('-inf')
        m_cpu[::17, :, :, 1] = 0
        p_cpu[::17] = 0
    partial, ml = p_cpu.cuda(), m_cpu.cuda()
    normalized = torch.empty((8, rows, 512), dtype=torch.float32, device='cuda')
    product = torch.empty((8, rows, 256), dtype=torch.float32, device='cuda')
    # Independent full CPU FP64 softmax merge and matrix-product oracle.
    m = m_cpu.double()
    gm = m[..., 0].amax(dim=2, keepdim=True)
    ex = torch.where(torch.isfinite(m[..., 0]), torch.exp2(m[..., 0] - gm), 0)
    gl = (ex * m[..., 1]).sum(dim=2)
    latent = (p_cpu.double() * ex.unsqueeze(-1)).sum(dim=2)
    latent *= torch.where(gl > 0, gl.reciprocal(), 0).unsqueeze(-1)
    oracle = torch.bmm(latent.permute(1, 0, 2).contiguous(), weight_cpu.double()).permute(1, 0, 2).contiguous()
    outputs, calls, buffers = {}, {}, {}
    arms = ['plow_tb8', 'fp32_bmm', 'fused_mfma', 'fused_bf16x2']
    if rows in kernels:
        arms.append('direct_fp32')
    for arm in arms:
        guard = torch.full((rows * 8 * 256 + 512,), 0x6B6B, dtype=torch.int16, device='cuda')
        out = guard[256:-256].view(torch.bfloat16).reshape(rows, 8, 256)
        if arm == 'plow_tb8':
            def call(out=out):
                stream = torch.cuda.current_stream().cuda_stream
                assert fold(out.data_ptr(), partial.data_ptr(), ml.data_ptr(), weight.data_ptr(), rows, 8, splits, stream) == 0
        elif arm == 'fused_bf16x2':
            def call(out=out):
                stream = torch.cuda.current_stream().cuda_stream
                assert bf16x2(out.data_ptr(), partial.data_ptr(), ml.data_ptr(), weight.data_ptr(), rows, 8, splits, stream) == 0
        elif arm == 'fused_mfma':
            def call(out=out):
                stream = torch.cuda.current_stream().cuda_stream
                assert mfma(out.data_ptr(), partial.data_ptr(), ml.data_ptr(), weight.data_ptr(), rows, 8, splits, stream) == 0
        elif arm == 'direct_fp32':
            kernel, choice = kernels[rows]
            data = bytearray.fromhex(choice['args_hex'])
            struct.pack_into('<4Q', data, 32, product.data_ptr(), product.data_ptr(), weight_float.data_ptr(), normalized.data_ptr())
            inline = ctypes.create_string_buffer(bytes(data))
            size = ctypes.c_size_t(len(data))
            extra = (ptr * 5)(1, ctypes.addressof(inline), 2, ctypes.addressof(size), 3)
            def call(out=out):
                stream = torch.cuda.current_stream().cuda_stream
                assert norm(normalized.data_ptr(), partial.data_ptr(), ml.data_ptr(), rows, 8, splits, stream) == 0
                assert hip.hipModuleLaunchKernel(kernel, *choice['grid'], *choice['block'], 0, stream, None, extra) == 0
                assert convert(out.data_ptr(), product.data_ptr(), rows, 8, stream) == 0
        else:
            def call(out=out):
                stream = torch.cuda.current_stream().cuda_stream
                assert norm(normalized.data_ptr(), partial.data_ptr(), ml.data_ptr(), rows, 8, splits, stream) == 0
                torch.bmm(normalized, weight_float, out=product)
                assert convert(out.data_ptr(), product.data_ptr(), rows, 8, stream) == 0
        errors = []
        for reuse in range(3):
            out.fill_(float('nan'))
            normalized.fill_(float('nan'))
            product.fill_(float('nan'))
            call()
            torch.cuda.synchronize()
            assert bool((guard[:256] == 0x6B6B).all()) and bool((guard[-256:] == 0x6B6B).all())
            actual = out.cpu().double()
            assert bool(torch.isfinite(actual).all())
            rel = float(torch.linalg.vector_norm(actual - oracle) / torch.linalg.vector_norm(oracle))
            assert rel < 0.003, (arm, splits, rows, rel)
            errors.append(rel)
        capture_diff = None
        if splits == 1:
            capture_diff = int((out.cpu().view(torch.int16) != expected[:rows].view(torch.int16)).sum())
            if arm == 'plow_tb8':
                assert capture_diff == 0
        outputs[arm] = out.clone()
        buffers[arm] = guard
        calls[arm] = call
        record = dict(arm=arm, rows=rows, splits=splits, relative_l2=errors, poisoned_reuses=3, guards=True, capture_differences=capture_diff)
        if arm == 'direct_fp32':
            reference = torch.bmm(normalized, weight_float)
            assert torch.equal(reference, product), 'direct assembly differs from library FP32 output'
            record['exact_library_fp32'] = True
        record.update(timing(call, out, [normalized, product] if arm in ['fp32_bmm', 'direct_fp32'] else []))
        records.append(record)
        print(json.dumps(record), flush=True)
        args.out.with_suffix('.partial.json').write_text(json.dumps(records, indent=2) + '\n')
    for record in records[-len(arms):]:
        record['vs_plow_differences'] = int((outputs['plow_tb8'].view(torch.int16) != outputs[record['arm']].view(torch.int16)).sum())
    del outputs, calls, buffers, out, guard, partial, ml, normalized, product, oracle, latent, p_cpu, m_cpu


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = dict(records=records, selection=selection, weight_conversion='2 MiB BF16 -> 4 MiB FP32 once, excluded from timing; no per-call conversion',
              timing='fp32_bmm graph includes normalize + FP32 bmm + convert; other arms use one kernel; cold flush 512 MiB',
              oracle='Complete CPU FP64 merge and matrix product, relative L2 < .003; three poisoned reuses and output guards',
              sources={str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('gemm_kernels.hip'), root/'helpers.so', capture/'fold-candidate.so', args.selected, args.object]},
              captures={n: sha(capture/f'{n}.prefill.bin') for n in ['weight','opart','mlpart','oat']})
args.out.write_text(json.dumps(report, indent=2)+'\n')
