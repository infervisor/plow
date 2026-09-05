"""Opt-in primitive parity; reuses the installed vLLM packed GDN oracle."""
import argparse, ctypes as C, json
p=argparse.ArgumentParser()
p.add_argument("--run-gpu", action="store_true")
p.add_argument("--library", required=True)
p.add_argument("--gdn-library")
p.add_argument("--fp8-gemm-only", action="store_true", help="run cuBLASLt FP8 M1 vs vLLM CUTLASS")
p.add_argument("--fp8-only", action="store_true", help="report exact activation quantization parity; exit 1 on any mismatch")
p.add_argument("--output", help="FP8 comparison JSON, written even when parity fails")
args=p.parse_args()
if not args.run_gpu: p.error("requires --run-gpu and an exclusive GPU slot")
import torch
lib=C.CDLL(args.library)
if not args.fp8_gemm_only:
    lib.qwen_test.argtypes=[C.c_uint,C.POINTER(C.c_void_p),C.POINTER(C.c_int),C.POINTER(C.c_float),C.c_void_p]
    lib.qwen_test.restype=C.c_int
def run(op,ts,ints,floats=(0.,0.)):
    code=lib.qwen_test(op,(C.c_void_p*8)(*[t.data_ptr() if t is not None else 0 for t in ts]),(C.c_int*8)(*ints),(C.c_float*2)(*floats),torch.cuda.current_stream().cuda_stream)
    assert code==0,code
def rand(*shape):return torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*0.2
def check_fp8_gemm():
    from vllm import _custom_ops as ops
    from pathlib import Path
    import statistics
    shapes = [("a_or_b",48,5120),("qkv",10240,5120),("z",6144,5120),
        ("gdn_out",5120,6144),("q_full",12288,5120),("k_or_v",1024,5120),
        ("gate_or_up",17408,5120),("down",5120,17408),("lm_head",248320,5120),
        ("fused_ba",96,5120),("fused_qkvz",16384,5120),
        ("fused_qkv",14336,5120),("fused_gtup",34816,5120)]
    lib.plow_fp8_m1_create.argtypes=[C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p)]
    lib.plow_fp8_m1_create.restype=C.c_int
    lib.plow_fp8_m1_run.argtypes=[C.c_void_p]*5
    lib.plow_fp8_m1_run.restype=C.c_int
    lib.plow_fp8_m1_destroy.argtypes=[C.c_void_p]
    lib.plow_fp8_m1_destroy.restype=None
    torch.manual_seed(1729)
    records=[]
    # Flush is outside the timed region; every iteration rereads the same weight
    # after evicting more than H100 L2. Outputs and descriptors are preallocated.
    evict=torch.zeros(700*1024*1024//4,device="cuda",dtype=torch.int32)
    def timing(fn):
        for _ in range(5):fn()
        torch.cuda.synchronize()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        times=[]
        for _ in range(30):
            evict.add_(1)
            start.record();fn();end.record();end.synchronize()
            times.append(start.elapsed_time(end)*1000)
        return statistics.median(times)
    for name,N,K in shapes:
        record=dict(name=name,N=N,K=K,logical_M=1,lm_head_bf16_island=name=="lm_head")
        handle=C.c_void_p()
        try:
            x=rand(1,K)
            w=rand(N,K)
            ws=(w.float().abs().amax()/448).reshape(1)
            wq,_=ops.scaled_fp8_quant(w,scale=ws)
            aq,asc=ops.scaled_fp8_quant(x,use_per_token_if_dynamic=True)
            del w
            attempts=[]
            for physical_m in (1,16):
                rc=lib.plow_fp8_m1_create(N,K,physical_m,ws.data_ptr(),asc.data_ptr(),C.byref(handle))
                attempts.append(dict(physical_M=physical_m,status=rc))
                if rc==0:break
            record["descriptor_attempts"]=attempts
            if not handle.value:
                record.update(passed=False,error="No cuBLASLt FP8 algorithm for M1 or padded16")
                records.append(record)
                continue
            ap=torch.zeros(physical_m,K,device="cuda",dtype=torch.float8_e4m3fn)
            ap[:1].copy_(aq)
            out=torch.empty(physical_m,N,device="cuda",dtype=torch.bfloat16)
            ref=torch.empty(1,N,device="cuda",dtype=torch.bfloat16)
            wt=wq.t()
            def native():
                rc=lib.plow_fp8_m1_run(handle,wq.data_ptr(),ap.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
                if rc:raise RuntimeError(f"cublasLtMatmul status {rc}")
            def reference():
                torch.ops._C.cutlass_scaled_mm(ref,aq,wt,asc,ws,None)
            native();reference();torch.cuda.synchronize()
            a=out[:1].float();b=ref.float()
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))
            max_abs=float((a-b).abs().max());max_ref=float(b.abs().max())
            # Reuse the existing GEMM comparator gate; retain exact-byte evidence.
            passed=finite and rel<=.006 and max_abs<=.05+.02*max_ref
            record.update(physical_M=physical_m,finite=finite,rel_l2=rel,max_abs=max_abs,
                bit_exact=bool(torch.equal(out[:1],ref)),passed=passed,
                cublasLt_cold_us=timing(native),vllm_cutlass_cold_us=timing(reference))
        except Exception as exc:
            record.update(passed=False,error=repr(exc))
        finally:
            torch.cuda.synchronize()
            lib.plow_fp8_m1_destroy(handle)
        records.append(record)
    report=dict(mode="true FP8 E4M3 x E4M3, FP32 accumulation, BF16 output",
        weight_scale="one FP32 scalar per synthetic packed matrix",
        activation_scale="vLLM CUDA per-token; one scalar at logical M1",
        timing="median30 CUDA-event samples,5warmups;700MiB L2 eviction outside eachsample; no quantization/padding/setup allocation timed",
        note="lm_head shape is a synthetic kernel stress case; model lm_head remains BF16",
        passed=all(r["passed"] for r in records),cases=records)
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report["passed"]

if args.fp8_gemm_only:
    raise SystemExit(0 if check_fp8_gemm() else 1)

def check_fp8():
    from vllm import _custom_ops as ops
    import vllm
    from pathlib import Path
    records=[]
    torch.manual_seed(1729)
    for M in (1,4,16):
        for K in (3840,5120,17408):
            for case in ("random", "zeros", "subtiny", "near_floor", "extreme_finite"):
                x=torch.randn(M,K,device="cuda",dtype=torch.float32)
                if case=="random":
                    x*=torch.logspace(-1,1,M,device="cuda")[:,None]
                elif case=="zeros":x.zero_()
                elif case=="subtiny":x*=1e-30
                elif case=="near_floor":
                    x=x.sign()*torch.logspace(-12,-2,M,device="cuda")[:,None]
                else:
                    x=x.sign()*torch.finfo(torch.bfloat16).max
                x=x.bfloat16()
                assert torch.isfinite(x).all().item()
                out=torch.empty_like(x,dtype=torch.float8_e4m3fn)
                scale=torch.empty(M,device="cuda",dtype=torch.float32)
                run(32,[out,x,scale],[M,K])
                ref,refscale=ops.scaled_fp8_quant(x,use_per_token_if_dynamic=True)
                torch.cuda.synchronize()
                a=out.view(torch.uint8).cpu();b=ref.view(torch.uint8).cpu()
                sa=scale.cpu();sb=refscale.flatten().cpu()
                different=(a!=b)
                loc=different.nonzero()[:8]
                records.append(dict(case=case,M=M,K=K,
                    quantized_bytes_exact=bool(torch.equal(a,b)),
                    mismatched_bytes=int(different.sum()),
                    scale_bits_exact=bool(torch.equal(sa.view(torch.int32),sb.view(torch.int32))),
                    plow_scales=sa.tolist(),vllm_scales=sb.tolist(),
                    plow_scale_bits=sa.view(torch.int32).tolist(),
                    vllm_scale_bits=sb.view(torch.int32).tolist(),
                    source_amax=x.float().abs().amax(dim=1).cpu().tolist(),
                    finite_outputs=bool(torch.isfinite(out.float()).all() and torch.isfinite(ref.float()).all()),
                    first_byte_mismatches=[dict(row=int(r),col=int(c),plow=int(a[r,c]),vllm=int(b[r,c])) for r,c in loc]))
    report=dict(torch_version=torch.__version__,vllm_version=vllm.__version__,
        device=torch.cuda.get_device_name(),seed=1729,reference="CUDA scaled_fp8_quant(use_per_token_if_dynamic=True)",
        plow_scale_floor=1e-12,vllm_native_fallback_scale_floor=1/(448*512),
        note="CUDA scale behavior is measured here; the Python native fallback floor is not assumed to apply to CUDA. No tolerance or edge-case exemption.",
        exact=all(r["quantized_bytes_exact"] and r["scale_bits_exact"] and r["finite_outputs"] for r in records),cases=records)
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report["exact"]

if args.fp8_only:
    raise SystemExit(0 if check_fp8() else 1)

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as oracle
results=[]
def check(name,out,ref,rtol=.01,atol=1e-5):
    torch.cuda.synchronize()
    torch.testing.assert_close(out,ref,rtol=rtol,atol=atol)
    results.append(dict(name=name,max_abs=(out.float()-ref.float()).abs().max().item(),exact=torch.equal(out,ref)))
torch.manual_seed(1729)
for B in (1,4,16):
    active=torch.ones(B,device="cuda",dtype=torch.int32)
    if B>1:active[0]=0
    live=active.bool()
    initial=torch.randn(B,48,128,128,device="cuda")*.01
    state=initial.clone()
    reference=torch.cat([torch.zeros_like(state[:1]),state.clone()])
    indices=torch.arange(1,B+1,device="cuda",dtype=torch.int32)
    indices[~live]=0
    alog=rand(48); bias=rand(48)
    history=rand(B,10240,3); refhist=history.clone(); weight=rand(10240,4)
    for step in range(4):
        if step==2:state[-1].zero_();reference[-1].zero_()
        raw=rand(B,10240);mixed=torch.zeros_like(raw)
        run(136,[mixed,raw,weight,history,active],[10240,4,B])
        conv=(refhist.float()*weight[:,:3].float()).sum(-1)+raw.float()*weight[:,-1].float()
        expected=torch.nn.functional.silu(conv).bfloat16()
        check(f"conv B{B} step{step}",mixed[live],expected[live])
        refhist[live]=torch.cat([refhist[live,:,1:],raw[live,:,None]],-1)
        check("conv history",history,refhist,0,0)
        a=rand(B,48);b=rand(B,48)
        output=torch.zeros(B,1,48,128,device="cuda",dtype=torch.bfloat16);expected=torch.empty_like(output)
        run(137,[output,mixed,a,b,alog,bias,state,active],[16,48,128,128,B,0],[128**-.5,1e-6])
        oracle(mixed,a,b,alog.float(),bias,128**-.5,reference,expected,indices,True)
        check(f"GDN B{B} step{step}",output[live],expected[live])
        check("GDN state",state,reference[1:],1e-4,1e-6)
        if B>1:check("inactive state",state[:1],initial[:1],0,0)
    x=rand(B,48,128);z=rand(B,48,128);gamma=rand(128);out=torch.zeros_like(x)
    run(138,[out,x,z,gamma,active],[48,128,B],[1e-6])
    ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*gamma.float()*torch.nn.functional.silu(z.float())).bfloat16()
    check("gated norm",out[live],ref[live])
    packed=rand(B,24,512);q=torch.zeros(B,24,256,device="cuda",dtype=torch.bfloat16);gate=torch.zeros_like(q)
    run(139,[q,gate,packed,active],[24,256,B])
    check("Q split",q[live],packed[live,:,:256],0,0)
    check("gate split",gate[live],packed[live,:,256:],0,0)
    run(140,[q,q,gate,active],[6144,B])
    check("sigmoid gate",q[live],(packed[:,:,:256]*gate.sigmoid())[live],0,0)
    x=rand(B,5120);gamma=rand(5120);out=torch.zeros_like(x)
    run(141,[out,x,gamma,active],[5120,B],[1e-6,1.])
    ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*(gamma.float()+1)).bfloat16()
    check("zero centered norm",out[live],ref[live])
    x=rand(B,4,256);gamma=rand(256);pos=torch.arange(B,device="cuda",dtype=torch.int32)+1
    angles=torch.randn(32,32,device="cuda");cos=angles.cos();sin=angles.sin()
    norm=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*(gamma.float()+1)).bfloat16().float()
    ref=norm.clone();c=cos[pos.long(),None,:].bfloat16().float();s=sin[pos.long(),None,:].bfloat16().float()
    ref[:,:,:32]=norm[:,:,:32]*c-norm[:,:,32:64]*s
    ref[:,:,32:64]=norm[:,:,32:64]*c+norm[:,:,:32]*s
    for ctx in (0,32):
        out=torch.zeros((B,4,ctx,256) if ctx else x.shape,device="cuda",dtype=torch.bfloat16)
        run(142,[out,x,gamma,cos,sin,pos,active],[4,256,64,B,ctx,1],[1e-6,1.])
        actual=out[torch.arange(B,device="cuda"),:,pos.long(),:] if ctx else out
        check("partial RoPE cache"+str(ctx),actual[live],ref.bfloat16()[live])
        if ctx:
            out.zero_()
            run(142,[out,x,None,None,None,pos,active],[4,256,0,B,ctx,0])
            actual=out[torch.arange(B,device="cuda"),:,pos.long(),:]
            check("V cache",actual[live],x[live],0,0)

for T in (1, 2, 128):
    channels=10240
    raw=rand(T,channels);weight=rand(channels,4);history=rand(1,channels,3)
    refhist=history.clone();out=torch.empty_like(raw);expected=torch.empty_like(raw)
    active=torch.ones(1,device="cuda",dtype=torch.int32)
    run(143,[out,raw,weight,history],[channels,4,T])
    for t in range(T):run(136,[expected[t],raw[t],weight,refhist,active],[channels,4,1])
    check("chunk conv T"+str(T),out,expected,0,0)
    check("chunk history T"+str(T),history,refhist,0,0)
    q=torch.empty(T,16,128,device="cuda",dtype=torch.bfloat16);k=torch.empty_like(q)
    v=torch.empty(T,48,128,device="cuda",dtype=torch.bfloat16)
    run(144,[q,k,v,out],[16,48,128,128,T],[1e-6])
    for name,tensor,start in (("Q",q,0),("K",k,2048)):
        source=out[:,start:start+2048].reshape(T,16,128).float()
        ref=(source/torch.sqrt(source.square().sum(-1,keepdim=True)+1e-6)).bfloat16()
        check("prepared "+name,tensor,ref)
    check("prepared V",v,out[:,4096:].reshape(T,48,128),0,0)
    a=rand(T,48);b=rand(T,48);alog=rand(48);bias=rand(48)
    alpha=torch.empty(T,48,device="cuda");beta=torch.empty_like(alpha)
    run(145,[alpha,beta,a,b,alog,bias],[48,T])
    check("prepared alpha",alpha,torch.exp(-torch.exp(alog.float())*torch.nn.functional.softplus(a.float()+bias.float())),1e-5,1e-7)
    check("prepared beta",beta,torch.sigmoid(b.float()).bfloat16().float(),0,0)
T=128
x=rand(T,4,256);gamma=rand(256);active=torch.ones(T,device="cuda",dtype=torch.int32)
pos=torch.arange(T,device="cuda",dtype=torch.int32)+128
angles=torch.randn(512,32,device="cuda");cos=angles.cos();sin=angles.sin()
out=torch.zeros(1,4,512,256,device="cuda",dtype=torch.bfloat16)
run(142,[out,x,gamma,cos,sin,pos,active],[4,256,64,T,512,1,1],[1e-6,1.])
norm=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*(gamma.float()+1)).bfloat16().float()
ref=norm.clone();c=cos[pos.long(),None,:].bfloat16().float();s=sin[pos.long(),None,:].bfloat16().float()
ref[:,:,:32]=norm[:,:,:32]*c-norm[:,:,32:64]*s
ref[:,:,32:64]=norm[:,:,32:64]*c+norm[:,:,:32]*s
actual=out[0,:,pos.long(),:].transpose(0,1)
check("prefill single slot KV rows",actual,ref.bfloat16())
assert not torch.count_nonzero(out[:,:,:128]).item()
assert not torch.count_nonzero(out[:,:,256:]).item()

if args.gdn_library:
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    gdn=C.CDLL(args.gdn_library)
    gdn.plow_gdn_create.argtypes=[C.c_int,C.POINTER(C.c_void_p)]
    gdn.plow_gdn_create.restype=C.c_int
    gdn.plow_gdn_destroy.argtypes=[C.c_void_p]
    gdn.plow_gdn_destroy.restype=C.c_int
    gdn.plow_gdn_run.argtypes=[C.c_void_p]*11+[C.c_int,C.c_void_p]
    gdn.plow_gdn_run.restype=C.c_int
    handle=C.c_void_p()
    assert gdn.plow_gdn_create(0,C.byref(handle))==0
    second=C.c_void_p()
    assert gdn.plow_gdn_create(0,C.byref(second))==-1001 and not second.value
    try:
        initial=torch.randn(1,48,128,128,device="cuda")*.01
        final=torch.empty_like(initial)
        output=torch.empty_like(v)
        maps=torch.empty(16896,device="cuda",dtype=torch.uint8)
        offsets=torch.tensor([0,128],device="cuda",dtype=torch.int64)
        for continuation in range(2):
            refout,refstate=chunk_gated_delta_rule(q,k,v,g=alpha,beta=beta,
                initial_state=initial.clone(),output_final_state=True,cu_seqlens=offsets,use_cp=False)
            rc=gdn.plow_gdn_run(handle,*[t.data_ptr() for t in
                (q,k,v,output,alpha,beta,final,initial,maps,offsets)],128,torch.cuda.current_stream().cuda_stream)
            assert rc==0,rc
            check("managed GDN output",output,refout,0,0)
            check("managed GDN state",final,refstate,0,0)
            check("managed GDN persistent copy",initial,final,0,0)
    finally:
        torch.cuda.synchronize()
        assert gdn.plow_gdn_destroy(handle)==0
    handle=C.c_void_p()
    assert gdn.plow_gdn_create(0,C.byref(handle))==0
    assert gdn.plow_gdn_destroy(handle)==0

print(json.dumps(results,indent=2))
print("PASS",len(results),"primitive comparisons")
