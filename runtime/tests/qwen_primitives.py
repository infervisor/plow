"""Opt-in primitive parity; reuses the installed vLLM packed GDN oracle."""
import argparse, ctypes as C, json
p=argparse.ArgumentParser()
p.add_argument("--run-gpu", action="store_true")
p.add_argument("--library", required=True)
p.add_argument("--gdn-library")
p.add_argument("--gdn-timing-only", action="store_true", help="native recurrent step vs reachable vLLM FLA packed decode, batches1/4/16")
p.add_argument("--attention-only", action="store_true", help="native attention vs installed serving FA3/FA4; excludes projections/cache writes")
p.add_argument("--attention-gf", type=int, choices=(0,2,4,6,8,16), default=0, help="benchmark-only native decode grouping; 0 preserves fixed controls")
p.add_argument("--attention-native-splits", default="baseline", help="baseline, fill, fill2, fill4, or positive integer; fill targets132 work items")
p.add_argument("--attention-kind", choices=("all","full","sliding"), default="all")
p.add_argument("--attention-contexts", default="1024,4096,8192,32768")
p.add_argument("--attention-batches", default="1,4,16", help="active decode batch; distinct from serving concurrency")
p.add_argument("--attention-model", choices=("all","gemma12","gemma31","qwen27"), default="all")
p.add_argument("--attention-phases", choices=("both","decode","prefill"), default="both")
p.add_argument("--attention-prefill-rows", default="1024,8192", help="trailing query chunks, capped at context; 0 means entire prompt")
p.add_argument("--attention-page-size", type=int, default=0)
p.add_argument("--attention-fa-splits", type=int, default=-1, help="-1 matches serving: FA3 graph decode32, otherwise0; nonnegative override recorded")
p.add_argument("--attention-iters", type=int, default=20)
p.add_argument("--attention-flush-mib", type=int, default=256)
p.add_argument("--attention-reverse", action="store_true")
p.add_argument("--fp8-prefill-only", action="store_true", help="run native FP8 TMA prefill vs installed CUTLASS")
p.add_argument("--fp8-prefill-rows", type=int, choices=(128,1024,4096,8192), default=128, help="runtime M for the native TMA prefill comparator")
p.add_argument("--fp8-rows", type=int, choices=(1,4,16,128), help="row count for the separate exact quantization gate")
p.add_argument("--fp8-tma", action="store_true", help="encode a 64-row FP8 weight map before timing")
p.add_argument("--fp8-interpreter-only", action="store_true", help="run interpreter FP8 M1 body vs vLLM CUTLASS")
p.add_argument("--fp8-gemm-only", action="store_true", help="run cuBLASLt FP8 M1 vs vLLM CUTLASS")
p.add_argument("--fp8-only", action="store_true", help="report exact activation quantization parity; exit 1 on any mismatch")
p.add_argument("--gemma-nrn-only", action="store_true", help="compare fused prefill NRN against native NormResidual plus RMSNorm")
p.add_argument("--output", help="FP8 comparison JSON, written even when parity fails")
args=p.parse_args()
if args.fp8_prefill_only:
    if args.fp8_gemm_only:p.error("prefill comparison uses the native interpreter")
    args.fp8_interpreter_only=True
if not args.run_gpu: p.error("requires --run-gpu and an exclusive GPU slot")
import torch
lib=C.CDLL(args.library)
if not args.fp8_gemm_only and not args.attention_only:
    lib.qwen_test.argtypes=[C.c_uint,C.POINTER(C.c_void_p),C.POINTER(C.c_int),C.POINTER(C.c_float),C.c_void_p]
    lib.qwen_test.restype=C.c_int
def run(op,ts,ints,floats=(0.,0.)):
    code=lib.qwen_test(op,(C.c_void_p*8)(*[t.data_ptr() if t is not None else 0 for t in ts]),(C.c_int*8)(*ints),(C.c_float*2)(*floats),torch.cuda.current_stream().cuda_stream)
    assert code==0,code
def rand(*shape):return torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*0.2

def check_attention():
    import hashlib, importlib.metadata, inspect, statistics
    from pathlib import Path
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as flash, get_scheduler_metadata
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
    shapes=[("gemma12","sliding",256,16,8,1024,1.,4,17),
            ("gemma12","full",512,16,1,0,1.,4,33),
            ("gemma31","sliding",256,32,16,1024,1.,4,17),
            ("gemma31","full",512,32,4,0,1.,4,33),
            ("qwen27","full",256,24,4,0,256**-.5,3,11)]
    contexts=[int(x) for x in args.attention_contexts.split(",")]
    batches=[int(x) for x in args.attention_batches.split(",")]
    chunks=[int(x) for x in args.attention_prefill_rows.split(",")]
    if (min(contexts+batches)<1 or min(chunks)<0 or (args.attention_page_size!=0 and args.attention_page_size<16) or
        args.attention_page_size%16 or args.attention_iters<3 or args.attention_fa_splits < -1 or
        args.attention_flush_mib<128):
        p.error("invalid attention geometry; use >=3 iterations and >=128 MiB eviction")
    if any(b not in (1,4,16) for b in batches):p.error("attention decode batches must be 1,4,16")
    lib.plow_attention.argtypes=[C.POINTER(C.c_void_p),C.POINTER(C.c_int),C.c_float,C.c_void_p]
    lib.plow_attention.restype=C.c_int
    if args.attention_gf:
        if args.attention_phases!="decode":p.error("--attention-gf requires --attention-phases decode")
        lib.plow_attention_gf.argtypes=lib.plow_attention.argtypes+[C.c_uint]
        lib.plow_attention_gf.restype=C.c_int
    if args.attention_native_splits not in ("baseline","fill","fill2","fill4"):
        try:valid_splits=int(args.attention_native_splits)>0
        except ValueError:valid_splits=False
        if not valid_splits:p.error("invalid native split policy")
    lib.plow_attention_maps.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int]
    lib.plow_attention_maps.restype=C.c_int
    source=Path(inspect.getsourcefile(flash))
    report=dict(boundary="standalone native operation bodies vs vLLM library kernels; not interpreter-role or transformer-block timing; native decode includes merge; excludes QKV projection, norm/RoPE, cache write/packing, output projection/gate",
        scheduling="FA3 scheduler metadata constructed untimed; both sides warmed and CUDA-graph captured symmetrically",
        vllm=importlib.metadata.version("vllm"),torch=torch.__version__,
        library=str(Path(args.library).resolve()),library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        reference_source=str(source),reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(),sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        arguments=vars(args),rows=[])
    if report["sm_count"]!=132 or torch.cuda.get_device_capability()!=(9,0):
        raise RuntimeError("frozen native wrapper requires the 132-SM H100")
    torch.manual_seed(1729)
    flush=torch.empty(args.attention_flush_mib*1024*1024,device="cuda",dtype=torch.uint8)
    def save():
        if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    def guarded(shape,dtype):
        n=1
        for dim in shape:n*=dim
        buf=torch.full((n+256,),37.,device="cuda",dtype=dtype)
        return buf[128:-128].reshape(shape),buf
    def capture(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        return graph
    def timing(native,reference):
        graphs={"native":capture(native),"vllm":capture(reference)}
        results={}
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for cold in (False,True):
            times={name:[] for name in graphs}
            for rep in range(args.attention_iters):
                order=["native","vllm"]
                if bool(rep%2)^args.attention_reverse:order.reverse()
                for name in order:
                    if cold:flush.zero_()  # Same stream, before timing; no subtraction.
                    start.record();graphs[name].replay();end.record();end.synchronize()
                    times[name].append(start.elapsed_time(end)*1000)
            for name,values in times.items():
                values.sort()
                results[("evicted_" if cold else "hot_")+name+"_us"]=statistics.median(values)
                results[("evicted_" if cold else "hot_")+name+"_p95_us"]=values[min(len(values)-1,int(.95*len(values)))]
        return results
    for model,kind,hd,nh,nkv,window,scale,version,decode_splits in shapes:
        if args.attention_model not in ("all",model):continue
        if args.attention_kind not in ("all",kind):continue
        gf=args.attention_gf or (4 if hd==512 else 2)
        if (nh//nkv)%gf:p.error(f"GF{gf} does not divide {model} {kind} GQA{nh//nkv}")
        page_size=args.attention_page_size or (784 if model=="qwen27" else 16)
        for ctx in contexts:
            cases=[]
            if args.attention_phases!="prefill":cases.extend(("decode",b,1) for b in batches)
            if args.attention_phases!="decode":cases.extend(("prefill",1,q) for q in sorted({min(ctx,x) if x else ctx for x in chunks}))
            for phase,batch,rows in cases:
                fa_splits=args.attention_fa_splits if args.attention_fa_splits>=0 else (32 if version==3 and phase=="decode" else 0)
                native_splits=decode_splits
                if args.attention_native_splits.startswith("fill"):
                    waves=int(args.attention_native_splits[4:] or "1")
                    groups=batch*(nh//gf)
                    native_splits=(132*waves+groups-1)//groups
                elif args.attention_native_splits!="baseline":native_splits=int(args.attention_native_splits)
                row=dict(model=model,kind=kind,phase=phase,batch=batch,query_rows=rows,context=ctx,
                    head_dim=hd,q_heads=nh,kv_heads=nkv,gqa=nh//nkv,window=window,scale=scale,
                    causal=True,query_position=ctx-rows,dtype="bfloat16",fa_version=version,
                    reference="vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func",
                    reference_layout="paged [pages,KVH,page,2HD] transposed/split views",
                    native_layout="separate contiguous [B,KVH,context,HD]",page_size=page_size,
                    native_grid=132,native_threads=256,native_gf=gf,
                    native_decode_registers={(256,2):72,(256,6):113,(512,4):82,(512,8):138,(512,16):234}.get((hd,gf)),
                    native_decode_smem_bytes=4*(gf*256+2*max(8,gf)+gf*hd//2+2048),
                    native_prefill_smem_bytes=201728 if hd==512 else 103424,
                    native_prefill_registers=194 if hd==512 else 124,
                    native_policy=args.attention_native_splits,
                    native_work_items=batch*(nh//gf)*native_splits if phase=="decode" else None,
                    native_splits=native_splits if phase=="decode" else 1,
                    native_prefill_tile=[64,32],native_prefill_tma=phase=="prefill",
                    fa_num_splits=fa_splits,scheduler_metadata="FA3 AOT" if version==3 else None,
                    precision_gate="relative L2 <= 0.003; finite outputs; output/partial canaries",
                    input="seed1729 random BF16; Gemma Q/K unit RMS, not checkpoint capture")
                report["rows"].append(row)
                try:
                    q=rand(batch*rows,nh,hd)
                    k=rand(batch,nkv,ctx,hd);v=rand(batch,nkv,ctx,hd)
                    if model.startswith("gemma"):
                        q=(q.float()*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
                        k=(k.float()*torch.rsqrt(k.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
                    pages=(ctx+page_size-1)//page_size
                    cache=torch.zeros(batch*pages,nkv,page_size,2*hd,device="cuda",dtype=torch.bfloat16)
                    kc,vc=cache.transpose(1,2).split(hd,dim=-1)
                    for b in range(batch):
                        # Each request owns disjoint pages. Copy/layout work is untimed.
                        packed=torch.zeros(pages*page_size,nkv,2*hd,device="cuda",dtype=torch.bfloat16)
                        packed[:ctx,:,:hd]=k[b].transpose(0,1);packed[:ctx,:,hd:]=v[b].transpose(0,1)
                        cache[b*pages:(b+1)*pages].copy_(packed.reshape(pages,page_size,nkv,2*hd).transpose(1,2))
                    kc=canonicalize_singleton_dim_strides(kc)
                    vc=canonicalize_singleton_dim_strides(vc)
                    table=torch.arange(batch*pages,device="cuda",dtype=torch.int32).reshape(batch,pages)
                    lengths=torch.full((batch,),ctx,device="cuda",dtype=torch.int32)
                    cuq=torch.arange(batch+1,device="cuda",dtype=torch.int32)*rows
                    scheduler=None
                    if version==3:
                        scheduler=get_scheduler_metadata(batch_size=batch,max_seqlen_q=rows,
                            max_seqlen_k=ctx,num_heads_q=nh,num_heads_kv=nkv,headdim=hd,
                            cache_seqlens=lengths,qkv_dtype=torch.bfloat16,cu_seqlens_q=cuq,
                            page_size=page_size,causal=True,window_size=(-1,-1),num_splits=fa_splits)
                    out,ob=guarded(q.shape,torch.bfloat16);ref,rb=guarded(q.shape,torch.bfloat16)
                    ns=row["native_splits"]
                    op,pb=guarded((batch,nh,ns,hd) if phase=="decode" else (1,),torch.float32)
                    ml,mb=guarded((batch,nh,ns,2) if phase=="decode" else (1,),torch.float32)
                    maps=torch.empty(256,device="cuda",dtype=torch.uint8)
                    if phase=="prefill":
                        rc=lib.plow_attention_maps(maps.data_ptr(),k.data_ptr(),v.data_ptr(),hd,ctx,nkv)
                        if rc:raise RuntimeError(f"tensor map encoding failed: {rc}")
                    tensors=(C.c_void_p*8)(*[t.data_ptr() for t in (q,k,v,out,op,ml,lengths,maps)])
                    ints=(C.c_int*12)(phase=="prefill",hd,batch,rows,ctx,nh,nkv,window,ns,ctx-rows,-1,ctx)
                    def native():
                        if args.attention_gf:
                            rc=lib.plow_attention_gf(tensors,ints,scale,torch.cuda.current_stream().cuda_stream,gf)
                        else:
                            rc=lib.plow_attention(tensors,ints,scale,torch.cuda.current_stream().cuda_stream)
                        if rc:raise RuntimeError(f"native attention launch failed: {rc}")
                    def reference():
                        return flash(q,kc,vc,rows,cuq,ctx,seqused_k=lengths,out=ref,
                            softmax_scale=scale,causal=True,window_size=(window-1,0) if window else (-1,-1),
                            block_table=table,fa_version=version,num_splits=fa_splits,scheduler_metadata=scheduler)
                    native();reference();torch.cuda.synchronize()
                    grouping_pass=True
                    if args.attention_gf:
                        control,cb=guarded(q.shape,torch.bfloat16)
                        control_tensors=(C.c_void_p*8)(*tensors)
                        control_tensors[3]=control.data_ptr()
                        rc=lib.plow_attention(control_tensors,ints,scale,torch.cuda.current_stream().cuda_stream)
                        if rc:raise RuntimeError(f"native grouping control failed: {rc}")
                        torch.cuda.synchronize()
                        relative=((out.float()-control.float()).norm()/control.float().norm().clamp_min(1e-30)).item()
                        grouping_pass=bool(torch.isfinite(control).all() and (cb[:128]==37).all() and (cb[-128:]==37).all()) and relative<=.003
                        row.update(native_grouping_control_gf=4 if hd==512 else 2,
                            native_grouping_control_exact=torch.equal(out,control),native_grouping_control_relative_l2=relative,
                            native_grouping_control_passed=grouping_pass)
                    delta=out.float()-ref.float()
                    row.update(relative_l2=(delta.norm()/ref.float().norm().clamp_min(1e-30)).item(),
                        max_abs=delta.abs().max().item(),exact=torch.equal(out,ref))
                    finite=bool(torch.isfinite(out).all() and torch.isfinite(ref).all())
                    canaries=all(bool((buf[:128]==37).all() and (buf[-128:]==37).all()) for buf in (ob,rb,pb,mb))
                    row.update(finite=finite,canaries=canaries,passed=finite and canaries and grouping_pass and row["relative_l2"]<=.003)
                    if not row["passed"]:raise AssertionError("attention numerical/canary gate failed")
                    row.update(timing(native,reference))
                    print(json.dumps(row),flush=True);save()
                except Exception as exc:
                    row.update(passed=False,error=f"{type(exc).__name__}: {exc}");save();raise
    if not report["rows"]:raise RuntimeError("no attention operator cells selected")
    save()
    print("PASS",len(report["rows"]),"matched attention operator cells")
    return True
if args.attention_only:
    raise SystemExit(0 if check_attention() else 1)

def check_gemma_nrn():
    from pathlib import Path
    import statistics
    torch.manual_seed(1729)
    records=[]
    def timing(fn):
        for _ in range(5):fn()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(30):fn()
        for _ in range(5):graph.replay()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        times=[]
        for _ in range(3):
            start.record()
            graph.replay()
            end.record();end.synchronize()
            times.append(start.elapsed_time(end)*1000/30)
        return statistics.median(times)
    for feat in (3840,5376):
        for rows in (32,128,1024,4096):
            for scale in (1.,0.052978515625):
                a,b=rand(rows,feat),rand(rows,feat)
                gb,gn=rand(feat)+1,rand(feat)+1
                for weighted in ((True,False) if rows==32 else (True,)):
                    wb,wn=(gb,gn) if weighted else (None,None)
                    residual,out=torch.empty_like(a),torch.empty_like(a)
                    fused_residual,fused_out=torch.empty_like(a),torch.empty_like(a)
                    def split():
                        run(16,[residual,a,b,wb],[rows,feat],[1e-6,scale])
                        run(1,[out,residual,wn],[rows,feat],[1e-6,0.])
                    def fused():
                        run(23,[fused_out,fused_residual,a,b,wb,wn],[rows,feat],[1e-6,scale])
                    split();fused();torch.cuda.synchronize()
                    exact=bool(torch.equal(residual,fused_residual) and torch.equal(out,fused_out))
                    finite=bool(torch.isfinite(out).all() and torch.isfinite(fused_out).all())
                    aliased=a.clone();aliased_out=torch.empty_like(a)
                    run(23,[aliased_out,aliased,aliased,b,wb,wn],[rows,feat],[1e-6,scale])
                    torch.cuda.synchronize()
                    alias_exact=bool(torch.equal(residual,aliased) and torch.equal(out,aliased_out))
                    record=dict(rows=rows,feat=feat,scale=scale,weighted=weighted,
                        bit_exact=exact,alias_bit_exact=alias_exact,finite=finite,
                        residual_mismatches=int((residual!=fused_residual).sum()),
                        output_mismatches=int((out!=fused_out).sum()),
                        split_us=timing(split),fused_us=timing(fused),passed=exact and alias_exact and finite)
                    records.append(record)
    report=dict(candidate="native fused NormResidualNorm",reference="native NormResidual + RMSNorm",
        timing="median3 CUDA-event graph replays of30 calls,5graph warmups; capture, allocation and correctness outside timing",
        passed=all(r["passed"] for r in records),cases=records)
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report["passed"]

if args.gemma_nrn_only:
    raise SystemExit(0 if check_gemma_nrn() else 1)

def check_fp8_gemm():
    from vllm import _custom_ops as ops
    from pathlib import Path
    import statistics
    shapes = [("a_or_b",48,5120),("qkv",10240,5120),("z",6144,5120),
        ("gdn_out",5120,6144),("q_full",12288,5120),("k_or_v",1024,5120),
        ("gate_or_up",17408,5120),("down",5120,17408),("lm_head",248320,5120),
        ("fused_ba",96,5120),("fused_qkvz",16384,5120),
        ("fused_qkv",14336,5120),("fused_gtup",34816,5120),
        ("k_tail_128",1024,640),("k_tail_16",1024,528)]
    logical_m=args.fp8_prefill_rows if args.fp8_prefill_only else 1
    if args.fp8_prefill_only:
        shapes=[shape for shape in shapes if shape[0] not in
            ("lm_head","fused_ba","fused_qkvz","fused_qkv","fused_gtup")]
    if not args.fp8_interpreter_only:
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
        record=dict(name=name,N=N,K=K,logical_M=logical_m,lm_head_bf16_island=name=="lm_head")
        handle=C.c_void_p()
        try:
            x=rand(logical_m,K)
            w=rand(N,K)
            ws=(w.float().abs().amax()/448).reshape(1)
            wq,_=ops.scaled_fp8_quant(w,scale=ws)
            aq,asc=ops.scaled_fp8_quant(x,use_per_token_if_dynamic=True)
            del w
            if args.fp8_interpreter_only:
                physical_m=logical_m
                record["descriptor_attempts"]=[]
                checked=torch.empty_like(aq)
                checked_scale=torch.empty_like(asc)
                run(32,[checked,x,checked_scale],[logical_m,K])
                torch.cuda.synchronize()
                record["activation_bytes_scales_exact"]=bool(torch.equal(checked.view(torch.uint8),aq.view(torch.uint8)) and torch.equal(checked_scale.view(torch.int32),asc.view(torch.int32)))
                record["quantized_bytes_exact"]=bool(torch.equal(checked.view(torch.uint8),aq.view(torch.uint8)))
                record["scale_bits_exact"]=bool(torch.equal(checked_scale.view(torch.int32),asc.view(torch.int32)))
                record["quantized_byte_mismatches"]=int((checked.view(torch.uint8)!=aq.view(torch.uint8)).sum())
                record["scale_bit_mismatches"]=int((checked_scale.view(torch.int32)!=asc.view(torch.int32)).sum())
                if not args.fp8_prefill_only:aq,asc=checked,checked_scale
            else:
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
            ap[:logical_m].copy_(aq)
            guarded_out=None
            if args.fp8_prefill_only:
                guarded_out=torch.full((physical_m*N+16,),-123.,device="cuda",dtype=torch.bfloat16)
                out=guarded_out[8:-8].view(physical_m,N)
            else:out=torch.empty(physical_m,N,device="cuda",dtype=torch.bfloat16)
            ref=torch.empty(logical_m,N,device="cuda",dtype=torch.bfloat16)
            wt=wq.t()
            ws_vector=ws.expand(N).contiguous()
            if args.fp8_prefill_only:
                ws_vector=ws_vector*torch.linspace(.5,1.5,N,device="cuda",dtype=torch.float32)
            ref_ws=ws_vector if args.fp8_prefill_only else ws
            weight_map=activation_map=None
            if args.fp8_tma or args.fp8_prefill_only:
                assert args.fp8_interpreter_only
                driver=C.CDLL("libcuda.so.1")
                encode=driver.cuTensorMapEncodeTiled
                encode.argtypes=[C.c_void_p,C.c_int,C.c_uint,C.c_void_p,
                    C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),
                    C.POINTER(C.c_uint),C.POINTER(C.c_uint),C.c_int,C.c_int,C.c_int,C.c_int]
                encode.restype=C.c_int
                def tensor_map(tensor,rows,box_rows):
                    storage=C.create_string_buffer(255)
                    aligned=(C.addressof(storage)+127)&~127
                    rc=encode(aligned,0,2,tensor.data_ptr(),(C.c_uint64*2)(K,rows),
                        (C.c_uint64*1)(K),(C.c_uint*2)(128,box_rows),(C.c_uint*2)(1,1),0,3,2,0)
                    record["descriptor_attempts"].append(dict(rows=rows,k=K,box_rows=box_rows,status=rc))
                    assert rc==0, f"cuTensorMapEncodeTiled status {rc}"
                    result=torch.tensor(list(C.string_at(aligned,128)),device="cuda",dtype=torch.uint8)
                    assert result.data_ptr()%128==0
                    return result
                weight_map=tensor_map(wq,N,128 if args.fp8_prefill_only else 64)
                if args.fp8_prefill_only:activation_map=tensor_map(ap,logical_m,128)

            def native():
                if args.fp8_interpreter_only:
                    run(33,[out,ap,wq,asc,ws_vector,None,activation_map,weight_map],[logical_m,N,K])
                else:
                    rc=lib.plow_fp8_m1_run(handle,wq.data_ptr(),ap.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
                    if rc:raise RuntimeError(f"cublasLtMatmul status {rc}")
            def reference():
                torch.ops._C.cutlass_scaled_mm(ref,aq,wt,asc,ref_ws,None)
            native();reference();torch.cuda.synchronize()
            a=out[:logical_m].float();b=ref.float()
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))
            max_abs=float((a-b).abs().max());max_ref=float(b.abs().max())
            # Reuse the existing GEMM comparator gate; retain exact-byte evidence.
            record["output_canaries_pass"]=guarded_out is None or bool((guarded_out[:8]==-123).all() and (guarded_out[-8:]==-123).all())
            passed=record["output_canaries_pass"] and finite and rel<=.006 and max_abs<=.05+.02*max_ref and record.get("activation_bytes_scales_exact",True)
            record.update(physical_M=physical_m,finite=finite,rel_l2=rel,max_abs=max_abs,
                bit_exact=bool(torch.equal(out[:logical_m],ref)),passed=passed,
                candidate_cold_us=timing(native),vllm_cutlass_cold_us=timing(reference))
        except Exception as exc:
            record.update(passed=False,error=repr(exc))
        finally:
            torch.cuda.synchronize()
            if not args.fp8_interpreter_only:lib.plow_fp8_m1_destroy(handle)
        records.append(record)
    report=dict(backend=f"Plow native uniform TMA M{logical_m}" if args.fp8_prefill_only else "Plow interpreter d_gemm_w8a8 M1" if args.fp8_interpreter_only else "cuBLASLt comparison",
        mode="true FP8 E4M3 x E4M3, FP32 accumulation, BF16 output",
        weight_scale="heterogeneous FP32 per-N scales" if args.fp8_prefill_only else "one FP32 scalar per synthetic packed matrix",
        activation_scale="vLLM CUDA per-token; quant byte/scale parity checked separately, GEMM shares reference-quantized operands" if args.fp8_prefill_only else "vLLM CUDA per-token; one scalar at logical M1",
        timing="median30 CUDA-event samples,5warmups;700MiB L2 eviction outside eachsample; no quantization/padding/setup allocation timed",
        note=f"M{logical_m} actual projection shapes plus K528/640 tails; BF16 model head excluded" if args.fp8_prefill_only else "lm_head shape is a synthetic kernel stress case; model lm_head remains BF16",
        passed=all(r["passed"] for r in records),cases=records)
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report["passed"]

if args.fp8_gemm_only or args.fp8_interpreter_only:
    raise SystemExit(0 if check_fp8_gemm() else 1)

def check_fp8():
    from vllm import _custom_ops as ops
    import vllm
    from pathlib import Path
    records=[]
    torch.manual_seed(1729)
    for M in ((args.fp8_rows,) if args.fp8_rows is not None else (1,4,16)):
        for K in ((5120,6144,17408,528,640) if M==128 else (3840,5120,17408)):
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
if args.gdn_timing_only:
    import hashlib, inspect, statistics
    from pathlib import Path
    torch.manual_seed(1729)
    source=Path(inspect.getsourcefile(oracle))
    report=dict(boundary="recurrent step only; excludes causal conv, projections, gated output norm",
        reference=str(source),reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        initial_state="random FP32, identical reset before each untimed pair; not a captured context checkpoint",
        reference_state_indices="1..B; FLA index0 is reserved NULL_BLOCK_ID",
        context="fixed recurrent state dimensions; 1K/4K/8K/32K histories require separately captured states",
        rows=[])
    for B in (1,4,16):
        active=torch.ones(B,device="cuda",dtype=torch.int32)
        mixed=rand(B,10240);a=rand(B,48);b=rand(B,48);alog=rand(48);bias=rand(48)
        alog32=alog.float()
        initial=torch.randn(B,48,128,128,device="cuda")*.01
        state=initial.clone()
        reference_initial=torch.cat([torch.zeros_like(initial[:1]),initial])
        reference=reference_initial.clone()
        indices=torch.arange(1,B+1,device="cuda",dtype=torch.int32)
        out=torch.empty(B,1,48,128,device="cuda",dtype=torch.bfloat16);ref=torch.empty_like(out)
        def native():run(137,[out,mixed,a,b,alog,bias,state,active],[16,48,128,128,B,0],[128**-.5,1e-6])
        def reference_step():oracle(mixed,a,b,alog32,bias,128**-.5,reference,ref,indices,True)
        native();reference_step();torch.cuda.synchronize()
        torch.testing.assert_close(out,ref,rtol=.01,atol=1e-5)
        torch.testing.assert_close(state,reference[1:],rtol=1e-4,atol=1e-6)
        torch.testing.assert_close(reference[0],reference_initial[0],rtol=0,atol=0)
        graphs={}
        for name,fn in (("native",native),("vllm_fla",reference_step)):
            for _ in range(3):fn()
            torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):fn()
            graphs[name]=g
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        times={name:[] for name in graphs}
        for rep in range(max(3,args.attention_iters)):
            order=list(graphs)
            if rep%2:order.reverse()
            for name in order:
                state.copy_(initial);reference.copy_(reference_initial)
                start.record();graphs[name].replay();end.record();end.synchronize()
                times[name].append(start.elapsed_time(end)*1000)
        row=dict(batch=B,HK=16,HV=48,DK=128,DV=128,passed=True,
            native_us=statistics.median(times["native"]),vllm_fla_us=statistics.median(times["vllm_fla"]))
        report["rows"].append(row);print(json.dumps(row),flush=True)
        if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    raise SystemExit(0)
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
