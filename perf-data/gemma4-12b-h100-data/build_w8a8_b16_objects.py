import pathlib,subprocess,os,json
r=pathlib.Path("plans/gemma4-12b-roofline");out=r/"cubin-w8a8-b16";out.mkdir(exist_ok=True)
env=dict(PATH="/usr/local/cuda/bin:/usr/bin:/bin",LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu")
common=["-std=c++17","-arch=sm_90a","-O3","-cubin","-Xptxas=-v","-I","runtime/common","-I","runtime/nvidia"]
common += ["-D"+s for s in ["PLOW_NV_GEMMA=1","PLOW_NV_FA_GF=2","PLOW_NV_EMBED_SMEM=1","PLOW_NV_MLA=0","PLOW_NV_MAMBA=0","PLOW_NV_DSA=0","PLOW_NV_GEMV_RB=1","PLOW_MOE_DOWN_LANESPLIT=1","PLOW_NV_FA_WPR=1","PLOW_NV_FP8_RB=4"]]
configs=[("interp_sm90a",["PLOW_NV_FA_GF_FULL=4"]),("interp_sm90a_pf",["PLOW_NV_PREFILL=1","PLOW_NV_TMA_GEMM=1","PLOW_NV_FA_PIPE=1","PGM90_TMA_STAGES=3"])]
for packed in [0,1]:
 suffix="pfpacked" if packed else "pf"
 for role in ["seg","gemm","fa"]:
  name="interp_sm90a_"+suffix+role
  flags=["PLOW_NV_PREFILL=1","PLOW_NV_SEGMENTS=1",f"PLOW_NV_PACKED_REQUEST={packed}","PLOW_NV_TMA_GEMM=1","PGM90_TMA_STAGES=3"]
  if packed:flags += ["PLOW_NV_MASKED_PADDING=1"]
  if role=="seg":flags += ["PLOW_NV_FATLITE=1"]
  if role=="gemm":flags += ["PLOW_NV_SEG_WS384=1","PGM90_UNI_BN256=1","PLOW_NV_SEG_GEMM=1","PLOW_NV_GEMM_ONLY=1","PGM90_WS384_PREFETCH=1","PGM90_WS384_ISSUE_CURSOR=1","PGM90_WS384_SMEPI=1"]
  if role=="fa":flags += ["PLOW_NV_FA_ONLY=1","PLOW_NV_FA_ONLY_HD256=1"]
  configs.append((name,flags))
for name,flags in configs:
 if name != "interp_sm90a": flags += ["PLOW_NV_W8A8=1","PGM90_FP8_PROMOTE=1"]
 cmd=["nvcc",*common,*["-D"+s for s in flags],"-o",str(out/(name+".cubin")),"runtime/nvidia/interp_sm90a.cu"]
 with (r/("w8a8-b16-build-"+name+".log")).open("w") as f:
  f.write(json.dumps(cmd)+"\n");f.flush();subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,"PASS",flush=True)
