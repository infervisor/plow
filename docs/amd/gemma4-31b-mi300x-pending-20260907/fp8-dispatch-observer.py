import os
if os.environ.get('PLOW_QUANT_CAPTURE'):
    import json
    from pathlib import Path
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm import _custom_ops
    quant=torch.ops._C.dynamic_per_token_scaled_fp8_quant.default
    fused_quant=torch.ops._C.rms_norm_dynamic_per_token_quant.default
    records={}
    fused_records={}
    ordinal=0
    fused_ordinal=0
    class ObserveQuant(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            global ordinal,fused_ordinal
            fields={a.name:v for a,v in zip(func._schema.arguments,args)} if func==fused_quant else {}
            fields.update(kwargs or {})
            capture_fused=func==fused_quant and fields['input'].shape[0]==128
            slot=fused_ordinal%120
            if capture_fused and slot<2:
                fused_records[slot]={name:value.detach().clone() for name,value in fields.items() if isinstance(value,torch.Tensor) and name not in ('result','scale')}
            result=func(*args,**(kwargs or {}))
            if capture_fused:
                fused_ordinal+=1
                if slot<2:
                    for name,value in fields.items():
                        if isinstance(value,torch.Tensor) and name in ('result','scale'):
                            fused_records[slot][name]=value.detach().clone()
            if func==quant:
                x=args[1]
                if x.shape[0]==128:
                    index=ordinal%240
                    ordinal+=1
                    if index<4:
                        records[index]=x.detach().clone()
            return result
    from vllm.v1.worker.gpu_worker import Worker
    original_execute=Worker.execute_model
    def execute(self,*args,**kwargs):
        with ObserveQuant():
            return original_execute(self,*args,**kwargs)
    Worker.execute_model=execute
    def dump(worker):
        directory=Path(os.environ['PLOW_QUANT_CAPTURE'])
        directory.mkdir(parents=True,exist_ok=True)
        meta={'ordinal':ordinal,'fused_ordinal':fused_ordinal,'recorded':[],'fused_recorded':[],'fused_schema':str(fused_quant._schema),'mechanism':'TorchDispatchMode observing original op after compilation'}
        for index,value in records.items():
            v=value.detach().cpu().contiguous()
            path=directory/f'quant{index:02}.bf16'
            path.write_bytes(v.view(torch.uint16).numpy().astype('<u2').tobytes())
            meta['recorded'].append({'index':index,'shape':list(value.shape),'file':path.name})
        for index,fields in fused_records.items():
            for name,value in fields.items():
                v=value.detach().cpu().contiguous()
                path=directory/f'fused{index:02}.{name}.bin'
                path.write_bytes(v.view(torch.uint8).numpy().tobytes())
                meta['fused_recorded'].append({'index':index,'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'file':path.name})
        (directory/'manifest.json').write_text(json.dumps(meta,indent=2))
        return meta
    Worker._plow_dump_quant=dump
    from vllm import LLM
    original_generate=LLM.generate
    def generate(self,*args,**kwargs):
        result=original_generate(self,*args,**kwargs)
        self.collective_rpc('_plow_dump_quant')
        return result
    LLM.generate=generate
    capture_ready=True
