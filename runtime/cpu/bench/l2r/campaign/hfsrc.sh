#!/usr/bin/env bash
# hfsrc.sh: copy the vLLM image's transformers gemma4 modeling source to /tmp/g4c/l2r/hf_gemma4/
mkdir -p /tmp/g4c/l2r/hf_gemma4
/tmp/g4c/bin/vllm-py -c "import transformers, os, shutil; d = os.path.dirname(transformers.__file__) + '/models/gemma4'; print(transformers.__version__, d); [shutil.copy(os.path.join(d, f), '/tmp/g4c/l2r/hf_gemma4/') for f in os.listdir(d) if f.endswith('.py')]"
ls /tmp/g4c/l2r/hf_gemma4
