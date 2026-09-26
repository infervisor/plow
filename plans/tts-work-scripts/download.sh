#!/usr/bin/env bash
export HF_HOME=/root/tts-work/hf HF_HUB_ENABLE_HF_TRANSFER=1
PY=/root/tts-work/venv-ref/bin/python
/root/tts-work/uvboot/bin/uv pip install --python $PY -q hf_transfer
for m in maya-research/Veena ResembleAI/chatterbox ResembleAI/chatterbox-turbo; do
  $PY -c "from huggingface_hub import snapshot_download as s; print(s('$m'))"
done
echo DL_DONE
