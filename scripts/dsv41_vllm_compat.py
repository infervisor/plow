"""Would vLLM's DeepSeek-V4 model load THIS checkpoint? Decided on CPU.

vLLM builds a compressor for every layer whose compress_ratio > 1
(vllm/models/deepseek_v4/attention.py:227,360). V4.1 instead names the layers
that own a compressor in kv_source_layer_ids, and every other layer READS that
cache. If the two disagree, the load asks for weights that do not exist.
"""
import json

HF = "/workspace/models/DeepSeek-V4.1-Flash"
cfg = json.load(open(f"{HF}/config.json"))["text_config"]
idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]

L = cfg["num_hidden_layers"]
ratios = cfg["compress_ratios"]

vllm_wants = [l for l in range(L) if max(1, ratios[l]) > 1]
ckpt_has = sorted({int(k.split(".")[1]) for k in idx if ".attn.compressor.wkv" in k})
declared = cfg["kv_source_layer_ids"]

print(f"layers                       : {L}")
print(f"vLLM would build compressors : {len(vllm_wants)} layers -> {vllm_wants}")
print(f"checkpoint actually has      : {len(ckpt_has)} layers -> {ckpt_has}")
print(f"config kv_source_layer_ids   : {declared}")
print()
missing = [l for l in vllm_wants if l not in ckpt_has]
extra = [l for l in ckpt_has if l not in vllm_wants]
print(f"weights vLLM wants but checkpoint lacks : {len(missing)} -> {missing}")
print(f"compressors checkpoint has that vLLM    : {len(extra)} -> {extra}")
print("  would never instantiate")
print()

eng = sorted({int(k.split(".")[1]) for k in idx if ".engram." in k})
print(f"engram layers in checkpoint  : {eng}  (vLLM has no engram support at all)")
n_eng = sum(1 for k in idx if ".engram." in k)
print(f"engram tensors               : {n_eng}")
print()
print("VERDICT:", "INCOMPATIBLE" if (missing or extra or eng) else "compatible")
