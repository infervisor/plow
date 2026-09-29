# ASR on NVIDIA (H200, sm_90a) — plan

Rule: plowc emits packets → plowrt runs them (PacketRuntime + interpreter arms). No bespoke .so kernels.

Models: Qwen3-ASR-1.7B (`causal.v1`: audio-encoder packet + Qwen3 decoder packet),
Nemotron 3.5 streaming 0.6B (`rnnt.greedy.v1`: subsampling + FastConformer + LSTM predictor + joint).
Checkpoints: /root/plow/models/{Qwen3-ASR-1.7B,nemotron-3.5-asr-streaming-0.6b}. Audio: /root/asr-work/audio (73 LS clips, 481 s).
Reference venv: /root/asr-work/venv (vLLM 0.30, torch 2.13, transformers 5.17). Results: /root/asr-work/results.

## Steps
1. Baselines (scripts/asr/nvidia/ref_bench.py): vLLM Qwen3-ASR; HF Nemotron offline + cache-aware stream; CUPTI top kernels.
2. CUDA interp arms for FP32 speech ops 163-180 (port runtime/cpu/dev/golden/f32_primitives.c 1:1, block-sliced),
   gated object so text-LLM cubins unchanged. Gate vs CPU golden on random tensors.
3. CUDA `PacketRuntime` (exec/packet_runtime.rs load_packet_runtime "cuda"/auto arm): packet w/ embedded weights,
   device tensors by name, H2D/D2H/D2D, run(program) = interpreter launch.
4. Nemotron: move packet compile from plowrt example → plowc (`--asr nemotron --gguf ...` or hf safetensors). E2E vs HF transcript.
5. Qwen: plowc emits encoder.pkt + model.pkt; CUDA QwenExecution: encoder via PacketRuntime, decoder via GpuEngine
   with EmbedOverlayBf16 (179) splice. E2E vs vLLM transcript.
6. Streaming: cache-aware chunked encoder programs (att cache + conv cache tensors as packet state, chunk = 1+8r / 8(r+1) mel),
   RNNT state carried across chunks; WS partials. Match HF streaming transcript.
7. Per-kernel perf: per-program/opcode timing in plowrt vs CUPTI kernel table of reference; optimize hot ops
   (Q8Gemm → tensor cores, rel-attention, LSTM/joint loop latency).

## Baselines (H200, 73 LS dummy clips, 481 s; results in /root/asr-work/results)
| system | WER | p50 ms | p90 ms | seq RTFx | batch RTFx |
|---|---:|---:|---:|---:|---:|
| vLLM 0.30 Qwen3-ASR-1.7B (bf16, graphs) | 3.652% | 62.7 | 108.7 | 93 | 1534 |
| HF Nemotron 3.5 offline (bf16, r=3) | 5.478% | 155.2 | 290.7 | 35 | 151 (B16) |
| HF Nemotron 3.5 cache-aware stream (r=3, 320 ms) | 5.478% | 613.2 | 1209.5 | 9.1 | — |
Qwen decode kernels (1 clip): nvjet GEMV ~63%, FA3 decode ~16%, rmsnorm/silu/kv-write ~12%.
Nemotron: small GEMMs + argmax + DtoD/DtoH copies dominate (host-driven RNNT loop).
CUDA_HOME for vLLM JIT: scripts/asr/nvidia/mk_cuda_home.sh → /root/asr-work/cuda_home.

## Blockers
- nix develop cannot build new devshell: stale /homeless-shelter (cargo lockfiles, 06:45) → needs user decision.
