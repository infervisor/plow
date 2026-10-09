# Default voice-agent profile: Silero VAD (CPU) + Qwen3-ASR 1.7B + Chatterbox Multilingual + Gemma-4 E4B.
# ASR context 768 (30 s of audio + ~330 transcript tokens), MTL 1024 (~35 s of speech per request),
# E4B at its full 8K context with sliding-window rings allocated per live request.
MODELS=silero-vad qwen3-asr chatterbox-mtl gemma-4-e4b
LIVE_CTX=qwen3-asr=768,chatterbox-mtl=1024
ENV=PLOW_VMM_LIVE_RINGS_MODELS=gemma-4-e4b
ARGS=--pin-resident
