# Voice-agent profile with Veena (Indic + English voices) as the TTS: Silero VAD + Qwen3-ASR 1.7B + Veena + Gemma-4 E4B.
# Veena context 1536 holds its full 1400-token segment budget (long input is spoken in segments).
MODELS=silero-vad qwen3-asr veena gemma-4-e4b
LIVE_CTX=qwen3-asr=768,veena=1536
ENV=PLOW_VMM_LIVE_RINGS_MODELS=gemma-4-e4b
ARGS=--pin-resident
