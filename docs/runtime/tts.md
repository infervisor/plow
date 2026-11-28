# TTS Runtime Pipeline

Crisp technical specification of Text-to-Speech (TTS) serving and codec streaming in `plowrt`.

---

## 1. Synthesis Pipeline

```
Text Tokens → Acoustic Backbone (Flow Matching / Diffusion) → Neural Vocoder (Mimi / EnCodec) → Audio PCM
```

* **Acoustic Backbone:** Generates multi-codebook latent acoustic frames from input text and prosody conditioning.
* **Neural Vocoder:** Decodes discrete acoustic tokens into 24kHz / 48kHz PCM audio waveform.
* **Codec Ring Buffers:** Slot-indexed circular ring buffers (`PLOW_CODEC_RING_KV=1`) stream audio chunks with sub-50ms Time-to-First-Audio (TTFA).

---

## 2. Real-Time Streaming Controls

* **Device-Side Sampling (`PLOW_TTS_DEVICE_CONTROL=1`):** Eliminates host round-trips for top-$p$ acoustic codebook sampling.
* **Lookahead Framing:** First audio chunk emitted within 2–4 lookahead frames (`PLOW_TTS_FIRST_LOOKAHEAD=2`).
* **Turn Budgets:** Enforces end-to-end voice latency SLA ($T_{\text{TTFA}} \le 120\text{ ms}$).

---

## 3. Verification

* Acoustic parity validated via `scripts/tts/gate.py` against reference PyTorch model.
