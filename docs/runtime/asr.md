# ASR Runtime Pipeline

Crisp technical specification of Automatic Speech Recognition (ASR) serving in `plowrt`.

---

## 1. Pipeline Architecture

```
Raw Audio (PCM 16kHz) → Mel Filterbank → Audio Encoder → CTC / Decoder → Streaming Text
```

* **Audio Front-End:** 80-channel log-mel spectrogram extraction executed on device.
* **Conformer/Transformer Encoder:** Downsamples spectrogram frames (4× temporal subsampling) into acoustic embeddings.
* **Autoregressive / CTC Decoder:** Emits text tokens subject to turn latency budgets ($T_{\text{turn}} \le 200\text{ ms}$).

---

## 2. Streaming & Turn Governance

* **Chunked Streaming:** Audio frames ingested in 100–200 ms windows with lookahead buffering.
* **VAD Integration:** Voice Activity Detection packet marks utterance boundaries (`PLOW_ASR_VAD_PACKET=1`).
* **Partial Results:** Emits speculative partial transcripts, finalized upon endpoint detection.

---

## 3. Verification

* Validate ASR accuracy and greedy continuation:
  ```bash
  python3 scripts/asr/gate.py --candidate <scratch>/res --ref <scratch>/hf_ref
  ```
