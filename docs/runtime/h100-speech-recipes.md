# H100 Speech Serving Recipes

Crisp technical summary of speech recognition (ASR) and synthesis (TTS) models on NVIDIA H100 (`sm_90a`).

---

## 1. Qualified Models

* **Qwen3-ASR:** Conformer-based speech recognition; real-time streaming factor $\text{RTF} \le 0.04$.
* **Mimi-TTS:** Neural vocoder and flow-matching speech synthesis; Time-to-First-Audio $\le 80\text{ ms}$.

---

## 2. Deployment Configurations

* Deployed using slot-indexed codec ring buffers (`PLOW_CODEC_RING_KV=1`) and on-device sampling (`PLOW_TTS_DEVICE_CONTROL=1`).
* Full deployment recipes located in `recipes/infervisor/speech/`.
