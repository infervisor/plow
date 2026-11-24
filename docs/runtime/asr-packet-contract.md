# ASR Packet Contract

Crisp specification of the ASR packet ABI and symbol table contract in `plowrt`.

---

## 1. Packet ABI Invariants

* **Symbol Table:** Declares entry points for `audio_fe`, `audio_enc`, and `text_dec`.
* **Tensor Strides:** Audio feature tensors require $[B, T_{\text{mel}}, D_{\text{mel}}]$ contiguous layout.
* **Vocabulary Binding:** Tokenizer vocabulary and special tokens (`<sos>`, `<eos>`, `<blank>`) bound in `serve.json`.

---

## 2. Verification

* Verified via `plowc qualify --bundle <bundle>` ensuring all checkpoints evaluate clean.
