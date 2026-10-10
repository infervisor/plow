/-
# Plow.MediaGeometry — endpoint `media_geometry.v1`.

Integer geometry of the speech and multimodal packet contracts, read from the packets' pipeline
sections and tensor shapes: audio-LM overlay capacity and context fit, RNNT frame transforms and
symbol bound, codec-LM token/code/PCM addressing, guided-LM token domains, VAD frame/state
layout, and multimodal soft-token ids. `check_sound` is the acceptance theorem; the lemmas after
each contract are what an accepted contract buys for every input up to the declared maxima.

All arithmetic is over `Nat`. Wire values are u32 token ids and u64 parameters; the Rust side
rejects a parameter that does not fit before building this input.
-/
import Lean.Data.Json

namespace Plow.MediaGeometry

def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

theorem div_mono {a b k : Nat} (h : a ≤ b) : a / k ≤ b / k := by
  cases k with
  | zero => simp
  | succ k => exact (Nat.le_div_iff_mul_le (Nat.succ_pos _)).2 (Nat.le_trans (Nat.div_mul_le_self a _) h)

theorem ceilDiv_mono {a b s : Nat} (h : a ≤ b) : ceilDiv a s ≤ ceilDiv b s :=
  div_mono (by omega)

/-! ## Audio-LM (`causal.v1` with an encoder overlay) -/

/-- Overlay rows of `frames` mel frames: whole chunks give `ceilDiv chunk stride` rows each, the
    tail `ceilDiv (frames % chunk) stride` (`asr::audio_lm::AudioChunking::rows`). -/
def audioRows (chunk stride frames : Nat) : Nat :=
  frames / chunk * ceilDiv chunk stride + ceilDiv (frames % chunk) stride

theorem audioRows_mono {chunk stride f g : Nat} (hc : 0 < chunk) (h : f ≤ g) :
    audioRows chunk stride f ≤ audioRows chunk stride g := by
  unfold audioRows
  have hq : f / chunk ≤ g / chunk := div_mono h
  have tail : ∀ n, ceilDiv (n % chunk) stride ≤ ceilDiv chunk stride := fun n =>
    ceilDiv_mono (Nat.le_of_lt (Nat.mod_lt _ hc))
  rcases Nat.lt_or_eq_of_le hq with lt | eq
  · have := tail f
    have hstep : f / chunk * ceilDiv chunk stride + ceilDiv chunk stride ≤
        g / chunk * ceilDiv chunk stride := by
      have : (f / chunk + 1) * ceilDiv chunk stride ≤ g / chunk * ceilDiv chunk stride :=
        Nat.mul_le_mul_right _ lt
      simpa [Nat.add_mul] using this
    omega
  · -- Same chunk count: the tail is monotone because `f % chunk ≤ g % chunk`.
    have hf := Nat.div_add_mod f chunk
    have hg := Nat.div_add_mod g chunk
    have hm : f % chunk ≤ g % chunk := by
      have : chunk * (f / chunk) = chunk * (g / chunk) := by rw [eq]
      omega
    rw [eq]
    exact Nat.add_le_add_left (ceilDiv_mono hm) _

structure AudioLm where
  sampleRate : Nat
  maxSeconds : Nat
  encoderSampleRate : Nat
  encoderMaxSamples : Nat
  hop : Nat
  chunkFrames : Nat
  frameStride : Nat
  overlayRows : Nat
  encoderOutputRows : Nat
  hidden : Nat
  encoderWidth : Nat
  maxContext : Nat
  maxTokens : Nat
  reservePerRow : Nat
  reserveExtra : Nat
  audioToken : Nat
  stops : List Nat
  deriving Repr

/-- Mel frames of the longest accepted recording (an upper bound for either STFT framing). -/
def AudioLm.maxFrames (a : AudioLm) : Nat := a.encoderMaxSamples / a.hop + 1

def AudioLm.maxRows (a : AudioLm) : Nat := audioRows a.chunkFrames a.frameStride a.maxFrames

def AudioLm.Contract (a : AudioLm) : Prop :=
  0 < a.hop ∧ 0 < a.chunkFrames ∧ 0 < a.frameStride ∧
  a.encoderSampleRate = a.sampleRate ∧ a.encoderMaxSamples = a.maxSeconds * a.sampleRate ∧
  a.hidden = a.encoderWidth ∧
  a.maxRows ≤ a.overlayRows ∧ a.maxRows ≤ a.encoderOutputRows ∧
  a.maxRows + a.reservePerRow * a.maxRows + a.reserveExtra < a.maxContext ∧
  a.maxRows + a.maxTokens ≤ a.maxContext ∧ a.maxContext < 2 ^ 32 ∧
  a.audioToken < 2 ^ 31 ∧ (∀ s ∈ a.stops, s < 2 ^ 31 ∧ s ≠ a.audioToken)

instance (a : AudioLm) : Decidable a.Contract := by unfold AudioLm.Contract; exact inferInstance

/-- Every recording up to the declared maximum fits the overlay and the encoder output, and
    leaves the transcript reserve inside the context. -/
theorem AudioLm.every_recording_fits (a : AudioLm) (h : a.Contract) (frames : Nat)
    (hf : frames ≤ a.maxFrames) :
    audioRows a.chunkFrames a.frameStride frames ≤ a.overlayRows ∧
    audioRows a.chunkFrames a.frameStride frames ≤ a.encoderOutputRows ∧
    audioRows a.chunkFrames a.frameStride frames + a.maxTokens ≤ a.maxContext := by
  obtain ⟨_, hc, _, _, _, _, ho, he, _, ht, _⟩ := h
  have m := audioRows_mono (stride := a.frameStride) hc hf
  unfold AudioLm.maxRows at ho he ht
  omega

/-! ## RNNT / TDT (`rnnt.greedy.v1`) -/

structure FrameTransform where
  kernel : Nat
  stride : Nat
  padBefore : Nat
  padAfter : Nat
  deriving Repr

def FrameTransform.out (t : FrameTransform) (n : Nat) : Nat :=
  (n + t.padBefore + t.padAfter - t.kernel) / t.stride + 1

def chain : List FrameTransform → Nat → Nat
  | [], n => n
  | t :: rest, n => chain rest (t.out n)

theorem FrameTransform.out_mono (t : FrameTransform) {n m : Nat} (h : n ≤ m) : t.out n ≤ t.out m := by
  unfold FrameTransform.out
  exact Nat.add_le_add_right (div_mono (by omega)) 1

theorem chain_mono : ∀ (ts : List FrameTransform) {n m : Nat}, n ≤ m → chain ts n ≤ chain ts m
  | [], _, _, h => h
  | t :: rest, _, _, h => chain_mono rest (t.out_mono h)

/-- `blank` is a vocabulary piece or the extra joint class `vocab` (NeMo puts it last). -/
structure Rnnt where
  hop : Nat
  maxSamples : Nat
  inputFrames : Nat
  transforms : List FrameTransform
  frames : Nat
  jointRows : Nat
  blank : Nat
  vocab : Nat
  maxSymbolsPerFrame : Nat
  deriving Repr

def Rnnt.Contract (r : Rnnt) : Prop :=
  0 < r.hop ∧ r.maxSamples / r.hop + 1 ≤ r.inputFrames ∧
  (∀ t ∈ r.transforms, 0 < t.stride ∧ 0 < t.kernel ∧ t.kernel ≤ t.padBefore + t.padAfter + 1) ∧
  chain r.transforms r.inputFrames = r.frames ∧ r.frames ≤ r.jointRows ∧
  r.blank ≤ r.vocab ∧ r.vocab < 2 ^ 31 ∧ 0 < r.maxSymbolsPerFrame

instance (r : Rnnt) : Decidable r.Contract := by unfold Rnnt.Contract; exact inferInstance

/-- Every input up to the declared frame capacity yields encoder rows inside the joint input. -/
theorem Rnnt.every_input_fits (r : Rnnt) (h : r.Contract) (n : Nat) (hn : n ≤ r.inputFrames) :
    chain r.transforms n ≤ r.jointRows := by
  obtain ⟨_, _, _, hc, hj, _⟩ := h
  have := chain_mono r.transforms hn
  omega

/-- Greedy decoding emits at most `maxSymbolsPerFrame` symbols per encoder row, so a
    recording's decode is bounded by `frames * (maxSymbolsPerFrame + 1)` joint steps. -/
theorem Rnnt.decode_steps_bounded (r : Rnnt) (perFrame : Fin r.frames → Nat)
    (hp : ∀ f, perFrame f ≤ r.maxSymbolsPerFrame + 1) :
    (List.finRange r.frames).foldr (fun f acc => perFrame f + acc) 0 ≤
      r.frames * (r.maxSymbolsPerFrame + 1) := by
  have : ∀ (l : List (Fin r.frames)),
      l.foldr (fun f acc => perFrame f + acc) 0 ≤ l.length * (r.maxSymbolsPerFrame + 1) := by
    intro l
    induction l with
    | nil => simp
    | cons f rest ih =>
      simp only [List.foldr_cons, List.length_cons, Nat.succ_mul]
      have := hp f
      omega
  simpa using this (List.finRange r.frames)

/-! ## Codec LM (`tts.codec_lm.v1` + `codec.v1`) -/

structure CodecLm where
  lmSampleRate : Nat
  lmCodebook : Nat
  lmFrameCodes : Nat
  lmFrameSamples : Nat
  tokenBase : Nat
  maxNewTokens : Nat
  promptTokens : Nat
  maxContext : Nat
  stops : List Nat
  codecSampleRate : Nat
  codebook : Nat
  frameCodes : Nat
  frameSamples : Nat
  codesCapacity : Nat
  pcmCapacity : Nat
  windowFrames : Nat
  lookaheadFrames : Nat
  deriving Repr

def CodecLm.frames (c : CodecLm) : Nat := c.codesCapacity / c.frameCodes

def CodecLm.Contract (c : CodecLm) : Prop :=
  c.lmSampleRate = c.codecSampleRate ∧ c.lmCodebook = c.codebook ∧
  c.lmFrameCodes = c.frameCodes ∧ c.lmFrameSamples = c.frameSamples ∧
  0 < c.frameCodes ∧ 0 < c.codebook ∧ c.codesCapacity % c.frameCodes = 0 ∧
  c.frames * c.frameSamples ≤ c.pcmCapacity ∧
  0 < c.windowFrames ∧ c.windowFrames + c.lookaheadFrames ≤ c.frames ∧
  c.tokenBase + c.frameCodes * c.codebook ≤ 2 ^ 31 ∧
  c.maxNewTokens + c.promptTokens < c.maxContext ∧ c.maxContext < 2 ^ 32 ∧
  (∀ s ∈ c.stops, s < 2 ^ 31 ∧ (s < c.tokenBase ∨ c.tokenBase + c.frameCodes * c.codebook ≤ s))

instance (c : CodecLm) : Decidable c.Contract := by unfold CodecLm.Contract; exact inferInstance

/-- A frame's codes and samples stay inside the codec's code and PCM tensors. -/
theorem CodecLm.addresses_in_bounds (c : CodecLm) (h : c.Contract) {f j t : Nat}
    (hf : f < c.frames) (hj : j < c.frameCodes) (ht : t < c.frameSamples) :
    f * c.frameCodes + j < c.codesCapacity ∧ f * c.frameSamples + t < c.pcmCapacity := by
  obtain ⟨_, _, _, _, hfc, _, hmod, hpcm, _⟩ := h
  have hcap : c.frames * c.frameCodes = c.codesCapacity := by
    unfold CodecLm.frames; exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
  constructor
  · have : (f + 1) * c.frameCodes ≤ c.frames * c.frameCodes := Nat.mul_le_mul_right _ hf
    rw [Nat.add_mul, hcap] at this
    omega
  · have : (f + 1) * c.frameSamples ≤ c.frames * c.frameSamples := Nat.mul_le_mul_right _ hf
    rw [Nat.add_mul] at this
    omega

/-- An audio token decodes to a code position below `frameCodes` and a code below the
    codebook, and no stop token is an audio token. -/
theorem CodecLm.token_decodes (c : CodecLm) (h : c.Contract) {tok : Nat}
    (lo : c.tokenBase ≤ tok) (hi : tok < c.tokenBase + c.frameCodes * c.codebook) :
    (tok - c.tokenBase) / c.codebook < c.frameCodes ∧ (tok - c.tokenBase) % c.codebook < c.codebook := by
  obtain ⟨_, _, _, _, _, hcb, _⟩ := h
  refine ⟨?_, Nat.mod_lt _ hcb⟩
  rw [Nat.div_lt_iff_lt_mul hcb]
  omega

/-! ## Guided speech LM (`tts.guided_lm.v1`) -/

structure GuidedLm where
  speechVocab : Nat
  startSpeech : Nat
  stopSpeech : Nat
  validBelow : Nat
  textVocab : Nat
  startText : Nat
  stopText : Nat
  maxSpeechTokens : Nat
  speechPositions : Nat
  overlayRows : Nat
  maxContext : Nat
  deriving Repr

def GuidedLm.Contract (g : GuidedLm) : Prop :=
  g.startSpeech < g.speechVocab ∧ g.stopSpeech < g.speechVocab ∧
  g.validBelow ≤ g.startSpeech ∧ g.validBelow ≤ g.stopSpeech ∧
  g.startText < g.textVocab ∧ g.stopText < g.textVocab ∧
  g.maxSpeechTokens + 2 ≤ g.speechPositions ∧ g.overlayRows ≤ g.maxContext ∧ g.maxContext < 2 ^ 32

instance (g : GuidedLm) : Decidable g.Contract := by unfold GuidedLm.Contract; exact inferInstance

/-! ## VAD (`vad.frame.v1`) -/

structure Vad where
  frameSamples : Nat
  contextSamples : Nat
  inputElements : Nat
  stateBanks : Nat
  stateSizes : List Nat
  minSpeechMs : Nat
  maxDurationMs : Nat
  deriving Repr

def Vad.Contract (v : Vad) : Prop :=
  0 < v.frameSamples ∧ v.frameSamples + v.contextSamples = v.inputElements ∧
  0 < v.stateBanks ∧ v.stateSizes.length = 2 * v.stateBanks ∧
  (∀ s ∈ v.stateSizes, 0 < s ∧ s = v.stateSizes.headD 0) ∧ v.minSpeechMs ≤ v.maxDurationMs

instance (v : Vad) : Decidable v.Contract := by unfold Vad.Contract; exact inferInstance

/-! ## Multimodal soft tokens (`plow.multimodal.v1`) -/

structure Multimodal where
  hidden : Nat
  lmHidden : Nat
  padToken : Nat
  tokens : List Nat
  slabRows : Nat
  tableCapacity : Nat
  slabBytes : Nat
  tableBytes : Nat
  deriving Repr

def Multimodal.Contract (m : Multimodal) : Prop :=
  m.hidden = m.lmHidden ∧ m.padToken < 2 ^ 31 ∧ (∀ t ∈ m.tokens, t < 2 ^ 31) ∧
  m.tableCapacity ∈ (List.range 32).map (2 ^ ·) ∧ 0 < m.slabRows ∧ m.slabRows ≤ m.tableCapacity ∧
  m.slabBytes = m.slabRows * m.hidden * 2 ∧ m.tableBytes = m.tableCapacity * 8

instance (m : Multimodal) : Decidable m.Contract := by unfold Multimodal.Contract; exact inferInstance

/-- A soft-token id (bit 31 over a 31-bit content hash) fits u32 and is never the pad token or a
    placeholder/begin/end token. -/
theorem Multimodal.soft_ids_disjoint (m : Multimodal) (h : m.Contract) {hash : Nat} (hh : hash < 2 ^ 31) :
    2 ^ 31 + hash < 2 ^ 32 ∧ 2 ^ 31 + hash ≠ m.padToken ∧ ∀ t ∈ m.tokens, 2 ^ 31 + hash ≠ t := by
  obtain ⟨_, hp, ht, _⟩ := h
  refine ⟨by omega, by omega, fun t mem => ?_⟩
  have := ht t mem
  omega

/-! ## Checker -/

inductive Family
  | audioLm (a : AudioLm)
  | rnnt (r : Rnnt)
  | codecLm (c : CodecLm)
  | guidedLm (g : GuidedLm)
  | vad (v : Vad)
  | multimodal (m : Multimodal)
  deriving Repr

def Family.Contract : Family → Prop
  | .audioLm a => a.Contract
  | .rnnt r => r.Contract
  | .codecLm c => c.Contract
  | .guidedLm g => g.Contract
  | .vad v => v.Contract
  | .multimodal m => m.Contract

instance (f : Family) : Decidable f.Contract := by
  cases f <;> unfold Family.Contract <;> exact inferInstance

def check (families : List Family) : Bool :=
  !families.isEmpty && families.all fun f => decide f.Contract

theorem check_sound (families : List Family) (h : check families = true) :
    families ≠ [] ∧ ∀ f ∈ families, f.Contract := by
  simp only [check, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false, List.all_eq_true,
    decide_eq_true_eq] at h
  exact ⟨by intro e; simp [e] at h, h.2⟩

/-! ## JSON -/

open Lean (Json)

/-- Exactly these keys: an unknown field is a schema mismatch, never ignored. -/
def exactKeys (j : Json) (keys : List String) : Except String Unit := do
  let obj ← j.getObj?
  let present := obj.toArray.map (·.1) |>.toList
  for k in present do
    if !keys.contains k then throw s!"unknown field {k}"
  for k in keys do
    if !present.contains k then throw s!"missing field {k}"

def nat (j : Json) (k : String) : Except String Nat := j.getObjValAs? Nat k
def nats (j : Json) (k : String) : Except String (List Nat) := j.getObjValAs? (List Nat) k

def parseTransform (j : Json) : Except String FrameTransform := do
  exactKeys j ["kernel", "stride", "pad_before", "pad_after"]
  return {
    kernel := (← nat j "kernel"), stride := (← nat j "stride"),
    padBefore := (← nat j "pad_before"), padAfter := (← nat j "pad_after") }

def parseFamily (j : Json) : Except String Family := do
  match ← j.getObjValAs? String "kind" with
  | "audio_lm" =>
    exactKeys j ["kind", "sample_rate", "max_seconds", "encoder_sample_rate", "encoder_max_samples",
      "hop", "chunk_frames", "frame_stride", "overlay_rows", "encoder_output_rows", "hidden",
      "encoder_width", "max_context", "max_tokens", "reserve_per_row", "reserve_extra",
      "audio_token", "stops"]
    return .audioLm {
      sampleRate := (← nat j "sample_rate"), maxSeconds := (← nat j "max_seconds"),
      encoderSampleRate := (← nat j "encoder_sample_rate"), encoderMaxSamples := (← nat j "encoder_max_samples"),
      hop := (← nat j "hop"), chunkFrames := (← nat j "chunk_frames"), frameStride := (← nat j "frame_stride"),
      overlayRows := (← nat j "overlay_rows"), encoderOutputRows := (← nat j "encoder_output_rows"),
      hidden := (← nat j "hidden"), encoderWidth := (← nat j "encoder_width"),
      maxContext := (← nat j "max_context"), maxTokens := (← nat j "max_tokens"),
      reservePerRow := (← nat j "reserve_per_row"), reserveExtra := (← nat j "reserve_extra"),
      audioToken := (← nat j "audio_token"), stops := (← nats j "stops") }
  | "rnnt" =>
    exactKeys j ["kind", "hop", "max_samples", "input_frames", "transforms", "frames", "joint_rows",
      "blank", "vocab", "max_symbols_per_frame"]
    let ts ← j.getObjValAs? (List Json) "transforms"
    return .rnnt {
      hop := (← nat j "hop"), maxSamples := (← nat j "max_samples"),
      inputFrames := (← nat j "input_frames"), transforms := (← ts.mapM parseTransform),
      frames := (← nat j "frames"), jointRows := (← nat j "joint_rows"), blank := (← nat j "blank"),
      vocab := (← nat j "vocab"), maxSymbolsPerFrame := (← nat j "max_symbols_per_frame") }
  | "codec_lm" =>
    exactKeys j ["kind", "lm_sample_rate", "lm_codebook", "lm_frame_codes", "lm_frame_samples",
      "token_base", "max_new_tokens", "prompt_tokens", "max_context", "stops", "codec_sample_rate",
      "codebook", "frame_codes", "frame_samples", "codes_capacity", "pcm_capacity", "window_frames",
      "lookahead_frames"]
    return .codecLm {
      lmSampleRate := (← nat j "lm_sample_rate"), lmCodebook := (← nat j "lm_codebook"),
      lmFrameCodes := (← nat j "lm_frame_codes"), lmFrameSamples := (← nat j "lm_frame_samples"),
      tokenBase := (← nat j "token_base"), maxNewTokens := (← nat j "max_new_tokens"),
      promptTokens := (← nat j "prompt_tokens"), maxContext := (← nat j "max_context"),
      stops := (← nats j "stops"), codecSampleRate := (← nat j "codec_sample_rate"),
      codebook := (← nat j "codebook"), frameCodes := (← nat j "frame_codes"),
      frameSamples := (← nat j "frame_samples"), codesCapacity := (← nat j "codes_capacity"),
      pcmCapacity := (← nat j "pcm_capacity"), windowFrames := (← nat j "window_frames"),
      lookaheadFrames := (← nat j "lookahead_frames") }
  | "guided_lm" =>
    exactKeys j ["kind", "speech_vocab", "start_speech", "stop_speech", "valid_below", "text_vocab",
      "start_text", "stop_text", "max_speech_tokens", "speech_positions", "overlay_rows", "max_context"]
    return .guidedLm {
      speechVocab := (← nat j "speech_vocab"), startSpeech := (← nat j "start_speech"),
      stopSpeech := (← nat j "stop_speech"), validBelow := (← nat j "valid_below"),
      textVocab := (← nat j "text_vocab"), startText := (← nat j "start_text"), stopText := (← nat j "stop_text"),
      maxSpeechTokens := (← nat j "max_speech_tokens"), speechPositions := (← nat j "speech_positions"),
      overlayRows := (← nat j "overlay_rows"), maxContext := (← nat j "max_context") }
  | "vad" =>
    exactKeys j ["kind", "frame_samples", "context_samples", "input_elements", "state_banks",
      "state_sizes", "min_speech_ms", "max_duration_ms"]
    return .vad {
      frameSamples := (← nat j "frame_samples"), contextSamples := (← nat j "context_samples"),
      inputElements := (← nat j "input_elements"), stateBanks := (← nat j "state_banks"),
      stateSizes := (← nats j "state_sizes"), minSpeechMs := (← nat j "min_speech_ms"),
      maxDurationMs := (← nat j "max_duration_ms") }
  | "multimodal" =>
    exactKeys j ["kind", "hidden", "lm_hidden", "pad_token", "tokens", "slab_rows", "table_capacity",
      "slab_bytes", "table_bytes"]
    return .multimodal {
      hidden := (← nat j "hidden"), lmHidden := (← nat j "lm_hidden"),
      padToken := (← nat j "pad_token"), tokens := (← nats j "tokens"), slabRows := (← nat j "slab_rows"),
      tableCapacity := (← nat j "table_capacity"), slabBytes := (← nat j "slab_bytes"),
      tableBytes := (← nat j "table_bytes") }
  | other => throw s!"unknown media family {other}"

def Family.name : Family → String
  | .audioLm _ => "audio_lm"
  | .rnnt _ => "rnnt"
  | .codecLm _ => "codec_lm"
  | .guidedLm _ => "guided_lm"
  | .vad _ => "vad"
  | .multimodal _ => "multimodal"

def run (j : Json) : Except String String := do
  exactKeys j ["schema", "families"]
  if (← nat j "schema") != 1 then throw "unsupported media_geometry schema"
  let raw ← j.getObjValAs? (List Json) "families"
  let families ← raw.mapM parseFamily
  if families.isEmpty then throw "empty media geometry"
  for f in families do
    if !decide f.Contract then throw s!"{f.name} contract violated"
  if !check families then throw "media geometry rejected"
  return s!"check_sound: {families.length} media contracts ({", ".intercalate (families.map Family.name)}) hold over Nat; kernel arithmetic, frontend numerics and driver loops are outside this scope"

end Plow.MediaGeometry
