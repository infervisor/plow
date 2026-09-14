use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use gguf_rs_lib::format::metadata::Metadata;

use crate::asr::{Transcriber, Transcript};
use crate::asset::gguf::GgufFile;
use crate::{Result, RuntimeError};

type Status = i32;
const OK: Status = 0;

#[repr(C)]
struct BackendConfig {
    size: usize,
    gpu: i32,
}

#[repr(C)]
struct ModelConfig {
    size: usize,
    path: *const c_char,
    name: *const c_char,
}

#[repr(C)]
struct RecognizerConfig {
    size: usize,
    backend: *const BackendConfig,
    model: *const ModelConfig,
    streaming: *const c_void,
    decoder: *const c_void,
    vad: *const c_void,
    endpointing: *const c_void,
    postproc: *const c_void,
    diar: *const c_void,
    batching: *const c_void,
}

#[repr(C)]
struct RecognitionOptions {
    size: usize,
    request_id: *const c_char,
    language_code: *const c_char,
    interim_results: bool,
    enable_word_time_offsets: bool,
    enable_automatic_punctuation: bool,
    verbatim_transcripts: bool,
    profanity_filter: bool,
    stop_history_eou_ms: i32,
    speech_contexts: *const c_void,
    speech_context_count: usize,
    max_alternatives: i32,
    enable_speaker_diarization: bool,
    max_speaker_count: i32,
}

type Create = unsafe extern "C" fn(*const RecognizerConfig, *mut *mut c_void) -> Status;
type Destroy = unsafe extern "C" fn(*mut c_void);
type OptionsDefault = unsafe extern "C" fn() -> RecognitionOptions;
type Recognize = unsafe extern "C" fn(
    *mut c_void,
    *const RecognitionOptions,
    *const f32,
    usize,
    i32,
    *mut *mut c_void,
) -> Status;
type AlternativeCount = unsafe extern "C" fn(*const c_void) -> usize;
type ResultTranscript = unsafe extern "C" fn(*const c_void, usize) -> *const c_char;
type LanguageCount = unsafe extern "C" fn(*const c_void, usize) -> usize;
type LanguageCode = unsafe extern "C" fn(*const c_void, usize, usize) -> *const c_char;
type ResultDestroy = unsafe extern "C" fn(*mut c_void);
type LastError = unsafe extern "C" fn() -> *const c_char;

struct Api {
    create: Create,
    destroy: Destroy,
    options_default: OptionsDefault,
    recognize: Recognize,
    alternative_count: AlternativeCount,
    transcript: ResultTranscript,
    language_count: LanguageCount,
    language_code: LanguageCode,
    result_destroy: ResultDestroy,
    last_error: LastError,
}

impl Api {
    unsafe fn load(library: &libloading::Library) -> Result<Self> {
        unsafe fn get<T: Copy>(library: &libloading::Library, name: &[u8]) -> Result<T> {
            // SAFETY: the symbol names and signatures are pinned to nemo_speech/asr.h.
            unsafe { library.get::<T>(name) }
                .map(|symbol| *symbol)
                .map_err(|error| RuntimeError::Device(format!("NeMo ASR symbol: {error}")))
        }
        Ok(Self {
            create: unsafe { get(library, b"nemo_speech_asr_create\0")? },
            destroy: unsafe { get(library, b"nemo_speech_asr_destroy\0")? },
            options_default: unsafe {
                get(library, b"nemo_speech_asr_recognition_options_default\0")?
            },
            recognize: unsafe { get(library, b"nemo_speech_asr_recognize_f32\0")? },
            alternative_count: unsafe {
                get(library, b"nemo_speech_asr_result_alternative_count\0")?
            },
            transcript: unsafe { get(library, b"nemo_speech_asr_result_transcript\0")? },
            language_count: unsafe { get(library, b"nemo_speech_asr_result_language_count\0")? },
            language_code: unsafe { get(library, b"nemo_speech_asr_result_language_code\0")? },
            result_destroy: unsafe { get(library, b"nemo_speech_asr_result_destroy\0")? },
            last_error: unsafe { get(library, b"nemo_speech_asr_last_error\0")? },
        })
    }

    fn error(&self, operation: &str, status: Status) -> RuntimeError {
        // SAFETY: the ABI guarantees a thread-local NUL-terminated string after failure.
        let message = unsafe {
            let pointer = (self.last_error)();
            (!pointer.is_null())
                .then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
                .unwrap_or_else(|| "no error detail".into())
        };
        RuntimeError::Device(format!("NeMo ASR {operation} failed ({status}): {message}"))
    }
}

pub struct NemotronAsr {
    api: Api,
    recognizer: *mut c_void,
    _library: libloading::Library,
}

// The C ABI documents one recognizer as safe for independent calls from multiple
// threads. Transcriber still grants exclusive mutable access to each call.
unsafe impl Send for NemotronAsr {}

impl NemotronAsr {
    pub fn load(library_path: &Path, model_path: &Path, gpu: Option<u32>) -> Result<Self> {
        validate_model(model_path)?;
        let model_path = model_path
            .to_str()
            .ok_or_else(|| RuntimeError::Rejected("NeMo model path must be UTF-8".into()))?;
        let model_path = CString::new(model_path)
            .map_err(|_| RuntimeError::Rejected("NeMo model path contains NUL".into()))?;
        // SAFETY: the library remains owned by Self for every copied function pointer.
        let library = unsafe { libloading::Library::new(library_path) }
            .map_err(|error| RuntimeError::Device(format!("load NeMo ASR library: {error}")))?;
        // SAFETY: symbol signatures are checked against the installed stable header.
        let api = unsafe { Api::load(&library)? };
        let backend = BackendConfig {
            size: std::mem::size_of::<BackendConfig>(),
            gpu: gpu
                .map(i32::try_from)
                .transpose()
                .map_err(|_| RuntimeError::Rejected("NeMo GPU ordinal is too large".into()))?
                .unwrap_or(-1),
        };
        let model = ModelConfig {
            size: std::mem::size_of::<ModelConfig>(),
            path: model_path.as_ptr(),
            name: std::ptr::null(),
        };
        let config = RecognizerConfig {
            size: std::mem::size_of::<RecognizerConfig>(),
            backend: &backend,
            model: &model,
            streaming: std::ptr::null(),
            decoder: std::ptr::null(),
            vad: std::ptr::null(),
            endpointing: std::ptr::null(),
            postproc: std::ptr::null(),
            diar: std::ptr::null(),
            batching: std::ptr::null(),
        };
        let mut recognizer = std::ptr::null_mut();
        // SAFETY: all size-prefixed inputs match the published ABI and live through the call.
        let status = unsafe { (api.create)(&config, &mut recognizer) };
        if status != OK || recognizer.is_null() {
            return Err(api.error("create", status));
        }
        Ok(Self {
            api,
            recognizer,
            _library: library,
        })
    }
}

fn validate_model(path: &Path) -> Result<()> {
    let model = GgufFile::open(path)?;
    validate_model_catalog(model.metadata(), model.tensor_count())
        .map_err(|reason| RuntimeError::Rejected(format!("invalid Nemotron GGUF: {reason}")))
}

fn validate_model_catalog(
    metadata: &Metadata,
    tensor_count: usize,
) -> std::result::Result<(), String> {
    let architecture = metadata
        .get_string("general.architecture")
        .ok_or("missing general.architecture")?;
    if architecture != "asr" {
        return Err(format!(
            "expected general.architecture=asr, found {architecture:?}"
        ));
    }
    let head = metadata
        .get_string("asr.head_type")
        .ok_or("missing asr.head_type")?;
    if head != "rnnt" {
        return Err(format!("expected asr.head_type=rnnt, found {head:?}"));
    }
    if tensor_count == 0 {
        return Err("model has no tensors".into());
    }
    Ok(())
}

impl Transcriber for NemotronAsr {
    fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        Ok(requested.map(str::to_owned))
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<Transcript> {
        if !context.is_empty() {
            return Err(RuntimeError::Rejected(
                "Nemotron speech context is not implemented".into(),
            ));
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(RuntimeError::Rejected("ASR cancelled".into()));
        }
        let language = language
            .map(CString::new)
            .transpose()
            .map_err(|_| RuntimeError::Rejected("ASR language contains NUL".into()))?;
        // SAFETY: the library constructs the size and defaults for its current ABI version.
        let mut options = unsafe { (self.api.options_default)() };
        options.language_code = language.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
        options.enable_automatic_punctuation = true;
        let mut result = std::ptr::null_mut();
        // SAFETY: samples/options remain valid through this synchronous call; result is released below.
        let status = unsafe {
            (self.api.recognize)(
                self.recognizer,
                &options,
                samples.as_ptr(),
                samples.len(),
                crate::asr::frontend::SAMPLE_RATE as i32,
                &mut result,
            )
        };
        if status != OK {
            return Err(self.api.error("recognize", status));
        }
        if result.is_null() {
            return Err(RuntimeError::Device("NeMo ASR returned no result".into()));
        }
        let output = (|| {
            // SAFETY: accessors borrow the live result handle and return result-owned strings.
            if unsafe { (self.api.alternative_count)(result) } == 0 {
                return Err(RuntimeError::Device(
                    "NeMo ASR returned no transcript alternatives".into(),
                ));
            }
            let text = unsafe { (self.api.transcript)(result, 0) };
            if text.is_null() {
                return Err(RuntimeError::Device("NeMo ASR transcript is null".into()));
            }
            let text = unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned();
            let language = if unsafe { (self.api.language_count)(result, 0) } == 0 {
                language
                    .as_ref()
                    .map(|value| value.to_string_lossy().into_owned())
            } else {
                let value = unsafe { (self.api.language_code)(result, 0, 0) };
                (!value.is_null())
                    .then(|| unsafe { CStr::from_ptr(value).to_string_lossy().into_owned() })
            };
            Ok(Transcript { text, language })
        })();
        // SAFETY: successful recognition transferred exactly one result handle.
        unsafe { (self.api.result_destroy)(result) };
        if cancel.load(Ordering::Relaxed) {
            return Err(RuntimeError::Rejected("ASR cancelled".into()));
        }
        output
    }
}

impl Drop for NemotronAsr {
    fn drop(&mut self) {
        // SAFETY: recognizer was created by this API and is destroyed once before the library.
        unsafe { (self.api.destroy)(self.recognizer) };
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use gguf_rs_lib::format::metadata::MetadataValue;
    use gguf_rs_lib::prelude::GGUFBuilder;
    use gguf_rs_lib::reader::file_reader::GGUFFileReader;

    use super::validate_model_catalog;

    fn model(architecture: &str, head: &str) -> Vec<u8> {
        GGUFBuilder::new()
            .add_metadata(
                "general.architecture",
                MetadataValue::String(architecture.into()),
            )
            .add_metadata("asr.head_type", MetadataValue::String(head.into()))
            .add_f32_tensor("encoder.weight", vec![1], vec![0.0])
            .unwrap()
            .build_to_bytes()
            .unwrap()
            .0
    }

    #[test]
    fn validates_rnnt_model_before_loading_external_runtime() {
        let validate = |bytes| {
            let model = GGUFFileReader::new(Cursor::new(bytes)).unwrap();
            validate_model_catalog(model.metadata(), model.tensor_count())
        };
        validate(model("asr", "rnnt")).unwrap();
        assert!(validate(model("llama", "rnnt")).is_err());
        assert!(validate(model("asr", "ctc")).is_err());
    }
}
