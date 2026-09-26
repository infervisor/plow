//! The S3Gen stage of a Chatterbox pipeline: speech tokens -> 24 kHz waveform, by the native
//! `codec/libplow_s3gen.so` (runtime/nvidia/s3gen: conformer encoder, flow-matching decoder with
//! guidance, HiFT vocoder). Weights `codec/s3gen.bin`; one blob per voice under `codec/voices/`.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::Path;

use crate::{Result, RuntimeError};

pub const LIBRARY: &str = "codec/libplow_s3gen.so";
pub const WEIGHTS: &str = "codec/s3gen.bin";
pub const VOICES: &str = "codec/voices";

type Create = unsafe extern "C" fn(i32, *const i8, i32, i32, *mut *mut c_void) -> i32;
type AddVoice = unsafe extern "C" fn(*mut c_void, *const i8, *mut i32) -> i32;
type Synth = unsafe extern "C" fn(*mut c_void, i32, *const i32, i32, *mut f32, i32, u64, *mut i32) -> i32;

/// Owned by one thread (the handle is not thread-safe).
pub struct S3Gen {
    _lib: libloading::Library,
    h: usize,
    synth: Synth,
    voices: HashMap<String, i32>,
    max_tokens: usize,
}

/// Samples of 24 kHz audio per speech token (25 tokens/s).
pub const SAMPLES_PER_TOKEN: usize = 960;

impl S3Gen {
    pub fn load(assets: &Path, max_tokens: usize) -> Result<Self> {
        let lib_path = assets.join(LIBRARY);
        // SAFETY: the library implements the plow_s3gen C ABI (runtime/nvidia/s3gen/s3gen.cu).
        let lib = unsafe { libloading::Library::new(&lib_path) }
            .map_err(|e| RuntimeError::Device(format!("load {}: {e}", lib_path.display())))?;
        let sym = |e: libloading::Error| RuntimeError::Device(format!("{}: {e}", lib_path.display()));
        let (create, add_voice, synth) = unsafe {
            (
                *lib.get::<Create>(b"plow_s3gen_create\0").map_err(sym)?,
                *lib.get::<AddVoice>(b"plow_s3gen_add_voice\0").map_err(sym)?,
                *lib.get::<Synth>(b"plow_s3gen_synthesize_host\0").map_err(sym)?,
            )
        };
        let weights = assets.join(WEIGHTS);
        let w = CString::new(weights.to_string_lossy().as_bytes()).unwrap_or_default();
        let mut h = std::ptr::null_mut();
        let rc = unsafe { create(0, w.as_ptr(), 1, max_tokens as i32, &mut h) };
        if rc != 0 || h.is_null() {
            return Err(RuntimeError::Device(format!("plow_s3gen_create failed ({rc}) for {}", weights.display())));
        }
        let mut voices = HashMap::new();
        let vdir = assets.join(VOICES);
        for entry in std::fs::read_dir(&vdir).map_err(|source| RuntimeError::Io { path: vdir.clone(), source })? {
            let p = entry.map_err(|source| RuntimeError::Io { path: vdir.clone(), source })?.path();
            if p.extension().is_some_and(|e| e == "bin") {
                let c = CString::new(p.to_string_lossy().as_bytes()).unwrap_or_default();
                let mut id = -1;
                let rc = unsafe { add_voice(h, c.as_ptr(), &mut id) };
                if rc != 0 {
                    return Err(RuntimeError::Device(format!("plow_s3gen_add_voice {} failed ({rc})", p.display())));
                }
                voices.insert(p.file_stem().unwrap_or_default().to_string_lossy().into_owned(), id);
            }
        }
        Ok(S3Gen { _lib: lib, h: h as usize, synth, voices, max_tokens })
    }

    pub fn has_voice(&self, voice: &str) -> bool {
        self.voices.contains_key(voice)
    }

    /// Tokens (S3 ids, all < 6561) -> PCM f32 at 24 kHz.
    pub fn synthesize(&mut self, voice: &str, tokens: &[u32], seed: u64) -> Result<Vec<f32>> {
        let id = *self.voices.get(voice).ok_or_else(|| RuntimeError::Rejected(format!("unknown voice {voice:?}")))?;
        if tokens.is_empty() || tokens.len() > self.max_tokens {
            return Err(RuntimeError::Rejected(format!("{} speech tokens outside 1..={}", tokens.len(), self.max_tokens)));
        }
        let t: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        let cap = (tokens.len() + 8) * SAMPLES_PER_TOKEN;
        let mut out = vec![0f32; cap];
        let mut n = 0i32;
        // SAFETY: buffers sized as declared; the call is synchronous on the library's stream.
        let rc = unsafe { (self.synth)(self.h as *mut c_void, id, t.as_ptr(), t.len() as i32, out.as_mut_ptr(), cap as i32, seed, &mut n) };
        if rc != 0 {
            return Err(RuntimeError::Device(format!("plow_s3gen_synthesize_host failed ({rc})")));
        }
        out.truncate(n.max(0) as usize);
        Ok(out)
    }
}
