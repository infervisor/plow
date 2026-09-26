//! The codec stage of a speech pipeline: the native SNAC-24k decoder (`codec/libplow_snac.so`,
//! built from runtime/nvidia/snac by `plowc --emit devblob+cubin --tts-profile ...`) driven by one
//! worker thread. Jobs with the same frame count are decoded in ONE call, so concurrent streams
//! share each codec launch the way they share LM decode steps.

use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use parking_lot::Mutex;

pub const LIBRARY: &str = "codec/libplow_snac.so";
pub const WEIGHTS: &str = "codec/snac24k.bin";
const CODES: usize = 7;
const SAMPLES: usize = 2048;

type Create = unsafe extern "C" fn(i32, *const i8, i32, i32, *mut *mut c_void) -> i32;
type DecodeHost = unsafe extern "C" fn(*mut c_void, *const i32, i32, i32, *mut f32, u64) -> i32;

struct Job {
    codes: Vec<i32>,
    frames: usize,
    seed: u64,
    reply: tokio::sync::oneshot::Sender<Result<Vec<f32>, String>>,
}

pub struct Codec {
    tx: Mutex<mpsc::Sender<Job>>,
    pub max_frames: usize,
}

impl Codec {
    /// Loads the codec object and weights shipped in the asset directory.
    pub fn load(assets: &Path, max_batch: usize, max_frames: usize) -> Result<Self, String> {
        let (library, weights) = (assets.join(LIBRARY), assets.join(WEIGHTS));
        if !weights.is_file() {
            return Err(format!(
                "{} missing: export it with scripts/tts/snac_prep.py (see docs/runtime/tts.md)",
                weights.display()
            ));
        }
        Self::start(library, weights, max_batch, max_frames)
    }

    fn start(library: PathBuf, weights: PathBuf, max_batch: usize, max_frames: usize) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("plow-tts-codec".into())
            .spawn(move || {
                // SAFETY: the library implements the plow_snac C ABI (runtime/nvidia/snac/snac.cu).
                let lib = match unsafe { libloading::Library::new(&library) } {
                    Ok(l) => l,
                    Err(e) => return drop(ready_tx.send(Err(format!("load {}: {e}", library.display())))),
                };
                let (create, decode) = match unsafe {
                    (lib.get::<Create>(b"plow_snac_create\0"), lib.get::<DecodeHost>(b"plow_snac_decode_host\0"))
                } {
                    (Ok(c), Ok(d)) => (*c, *d),
                    _ => return drop(ready_tx.send(Err("codec object lacks plow_snac_create/decode_host".into()))),
                };
                let path = CString::new(weights.to_string_lossy().as_bytes()).unwrap_or_default();
                let mut h = std::ptr::null_mut();
                // Device 0 of the visible set: the device the engine serves on.
                let rc = unsafe { create(0, path.as_ptr(), max_batch as i32, max_frames as i32, &mut h) };
                if rc != 0 || h.is_null() {
                    return drop(ready_tx.send(Err(format!("plow_snac_create failed ({rc})"))));
                }
                let _ = ready_tx.send(Ok(()));
                run(rx, h, decode, max_batch, max_frames);
                drop(lib);
            })
            .map_err(|e| e.to_string())?;
        ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Codec { tx: Mutex::new(tx), max_frames })
    }

    /// `frames * 7` codebook ids -> `frames * 2048` samples.
    pub async fn decode(&self, codes: Vec<i32>, frames: usize, seed: u64) -> Result<Vec<f32>, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .lock()
            .send(Job { codes, frames, seed, reply })
            .map_err(|_| "codec worker stopped".to_string())?;
        rx.await.map_err(|_| "codec worker dropped the job".to_string())?
    }
}

fn run(rx: mpsc::Receiver<Job>, h: *mut c_void, decode: DecodeHost, max_batch: usize, max_frames: usize) {
    let mut pending: Vec<Job> = Vec::new();
    while let Ok(first) = rx.recv() {
        pending.push(first);
        pending.extend(rx.try_iter());
        while !pending.is_empty() {
            let f = pending[0].frames;
            let (mut batch, mut rest) = (Vec::new(), Vec::new());
            for j in pending.drain(..) {
                if j.frames == f && batch.len() < max_batch {
                    batch.push(j);
                } else {
                    rest.push(j);
                }
            }
            pending = rest;
            if f == 0 || f > max_frames {
                for j in batch {
                    let _ = j.reply.send(Err(format!("{f} frames outside 1..={max_frames}")));
                }
                continue;
            }
            let b = batch.len();
            let codes: Vec<i32> = batch.iter().flat_map(|j| j.codes.iter().copied()).collect();
            debug_assert_eq!(codes.len(), b * f * CODES);
            let mut pcm = vec![0f32; b * f * SAMPLES];
            // SAFETY: buffers are [b][f][7] and [b][f*2048]; the call is synchronous.
            let rc = unsafe { decode(h, codes.as_ptr(), b as i32, f as i32, pcm.as_mut_ptr(), batch[0].seed) };
            for (i, j) in batch.into_iter().enumerate() {
                let _ = j.reply.send(if rc == 0 {
                    Ok(pcm[i * f * SAMPLES..(i + 1) * f * SAMPLES].to_vec())
                } else {
                    Err(format!("plow_snac_decode_host failed ({rc})"))
                });
            }
        }
    }
}
