//! Apple Neural Engine executor (plans/apple-silicon-backend.md §4.5, rung 4).
//!
//! The ANE is reachable only through CoreML, so an "ANE kernel" is a CoreML model: static
//! shapes, weights baked in, one asynchronous call. This module builds such a model for a dense
//! GEMM (`y[T][N] = x[T][K] . W[N][K]^T`) directly as CoreML protobuf (the NeuralNetwork
//! `innerProduct` layer, fp16 weights; no Python, no protoc), compiles it with the system
//! compiler, loads it with `MLComputeUnits::CPUAndNeuralEngine`, and runs it on a caller-owned
//! f32 buffer without copying the input.
//!
//! Where it sits in the walk: the Metal engine runs the ANE op at a segment boundary of the
//! persistent walk (Event mode, §4.3): dispatch everything before it, run the CoreML program on
//! the host thread, bump the op's successor counters, dispatch the rest.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AllocAnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

use crate::{Result, RuntimeError};

// ---- minimal protobuf writer (proto3 wire format) ---------------------------------------------
fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}
fn key(out: &mut Vec<u8>, field: u32, wire: u8) {
    varint(out, ((field as u64) << 3) | wire as u64);
}
fn f_varint(out: &mut Vec<u8>, field: u32, v: u64) {
    key(out, field, 0);
    varint(out, v);
}
fn f_bytes(out: &mut Vec<u8>, field: u32, b: &[u8]) {
    key(out, field, 2);
    varint(out, b.len() as u64);
    out.extend_from_slice(b);
}
fn f_str(out: &mut Vec<u8>, field: u32, s: &str) {
    f_bytes(out, field, s.as_bytes());
}
fn f_msg(out: &mut Vec<u8>, field: u32, m: &[u8]) {
    f_bytes(out, field, m);
}
fn packed_i64(out: &mut Vec<u8>, field: u32, vals: &[i64]) {
    let mut p = Vec::new();
    for &v in vals {
        varint(&mut p, v as u64);
    }
    f_bytes(out, field, &p);
}

/// f32 -> IEEE half (RNE), the ANE's native weight and activation type.
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e);
        let mut h = (m >> 13) as u16;
        let rem = m & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10) as u16 | (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h = h.wrapping_add(1);
    }
    sign | h
}

/// CoreML `.mlmodel` bytes for `y[T][N] = x[T][K] . W^T` with `W` as fp16 `[N][K]` row-major.
/// I/O is f32 `[T, K]` -> `[T, N]` (exact array mapping); CoreML converts to fp16 on the ANE.
pub fn gemm_mlmodel(t: usize, k: usize, n: usize, w_f16: &[u16]) -> Vec<u8> {
    assert_eq!(w_f16.len(), n * k);
    let mut wbytes = Vec::with_capacity(n * k * 2);
    for &h in w_f16 {
        wbytes.extend_from_slice(&h.to_le_bytes());
    }
    let feat = |name: &str, shape: &[i64]| -> Vec<u8> {
        let mut arr = Vec::new();
        packed_i64(&mut arr, 1, shape);
        f_varint(&mut arr, 2, 65568); // FLOAT32
        let mut ty = Vec::new();
        f_msg(&mut ty, 5, &arr); // multiArrayType
        let mut fd = Vec::new();
        f_str(&mut fd, 1, name);
        f_msg(&mut fd, 3, &ty);
        fd
    };
    let mut desc = Vec::new();
    f_msg(&mut desc, 1, &feat("x", &[t as i64, k as i64]));
    f_msg(&mut desc, 10, &feat("y", &[t as i64, n as i64]));

    let mut wp = Vec::new();
    f_bytes(&mut wp, 2, &wbytes); // float16Value
    let mut ip = Vec::new();
    f_varint(&mut ip, 1, k as u64);
    f_varint(&mut ip, 2, n as u64);
    f_varint(&mut ip, 10, 0);
    f_msg(&mut ip, 20, &wp);
    let mut layer = Vec::new();
    f_str(&mut layer, 1, "gemm");
    f_str(&mut layer, 2, "x");
    f_str(&mut layer, 3, "y");
    f_msg(&mut layer, 140, &ip); // innerProduct
    let mut nn = Vec::new();
    f_msg(&mut nn, 1, &layer);
    f_varint(&mut nn, 5, 1); // EXACT_ARRAY_MAPPING

    let mut model = Vec::new();
    f_varint(&mut model, 1, 5); // specificationVersion
    f_msg(&mut model, 2, &desc);
    f_msg(&mut model, 500, &nn);
    model
}

/// One compiled GEMM program resident on the ANE.
pub struct AneGemm {
    model: Retained<MLModel>,
    pub t: usize,
    pub k: usize,
    pub n: usize,
    pub compile_ms: f64,
    pub last_ms: f64,
}

// SAFETY: the engine drives the ANE from one thread at a time; MLModel itself is thread-safe.
unsafe impl Send for AneGemm {}

impl AneGemm {
    /// Build, compile, and load. `dir` holds the `.mlmodel` and its compiled form (cached by name).
    pub fn new(
        dir: &Path,
        name: &str,
        t: usize,
        k: usize,
        n: usize,
        w_f16: &[u16],
        units: MLComputeUnits,
    ) -> Result<AneGemm> {
        let t0 = Instant::now();
        std::fs::create_dir_all(dir)
            .map_err(|e| RuntimeError::Device(format!("ane: mkdir {}: {e}", dir.display())))?;
        let src: PathBuf = dir.join(format!("{name}.mlmodel"));
        let compiled: PathBuf = dir.join(format!("{name}.mlmodelc"));
        if !compiled.exists() {
            std::fs::write(&src, gemm_mlmodel(t, k, n, w_f16))
                .map_err(|e| RuntimeError::Device(format!("ane: write {}: {e}", src.display())))?;
            // SAFETY: plain Foundation/CoreML calls with owned arguments.
            let out = unsafe {
                MLModel::compileModelAtURL_error(&NSURL::fileURLWithPath(&NSString::from_str(
                    src.to_str().unwrap(),
                )))
            }
            .map_err(|e| RuntimeError::Device(format!("ane: compile: {e}")))?;
            let out_path = unsafe { out.path() }
                .map(|p| p.to_string())
                .unwrap_or_default();
            let _ = std::fs::remove_dir_all(&compiled);
            std::fs::rename(&out_path, &compiled).map_err(|e| {
                RuntimeError::Device(format!(
                    "ane: move {out_path} -> {}: {e}",
                    compiled.display()
                ))
            })?;
        }
        let cfg = unsafe { MLModelConfiguration::init(MLModelConfiguration::alloc()) };
        unsafe { cfg.setComputeUnits(units) };
        let url = NSURL::fileURLWithPath(&NSString::from_str(compiled.to_str().unwrap()));
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg) }
            .map_err(|e| RuntimeError::Device(format!("ane: load: {e}")))?;
        Ok(AneGemm {
            model,
            t,
            k,
            n,
            compile_ms: t0.elapsed().as_secs_f64() * 1e3,
            last_ms: 0.0,
        })
    }

    /// `y[T][N] = x[T][K] . W^T`, f32 in and out; `x` is used in place (no copy), `y` is copied out.
    pub fn run(&mut self, x: &[f32], y: &mut [f32]) -> Result<()> {
        assert_eq!(x.len(), self.t * self.k);
        assert_eq!(y.len(), self.t * self.n);
        let t0 = Instant::now();
        let shape = NSArray::from_retained_slice(&[
            NSNumber::new_usize(self.t),
            NSNumber::new_usize(self.k),
        ]);
        let strides =
            NSArray::from_retained_slice(&[NSNumber::new_usize(self.k), NSNumber::new_usize(1)]);
        // SAFETY: `x` outlives the prediction (synchronous), no deallocator.
        let arr = unsafe {
            MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                MLMultiArray::alloc(),
                NonNull::new(x.as_ptr() as *mut c_void).unwrap(),
                &shape,
                MLMultiArrayDataType::Float32,
                &strides,
                None,
            )
        }
        .map_err(|e| RuntimeError::Device(format!("ane: input array: {e}")))?;
        let fv = unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) };
        let dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::from_retained_objects(
                &[&*NSString::from_str("x")],
                &[Retained::into_super(Retained::into_super(fv))],
            );
        let prov = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| RuntimeError::Device(format!("ane: provider: {e}")))?;
        let out = unsafe {
            self.model
                .predictionFromFeatures_error(ProtocolObject::from_ref(&*prov))
        }
        .map_err(|e| RuntimeError::Device(format!("ane: predict: {e}")))?;
        let yv = unsafe { out.featureValueForName(&NSString::from_str("y")) }
            .and_then(|v| unsafe { v.multiArrayValue() })
            .ok_or_else(|| RuntimeError::Device("ane: no output y".into()))?;
        // SAFETY: the output array holds T*N f32 (its dataType is FLOAT32 by the model description).
        unsafe {
            let p = yv.dataPointer().as_ptr() as *const f32;
            let n = (yv.count() as usize).min(y.len());
            std::ptr::copy_nonoverlapping(p, y.as_mut_ptr(), n);
        }
        self.last_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }
}

/// Feature provider protocol object type, re-exported for callers that build their own.
pub type FeatureProvider = ProtocolObject<dyn MLFeatureProvider>;
