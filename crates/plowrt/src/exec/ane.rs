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
            // The compiled form carries the weights; the source doubles the disk footprint.
            let _ = std::fs::remove_file(&src);
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

// ---- graphs: one CoreML program per transformer layer (plans/apple-heterogeneous-emit.md §4) --

/// A layer of an [`NetSpec`]. Only rank-generic NeuralNetwork layers are used, so every blob is
/// a plain rank-2 `[T, C]` array under `EXACT_ARRAY_MAPPING`.
pub enum Layer {
    /// `y[T][N] = x[T][K] . W^T`, `W` fp16 `[N][K]`.
    InnerProduct {
        input: String,
        output: String,
        k: usize,
        n: usize,
        w_f16: Vec<u16>,
    },
    /// `y = x * rsqrt(mean_C(x^2) + eps)` — RMSNorm without gamma (gamma is folded into the
    /// consuming InnerProduct's weights). Two layers: reduceSumSquare(keepdims) + unary rsqrt
    /// with `scale = 1/C`, then a broadcast multiply.
    RmsNorm {
        input: String,
        output: String,
        c: usize,
        eps: f32,
    },
    Sigmoid {
        input: String,
        output: String,
    },
    /// tanh-approximation GELU.
    Gelu {
        input: String,
        output: String,
    },
    Mul {
        a: String,
        b: String,
        output: String,
    },
    Add {
        a: String,
        b: String,
        output: String,
    },
}

/// A CoreML NeuralNetwork program: named f32 inputs/outputs of rank 2 and a layer list. `t_enum`
/// lists every row count the program must accept (enumerated shapes: one compiled program, the
/// ANE specializes per shape); `shape[0]` of every input/output is the row axis.
pub struct NetSpec {
    pub inputs: Vec<(String, usize)>,
    pub outputs: Vec<(String, usize)>,
    pub t_enum: Vec<usize>,
    /// Declare the enumerated shapes on the outputs too (else only the inputs are flexible).
    pub flex_outputs: bool,
    /// Use a shape RANGE `[t_enum[0], t_enum.last()]` on the row axis instead of an enumeration.
    pub range: bool,
    /// Outputs use a shape range even when the inputs are enumerated.
    pub out_range: bool,
    pub layers: Vec<Layer>,
}

fn layer_msg(name: &str, inputs: &[&str], outputs: &[&str], field: u32, params: &[u8]) -> Vec<u8> {
    let mut l = Vec::new();
    f_str(&mut l, 1, name);
    for i in inputs {
        f_str(&mut l, 2, i);
    }
    for o in outputs {
        f_str(&mut l, 3, o);
    }
    f_msg(&mut l, field, params);
    l
}

/// The `.mlmodel` bytes of `spec`.
pub fn net_mlmodel(spec: &NetSpec) -> Vec<u8> {
    let feat = |name: &str, cols: usize, flex: bool, out: bool| -> Vec<u8> {
        let mut arr = Vec::new();
        packed_i64(&mut arr, 1, &[spec.t_enum[0] as i64, cols as i64]);
        f_varint(&mut arr, 2, 65568); // FLOAT32
        if flex && (spec.range || (out && spec.out_range)) && spec.t_enum.len() > 1 {
            // ArrayFeatureType.shapeRange (31): repeated SizeRange { uint64 lowerBound = 1; int64 upperBound = 2 }
            let mut sr = Vec::new();
            let mut r0 = Vec::new();
            f_varint(&mut r0, 1, spec.t_enum[0] as u64);
            f_varint(&mut r0, 2, *spec.t_enum.last().unwrap() as u64);
            f_msg(&mut sr, 1, &r0);
            let mut r1 = Vec::new();
            f_varint(&mut r1, 1, cols as u64);
            f_varint(&mut r1, 2, cols as u64);
            f_msg(&mut sr, 1, &r1);
            f_msg(&mut arr, 31, &sr);
        } else if flex && spec.t_enum.len() > 1 {
            // ArrayFeatureType.enumeratedShapes (21): repeated Shape { repeated int64 shape = 1 }
            let mut en = Vec::new();
            for &t in &spec.t_enum {
                let mut sh = Vec::new();
                packed_i64(&mut sh, 1, &[t as i64, cols as i64]);
                f_msg(&mut en, 1, &sh);
            }
            f_msg(&mut arr, 21, &en);
        }
        let mut ty = Vec::new();
        f_msg(&mut ty, 5, &arr);
        let mut fd = Vec::new();
        f_str(&mut fd, 1, name);
        f_msg(&mut fd, 3, &ty);
        fd
    };
    let mut desc = Vec::new();
    for (n, c) in &spec.inputs {
        f_msg(&mut desc, 1, &feat(n, *c, true, false));
    }
    for (n, c) in &spec.outputs {
        f_msg(&mut desc, 10, &feat(n, *c, spec.flex_outputs, true));
    }
    let mut nn = Vec::new();
    let push = |nn: &mut Vec<u8>, m: Vec<u8>| f_msg(nn, 1, &m);
    for (li, layer) in spec.layers.iter().enumerate() {
        let lname = format!("l{li}");
        match layer {
            Layer::InnerProduct {
                input,
                output,
                k,
                n,
                w_f16,
            } => {
                assert_eq!(w_f16.len(), n * k);
                let mut wbytes = Vec::with_capacity(n * k * 2);
                for &h in w_f16 {
                    wbytes.extend_from_slice(&h.to_le_bytes());
                }
                let mut wp = Vec::new();
                f_bytes(&mut wp, 2, &wbytes);
                let mut ip = Vec::new();
                f_varint(&mut ip, 1, *k as u64);
                f_varint(&mut ip, 2, *n as u64);
                f_varint(&mut ip, 10, 0);
                f_msg(&mut ip, 20, &wp);
                push(&mut nn, layer_msg(&lname, &[input], &[output], 140, &ip));
            }
            Layer::RmsNorm {
                input,
                output,
                c,
                eps,
            } => {
                let ss = format!("{lname}_ss");
                let rs = format!("{lname}_rs");
                // reduceSumSquare (1290): axes=[-1] keepDims=true
                let mut rp = Vec::new();
                packed_i64(&mut rp, 1, &[-1]);
                f_varint(&mut rp, 2, 1);
                push(
                    &mut nn,
                    layer_msg(&format!("{lname}a"), &[input], &[&ss], 1290, &rp),
                );
                // unary (220): RSQRT (type 1): y = 1/sqrt(scale*x + shift + epsilon)
                let mut up = Vec::new();
                f_varint(&mut up, 1, 1);
                key(&mut up, 3, 5);
                up.extend_from_slice(&eps.to_le_bytes());
                key(&mut up, 5, 5);
                up.extend_from_slice(&(1.0f32 / *c as f32).to_le_bytes());
                push(
                    &mut nn,
                    layer_msg(&format!("{lname}b"), &[&ss], &[&rs], 220, &up),
                );
                // multiplyBroadcastable (900): [T,C] * [T,1]
                push(
                    &mut nn,
                    layer_msg(&format!("{lname}c"), &[input, &rs], &[output], 900, &[]),
                );
            }
            Layer::Sigmoid { input, output } => {
                let mut ap = Vec::new();
                f_msg(&mut ap, 40, &[]); // ActivationParams.sigmoid
                push(&mut nn, layer_msg(&lname, &[input], &[output], 130, &ap));
            }
            Layer::Gelu { input, output } => {
                let mut gp = Vec::new();
                f_varint(&mut gp, 1, 1); // TANH_APPROXIMATION
                push(&mut nn, layer_msg(&lname, &[input], &[output], 795, &gp));
            }
            Layer::Mul { a, b, output } => {
                push(&mut nn, layer_msg(&lname, &[a, b], &[output], 900, &[]));
            }
            Layer::Add { a, b, output } => {
                push(&mut nn, layer_msg(&lname, &[a, b], &[output], 880, &[]));
            }
        }
    }
    f_varint(&mut nn, 5, 1); // EXACT_ARRAY_MAPPING
    let mut model = Vec::new();
    f_varint(&mut model, 1, 5);
    f_msg(&mut model, 2, &desc);
    f_msg(&mut model, 500, &nn);
    model
}

/// One compiled graph program.
pub struct AneNet {
    model: Retained<MLModel>,
    inputs: Vec<(String, usize)>,
    outputs: Vec<(String, usize)>,
    pub compile_ms: f64,
    pub last_ms: f64,
}

// SAFETY: as for `AneGemm`.
unsafe impl Send for AneNet {}

impl AneNet {
    pub fn cached(dir: &Path, name: &str) -> bool {
        dir.join(format!("{name}.mlmodelc")).exists()
    }

    /// Build (unless `<dir>/<name>.mlmodelc` exists), compile and load. `spec` is only called
    /// when the compiled form is missing, so the caller can defer the weight conversion.
    pub fn new(
        dir: &Path,
        name: &str,
        spec: impl FnOnce() -> NetSpec,
        io: (Vec<(String, usize)>, Vec<(String, usize)>),
        units: MLComputeUnits,
    ) -> Result<AneNet> {
        let t0 = Instant::now();
        std::fs::create_dir_all(dir)
            .map_err(|e| RuntimeError::Device(format!("ane: mkdir {}: {e}", dir.display())))?;
        let src: PathBuf = dir.join(format!("{name}.mlmodel"));
        let compiled: PathBuf = dir.join(format!("{name}.mlmodelc"));
        if !compiled.exists() {
            let bytes = net_mlmodel(&spec());
            std::fs::write(&src, bytes)
                .map_err(|e| RuntimeError::Device(format!("ane: write {}: {e}", src.display())))?;
            let out = unsafe {
                MLModel::compileModelAtURL_error(&NSURL::fileURLWithPath(&NSString::from_str(
                    src.to_str().unwrap(),
                )))
            }
            .map_err(|e| RuntimeError::Device(format!("ane: compile {name}: {e}")))?;
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
            let _ = std::fs::remove_file(&src);
        }
        let cfg = unsafe { MLModelConfiguration::init(MLModelConfiguration::alloc()) };
        unsafe { cfg.setComputeUnits(units) };
        let url = NSURL::fileURLWithPath(&NSString::from_str(compiled.to_str().unwrap()));
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg) }
            .map_err(|e| RuntimeError::Device(format!("ane: load {name}: {e}")))?;
        Ok(AneNet {
            model,
            inputs: io.0,
            outputs: io.1,
            compile_ms: t0.elapsed().as_secs_f64() * 1e3,
            last_ms: 0.0,
        })
    }

    /// Run with `t` rows: `inputs[i]` holds `t * cols_i` f32 (used in place), `outputs[i]`
    /// receives `t * cols_i` f32.
    pub fn run(&mut self, t: usize, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
        assert_eq!(inputs.len(), self.inputs.len());
        assert_eq!(outputs.len(), self.outputs.len());
        let t0 = Instant::now();
        let mut keys: Vec<Retained<NSString>> = Vec::with_capacity(inputs.len());
        let mut vals: Vec<Retained<objc2::runtime::AnyObject>> = Vec::with_capacity(inputs.len());
        for (i, x) in inputs.iter().enumerate() {
            let cols = self.inputs[i].1;
            assert_eq!(x.len(), t * cols, "input {} rows", self.inputs[i].0);
            let shape =
                NSArray::from_retained_slice(&[NSNumber::new_usize(t), NSNumber::new_usize(cols)]);
            let strides =
                NSArray::from_retained_slice(&[NSNumber::new_usize(cols), NSNumber::new_usize(1)]);
            // SAFETY: `x` outlives the synchronous prediction; no deallocator.
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
            keys.push(NSString::from_str(&self.inputs[i].0));
            vals.push(Retained::into_super(Retained::into_super(fv)));
        }
        let krefs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
        let dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::from_retained_objects(&krefs, &vals);
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
        for (i, y) in outputs.iter_mut().enumerate() {
            let (name, cols) = &self.outputs[i];
            let yv = unsafe { out.featureValueForName(&NSString::from_str(name)) }
                .and_then(|v| unsafe { v.multiArrayValue() })
                .ok_or_else(|| RuntimeError::Device(format!("ane: no output {name}")))?;
            let want = t * cols;
            // SAFETY: the output array holds `count` f32 (FLOAT32 by the model description).
            unsafe {
                let n = yv.count() as usize;
                if n < want {
                    return Err(RuntimeError::Device(format!(
                        "ane: output {name} has {n} elements, expected {want}"
                    )));
                }
                let p = yv.dataPointer().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(p, y.as_mut_ptr(), want);
            }
        }
        self.last_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }
}
