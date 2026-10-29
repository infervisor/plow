use super::aot::{Exemplar, Image, Kernel, Launch};
use super::{CudaBackend, CudaStream};
use crate::{
    device::{Backend, DeviceMem},
    Result, RuntimeError,
};
use std::{path::Path, sync::Arc};

const MAX_TOKENS: usize = 8192;
const MAP_BYTES: u64 = 16896;
/// Per-sequence recurrent state `[HV=48, V=128, K=128]` f32, the library's fixed geometry.
pub(crate) const STATE_BYTES: u64 = 48 * 128 * 128 * 4;
/// Bytes per row of the conv core tensors (`validate_n` sizes 2 and 3).
pub(crate) const CORE_ROW_BYTES: u64 = 12288;
/// Packed sequences one launch takes (cu_seqlens entries - 1).
pub(crate) const MAX_SEQS: usize = 64;
/// The packet object (`scripts/build_sm90a_qwen_gdn.py`) and the host stub this file implements.
pub(crate) const OBJECT: &str = "plow_gdn_prefill.aot";
const ENTRY: &str = "gdn_prefill";
const HOST_STUB: &str = "gdn_sm90_prefill";
const HOST_STUB_VERSION: u32 = 1;
const ABI: &str = "plow_gdn_prefill:v1:hk=16:hv=48:d=128:dtype=bf16:state=f32_vk";
const UPSTREAM: [(&str, &str); 3] = [
    (
        "delta_rule_sm90_sha256",
        "c72398f8f6e57ed2d708d364dd0299ce414e75a27d044ac83c6278c0954290a4",
    ),
    ("flashinfer", "0.6.16.post3"),
    ("nvidia_cutlass_dsl", "4.6.2"),
];
/// cuda.h enums the export's tensor maps use.
/// Bytes per parameter the export's host code defines (the builder's `DEFINED`): an exemplar
/// masking any of them is refused, so the check cannot be weakened by the manifest.
const DEFINED: [usize; 22] = [
    12, 12, 64, 4, 64, 4, 64, 4, 64, 4, 12, 12, 8, 12, 4, 4, 4, 4, 4, 4, 4, 4,
];
/// Tensor-map bits the export's own encoder sets that `cuTensorMapEncodeTiled` (driver 595) does
/// not: byte 8 bit 1 and byte 10 bit 5 (below one 64-row box the export leaves the latter to
/// uninitialized stack). The exemplar check ignores exactly these; everything else must match.
const ENCODER_ONLY_BITS: [(usize, u8); 2] = [(8, 0x02), (10, 0x20)];
const TMAP_PARAMS: [usize; 4] = [2, 4, 6, 8];
const BFLOAT16: u32 = 9;
const SWIZZLE_32B: u32 = 1;
const SWIZZLE_128B: u32 = 3;
const L2_PROMOTION_128B: u32 = 2;

/// Device addresses of one launch, in the export's argument order.
#[derive(Clone, Copy)]
struct Operands {
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    alpha: u64,
    beta: u64,
    state: u64,
    initial: u64,
    maps: u64,
    offsets: u64,
}

/// The export's host code for `FullyFusedDeltaRuleSm90` (cute_dsl_plow_gdn_sm90_wrapper):
/// Q/K `[tokens][16][128]` and V/O `[tokens][48][128]` bf16 tensor maps, `nseq * 48` CTAs.
fn build(be: &CudaBackend, l: &mut Launch, tokens: u64, nseq: u64, p: &Operands) -> Result<()> {
    build_with(
        &|base, heads, row, box0, swizzle| {
            be.encode_tmap_tiled(
                BFLOAT16,
                base,
                &[128, tokens, heads],
                &[row, 256],
                &[box0, 64, 1],
                &[1, 1, 1],
                0,
                swizzle,
                L2_PROMOTION_128B,
                0,
            )
        },
        l,
        tokens,
        nseq,
        p,
    )
}

type MapFn<'a> = &'a dyn Fn(u64, u64, u64, u32, u32) -> Result<[u8; 128]>;

/// `build` with the tensor-map encoder `(base, heads, row bytes, box dim 0, swizzle)` supplied.
fn build_with(map: MapFn, l: &mut Launch, tokens: u64, nseq: u64, p: &Operands) -> Result<()> {
    let gate = (tokens * 48) as u32;
    let state = (nseq * 48 * 128 * 128) as u32;
    let ptr_len = |l: &mut Launch, i: usize, ptr: u64, n: u32| {
        l.u64(i, 0, ptr);
        l.u32(i, 8, n);
        l.u32(i, 12, 0);
    };
    ptr_len(l, 0, p.alpha, gate);
    ptr_len(l, 1, p.beta, gate);
    l.tmap(2, &map(p.q, 16, 4096, 64, SWIZZLE_128B)?);
    l.u64(3, 0, tokens);
    l.tmap(4, &map(p.k, 16, 4096, 64, SWIZZLE_128B)?);
    l.u64(5, 0, tokens);
    l.tmap(6, &map(p.v, 48, 12288, 64, SWIZZLE_128B)?);
    l.u64(7, 0, tokens);
    l.tmap(8, &map(p.o, 48, 12288, 16, SWIZZLE_32B)?);
    l.u64(9, 0, tokens);
    ptr_len(l, 10, p.state, state);
    ptr_len(l, 11, p.initial, state);
    l.u64(12, 0, p.maps);
    ptr_len(l, 13, p.offsets, (nseq + 1) as u32);
    // 128^-0.5 rounded to f32, the export's `scale` literal.
    l.u32(14, 0, 0x3db5_04f3);
    for (i, v) in [
        (15, 16),
        (16, 16),
        (17, 48),
        (18, 48),
        (19, nseq as u32),
        (20, 1),
        (21, 0),
    ] {
        l.u32(i, 0, v);
    }
    l.grid = [(nseq * 48) as u32, 1, 1];
    l.block = [512, 1, 1];
    l.smem = 185728;
    Ok(())
}

fn rejected(why: String) -> RuntimeError {
    RuntimeError::Rejected(format!("GDN prefill object: {why}"))
}

/// Loads the kernel and proves the host stub: manifest identity, then every exemplar rebuilt.
fn load_kernel(be: &CudaBackend, path: &Path) -> Result<Kernel> {
    let bytes = std::fs::read(path).map_err(|e| {
        rejected(format!(
            "{}: {e} (packets from before the cubin route carry libplow_gdn_prefill.so; re-emit)",
            path.display()
        ))
    })?;
    let image = Image::parse(bytes)?;
    let m = &image.manifest;
    let upstream: Vec<(&str, &str)> = m
        .upstream
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    if m.host_stub != HOST_STUB
        || m.host_stub_version != HOST_STUB_VERSION
        || m.abi != ABI
        || upstream != UPSTREAM
    {
        return Err(rejected(format!(
            "built for host stub {} v{} abi {} upstream {upstream:?}; this plowrt implements \
             {HOST_STUB} v{HOST_STUB_VERSION} abi {ABI} upstream {UPSTREAM:?}",
            m.host_stub, m.host_stub_version, m.abi
        )));
    }
    let entry = image.entry(ENTRY)?;
    let kernel = Kernel::load(be, &image, entry)?;
    for x in &entry.exemplars {
        let (tokens, nseq, p) = exemplar_operands(x)?;
        if x.params.len() != DEFINED.len()
            || x.params
                .iter()
                .zip(DEFINED)
                .any(|(h, d)| h.get(..2 * d).is_none_or(|h| h.contains("..")))
        {
            return Err(rejected(format!(
                "exemplar tokens={tokens} nseq={nseq} masks a defined parameter byte"
            )));
        }
        let mut l = Launch::new(&kernel.layout);
        build(be, &mut l, tokens, nseq, &p)?;
        l.matches(&driver_view(x))
            .map_err(|e| rejected(format!("exemplar tokens={tokens} nseq={nseq}: {e}")))?;
    }
    Ok(kernel)
}

/// `x` with [`ENCODER_ONLY_BITS`] cleared in its captured tensor maps.
fn driver_view(x: &Exemplar) -> Exemplar {
    let mut params = x.params.clone();
    for i in TMAP_PARAMS {
        for (b, bits) in ENCODER_ONLY_BITS {
            let Some(cap) = params[i].get(2 * b..2 * b + 2) else {
                continue;
            };
            if let Ok(v) = u8::from_str_radix(cap, 16) {
                params[i].replace_range(2 * b..2 * b + 2, &format!("{:02x}", v & !bits));
            }
        }
    }
    Exemplar {
        args: x.args.clone(),
        grid: x.grid,
        block: x.block,
        smem: x.smem,
        attrs: x.attrs.clone(),
        kernel_attrs: x.kernel_attrs.clone(),
        params,
    }
}

fn exemplar_operands(x: &Exemplar) -> Result<(u64, u64, Operands)> {
    let &[tokens, nseq, q, k, v, o, alpha, beta, state, initial, maps, offsets] = x.args.as_slice()
    else {
        return Err(rejected(format!("exemplar args {:?}", x.args)));
    };
    let a = |v: i64| v as u64;
    Ok((
        a(tokens),
        a(nseq),
        Operands {
            q: a(q),
            k: a(k),
            v: a(v),
            o: a(o),
            alpha: a(alpha),
            beta: a(beta),
            state: a(state),
            initial: a(initial),
            maps: a(maps),
            offsets: a(offsets),
        },
    ))
}

pub(crate) struct NativeGdn {
    kernel: Kernel,
    launch: Launch,
    maps: DeviceMem,
    /// `{0, tokens}` cu_seqlens per single-sequence token count.
    offsets: DeviceMem,
    /// cu_seqlens of the current packed launch (device, MAX_SEQS + 1 i64) and its host image,
    /// which the async upload reads until the caller's stream synchronize.
    cu: DeviceMem,
    cu_host: Vec<i64>,
    stream: Option<usize>,
    backend: Arc<CudaBackend>,
}

impl NativeGdn {
    pub(crate) fn load(backend: Arc<CudaBackend>, path: &Path) -> Result<Self> {
        backend.bind()?;
        let kernel = load_kernel(&backend, path)?;
        let launch = Launch::new(&kernel.layout);
        let cu = backend.alloc(backend.device_ordinal, ((MAX_SEQS + 1) * 8) as u64)?;
        let maps = backend.alloc(backend.device_ordinal, MAP_BYTES)?;
        if maps.base % 128 != 0 {
            return Err(RuntimeError::Device(
                "native GDN maps require 128-byte alignment".into(),
            ));
        }
        let offsets = backend.alloc(backend.device_ordinal, (MAX_TOKENS * 16) as u64)?;
        let mut pairs = Vec::with_capacity(MAX_TOKENS * 16);
        for tokens in 1..=MAX_TOKENS {
            pairs.extend_from_slice(&0_i64.to_le_bytes());
            pairs.extend_from_slice(&(tokens as i64).to_le_bytes());
        }
        backend.memcpy_htod(offsets.base, &pairs)?;
        Ok(Self {
            kernel,
            launch,
            maps,
            offsets,
            cu,
            cu_host: Vec::with_capacity(MAX_SEQS + 1),
            stream: None,
            backend,
        })
    }

    fn enqueue(
        &mut self,
        tensors: &[&DeviceMem; 8],
        tokens: usize,
        nseq: usize,
        offsets: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        if stream.keep.ctx != self.backend.ctx || self.stream.is_some_and(|s| s != stream.raw) {
            return Err(RuntimeError::Device(
                "native GDN requires one stream in its selected context".into(),
            ));
        }
        self.backend.bind()?;
        self.stream = Some(stream.raw);
        let p = Operands {
            q: tensors[0].base,
            k: tensors[1].base,
            v: tensors[2].base,
            o: tensors[3].base,
            alpha: tensors[4].base,
            beta: tensors[5].base,
            state: tensors[6].base,
            initial: tensors[7].base,
            maps: self.maps.base,
            offsets,
        };
        build(
            &self.backend,
            &mut self.launch,
            tokens as u64,
            nseq as u64,
            &p,
        )?;
        self.kernel.launch(&self.backend, &mut self.launch, stream)
    }

    /// Enqueues BF16 Q/K/V/output, F32 alpha/beta, and F32 V-first states, then copies the
    /// output state back to the initial state on this stream.
    ///
    /// # Safety
    /// Views must remain live in this context with the stated layouts until stream
    /// completion. No other stream may race writable tensors. Keep this stream
    /// alive until the adapter is dropped.
    pub(crate) unsafe fn launch(
        &mut self,
        tensors: [&DeviceMem; 8],
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        validate(&tensors, tokens)?;
        let offsets = self.offsets.base + ((tokens - 1) * 16) as u64;
        self.enqueue(&tensors, tokens, 1, offsets, stream)?;
        self.backend
            .memcpy_dtod_async(tensors[7].base, tensors[6].base, STATE_BYTES, stream)
    }
}

fn validate(tensors: &[&DeviceMem; 8], tokens: usize) -> Result<()> {
    validate_n(tensors, tokens, 1)
}

/// `tensors[6]` / `[7]` hold `nseq` contiguous state slots.
fn validate_n(tensors: &[&DeviceMem; 8], tokens: usize, nseq: usize) -> Result<()> {
    if !(1..=MAX_TOKENS).contains(&tokens) {
        return Err(RuntimeError::Device(
            "native GDN tokens must be in 1..=8192".into(),
        ));
    }
    if !(1..=MAX_SEQS).contains(&nseq) || nseq > tokens {
        return Err(RuntimeError::Device(
            "native GDN packed sequences must be in 1..=64 and non-empty".into(),
        ));
    }
    let t = tokens as u64;
    let sizes = [
        t * 4096,
        t * 4096,
        t * 12288,
        t * 12288,
        t * 192,
        t * 192,
        nseq as u64 * STATE_BYTES,
        nseq as u64 * STATE_BYTES,
    ];
    let mut ends = [0; 8];
    for (i, tensor) in tensors.iter().enumerate() {
        let alignment = if i == 4 || i == 5 { 4 } else { 16 };
        if tensor.base == 0 || tensor.base % alignment != 0 || tensor.len < sizes[i] {
            return Err(RuntimeError::Device(format!(
                "native GDN tensor {i}: invalid extent/alignment"
            )));
        }
        ends[i] = tensor
            .base
            .checked_add(sizes[i])
            .ok_or_else(|| RuntimeError::Device("native GDN address overflow".into()))?;
    }
    for i in [3, 6, 7] {
        for j in 0..8 {
            if i != j && tensors[i].base < ends[j] && tensors[j].base < ends[i] {
                return Err(RuntimeError::Device(
                    "native GDN writable tensor aliases another operand".into(),
                ));
            }
        }
    }
    Ok(())
}

impl NativeGdn {
    /// `lens.len()` packed sequences, rows contiguous in `lens` order; `tensors[6]` / `[7]` are
    /// `lens.len()` contiguous final / initial state slots (the caller gathers and scatters the
    /// per-slot states; nothing is copied back here).
    ///
    /// # Safety
    /// As [`Self::launch`]; the caller synchronizes the stream before the next packed launch
    /// (the cu_seqlens upload reads a host buffer owned by the adapter).
    pub(crate) unsafe fn launch_varlen(
        &mut self,
        tensors: [&DeviceMem; 8],
        lens: &[u32],
        stream: &CudaStream,
    ) -> Result<()> {
        if lens.iter().any(|&l| l == 0) {
            return Err(RuntimeError::Device(
                "native GDN packed sequence of 0 rows".into(),
            ));
        }
        let tokens: usize = lens.iter().map(|&l| l as usize).sum();
        validate_n(&tensors, tokens, lens.len())?;
        if stream.keep.ctx != self.backend.ctx || self.stream.is_some_and(|s| s != stream.raw) {
            return Err(RuntimeError::Device(
                "native GDN requires one stream in its selected context".into(),
            ));
        }
        self.backend.bind()?;
        self.cu_host.clear();
        self.cu_host.push(0);
        let mut end = 0i64;
        for &l in lens {
            end += i64::from(l);
            self.cu_host.push(end);
        }
        // SAFETY: cu_host lives on self until the caller's stream synchronize.
        unsafe {
            self.backend.memcpy_htod_async(
                self.cu.base,
                bytemuck::cast_slice(&self.cu_host),
                stream,
            )?;
        }
        let cu = self.cu.base;
        self.enqueue(&tensors, tokens, lens.len(), cu, stream)
    }
}

impl Drop for NativeGdn {
    fn drop(&mut self) {
        // Destruction is cold; retire queued work before the module unloads.
        if self.backend.bind().is_ok() {
            let _ = self.backend.synchronize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensors() -> [DeviceMem; 8] {
        std::array::from_fn(|i| DeviceMem::view((i as u64 + 1) << 30, 128 << 20))
    }

    /// CPU: the stub rebuilds every exemplar's non-tensor-map bytes and config (tensor maps are
    /// taken from the capture; `layer_dump` on a GPU checks the driver's encoding at load).
    #[test]
    #[ignore = "GDN_AOT_DIR=<dir holding plow_gdn_prefill.aot>"]
    fn stub_rebuilds_exemplar_scalars() {
        let dir = std::path::PathBuf::from(std::env::var("GDN_AOT_DIR").expect("GDN_AOT_DIR"));
        let image = Image::parse(std::fs::read(dir.join(OBJECT)).unwrap()).unwrap();
        let e = image.entry(ENTRY).unwrap();
        let layout: Vec<(usize, usize)> = e
            .params
            .iter()
            .map(|&[o, s]| (o as usize, s as usize))
            .collect();
        assert_eq!(e.exemplars.len(), 3);
        for x in &e.exemplars {
            let (tokens, nseq, p) = exemplar_operands(x).unwrap();
            let captured = |i: usize| -> [u8; 128] {
                std::array::from_fn(|b| {
                    u8::from_str_radix(&x.params[i][2 * b..2 * b + 2], 16).unwrap_or(0)
                })
            };
            let map = |base: u64, _: u64, _: u64, _: u32, _: u32| {
                let i = [p.q, p.k, p.v, p.o]
                    .iter()
                    .position(|&b| b == base)
                    .unwrap();
                Ok(captured(2 + 2 * i))
            };
            let mut l = Launch::new(&layout);
            build_with(&map, &mut l, tokens, nseq, &p).unwrap();
            l.matches(x).unwrap();
        }
    }

    /// Writes each case's inputs (`<case>.in<i>`) and outputs (`<case>.out<i>`: o, final state,
    /// initial state after the launch) under `GDN_AOT_DIR` for a byte comparison against another
    /// route, and prints the mean launch time over 20 back-to-back launches.
    #[test]
    #[ignore = "requires a leased SM90 GPU; GDN_AOT_DIR=<dir holding plow_gdn_prefill.aot>"]
    fn layer_dump() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("GDN_AOT_DIR").expect("GDN_AOT_DIR"));
        let be = Arc::new(CudaBackend::new(0)?);
        let mut g = NativeGdn::load(be.clone(), &dir.join(OBJECT))?;
        let stream = be.stream_create()?;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |lo: f32, hi: f32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            lo + (hi - lo) * ((seed >> 40) as f32 / (1u64 << 24) as f32)
        };
        let cases: [(&str, &[u32]); 4] = [
            ("t8192", &[8192]),
            ("t1000", &[1000]),
            ("t1", &[1]),
            ("var4", &[300, 1, 700, 2048]),
        ];
        for (name, lens) in cases {
            let t: usize = lens.iter().map(|&l| l as usize).sum();
            let n = lens.len();
            let elems = [
                t * 2048,
                t * 2048,
                t * 6144,
                t * 6144,
                t * 48,
                t * 48,
                n * 48 * 128 * 128,
                n * 48 * 128 * 128,
            ];
            let mut bufs = Vec::new();
            for (i, &e) in elems.iter().enumerate() {
                let bytes: Vec<u8> = match i {
                    0 | 1 => (0..e)
                        .flat_map(|_| ((next(-0.15, 0.15).to_bits() >> 16) as u16).to_le_bytes())
                        .collect(),
                    2 => (0..e)
                        .flat_map(|_| ((next(-1.0, 1.0).to_bits() >> 16) as u16).to_le_bytes())
                        .collect(),
                    3 | 6 => vec![0u8; e * if i == 3 { 2 } else { 4 }],
                    4 => (0..e).flat_map(|_| next(0.9, 1.0).to_le_bytes()).collect(),
                    5 => (0..e).flat_map(|_| next(0.0, 1.0).to_le_bytes()).collect(),
                    _ => (0..e)
                        .flat_map(|_| next(-0.01, 0.01).to_le_bytes())
                        .collect(),
                };
                std::fs::write(dir.join(format!("{name}.in{i}")), &bytes).unwrap();
                let m = be.alloc(0, bytes.len() as u64)?;
                be.upload(&m, 0, &bytes)?;
                bufs.push((m, bytes.len()));
            }
            let views: [DeviceMem; 8] =
                std::array::from_fn(|i| DeviceMem::view(bufs[i].0.base, bufs[i].0.len));
            let run = |g: &mut NativeGdn| -> Result<()> {
                // SAFETY: the views outlive the synchronize below; one stream.
                unsafe {
                    if n == 1 {
                        g.launch(views.each_ref(), t, &stream)
                    } else {
                        g.launch_varlen(views.each_ref(), lens, &stream)
                    }
                }
            };
            run(&mut g)?;
            be.synchronize()?;
            for i in [3, 6, 7] {
                let mut out = vec![0u8; bufs[i].1];
                be.download(&bufs[i].0, 0, &mut out)?;
                std::fs::write(dir.join(format!("{name}.out{i}")), &out).unwrap();
            }
            let start = std::time::Instant::now();
            for _ in 0..20 {
                run(&mut g)?;
                if n > 1 {
                    be.synchronize()?;
                }
            }
            be.synchronize()?;
            println!(
                "GDN_AOT {name} tokens={t} nseq={n} us_per_launch={:.1}",
                start.elapsed().as_secs_f64() * 1e6 / 20.0
            );
        }
        Ok(())
    }

    #[test]
    fn accepts_token_endpoints_and_readonly_aliases() {
        let mut t = tensors();
        assert!(validate(&t.each_ref(), 1).is_ok());
        assert!(validate(&t.each_ref(), MAX_TOKENS).is_ok());
        t[1] = DeviceMem::view(t[0].base, t[0].len);
        assert!(validate(&t.each_ref(), MAX_TOKENS).is_ok());
    }

    #[test]
    fn rejects_invalid_launch_ranges_before_ffi() {
        let mut t = tensors();
        assert!(validate(&t.each_ref(), 0).is_err());
        assert!(validate(&t.each_ref(), MAX_TOKENS + 1).is_err());
        t[2].len = 12287;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[0].base += 2;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[7].base = t[6].base;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[3].base = t[0].base + 16;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[0].base = u64::MAX - 15;
        assert!(validate(&t.each_ref(), 1).is_err());
    }
}
