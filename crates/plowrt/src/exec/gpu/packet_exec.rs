//! A self-contained packet (weights embedded, no language-model step protocol) on the CUDA
//! interpreter: the `PacketRuntime` for forward pipelines such as the Qwen3-ASR audio encoder.
//! Every program runs as one cooperative launch per global-queue segment of the speech object
//! (`interp_sm90a_speech.cubin`, `interp_sm120_speech.cubin` or `interp_sm89_speech.cubin`, the
//! interpreter with the FP32 speech arms).

use std::path::Path;
use std::sync::Arc;

use packet::dev::{DevProgram, CTR_STRIDE};

use super::{pod_bytes, BLOCK};
use crate::asset::devblob::DevBlob;
use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, GraphExec, KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::exec::packet_runtime::{
    check_copy, check_region, check_transfer, PacketBuffer, PacketData, PacketRuntime,
    PacketTensor, PacketWrite,
};
use crate::{Result, RuntimeError};

/// The speech object and its entry symbol for a compute capability.
pub(super) fn speech_object(cc: (u32, u32)) -> (&'static str, &'static str) {
    match cc {
        (8, 9) => (
            "interp_sm89_speech.cubin",
            "_Z18interp_sm89_speech11PlowProgram",
        ),
        (10, _) | (12, _) => (
            "interp_sm120_speech.cubin",
            "_Z19interp_sm120_speech11PlowProgram",
        ),
        _ => (
            "interp_sm90a_speech.cubin",
            "_Z19interp_sm90a_speech11PlowProgram",
        ),
    }
}
fn in_range(mem: &DeviceMem, offset: usize, bytes: usize) -> Result<()> {
    if offset as u64 + bytes as u64 > mem.len {
        return Err(RuntimeError::Device(format!(
            "packet transfer out of range: off {offset} + {bytes} > {}",
            mem.len
        )));
    }
    Ok(())
}

/// FP32 speech ops are 163..=204, bit `op - 163` of `PLOW_SPEECH_OPS` (`op_speech_f32.cuh`).
const SPEECH_OP0: u32 = 163;

fn speech_ops(blob: &DevBlob) -> u64 {
    blob.progs
        .iter()
        .flat_map(|p| &p.insts)
        .map(|d| u32::from(d.op))
        .filter(|op| (SPEECH_OP0..=204).contains(op))
        .fold(0, |mask, op| mask | 1 << (op - SPEECH_OP0))
}

/// Speech-object capability bits the instructions need (`plow_speech_caps_speech`): bit 0 a
/// GemvF32Out row view (flag bit 12), bit 1 an AttentionF32 query view (flag bit 6).
pub(super) fn speech_caps_required<'a>(
    insts: impl IntoIterator<Item = &'a packet::dev::DevInst64>,
) -> u32 {
    use packet::dev::DevOp;
    insts.into_iter().fold(0, |caps, d| {
        caps | match DevOp::from_u16(d.op) {
            Some(DevOp::GemvF32Out) if d.i[7] & 4096 != 0 => 1,
            Some(DevOp::AttentionF32) if d.i[6] & 64 != 0 => 2,
            _ => 0,
        }
    })
}

/// The capability bits `required` that an object reporting `compiled` (`None` = no symbol) lacks.
pub(super) fn missing_speech_caps(required: u32, compiled: Option<u32>) -> u32 {
    required & !compiled.unwrap_or(0)
}

pub(super) fn missing_speech_ops(required: u64, compiled: u64) -> Vec<u32> {
    let missing = required & !compiled;
    (0..64)
        .filter(|b| missing >> b & 1 != 0)
        .map(|b| b + SPEECH_OP0)
        .collect()
}

#[cfg(test)]
#[test]
fn a_specialized_speech_object_must_cover_the_packet_ops() {
    assert!(missing_speech_ops(0b1011, 0b1111).is_empty());
    assert_eq!(missing_speech_ops(0b1011, 0b0011), vec![SPEECH_OP0 + 3]);
    assert_eq!(missing_speech_ops(1 << 41, 0), vec![204]);
}

#[cfg(test)]
#[test]
fn test_speech_object_resolution() {
    assert_eq!(speech_object((8, 9)).0, "interp_sm89_speech.cubin");
    assert_eq!(speech_object((9, 0)).0, "interp_sm90a_speech.cubin");
    assert_eq!(speech_object((10, 0)).0, "interp_sm120_speech.cubin");
    assert_eq!(speech_object((12, 0)).0, "interp_sm120_speech.cubin");
}

#[cfg(test)]
#[test]
fn speech_object_symbols_follow_the_cmake_mangling() {
    for cc in [(8, 9), (9, 0), (10, 0), (12, 0)] {
        let (file, symbol) = speech_object(cc);
        let stem = file.strip_suffix(".cubin").unwrap();
        assert_eq!(symbol, format!("_Z{}{stem}11PlowProgram", stem.len()));
    }
}

struct Program {
    kernarg: DevProgram,
    segments: usize,
    counters: u64,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

pub struct CudaPacketRuntime {
    be: Arc<CudaBackend>,
    stream: CudaStream,
    events: (CudaEvent, CudaEvent),
    module: Module,
    function: KernelFn,
    grid: u32,
    smem: u32,
    names: Vec<String>,
    tensors: Vec<DeviceMem>,
    _table: DeviceMem,
    /// Every program's counters and cursors, in program order: a run of consecutive programs
    /// zeroes its counters with one memset.
    _counters: DeviceMem,
    programs: Vec<Program>,
    /// One CUDA graph per program sequence (every pointer is fixed at load, so a capture stays
    /// valid); `None` once capture failed for that sequence.
    graphs: std::collections::HashMap<Vec<usize>, Option<GraphExec>>,
    last_us: f64,
    input_stage: Option<PinnedHost>,
    input_failed: bool,
    /// Last field: dropped after every allocation above, so [`resident_in`] turns false only once
    /// the device memory is back.
    _resident: Arc<()>,
}

type Residents = parking_lot::Mutex<Vec<(std::path::PathBuf, u8, std::sync::Weak<()>)>>;

/// Live runtimes by the asset directory their packet sits in and their device.
fn residents() -> &'static Residents {
    static R: std::sync::OnceLock<Residents> = std::sync::OnceLock::new();
    R.get_or_init(Default::default)
}

/// Whether a packet runtime loaded from `dir` onto `device` still holds device memory.
pub fn resident_in(dir: &Path, device: u8) -> bool {
    let mut r = residents().lock();
    r.retain(|(_, _, live)| live.strong_count() > 0);
    r.iter().any(|(d, o, _)| d == dir && *o == device)
}

// SAFETY: the runtime owns its stream, module and allocations and is used by one thread at a time
// (`PacketRuntime: Send`, every method takes the runtime by reference).
unsafe impl Send for CudaPacketRuntime {}

impl CudaPacketRuntime {
    pub fn load(path: &Path, device: u8) -> Result<Self> {
        let be = Arc::new(CudaBackend::new(device)?);
        Self::load_on(be, path)
    }

    pub fn load_on(be: Arc<CudaBackend>, path: &Path) -> Result<Self> {
        let ordinal = be.device_ordinal;
        let image = crate::asset::packet_image::PacketImage::open(path)?;
        image.blob.check_l2_dispatch(false)?;
        let blob = &image.blob;
        if !blob.gen.is_empty() || blob.tp.is_some() {
            return Err(RuntimeError::Rejected(format!(
                "{}: generated tensors or TP are not supported by the CUDA packet runtime",
                path.display()
            )));
        }
        let dir = path.parent().unwrap_or(Path::new("."));
        let (file, symbol) = speech_object(be.compute_capability());
        let object = dir.join(file);
        let image = std::fs::read(&object).map_err(|source| RuntimeError::Io {
            path: object.clone(),
            source,
        })?;
        let module = be.module_load(&image)?;
        let function = be.get_function(&module, symbol)?;
        if let (Some(lo), Some(hi)) = (
            be.module_global_u32(&module, "plow_speech_ops_lo_speech")?,
            be.module_global_u32(&module, "plow_speech_ops_hi_speech")?,
        ) {
            let ops = missing_speech_ops(speech_ops(&blob), u64::from(hi) << 32 | u64::from(lo));
            if !ops.is_empty() {
                return Err(RuntimeError::Rejected(format!(
                    "{}: {} lacks speech ops {ops:?}; rebuild it for this packet",
                    path.display(),
                    object.display()
                )));
            }
        }
        let caps = missing_speech_caps(
            speech_caps_required(blob.progs.iter().flat_map(|p| &p.insts)),
            be.module_global_u32(&module, "plow_speech_caps_speech")?,
        );
        if caps != 0 {
            return Err(RuntimeError::Rejected(format!(
                "{}: {} lacks speech capabilities {caps:#x}; rebuild it for this packet",
                path.display(),
                object.display()
            )));
        }
        let smem = be
            .module_global_u32(&module, "plow_arena_bytes_speech")?
            .unwrap_or(49152);
        if let Some(limit) = crate::config::RuntimeConfig::debug_max_inst() {
            if be.module_global_set_u32(&module, "plow_debug_max_inst_speech", limit)? {
                tracing::warn!(
                    limit,
                    "packet runtime: instructions at or past the limit are skipped"
                );
            }
        }
        be.set_max_dynamic_smem(function, smem)?;
        let capacity = be.occupancy_blocks_per_sm(function, BLOCK, smem as usize)? * be.sm_count();
        if blob.n_cu == 0 || blob.n_cu > capacity {
            return Err(RuntimeError::Rejected(format!(
                "{}: n_cu {} exceeds cooperative capacity {capacity}",
                path.display(),
                blob.n_cu
            )));
        }

        let mut tensors = Vec::with_capacity(blob.tensors.len());
        for t in &blob.tensors {
            let mem = be.alloc(0, t.bytes.max(4))?;
            match &t.init {
                Some(r) => be.upload(&mem, 0, &blob.init[r.clone()])?,
                None => be.memset_d8(mem.base, 0, t.bytes as usize)?,
            }
            tensors.push(mem);
        }
        let ptrs: Vec<u64> = tensors.iter().map(|m| m.base).collect();
        let table = be.alloc(0, (ptrs.len() * 8).max(8) as u64)?;
        be.upload(&table, 0, pod_bytes(&ptrs))?;

        let regions: Vec<(usize, usize)> = blob
            .progs
            .iter()
            .map(|p| {
                let segments = p.gq_seg_ofs.len().saturating_sub(1);
                (
                    (p.n_counter as usize * CTR_STRIDE as usize * 4).max(4),
                    segments * CTR_STRIDE as usize * 4,
                )
            })
            .collect();
        let slab_bytes: usize = regions
            .iter()
            .map(|&(c, k)| (c + k).next_multiple_of(256))
            .sum();
        let counter_slab = be.alloc(0, slab_bytes.max(4) as u64)?;
        let mut slab_offset = 0u64;
        let mut programs = Vec::with_capacity(blob.progs.len());
        for (p, &(counter_only, cursor_bytes)) in blob.progs.iter().zip(&regions) {
            if p.gq_seg_ofs.len() < 2 || p.l2_domains != 0 {
                return Err(RuntimeError::Rejected(format!(
                    "{}: program needs an unplaced global-queue stream",
                    path.display()
                )));
            }
            let segments = p.gq_seg_ofs.len() - 1;
            let upload = |bytes: &[u8]| -> Result<DeviceMem> {
                let mem = be.alloc(0, bytes.len().max(4) as u64)?;
                if !bytes.is_empty() {
                    be.upload(&mem, 0, bytes)?;
                }
                Ok(mem)
            };
            let d_inst = upload(pod_bytes(&p.insts))?;
            let d_stream = upload(pod_bytes(&p.stream))?;
            let d_sofs = upload(pod_bytes(&p.stream_ofs))?;
            let d_slen = upload(pod_bytes(&p.stream_len))?;
            let d_waits = upload(pod_bytes(&p.waits))?;
            let d_succs = upload(pod_bytes(&p.succs))?;
            let d_gq_stream = upload(pod_bytes(&p.gq_stream))?;
            let d_gq_seg = upload(pod_bytes(&p.gq_seg_ofs))?;
            let counters = counter_slab.base + slab_offset;
            let cursors = counters + counter_only as u64;
            slab_offset += (counter_only + cursor_bytes).next_multiple_of(256) as u64;
            let kernarg = DevProgram {
                insts: d_inst.base,
                stream: d_stream.base,
                stream_ofs: d_sofs.base,
                stream_len: d_slen.base,
                waits: d_waits.base,
                succs: d_succs.base,
                counters,
                tensors: table.base,
                trace: 0,
                cur_seg: 0,
                l2_domains: 0,
                hier_base: 0,
                n_seg: 1,
                gq_stream: d_gq_stream.base,
                gq_seg_ofs: d_gq_seg.base,
                gq_cursor: cursors,
                xctr: 0,
                peer_scratch: 0,
                rank: 0,
                n_gpu: 1,
                seg_ofs: 0,
                prefill_spans: 0,
                prefill_parked: 0,
                n_prefill_spans: 0,
                n_prefill_rows: 0,
                token_batch: 0,
            };
            programs.push(Program {
                kernarg,
                segments,
                counters,
                counter_bytes: counter_only + cursor_bytes,
                _tables: vec![
                    d_inst,
                    d_stream,
                    d_sofs,
                    d_slen,
                    d_waits,
                    d_succs,
                    d_gq_stream,
                    d_gq_seg,
                ],
            });
        }
        // The init copies and memsets ran on the legacy stream, which the non-blocking packet
        // stream does not wait for.
        be.synchronize()?;
        let stream = be.stream_create()?;
        let events = (be.event_create(true)?, be.event_create(true)?);
        tracing::info!(packet = %path.display(), programs = programs.len(), tensors = tensors.len(), grid = blob.n_cu, smem, "cuda packet runtime loaded");
        Ok(CudaPacketRuntime {
            be,
            stream,
            events,
            module,
            function,
            grid: blob.n_cu,
            smem,
            names: blob.tensors.iter().map(|t| t.name.clone()).collect(),
            tensors,
            _table: table,
            _counters: counter_slab,
            programs,
            graphs: std::collections::HashMap::new(),
            last_us: 0.0,
            input_stage: None,
            input_failed: false,
            _resident: {
                let token = Arc::new(());
                residents()
                    .lock()
                    .push((dir.to_path_buf(), ordinal, Arc::downgrade(&token)));
                token
            },
        })
    }

    /// Skip every instruction at or past `limit` (per-instruction cost by caps).
    pub fn set_debug_max_inst(&self, limit: u32) -> Result<bool> {
        self.be
            .module_global_set_u32(&self.module, "plow_debug_max_inst_speech", limit)
    }

    pub fn backend(&self) -> &Arc<CudaBackend> {
        &self.be
    }

    /// Device address of a tensor, for device-to-device handoff to another engine on the same GPU.
    pub fn device_ptr(&self, tensor: PacketTensor) -> Option<u64> {
        self.tensors.get(tensor.handle).map(|m| m.base)
    }

    fn enqueue(&self, programs: &[usize]) -> Result<()> {
        let consecutive = programs.windows(2).all(|w| w[1] == w[0] + 1);
        if let (true, Some(&first), Some(&last)) = (consecutive, programs.first(), programs.last())
        {
            let (first, last) = (&self.programs[first], &self.programs[last]);
            let bytes = (last.counters + last.counter_bytes as u64 - first.counters) as usize;
            self.be
                .memset_d8_async(first.counters, 0, bytes, &self.stream)?;
        }
        for &program in programs {
            let p = &self.programs[program];
            if !consecutive {
                self.be
                    .memset_d8_async(p.counters, 0, p.counter_bytes, &self.stream)?;
            }
            for seg in 0..p.segments {
                let mut arg = p.kernarg;
                arg.gq_seg_ofs += (seg * 4) as u64;
                arg.gq_cursor += (seg * CTR_STRIDE as usize * 4) as u64;
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                self.be.launch_cooperative(
                    self.function,
                    self.grid,
                    BLOCK,
                    self.smem,
                    &mut params,
                    Some(&self.stream),
                )?;
            }
        }
        Ok(())
    }

    /// Every transfer goes through the packet stream: it is NON_BLOCKING, so a legacy-stream copy
    /// is not ordered before the next program (a DtoD copy returns before it lands).
    fn upload(&self, mem: &DeviceMem, offset: usize, bytes: &[u8]) -> Result<()> {
        in_range(mem, offset, bytes.len())?;
        // SAFETY: `bytes` outlives the copy (synchronized below); the range is inside `mem`.
        unsafe {
            self.be
                .memcpy_htod_async(mem.base + offset as u64, bytes, &self.stream)?
        };
        self.be.stream_synchronize(&self.stream)
    }

    fn download(&self, mem: &DeviceMem, offset: usize, bytes: &mut [u8]) -> Result<()> {
        in_range(mem, offset, bytes.len())?;
        // SAFETY: as `upload`.
        unsafe {
            self.be
                .memcpy_dtoh_async(bytes, mem.base + offset as u64, &self.stream)?
        };
        self.be.stream_synchronize(&self.stream)
    }

    fn mem(&self, t: PacketTensor) -> Result<&DeviceMem> {
        self.check_inputs()?;
        self.tensors.get(t.handle).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {} is missing", t.handle))
        })
    }

    fn check_inputs(&self) -> Result<()> {
        if self.input_failed {
            return Err(RuntimeError::Device(
                "packet input copies did not retire".into(),
            ));
        }
        Ok(())
    }

    fn reserve_inputs(&mut self, size: usize) -> Result<()> {
        if size > 0 && self.input_stage.as_ref().is_none_or(|s| s.len() < size) {
            self.input_stage = Some(self.be.host_alloc_pinned(size)?);
        }
        Ok(())
    }

    fn abandon_transfers(
        &mut self,
        tensors: impl IntoIterator<Item = PacketTensor>,
        error: RuntimeError,
    ) -> RuntimeError {
        self.input_failed = true;
        std::mem::forget(self.be.clone());
        std::mem::forget(self._resident.clone());
        for tensor in tensors {
            self.tensors[tensor.handle].quarantine();
        }
        #[cfg(target_os = "linux")]
        {
            self.be.quarantine_staging(&error)
        }
        #[cfg(not(target_os = "linux"))]
        {
            error
        }
    }
}

fn input_size(mut sizes: impl Iterator<Item = usize>) -> Result<usize> {
    sizes.try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .and_then(|n| n.checked_add(7))
            .map(|n| n & !7)
            .ok_or_else(|| RuntimeError::Device("packet input size overflow".into()))
    })
}

fn device_input_source(be: &CudaBackend, data: PacketData<'_>) -> Option<u64> {
    match data {
        PacketData::Buffer(source) => source
            .cuda_source()
            .filter(|(b, _)| be.shares_context(b))
            .map(|(_, address)| address),
        _ => None,
    }
}

fn host_input_size(be: &CudaBackend, data: PacketData<'_>) -> usize {
    match data {
        PacketData::Zero(_) => 0,
        _ if device_input_source(be, data).is_some() => 0,
        _ => data.len(),
    }
}

fn submit_writes(
    count: usize,
    mut enqueue: impl FnMut(usize) -> Result<()>,
    drain: impl FnOnce() -> Result<()>,
    quarantine: impl FnOnce(),
) -> Result<()> {
    let status = (0..count).try_for_each(&mut enqueue);
    if let Err(error) = drain() {
        quarantine();
        return Err(error);
    }
    status
}

#[cfg(test)]
mod write_tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[test]
    #[ignore = "GPU correctness; run through gpuq"]
    fn packet_pool_reuses_released_slots_and_preserves_held_views() {
        use crate::exec::packet_runtime::PacketBufferPool;
        let backend = Arc::new(CudaBackend::new(0).unwrap());
        let stream = backend.stream_create().unwrap();
        let source = Arc::new(backend.alloc(0, 64).unwrap());
        let bytes: Vec<u8> = (0..64).map(|i| (i * 37) as u8).collect();
        backend.upload(&source, 0, &bytes).unwrap();
        let mut pool = PacketBufferPool::cuda(backend.clone(), 48, 2).unwrap();
        let first = pool.acquire(37, source.clone()).unwrap();
        let first_address = first.address();
        let second = pool.acquire(41, source.clone()).unwrap();
        assert_eq!(second.address() - first_address, 48);
        assert!(pool.acquire(1, source.clone()).is_none());
        backend
            .memcpy_dtod_async(first.address(), source.base + 3, 37, &stream)
            .unwrap();
        backend
            .memcpy_dtod_async(second.address(), source.base + 7, 41, &stream)
            .unwrap();
        backend.stream_synchronize(&stream).unwrap();
        // SAFETY: both pending copies were drained above.
        let (first, second) = unsafe { (first.retire(), second.retire()) };
        let view = first.slice(5..29).unwrap();
        drop(first);
        assert!(pool.acquire(1, source.clone()).is_none());
        let mut actual = [0; 24];
        view.read(&mut actual).unwrap();
        assert_eq!(actual, bytes[8..32]);
        drop(view);
        let reused = pool.acquire(48, source.clone()).unwrap();
        assert_eq!(reused.address(), first_address);
        backend.upload(&source, 0, &[0xdd; 64]).unwrap();
        backend
            .memcpy_dtod_async(reused.address(), source.base, 48, &stream)
            .unwrap();
        drop(source);
        backend.stream_synchronize(&stream).unwrap();
        // SAFETY: the reused slot and its retained producer have completed.
        let reused = unsafe { reused.retire() };
        drop(pool);
        let mut old = [0; 41];
        second.read(&mut old).unwrap();
        assert_eq!(old, bytes[7..48]);
        let mut new = [0; 48];
        reused.read(&mut new).unwrap();
        assert_eq!(new, [0xdd; 48]);
    }

    #[test]
    #[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS; run through gpuq"]
    fn packet_snapshot_survives_producer_reuse_and_runtime_drop() {
        let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
        let mut producer = CudaPacketRuntime::load(&assets.join("codec.pkt"), 0).unwrap();
        let resident = Arc::downgrade(&producer._resident);
        let input = producer.create_tensor(64).unwrap();
        let bytes: Vec<u8> = (0..64).map(|i| (i * 37) as u8).collect();
        producer.write_tensor(input, &bytes).unwrap();
        let snapshot = producer.snapshot_tensor(input, 7, 41).unwrap();
        let view = snapshot.slice(3..35).unwrap();
        producer.write_tensor(input, &[0x55; 64]).unwrap();
        drop(producer);
        drop(snapshot);
        assert!(resident.upgrade().is_some());
        let mut output = [0u8; 32];
        view.read(&mut output).unwrap();
        assert_eq!(&output, &bytes[10..42]);
        std::thread::spawn(move || {
            let mut consumer = CudaPacketRuntime::load(&assets.join("codec.pkt"), 0).unwrap();
            let target = consumer.create_tensor(64).unwrap();
            consumer.write_tensor(target, &[0xa5; 64]).unwrap();
            assert!(consumer.input_stage.is_none());
            consumer.write_tensor_from(target, 5, &view).unwrap();
            assert!(
                consumer.input_stage.is_none(),
                "same-context copy fell back to host staging"
            );
            let mut actual = [0u8; 64];
            consumer.read_tensor(target, &mut actual).unwrap();
            let mut expected = [0xa5; 64];
            expected[5..37].copy_from_slice(&bytes[10..42]);
            assert_eq!(actual, expected);
            let first = view.slice(0..11).unwrap();
            let second = view.slice(7..24).unwrap();
            consumer
                .write_tensors(&[
                    PacketWrite {
                        tensor: target,
                        offset: 2,
                        data: PacketData::Zero(46),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 3,
                        data: PacketData::Buffer(&first),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 9,
                        data: PacketData::Buffer(&second),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 64,
                        data: PacketData::Zero(0),
                    },
                ])
                .unwrap();
            assert!(
                consumer.input_stage.is_none(),
                "device-only batch staged on host"
            );
            expected[2..48].fill(0);
            expected[3..14].copy_from_slice(&bytes[10..21]);
            expected[9..26].copy_from_slice(&bytes[17..34]);
            consumer.read_tensor(target, &mut actual).unwrap();
            assert_eq!(actual, expected);
            for data in [PacketData::Zero(usize::MAX), PacketData::Buffer(&view)] {
                assert!(consumer
                    .write_tensors(&[
                        PacketWrite {
                            tensor: target,
                            offset: 0,
                            data: PacketData::Zero(64)
                        },
                        PacketWrite {
                            tensor: target,
                            offset: 63,
                            data
                        },
                    ])
                    .is_err());
                consumer.read_tensor(target, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            for offset in [63, usize::MAX] {
                assert!(consumer.write_tensor_from(target, offset, &view).is_err());
                consumer.read_tensor(target, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            assert!(consumer.snapshot_tensor(target, 60, 5).is_err());
            assert!(consumer
                .write_tensor_from(
                    PacketTensor {
                        bytes: 63,
                        ..target
                    },
                    0,
                    &view
                )
                .is_err());
            let host: PacketBuffer = vec![9, 8, 7].into();
            consumer
                .write_tensors(&[
                    PacketWrite {
                        tensor: target,
                        offset: 1,
                        data: PacketData::Buffer(&host),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 6,
                        data: PacketData::Buffer(&first),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 12,
                        data: PacketData::Zero(2),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 15,
                        data: PacketData::Host(&[5, 4, 3, 2, 1]),
                    },
                ])
                .unwrap();
            assert!(consumer.input_stage.is_some());
            assert_eq!(consumer.input_stage.as_ref().unwrap().len(), 16);
            expected[1..4].copy_from_slice(&[9, 8, 7]);
            expected[6..17].copy_from_slice(&bytes[10..21]);
            expected[12..14].fill(0);
            expected[15..20].copy_from_slice(&[5, 4, 3, 2, 1]);
            consumer.read_tensor(target, &mut actual).unwrap();
            assert_eq!(actual, expected);
            drop(first);
            drop(second);
            drop(view);
            assert!(resident.upgrade().is_none());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn input_writes_drain_after_success_or_any_enqueue_failure() {
        for failure in [None, Some(0), Some(1), Some(2)] {
            for drain_fails in [false, true] {
                let seen = RefCell::new(Vec::new());
                let quarantined = Cell::new(false);
                let result = submit_writes(
                    3,
                    |i| {
                        seen.borrow_mut().push(i);
                        if failure == Some(i) {
                            Err(RuntimeError::Device("enqueue".into()))
                        } else {
                            Ok(())
                        }
                    },
                    || {
                        seen.borrow_mut().push(3);
                        if drain_fails {
                            Err(RuntimeError::Device("drain".into()))
                        } else {
                            Ok(())
                        }
                    },
                    || quarantined.set(true),
                );
                let mut expected: Vec<_> = (0..failure.map_or(3, |i| i + 1)).collect();
                expected.push(3);
                assert_eq!(*seen.borrow(), expected);
                assert_eq!(quarantined.get(), drain_fails);
                assert_eq!(result.is_err(), failure.is_some() || drain_fails);
            }
        }
    }

    #[test]
    #[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS; run through gpuq"]
    fn packet_bulk_writes_preserve_ranges_order_and_reuse() {
        let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
        let mut rt = CudaPacketRuntime::load(&assets.join("codec.pkt"), 0).unwrap();
        let t = rt.create_tensor(64).unwrap();
        for size in [7, 23, 3, 47, 7] {
            let initial = [0xa5; 64];
            rt.write_tensor(t, &initial).unwrap();
            let first = vec![0x37; size];
            let writes = [
                PacketWrite {
                    tensor: t,
                    offset: 5,
                    data: PacketData::Host(&first),
                },
                PacketWrite {
                    tensor: t,
                    offset: 9,
                    data: PacketData::Host(&[1, 2, 3, 4]),
                },
                PacketWrite {
                    tensor: t,
                    offset: 64,
                    data: PacketData::Host(&[]),
                },
            ];
            rt.write_tensors(&writes).unwrap();
            let mut expected = initial;
            expected[5..5 + size].copy_from_slice(&first);
            expected[9..13].copy_from_slice(&[1, 2, 3, 4]);
            let mut actual = [0; 64];
            rt.read_tensor(t, &mut actual).unwrap();
            assert_eq!(actual, expected);
            assert!(rt
                .write_tensors(&[
                    PacketWrite {
                        tensor: t,
                        offset: 0,
                        data: PacketData::Host(&[0xff]),
                    },
                    PacketWrite {
                        tensor: t,
                        offset: usize::MAX,
                        data: PacketData::Host(&[1]),
                    },
                ])
                .is_err());
            rt.read_tensor(t, &mut actual).unwrap();
            assert_eq!(actual, expected, "validation must precede all transfers");
            assert!(rt
                .write_tensors(&[PacketWrite {
                    tensor: PacketTensor { bytes: 63, ..t },
                    offset: 0,
                    data: PacketData::Host(&[1]),
                }])
                .is_err());
        }
        rt.prepare_writes(&[t, t, t]).unwrap();
        let pointer = rt.input_stage.as_ref().unwrap().as_ptr();
        rt.write_tensors(&[PacketWrite {
            tensor: t,
            offset: 0,
            data: PacketData::Host(&[0xa5; 64]),
        }])
        .unwrap();
        assert_eq!(
            pointer,
            rt.input_stage.as_ref().unwrap().as_ptr(),
            "prepared input staging grew"
        );
        assert!(input_size([usize::MAX].into_iter()).is_err());
    }
}

impl PacketRuntime for CudaPacketRuntime {
    fn tensor(&self, name: &str) -> Option<PacketTensor> {
        let handle = self.names.iter().position(|n| n == name)?;
        Some(PacketTensor {
            handle,
            bytes: self.tensors[handle].len as usize,
        })
    }

    fn write_tensors(&mut self, writes: &[PacketWrite<'_>]) -> Result<()> {
        self.check_inputs()?;
        for w in writes {
            let mem = self.mem(w.tensor)?;
            check_transfer(
                w.tensor,
                w.tensor.handle,
                mem.len as usize,
                mem.len as usize,
            )?;
            w.check()?;
        }
        if writes.iter().all(|w| w.data.is_empty()) {
            return Ok(());
        }
        let size = input_size(writes.iter().map(|w| host_input_size(&self.be, w.data)))?;
        self.reserve_inputs(size)?;
        let mut offset = 0;
        for w in writes {
            let len = host_input_size(&self.be, w.data);
            if len > 0 {
                let stage = self.input_stage.as_mut().unwrap();
                w.data
                    .read(&mut stage.as_mut_slice()[offset..offset + len])?;
                offset = (offset + len + 7) & !7;
            }
        }
        let mut offset = 0;
        let mut failed = false;
        let result = submit_writes(
            writes.len(),
            |i| {
                let w = &writes[i];
                let len = w.data.len();
                if len == 0 {
                    return Ok(());
                }
                let target = self.tensors[w.tensor.handle].base + w.offset as u64;
                if let PacketData::Zero(_) = w.data {
                    return self.be.memset_d8_async(target, 0, len, &self.stream);
                }
                if let Some(source) = device_input_source(&self.be, w.data) {
                    return self
                        .be
                        .memcpy_dtod_async(target, source, len as u64, &self.stream);
                }
                let stage = self.input_stage.as_ref().unwrap();
                let source = &stage.as_slice()[offset..offset + len];
                offset = (offset + len + 7) & !7;
                // SAFETY: validated ranges, owned pinned source, and the batch drains below.
                unsafe { self.be.memcpy_htod_async(target, source, &self.stream) }
            },
            || self.be.stream_synchronize(&self.stream),
            || failed = true,
        );
        if failed {
            // Unproven completion must retain buffers and the backend's small-copy kernel.
            if let Some(stage) = self.input_stage.take() {
                std::mem::forget(stage);
            }
            for w in writes {
                if let PacketData::Buffer(source) = w.data {
                    std::mem::forget(source.clone());
                }
            }
            return Err(
                self.abandon_transfers(writes.iter().map(|w| w.tensor), result.unwrap_err())
            );
        }
        result
    }

    fn prepare_writes(&mut self, tensors: &[PacketTensor]) -> Result<()> {
        self.check_inputs()?;
        for &t in tensors {
            let mem = self.mem(t)?;
            check_transfer(t, t.handle, mem.len as usize, mem.len as usize)?;
        }
        self.reserve_inputs(input_size(tensors.iter().map(|t| t.bytes))?)
    }

    fn snapshot_tensor(
        &mut self,
        tensor: PacketTensor,
        offset: usize,
        bytes: usize,
    ) -> Result<PacketBuffer> {
        let source = self.mem(tensor)?;
        check_transfer(
            tensor,
            tensor.handle,
            source.len as usize,
            source.len as usize,
        )?;
        check_region(tensor, offset, bytes)?;
        if bytes == 0 {
            return Ok(Vec::new().into());
        }
        let address = source.base + offset as u64;
        let mut snapshot = self.be.alloc(0, bytes as u64)?;
        let mut failed = false;
        let result = submit_writes(
            1,
            |_| {
                self.be
                    .memcpy_dtod_async(snapshot.base, address, bytes as u64, &self.stream)
            },
            || self.be.stream_synchronize(&self.stream),
            || failed = true,
        );
        if failed {
            snapshot.quarantine();
            return Err(self.abandon_transfers([tensor], result.unwrap_err()));
        }
        result?;
        Ok(PacketBuffer::completed_cuda(
            snapshot,
            self.be.clone(),
            bytes,
            self._resident.clone(),
        ))
    }

    fn write_tensor(&mut self, tensor: PacketTensor, bytes: &[u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, bytes.len())?;
        self.upload(mem, 0, bytes)
    }

    fn read_tensor(&self, tensor: PacketTensor, bytes: &mut [u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, bytes.len())?;
        self.download(mem, 0, bytes)
    }

    fn write_tensor_at(&mut self, tensor: PacketTensor, offset: usize, bytes: &[u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, mem.len as usize)?;
        self.upload(mem, offset, bytes)
    }

    fn read_tensor_at(&self, tensor: PacketTensor, offset: usize, bytes: &mut [u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, mem.len as usize)?;
        self.download(mem, offset, bytes)
    }

    fn copy_tensor(
        &mut self,
        source: PacketTensor,
        source_offset: usize,
        target: PacketTensor,
        target_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let (s, t) = (self.mem(source)?, self.mem(target)?);
        check_copy(
            source,
            s.len as usize,
            source_offset,
            target,
            t.len as usize,
            target_offset,
            bytes,
        )?;
        self.be.memcpy_dtod_async(
            t.base + target_offset as u64,
            s.base + source_offset as u64,
            bytes as u64,
            &self.stream,
        )
    }

    fn create_tensor(&mut self, bytes: usize) -> Result<PacketTensor> {
        let mem = self.be.alloc(0, (bytes as u64).max(4))?;
        self.be
            .memset_d8_async(mem.base, 0, mem.len as usize, &self.stream)?;
        self.be.stream_synchronize(&self.stream)?;
        self.tensors.push(mem);
        // Unnamed: `tensor(name)` never resolves it; only its handle does.
        self.names.push(String::new());
        Ok(PacketTensor {
            handle: self.tensors.len() - 1,
            bytes,
        })
    }

    fn run(&mut self, program: usize) -> Result<()> {
        self.run_sequence(&[program])
    }

    /// Every program is enqueued on the one stream (stream order is the dependency), as one
    /// CUDA graph after the first run of a sequence; one sync.
    fn run_sequence(&mut self, programs: &[usize]) -> Result<()> {
        self.check_inputs()?;
        if let Some(&bad) = programs.iter().find(|&&p| p >= self.programs.len()) {
            return Err(RuntimeError::Rejected(format!(
                "packet program {bad} is missing"
            )));
        }
        if !self.graphs.contains_key(programs) {
            let graph = match self
                .be
                .graph_capture(&self.stream, || self.enqueue(programs))
            {
                Ok(g) => Some(g),
                Err(error) => {
                    tracing::warn!(%error, programs = programs.len(), "packet sequence graph capture failed; launching directly");
                    None
                }
            };
            self.graphs.insert(programs.to_vec(), graph);
        }
        self.be.event_record(&self.events.0, &self.stream)?;
        match self.graphs.get(programs).and_then(Option::as_ref) {
            Some(graph) => self.be.graph_launch(graph, &self.stream)?,
            None => self.enqueue(programs)?,
        }
        self.be.event_record(&self.events.1, &self.stream)?;
        self.be.stream_synchronize(&self.stream)?;
        self.last_us = f64::from(self.be.event_elapsed_ms(&self.events.0, &self.events.1)?) * 1e3;
        Ok(())
    }

    fn last_run_us(&self) -> f64 {
        self.last_us
    }
}
